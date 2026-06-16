
import os
from pathlib import Path
REPO_ROOT = Path(os.environ.get("THESIS_ROOT", Path(__file__).resolve().parents[2]))
import os
import json
import time
import hashlib
import warnings
from datetime import datetime
import random
import numpy as np
import pandas as pd
import torchaudio
from faster_whisper import WhisperModel
from transformers import pipeline
from transformers.utils import logging as hf_logging
import evaluate
import torch
from tqdm.auto import tqdm

# =========================
# Repro & seeds
# =========================
seed = 42
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed); torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)

# =========================
# Silence noisy warnings
# =========================
os.environ["TOKENIZERS_PARALLELISM"] = "false"
hf_logging.set_verbosity_error()
warnings.filterwarnings(
    "ignore",
    message="In 2.9, this function's implementation will be changed to use torchaudio.load_with_torchcodec",
    category=UserWarning,
    module="torchaudio._backend.utils",
)
warnings.filterwarnings(
    "ignore",
    message=".*deprecated.*Stream.*",
    category=UserWarning,
    module="torchaudio._backend.ffmpeg",
)
warnings.filterwarnings("ignore", category=pd.errors.DtypeWarning)

# =========================
# Hardcoded variables (adapt paths as needed)
# =========================
DATA_FILES = [
    str(REPO_ROOT / "data_preparation/merged_datasets_plus_de/merged_train.tsv"),
    str(REPO_ROOT / "data_preparation/merged_datasets_plus_de/merged_valid.tsv"),
    str(REPO_ROOT / "data_preparation/merged_datasets_plus_de/merged_test.tsv"),
]

OUT_DIR = str(REPO_ROOT / "asr/bench_outputs")
os.makedirs(OUT_DIR, exist_ok=True)

# === Only large-v3-turbo in both backends ===
MODELS = {
    # CTranslate2 conversion of openai/whisper-large-v3-turbo for faster-whisper
    "faster-whisper:large-v3-turbo": {"type": "faster", "id": "h2oai/faster-whisper-large-v3-turbo"},
    # Original Transformers checkpoint
    "hf:whisper:openai/whisper-large-v3-turbo": {"type": "hf", "id": "openai/whisper-large-v3-turbo"},
}

MAX_SAMPLES_PER_LABEL = 300
SAMPLE_RATE = 16000

def pick_device_for_hf():
    try:
        if torch.cuda.is_available() and torch.backends.cudnn.is_available():
            return 0
    except Exception:
        pass
    return -1

HF_DEVICE = pick_device_for_hf()

# =========================
# Load & balance dataset (deterministic)
# =========================
print("Loading and preparing dataset...")
read_opts = dict(sep="\t", low_memory=False, dtype=str, quoting=3, on_bad_lines="skip")
dfs = [pd.read_csv(f, **read_opts) for f in DATA_FILES]
df = pd.concat(dfs, ignore_index=True)
df = df[["audio_path", "sentence", "label_id"]].dropna()

print(f"Total records before filtering: {len(df)}")

exists_mask = df["audio_path"].apply(os.path.isfile)
missing_count = (~exists_mask).sum()
if missing_count:
    print(f"⚠️ Skipping {missing_count} rows with missing files.")
df = df[exists_mask].reset_index(drop=True)

print(f"Records after file existence filtering: {len(df)}")

# Deterministic balanced sampling
df = df.sort_values(['label_id', 'audio_path']).reset_index(drop=True)
parts = []
sampling_stats = {}
for lab in sorted(df['label_id'].unique()):
    grp = df[df['label_id'] == lab]
    k = min(len(grp), MAX_SAMPLES_PER_LABEL)
    if k == 0:
        continue
    sampled = grp.sample(n=k, random_state=42).sort_values('audio_path')
    parts.append(sampled)
    sampling_stats[lab] = {'available': len(grp), 'sampled': k}
    print(f"Label {lab}: {k}/{len(grp)} samples")

if not parts:
    raise ValueError("No samples available after filtering!")

df_balanced = pd.concat(parts, ignore_index=True)
print(f"Final balanced dataset size: {len(df_balanced)}")

records = df_balanced.sort_values(['label_id', 'audio_path']).to_dict(orient="records")

# Dataset fingerprint
fp_hasher = hashlib.sha256()
for ex in sorted(records, key=lambda x: (x["label_id"], x["audio_path"])):
    fp_hasher.update((ex["audio_path"] + "\t" + ex["sentence"] + "\t" + str(ex["label_id"]) + "\n").encode("utf-8"))
dataset_fingerprint = fp_hasher.hexdigest()
print(f"Dataset fingerprint: {dataset_fingerprint[:16]}…")
print(f"This ensures both backends test the same {len(records)} sentences.\n")

# =========================
# Metrics
# =========================
wer_metric   = evaluate.load("wer")
cer_metric   = evaluate.load("cer")
bleu_metric  = evaluate.load("sacrebleu")
chrf_metric  = evaluate.load("chrf")
ter_metric   = evaluate.load("ter")

def _to_float_score(x):
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, dict):
        for k in ("score", "ter", "TER", "value"):
            if k in x:
                try:
                    return float(x[k])
                except Exception:
                    pass
    return float("nan")

def compute_all_metrics(preds, refs):
    if not preds or not refs or len(preds) != len(refs):
        return {"WER": float("nan"), "CER": float("nan"), "sacreBLEU": float("nan"), "chrF": float("nan"), "TER": float("nan")}
    refs_nested = [[r] for r in refs]
    try:
        wer   = _to_float_score(wer_metric.compute(predictions=preds, references=refs))
        cer   = _to_float_score(cer_metric.compute(predictions=preds, references=refs))
        bleu  = _to_float_score(bleu_metric.compute(predictions=preds, references=refs_nested))
        chrf  = _to_float_score(chrf_metric.compute(predictions=preds, references=refs))
        ter   = _to_float_score(ter_metric.compute(predictions=preds, references=refs))
        return {"WER": wer, "CER": cer, "sacreBLEU": bleu, "chrF": chrf, "TER": ter}
    except Exception as e:
        print(f"⚠️ Metric computation failed: {e}")
        return {"WER": float("nan"), "CER": float("nan"), "sacreBLEU": float("nan"), "chrF": float("nan"), "TER": float("nan")}

# =========================
# Audio loader
# =========================
def load_audio(file_path):
    wav, sr = torchaudio.load(file_path)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != SAMPLE_RATE:
        wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
    return wav.squeeze().numpy()

def load_faster_whisper(model_id):
    try:
        print(f"  Loading faster-whisper on CUDA...")
        return WhisperModel(model_id, device="cuda", compute_type="float16")
    except Exception as e:
        print(f"  CUDA failed ({e}), falling back to CPU (int8)...")
        return WhisperModel(model_id, device="cpu", compute_type="int8")

# =========================
# Run
# =========================
run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
results = {}
skipped_rows = []
total_tasks = len(MODELS) * len(records)

print("=" * 60)
print(f"Starting benchmark run: {run_id}")
print(f"Testing {len(MODELS)} backends on {len(records)} identical sentences")
print("=" * 60)

with tqdm(total=total_tasks, desc="All models", unit="utt") as global_bar:
    for model_name, info in MODELS.items():
        print(f"\n=== Running {model_name} ===")
        preds, refs, labels, paths = [], [], [], []
        model_obj = None

        with tqdm(total=len(records), desc=f"{model_name}", unit="utt", leave=False) as model_bar:

            if info["type"] == "faster":
                try:
                    model_obj = load_faster_whisper(info["id"])
                except Exception as e:
                    print(f"  ❌ Failed to load {model_name}: {e}")
                    for ex in records:
                        skipped_rows.append((model_name, ex["audio_path"], ex["label_id"], f"Model load failed: {e}"))
                        model_bar.update(1); global_bar.update(1)
                    continue

                for ex in records:
                    try:
                        segments, _ = model_obj.transcribe(
                            ex["audio_path"],
                            language="de",               # adjust if your refs are not German
                            vad_filter=True,
                            vad_parameters={"min_silence_duration_ms": 500},
                            condition_on_previous_text=False,
                            beam_size=5,
                        )
                        text = " ".join(seg.text for seg in segments).strip()
                        preds.append(text)
                        refs.append(ex["sentence"])
                        labels.append(ex["label_id"])
                        paths.append(ex["audio_path"])
                    except Exception as e:
                        skipped_rows.append((model_name, ex["audio_path"], ex["label_id"], str(e)))
                    finally:
                        model_bar.update(1); global_bar.update(1)

            elif info["type"] == "hf":
                try:
                    print(f"  Loading HF pipeline on {'GPU' if HF_DEVICE == 0 else 'CPU'}...")
                    asr = pipeline(
                        "automatic-speech-recognition",
                        model=info["id"],
                        device=HF_DEVICE,
                        torch_dtype="auto",
                    )
                except Exception as e:
                    print(f"  ❌ Failed to load {model_name}: {e}")
                    for ex in records:
                        skipped_rows.append((model_name, ex["audio_path"], ex["label_id"], f"Model load failed: {e}"))
                        model_bar.update(1); global_bar.update(1)
                    continue

                for ex in records:
                    try:
                        audio = load_audio(ex["audio_path"])
                        out = asr(
                            audio,
                            generate_kwargs={"task": "transcribe", "language": "de"},
                            return_timestamps=False,
                            # Uncomment if your files can be long:
                            # chunk_length_s=30, stride_length_s=5,
                        )
                        text = out["text"].strip() if isinstance(out, dict) else str(out).strip()
                        preds.append(text)
                        refs.append(ex["sentence"])
                        labels.append(ex["label_id"])
                        paths.append(ex["audio_path"])
                    except Exception as e:
                        skipped_rows.append((model_name, ex["audio_path"], ex["label_id"], str(e)))
                    finally:
                        model_bar.update(1); global_bar.update(1)

        # Metrics for this backend
        if len(preds) == 0:
            print("  ⚠️ No successful predictions for this backend.")
            overall = {k: float("nan") for k in ["WER","CER","sacreBLEU","chrF","TER"]}
            per_label = {}
        else:
            print(f"  Successfully processed {len(preds)}/{len(records)} samples")
            overall = compute_all_metrics(preds, refs)
            print("  Overall -> " + ", ".join(f"{k}: {v:.3f}" for k, v in overall.items() if not np.isnan(v)))

            per_label = {}
            for lab in sorted(set(labels), key=lambda x: str(x)):
                idx = [i for i, l in enumerate(labels) if l == lab]
                if not idx:
                    continue
                lp = [preds[i] for i in idx]
                lr = [refs[i]  for i in idx]
                per_label[lab] = compute_all_metrics(lp, lr)

        # Save raw predictions
        pred_df = pd.DataFrame({
            "audio_path": paths,
            "label_id": labels,
            "reference": refs,
            "prediction": preds,
        })
        pred_csv = os.path.join(OUT_DIR, f"{run_id}__{model_name.replace('/', '_').replace(':', '_')}_preds.csv")
        pred_df.to_csv(pred_csv, index=False)

        results[model_name] = {
            "overall": overall,
            "per_label": per_label,
            "preds_csv": pred_csv,
            "successful_predictions": len(preds),
            "total_records": len(records)
        }

# =========================
# Save summary + manifest + skipped
# =========================
summary_json  = os.path.join(OUT_DIR, f"{run_id}__metrics_summary.json")
manifest_json = os.path.join(OUT_DIR, f"{run_id}__run_manifest.json")
skipped_csv   = os.path.join(OUT_DIR, f"{run_id}__skipped.csv")

with open(summary_json, "w") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

rows = []
for m, r in results.items():
    row = {"model": m, "success_rate": f"{r['successful_predictions']}/{r['total_records']}", **r["overall"]}
    rows.append(row)
summary_df = pd.DataFrame(rows)
summary_df.to_csv(os.path.join(OUT_DIR, f"{run_id}__metrics_table.csv"), index=False)

manifest = {
    "run_id": run_id,
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    "data_files": DATA_FILES,
    "max_samples_per_label": MAX_SAMPLES_PER_LABEL,
    "sample_rate": SAMPLE_RATE,
    "models": MODELS,
    "records_count": len(records),
    "dataset_fingerprint_sha256": dataset_fingerprint,
    "device_hf": "cuda:0" if HF_DEVICE == 0 else "cpu",
    "missing_files_filtered": int(missing_count),
    "sampling_stats": sampling_stats,
}
with open(manifest_json, "w") as f:
    json.dump(manifest, f, indent=2, ensure_ascii=False)

if skipped_rows:
    pd.DataFrame(skipped_rows, columns=["model","audio_path","label_id","reason"]).to_csv(skipped_csv, index=False)
    print(f"\n⚠️ Skipped {len(skipped_rows)} items across all models. Details: {skipped_csv}")
else:
    print("\n✅ No skips across all models!")

# =========================
# Print summary
# =========================
print("\n" + "=" * 60)
print("FINAL BENCHMARK RESULTS")
print("=" * 60)
print(f"📊 Metrics summary: {summary_json}")
print(f"📋 Run manifest:    {manifest_json}")
print(f"📈 Summary table:   {os.path.join(OUT_DIR, f'{run_id}__metrics_table.csv')}")
print(f"\n🎯 Both backends tested on identical {len(records)} sentences")
print(f"📝 Samples per label: {dict(sampling_stats)}")
for model, res in results.items():
    print(f"\n📱 Backend: {model}")
    success_rate = res['successful_predictions'] / res['total_records'] * 100
    print(f"   ✅ Success rate: {res['successful_predictions']}/{res['total_records']} ({success_rate:.1f}%)")
    if res['successful_predictions'] > 0:
        valid_metrics = {k: v for k, v in res['overall'].items() if not np.isnan(v)}
        if valid_metrics:
            print("   📊 Overall: " + ", ".join(f"{k}: {v:.3f}" for k, v in valid_metrics.items()))
        else:
            print("   ⚠️  No valid overall metrics")
        if res["per_label"]:
            print(f"   📋 Per-label results available for {len(res['per_label'])} labels")
    else:
        print("   ❌ No successful predictions")
    print(f"   💾 Predictions: {res['preds_csv']}")
print(f"\n🎉 Benchmark completed! Run ID: {run_id}")
