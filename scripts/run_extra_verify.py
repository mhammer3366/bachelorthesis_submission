#!/usr/bin/env python3

import os
from pathlib import Path
REPO_ROOT = Path(os.environ.get("THESIS_ROOT", Path(__file__).resolve().parents[1]))
import json, csv, re
from collections import Counter, defaultdict
from pathlib import Path
import pandas as pd
from sklearn.metrics import confusion_matrix

BASE = REPO_ROOT
OUT = BASE / "extra_verification.json"
DATA_ROOT = Path(os.environ.get("DATA_ROOT", "/home/ai/AI-DataPool/Datasets"))
MODELS_ROOT = Path(os.environ.get("MODELS_ROOT", DATA_ROOT.parent / "Models"))
CHECKPOINTS_DIR = Path(os.environ.get("CHECKPOINTS_DIR", MODELS_ROOT / "TTS" / "chatterbox"))
results = {}

# Item 12 speaker-level CM via comparison_prints logic
META_BASE = BASE / "data_preparation/merged_datasets_plus_de"
preds_file = BASE / "models/7_linear_phoneme_binary/test_preds.csv"
speaker_map = {}
meta_file = META_BASE / "merged_test.tsv"
with open(meta_file, encoding="utf-8") as f:
    reader = csv.DictReader(f, delimiter="\t")
    for row in reader:
        speaker_map[row["audio_path"]] = row["client_id"]

speaker_predictions = defaultdict(lambda: {"preds": [], "true": None})
with open(preds_file, encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        audio_path = row.get("audio_path", "")
        client_id = None
        for path, cid in speaker_map.items():
            if path in audio_path or audio_path in path:
                client_id = cid
                break
        if not client_id and audio_path.startswith("concat_"):
            parts = audio_path.split("_")
            if len(parts) >= 2:
                potential_id = "_".join(parts[1:-2]) if len(parts) > 3 else parts[1]
                if potential_id in speaker_map.values():
                    client_id = potential_id
        if not client_id:
            continue
        pred = int(row["pred_binary"])
        true_label = int(row["binary_label"])
        if speaker_predictions[client_id]["true"] is None:
            speaker_predictions[client_id]["true"] = true_label
        speaker_predictions[client_id]["preds"].append(pred)

cm = [[0, 0], [0, 0]]
for data in speaker_predictions.values():
    majority_pred = Counter(data["preds"]).most_common(1)[0][0]
    cm[data["true"]][majority_pred] += 1

tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
acc = (tn + tp) / (tn + fp + fn + tp)
gr = tp / (fn + tp) if (fn + tp) else 0
nr = tn / (tn + fp) if (tn + fp) else 0
results["12_speaker_cm"] = {
    "cm": cm,
    "n_speakers": len(speaker_predictions),
    "accuracy": round(acc, 3),
    "german_recall": round(gr, 3),
    "non_german_recall": round(nr, 3),
    "balanced_accuracy": round((gr + nr) / 2, 3),
    "german_total": fn + tp,
}

with open(BASE / "models/7_linear_phoneme_binary/test_metrics.json") as f:
    chunk = json.load(f)
results["12_chunk_cm"] = chunk["confusion_matrix"]

# Item 15
df = pd.read_csv(BASE / "asr/bench_outputs/20250903_180349__faster-whisper_large-v3_preds.csv")
results["15_per_label"] = {int(k): int(v) for k, v in sorted(df["label_id"].value_counts().items())}

# Item 26 trainer_state details
for run in [
    "chatterbox_finetuned_vorarlberg_binary",
    "chatterbox_finetuned_stt_all",
    "vorarlberg_finetuned_10_epochs",
]:
    ts_path = CHECKPOINTS_DIR / run / "trainer_state.json"
    tr_path = ts_path.parent / "train_results.json"
    if not ts_path.exists():
        continue
    with open(ts_path) as f:
        ts = json.load(f)
    with open(tr_path) as f:
        tr = json.load(f)
    args = ts.get("args", {})
    steps = ts.get("max_steps") or ts.get("global_step")
    entry = {
        "max_steps": steps,
        "num_train_epochs": ts.get("num_train_epochs"),
        "per_device_train_batch_size": args.get("per_device_train_batch_size"),
        "gradient_accumulation_steps": args.get("gradient_accumulation_steps"),
        "world_size": args.get("world_size"),
        "train_runtime_s": tr.get("train_runtime"),
        "train_runtime_h": round(tr.get("train_runtime", 0) / 3600, 2),
        "train_loss": tr.get("train_loss"),
    }
    if steps and tr.get("train_runtime") and ts.get("num_train_epochs"):
        samples_per_epoch = steps * args.get("per_device_train_batch_size", 1) * args.get("gradient_accumulation_steps", 1) * args.get("world_size", 1) / ts.get("num_train_epochs")
        entry["derived_samples_per_epoch"] = round(samples_per_epoch)
    results[f"26_{run}"] = entry

# Item 25 ref wav
ref = Path(os.environ.get("REF_VOICE_WAV", str(BASE / "tts/chatterbox-finetuning/voice_samples/daniel_ganahl_unfall_montafonerisch.wav")))
results["25_ref_wav"] = {"exists": ref.exists(), "path": str(ref), "size_bytes": ref.stat().st_size if ref.exists() else None}

# Item 28 workflow
results["28_workflow_png"] = list(BASE.rglob("workflow_pipeline.png"))

# Item 5 podcast mentions in Vorarlberg json (limited search)
vbg = DATA_ROOT / "audio/Vorarlberg"
mentions = {}
for pattern in ["podcast.de", "Podimo", "LetsCast", "Kristbergbahn", "Lauterach", "Fröweis", "Fr\u00f6weis"]:
    count = 0
    sample = None
    for jf in list(vbg.glob("**/*sentences.json"))[:50]:
        try:
            text = jf.read_text(encoding="utf-8", errors="ignore")
            if pattern.lower() in text.lower():
                count += 1
                sample = str(jf)
        except Exception:
            pass
    mentions[pattern] = {"files_matched_in_sample": count, "sample": sample}
results["5_vbg_mentions"] = mentions

SDS_SPLITS = DATA_ROOT / "audio/Schweiz/SDS-200/SDS-200-Corpus/splits"
full_cantons = Counter()
for split in ["train", "valid", "test"]:
    sds = pd.read_csv(
        SDS_SPLITS / f"{split}.tsv",
        sep="\t", dtype=str, keep_default_na=False,
    )
    full_cantons.update(sds["canton"].str.upper().str.strip())
results["1_full_sds_cantons"] = dict(sorted(full_cantons.items()))
results["1_full_sds_n_cantons"] = len([k for k in full_cantons if k])

with open(OUT, "w") as f:
    json.dump(results, f, indent=2)
print(f"Wrote {OUT}")
