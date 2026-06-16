#!/usr/bin/env python3
import os
from pathlib import Path
REPO_ROOT = Path(os.environ.get("THESIS_ROOT", Path(__file__).resolve().parents[1]))

"""Comprehensive thesis number verification script."""
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

BASE = REPO_ROOT
MERGED = BASE / "data_preparation/merged_datasets"
MERGED_DE = BASE / "data_preparation/merged_datasets_plus_de"
DATA_ROOT = Path(os.environ.get("DATA_ROOT", "/home/ai/AI-DataPool/Datasets"))
SDS_SPLITS = DATA_ROOT / "audio/Schweiz/SDS-200/SDS-200-Corpus/splits"
STT_DIR = DATA_ROOT / "audio/Schweiz/STT4SG-350"
VBG_ROOT = DATA_ROOT / "audio/Vorarlberg"
MODELS_ROOT = Path(os.environ.get("MODELS_ROOT", DATA_ROOT.parent / "Models"))
CHATTERBOX = Path(os.environ.get("CHECKPOINTS_DIR", MODELS_ROOT / "TTS" / "chatterbox"))
REF_WAV = Path(os.environ.get("REF_VOICE_WAV", str(REPO_ROOT / "tts/chatterbox-finetuning/voice_samples/daniel_ganahl_unfall_montafonerisch.wav")))

CANTON_TO_DIALECT = {
    "AG": "Zürich", "ZH": "Zürich", "ZG": "Zürich",
    "LU": "Innerschweiz", "SZ": "Innerschweiz", "UR": "Innerschweiz", "NW": "Innerschweiz",
    "VS": "Wallis", "GR": "Graubünden",
    "TG": "Ostschweiz", "SG": "Ostschweiz", "AI": "Ostschweiz", "SH": "Ostschweiz",
    "BS": "Basel", "BL": "Basel", "BE": "Bern", "FR": "Bern",
}


def load_tsv(path):
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)


def item1_canton_coverage():
    cantons = Counter()
    for split in ["train", "valid", "test"]:
        df = load_tsv(MERGED / f"merged_{split}.tsv")
        sds = df[df["dataset"] == "SDS-200"]
        cantons.update(sds["canton"].str.upper())
    mapped = set(CANTON_TO_DIALECT.keys())
    present = set(cantons.keys())
    absent_mapped = sorted(mapped - present)
    extra = sorted(present - mapped)
    return {
        "cantons_in_merged_sds": dict(sorted(cantons.items())),
        "n_cantons": len(present),
        "cantons_present": sorted(present),
        "mapped_but_absent": absent_mapped,
        "extra_unmapped_in_merge": extra,
    }


def item2_dialect_distribution():
    regions = ["Basel", "Bern", "Graubünden", "Innerschweiz", "Ostschweiz", "Wallis", "Zürich"]
    per_split = {}
    train_only = Counter()
    all_splits = Counter()
    split_totals = {}
    for split in ["train", "valid", "test"]:
        df = load_tsv(MERGED / f"merged_{split}.tsv")
        vc = df["dialect_region"].value_counts()
        per_split[split] = {r: int(vc.get(r, 0)) for r in regions}
        per_split[split]["total"] = len(df)
        split_totals[split] = len(df)
        all_splits.update(df["dialect_region"].tolist())
        if split == "train":
            train_only.update(df["dialect_region"].tolist())
    train_sum = sum(per_split["train"][r] for r in regions)
    all_sum = sum(all_splits[r] for r in regions)
    return {
        "per_split": per_split,
        "split_totals": split_totals,
        "train_region_sum": train_sum,
        "all_splits_region_sum": all_sum,
        "all_splits_total_rows": sum(split_totals.values()),
        "train_only_region_counts": {r: train_only[r] for r in regions},
    }


def item3_speakers():
    per_split = {}
    all_ids = set()
    for split in ["train", "valid", "test"]:
        df = load_tsv(MERGED / f"merged_{split}.tsv")
        ids = set(df["client_id"].unique())
        per_split[split] = len(ids)
        all_ids |= ids
    return {
        "unique_client_id_per_split": per_split,
        "global_unique": len(all_ids),
        "sum_per_split": sum(per_split.values()),
        "disjoint": sum(per_split.values()) == len(all_ids),
    }


def item4_dropped_samples():
    # Source row counts
    sds_total = sum(len(load_tsv(SDS_SPLITS / f"{s}.tsv")) for s in ["train", "valid", "test"])
    stt_total = (
        len(load_tsv(STT_DIR / "train_all.tsv"))
        + len(load_tsv(STT_DIR / "valid.tsv"))
        + len(load_tsv(STT_DIR / "test.tsv"))
    )
    merged_total = sum(len(load_tsv(MERGED / f"merged_{s}.tsv")) for s in ["train", "valid", "test"])
    delta = (sds_total + stt_total) - merged_total

    # SDS unmappable cantons
    unmapped_by_canton = Counter()
    unmapped_by_split = {}
    for split in ["train", "valid", "test"]:
        df = load_tsv(SDS_SPLITS / f"{split}.tsv")
        df["canton_u"] = df["canton"].str.upper().str.strip()
        df["mapped"] = df["canton_u"].map(CANTON_TO_DIALECT)
        unmapped = df[df["mapped"].isna() | (df["mapped"] == "")]
        unmapped_by_split[split] = len(unmapped)
        unmapped_by_canton.update(unmapped["canton_u"].tolist())

    # duplicate paths across concat (should be 0 from source)
    dup_dropped = 0
    for split in ["train", "valid", "test"]:
        stt = load_tsv(STT_DIR / ("train_all.tsv" if split == "train" else f"{split}.tsv"))
        sds = load_tsv(SDS_SPLITS / f"{split}.tsv")
        stt_paths = set(stt.iloc[:, 0] if "path" not in stt.columns else stt["path"])
        if "clip_path" in sds.columns:
            sds_paths = set(sds["clip_path"])
        else:
            sds_paths = set(sds["path"])
        dup_dropped += len(stt_paths & sds_paths)

    return {
        "sds_source_rows": sds_total,
        "stt_source_rows": stt_total,
        "source_sum": sds_total + stt_total,
        "merged_total": merged_total,
        "delta_dropped": delta,
        "sds_unmapped_by_split": unmapped_by_split,
        "sds_unmapped_total": sum(unmapped_by_split.values()),
        "unmapped_canton_counts": dict(unmapped_by_canton.most_common()),
        "path_overlap_stt_sds": dup_dropped,
    }


def item6_cv_version():
    cv_dir = DATA_ROOT / "audio/Deutschland/cv22-de/cv-corpus-22.0-2025-06-20"
    return {
        "cv_dir_exists": cv_dir.exists(),
        "cv_dir_name": cv_dir.name if cv_dir.exists() else None,
        "code_label": "CV22-DE",
        "merge_german_comment": "Common Voice (v22)",
    }


def item7_vorarlberg_hours():
    """Sum ffprobe durations under */original/ trees."""
    total_sec = 0.0
    folder_hours = {}
    for sub in sorted(VBG_ROOT.iterdir()):
        if not sub.is_dir():
            continue
        orig = sub / "original"
        if not orig.is_dir():
            continue
        sec = 0.0
        for wav in orig.rglob("*"):
            if wav.suffix.lower() not in {".wav", ".mp3", ".flac", ".m4a", ".ogg"}:
                continue
            try:
                out = subprocess.check_output(
                    ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                     "-of", "default=noprint_wrappers=1:nokey=1", str(wav)],
                    stderr=subprocess.DEVNULL, text=True,
                ).strip()
                if out:
                    sec += float(out)
            except Exception:
                pass
        if sec > 0:
            folder_hours[sub.name] = round(sec / 3600, 6)
            total_sec += sec
    return {
        "folder_hours": folder_hours,
        "total_hours": round(total_sec / 3600, 6),
        "total_hours_4dp": round(total_sec / 3600, 4),
    }


def item11_binary_split():
    counts = {}
    for split, fname in [("train", "merged_train.tsv"), ("valid", "merged_valid.tsv"), ("test", "merged_test.tsv")]:
        df = load_tsv(MERGED_DE / fname)
        counts[split] = int((df["label_id"].astype(int) == 7).sum())
    return counts


def item12_confusion():
    metrics_path = BASE / "models/7_linear_phoneme_binary/test_metrics.json"
    with open(metrics_path) as f:
        m = json.load(f)
    cm_chunk = m["confusion_matrix"]
    # speaker-level from comparison script logic
    preds_path = BASE / "models/7_linear_phoneme_binary/test_preds.csv"
    df = pd.read_csv(preds_path)
    # group by client_id - majority vote or first? read comparison script
    return {"chunk_cm": cm_chunk, "metrics_file": str(metrics_path)}


def item15_asr_counts():
    summary_path = BASE / "asr/bench_outputs/20250903_180349__metrics_summary.json"
    with open(summary_path) as f:
        summary = json.load(f)
    fw = summary.get("faster-whisper:large-v3", {})
    preds_csv = fw.get("preds_csv")
    per_label = {}
    if preds_csv and Path(preds_csv).exists():
        df = pd.read_csv(preds_csv)
        label_col = "label_id" if "label_id" in df.columns else "dialect_region"
        if label_col in df.columns:
            per_label = df[label_col].value_counts().sort_index().to_dict()
    return {
        "successful_predictions": fw.get("successful_predictions"),
        "total_records": fw.get("total_records"),
        "per_label_counts": per_label,
        "overall_wer": fw["overall"]["WER"],
        "overall_cer": fw["overall"]["CER"],
        "label7_wer": fw["per_label"]["7"]["WER"],
    }


def item18_wer_cer():
    summary_path = BASE / "asr/bench_outputs/20250903_180349__metrics_summary.json"
    with open(summary_path) as f:
        summary = json.load(f)
    result = {}
    for key in ["faster-whisper:large-v3", "wav2vec2-german"]:
        if key in summary:
            result[key] = {
                "WER": summary[key]["overall"]["WER"],
                "CER": summary[key]["overall"]["CER"],
            }
    # find wav2vec2 key
    for k, v in summary.items():
        if "wav2vec" in k.lower() or "german" in k.lower():
            result[k] = {"WER": v["overall"]["WER"], "CER": v["overall"]["CER"]}
    fw = summary["faster-whisper:large-v3"]
    result["label7_wer"] = fw["per_label"]["7"]["WER"]
    return result


def item20_vorarlberg_counts():
    vbg_tsv = VBG_ROOT / "vorarlberger_daten_16000.tsv"
    fin_tsv = BASE / "eval/dialect_evaluation/vorarlberg_results/binary_classification/vorarlberg_finalized.tsv"
    counts = {}
    for name, path in [("vorarlberger_daten_16000", vbg_tsv), ("vorarlberg_finalized", fin_tsv)]:
        if path.exists():
            df = load_tsv(path)
            counts[name] = len(df)
        else:
            counts[name] = None
    return counts


def item21_26_27_chatterbox():
    runs = {}
    for d in sorted(CHATTERBOX.iterdir()):
        tr = d / "train_results.json"
        ts = d / "trainer_state.json"
        if not tr.exists():
            continue
        with open(tr) as f:
            tr_data = json.load(f)
        entry = {
            "train_loss": tr_data.get("train_loss"),
            "train_runtime": tr_data.get("train_runtime"),
            "train_steps_per_second": tr_data.get("train_steps_per_second"),
            "epoch": tr_data.get("epoch"),
        }
        if ts.exists():
            with open(ts) as f:
                ts_data = json.load(f)
            entry["max_steps"] = ts_data.get("max_steps") or (ts_data.get("global_step"))
            entry["num_train_epochs"] = ts_data.get("num_train_epochs")
            args = ts_data.get("args") or ts_data.get("training_args") or {}
            if isinstance(args, dict):
                entry["per_device_train_batch_size"] = args.get("per_device_train_batch_size")
                entry["gradient_accumulation_steps"] = args.get("gradient_accumulation_steps")
                entry["world_size"] = args.get("world_size")
        runs[d.name] = entry
    return runs


def item29_sds_full_vs_merged():
    # Full SDS corpus
    full_speakers = set()
    full_rows = 0
    full_hours = 0.0
    for split in ["train", "valid", "test"]:
        df = load_tsv(SDS_SPLITS / f"{split}.tsv")
        full_rows += len(df)
        full_speakers.update(df["client_id"].unique())
        if "duration" in df.columns:
            full_hours += pd.to_numeric(df["duration"], errors="coerce").sum() / 3600
        elif "clip_duration" in df.columns:
            full_hours += pd.to_numeric(df["clip_duration"], errors="coerce").sum() / 3600

    # Merged SDS subset
    merged_sds_speakers = set()
    merged_sds_rows = 0
    for split in ["train", "valid", "test"]:
        df = load_tsv(MERGED / f"merged_{split}.tsv")
        sds = df[df["dataset"] == "SDS-200"]
        merged_sds_rows += len(sds)
        merged_sds_speakers.update(sds["client_id"].unique())

    return {
        "full_sds_rows": full_rows,
        "full_sds_speakers": len(full_speakers),
        "full_sds_hours_from_duration_col": round(full_hours, 4) if full_hours else None,
        "merged_sds_rows": merged_sds_rows,
        "merged_sds_speakers": len(merged_sds_speakers),
    }


def main():
    results = {}
    print("Running verification...")
    results["1_canton"] = item1_canton_coverage()
    results["2_dialect"] = item2_dialect_distribution()
    results["3_speakers"] = item3_speakers()
    results["4_dropped"] = item4_dropped_samples()
    results["6_cv"] = item6_cv_version()
    print("Computing Vorarlberg hours (ffprobe)...")
    results["7_vbg_hours"] = item7_vorarlberg_hours()
    results["11_binary"] = item11_binary_split()
    results["15_asr"] = item15_asr_counts()
    results["18_wer"] = item18_wer_cer()
    results["20_vbg_counts"] = item20_vorarlberg_counts()
    results["21_26_27_chatterbox"] = item21_26_27_chatterbox()
    results["29_sds"] = item29_sds_full_vs_merged()

    # eval unit counts
    phoneme_preds = BASE / "models/4_linear_phoneme/test_preds.csv"
    test_phonemes = BASE / "data_preparation/feature_extraction/ohne_deutsch/saved_features_phoneme/test_phonemes.tsv"
    results["9_eval_units"] = {
        "phoneme_test_preds_rows": len(pd.read_csv(phoneme_preds)) if phoneme_preds.exists() else None,
        "test_phonemes_rows": len(load_tsv(test_phonemes)) if test_phonemes.exists() else None,
    }
    cnn_metrics = BASE / "models/1_melspec_cnn/test_metrics.json"
    if cnn_metrics.exists():
        with open(cnn_metrics) as f:
            cm = json.load(f)
        results["9_eval_units"]["cnn_test_samples"] = cm.get("n_test") or cm.get("test_size")

    out = BASE / "verification_output.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Wrote {out}")
    print(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    main()
