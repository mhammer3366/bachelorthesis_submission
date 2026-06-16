#!/usr/bin/env python3
# Add label_id to Wien TSV using speaker-level predictions.
# label_id: 0 = dialect, 1 = high_german  (change below if you want 7/8)

import os
import pandas as pd
import numpy as np

# ---- hardcoded paths ----
INPUT_TSV  = "/home/ai/AI-DataPool/Datasets/audio/Österreich/sliced_16000_mono/Wien/master_with_phonemes_plus.tsv"
LABEL_TSV  = "/home/ai/AI-DataPool/Datasets/audio/Österreich/sliced_16000_mono/Wien/nb_phoneme_binary/speaker_preds.tsv"
OUTPUT_TSV = "/home/ai/AI-DataPool/Datasets/audio/Österreich/sliced_16000_mono/Wien/master_with_phonemes_binary.tsv"

GERMAN_ID = 1  # high_german
VBG_ID    = 0  # dialect / non-High-German


def infer_is_german(row: pd.Series) -> int:
    """
    Return 1 if German (high_german), 0 if dialect, or -1 if unknown.
    Priority: pred_label_idx -> pred_label_name -> prob_high_german_mean.
    """
    # 1) pred_label_idx (0=dialect, 1=high_german)
    if "pred_label_idx" in row and pd.notna(row["pred_label_idx"]):
        try:
            return 1 if int(float(row["pred_label_idx"])) == 1 else 0
        except Exception:
            pass

    # 2) pred_label_name
    if "pred_label_name" in row and isinstance(row["pred_label_name"], str):
        name = row["pred_label_name"].strip().lower()
        if name in {"german", "high_german", "highgerman"}:
            return 1
        if name in {"dialect", "non_german", "nongerman", "vorarlberg"}:
            return 0

    # 3) probability threshold
    for prob_col in ["prob_high_german_mean", "prob_german", "prob_high_german"]:
        if prob_col in row and pd.notna(row[prob_col]):
            try:
                p = float(row[prob_col])
                return 1 if p >= 0.5 else 0
            except Exception:
                pass

    return -1


def main():
    # ---- Load utterance-level TSV ----
    print(f"📄 Reading utterance TSV: {INPUT_TSV}")
    df = pd.read_csv(INPUT_TSV, sep="\t", dtype=str, low_memory=False)

    for col in ["podcast", "episode", "speaker_id"]:
        if col not in df.columns:
            raise ValueError(f"INPUT_TSV must contain column '{col}' (has: {df.columns.tolist()})")

    # convenience column
    if "audio_path" not in df.columns and "path" in df.columns:
        df["audio_path"] = df["path"]

    # build speaker key
    df["speaker_key"] = (
        df["podcast"].astype(str)
        + "__"
        + df["episode"].astype(str)
        + "__"
        + df["speaker_id"].astype(str)
    )

    # ---- Load speaker-level predictions ----
    print(f"📄 Reading speaker preds: {LABEL_TSV}")
    preds = pd.read_csv(LABEL_TSV, sep="\t", dtype=str, low_memory=False)

    for col in ["podcast", "episode", "speaker_id"]:
        if col not in preds.columns:
            raise ValueError(f"LABEL_TSV must contain column '{col}' (has: {preds.columns.tolist()})")

    preds["speaker_key"] = (
        preds["podcast"].astype(str)
        + "__"
        + preds["episode"].astype(str)
        + "__"
        + preds["speaker_id"].astype(str)
    )

    # Infer german/dialect per speaker
    print("🧠 Inferring is_german per speaker_key ...")
    preds["is_german"] = preds.apply(infer_is_german, axis=1)

    # If there are duplicates per key, keep last (or by updated_at if present)
    if "updated_at" in preds.columns:
        preds["_ts"] = pd.to_datetime(preds["updated_at"], errors="coerce")
        preds = preds.sort_values("_ts")

    preds = preds.drop_duplicates(subset=["speaker_key"], keep="last")

    # Map is_german → label_id
    preds["label_id"] = preds["is_german"].map({1: GERMAN_ID, 0: VBG_ID})

    # ---- Merge onto utterance-level rows via speaker_key ----
    print("🔗 Merging labels onto utterance-level TSV by (podcast, episode, speaker_id) ...")
    out = df.merge(preds[["speaker_key", "label_id"]], on="speaker_key", how="left")

    # Report coverage
    n_total = len(out)
    n_labeled = out["label_id"].notna().sum()
    n_missing = n_total - n_labeled
    print(f"Rows total: {n_total:,} | with label_id: {n_labeled:,} | missing: {n_missing:,}")

    # Save (drop helper key)
    out = out.drop(columns=["speaker_key"])
    os.makedirs(os.path.dirname(OUTPUT_TSV), exist_ok=True)
    out.to_csv(OUTPUT_TSV, sep="\t", index=False)
    print(f"✅ Saved with label_id → {OUTPUT_TSV}")

    print("\nLabel counts (by label_id):")
    print(out["label_id"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()
