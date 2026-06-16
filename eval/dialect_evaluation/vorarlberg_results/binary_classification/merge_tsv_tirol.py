#!/usr/bin/env python3
# Add label_id to Tirol TSV using speaker-level predictions and also
# create a filtered file containing only dialect rows (label_id == 0)
# with only columns "path" and "text".

import os
import pandas as pd

# ---- hardcoded paths ----
INPUT_TSV  = "/home/ai/AI-DataPool/Datasets/audio/Österreich/sliced_16000_mono/Tirol/master.cleaned_with_phonemes.tsv"
LABEL_TSV  = "/home/ai/AI-DataPool/Datasets/audio/Österreich/sliced_16000_mono/Tirol/nb_phoneme_binary_60secs/speaker_preds.csv"

OUTPUT_TSV_FULL     = "/home/ai/AI-DataPool/Datasets/audio/Österreich/sliced_16000_mono/Tirol/master.cleaned_with_phonemes_with_label_id.tsv"
OUTPUT_TSV_FILTERED = "/home/ai/AI-DataPool/Datasets/audio/Österreich/sliced_16000_mono/Tirol/binary_classified_only_tirol.tsv"

GERMAN_ID = 1  # high_german
VBG_ID    = 0  # dialect / non-High-German


def infer_is_german(row: pd.Series) -> int:
    """Return 1 if German, 0 if dialect, -1 if unknown."""
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
        if name in {"dialect", "non_german", "nongerman"}:
            return 0

    # 3) probability fallback
    for col in ["prob_high_german_mean", "prob_german", "prob_high_german"]:
        if col in row and pd.notna(row[col]):
            try:
                return 1 if float(row[col]) >= 0.5 else 0
            except Exception:
                pass

    return -1


def main():
    print(f"📄 Reading utterance TSV: {INPUT_TSV}")
    df = pd.read_csv(INPUT_TSV, sep="\t", dtype=str, low_memory=False)

    if "speaker" not in df.columns:
        raise ValueError(f"INPUT_TSV must contain 'speaker' (has: {df.columns.tolist()})")

    if "audio_path" not in df.columns and "path" in df.columns:
        df["audio_path"] = df["path"]

    print(f"📄 Reading speaker preds: {LABEL_TSV}")
    preds = pd.read_csv(LABEL_TSV, dtype=str, low_memory=False)

    if "speaker_id" not in preds.columns:
        print("❌ Columns in LABEL_TSV:", preds.columns.tolist())
        raise ValueError("LABEL_TSV must contain 'speaker_id'")

    preds = preds.copy()
    preds["is_german"] = preds.apply(infer_is_german, axis=1)
    preds = preds.drop_duplicates(subset=["speaker_id"], keep="last")
    preds["label_id"] = preds["is_german"].map({1: GERMAN_ID, 0: VBG_ID})

    print("Speaker-level label_id distribution:")
    print(preds["label_id"].value_counts(dropna=False).to_string())

    print("🔗 Merging labels via speaker → label_id")
    label_map = preds.set_index("speaker_id")["label_id"]
    df["label_id"] = df["speaker"].map(label_map)

    # numeric view of label_id
    df["label_id_num"] = pd.to_numeric(df["label_id"], errors="coerce")

    n_total = len(df)
    n_labeled = df["label_id_num"].notna().sum()
    print(f"Rows total {n_total:,} | labeled {n_labeled:,} | missing {n_total-n_labeled:,}")

    print("Utterance-level label_id_num distribution:")
    print(df["label_id_num"].value_counts(dropna=False).to_string())

    # ---- save full file ----
    os.makedirs(os.path.dirname(OUTPUT_TSV_FULL), exist_ok=True)
    df.to_csv(OUTPUT_TSV_FULL, sep="\t", index=False)
    print(f"✅ Full TSV saved → {OUTPUT_TSV_FULL}")

    # ---- filtered dialect-only file ----
    print("🎯 Creating dialect-only TSV with columns ['path', 'text'] ...")

    dialect_mask = df["label_id_num"] == VBG_ID
    keep = df.loc[dialect_mask, ["path", "text"]]

    keep.to_csv(OUTPUT_TSV_FILTERED, sep="\t", index=False)
    print(f"✅ Filtered dialect-only TSV saved → {OUTPUT_TSV_FILTERED}")

    print("\nFiltered row count:", len(keep))
    print("Sample:")
    print(keep.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
