#!/usr/bin/env python3

import os
from pathlib import Path
REPO_ROOT = Path(os.environ.get("THESIS_ROOT", Path(__file__).resolve().parents[4]))
# Add label_id to Vorarlberg TSV using speaker-level predictions.
# label_id: 7 = German, 8 = Vorarlberg (dialect)

import os
import pandas as pd
import numpy as np

# ---- hardcoded paths ----
INPUT_TSV  = str(REPO_ROOT / "data_preparation/feature_extraction/vorarlberg/vorarlberg_with_phonemes_with_client.tsv")
LABEL_TSV  = str(REPO_ROOT / "eval/dialect_evaluation/vorarlberg_results/binary_classification/concat_30/speaker_preds.csv")
OUTPUT_TSV = str(REPO_ROOT / "eval/dialect_evaluation/vorarlberg_results/binary_classification/vorarlberg_finalized.tsv")

GERMAN_ID = 7
VBG_ID    = 8  # Vorarlberg dialect / non-German

def smart_read_table(path: str) -> pd.DataFrame:
    """Read CSV/TSV with best-effort sep inference."""
    # try pandas' python engine inference
    try:
        return pd.read_csv(path, sep=None, engine="python", dtype=str, low_memory=False)
    except Exception:
        pass
    # try common fallbacks
    for sep in [",", "\t", ";", "|"]:
        try:
            df = pd.read_csv(path, sep=sep, dtype=str, low_memory=False)
            # if it read as single column with separators inside, keep trying
            if df.shape[1] == 1 and (sep in df.columns[0]):
                continue
            return df
        except Exception:
            continue
    raise ValueError(f"Could not read table: {path}")

def infer_is_german(row: pd.Series) -> int:
    """
    Return 1 if German, 0 if Vorarlberg (dialect), or -1 if unknown.
    Priority: pred_label_idx -> pred_label_name -> prob_high_german_mean.
    """
    # pred_label_idx (0=dialect, 1=german)
    if "pred_label_idx" in row and pd.notna(row["pred_label_idx"]):
        try:
            return 1 if int(float(row["pred_label_idx"])) == 1 else 0
        except Exception:
            pass

    # pred_label_name
    if "pred_label_name" in row and isinstance(row["pred_label_name"], str):
        name = row["pred_label_name"].strip().lower()
        if name in {"german", "high_german", "highgerman"}:
            return 1
        if name in {"dialect", "non_german", "nongerman", "vorarlberg"}:
            return 0

    # probability threshold
    for prob_col in ["prob_high_german_mean", "prob_german", "prob_high_german"]:
        if prob_col in row and pd.notna(row[prob_col]):
            try:
                p = float(row[prob_col])
                return 1 if p >= 0.5 else 0
            except Exception:
                pass

    return -1

def main():
    # Load input (utterance-level)
    df = pd.read_csv(INPUT_TSV, sep="\t", dtype=str, low_memory=False)
    if "client_id" not in df.columns:
        raise ValueError("INPUT_TSV must contain 'client_id'")

    # Optional convenience column to match your target schema examples
    if "audio_path" not in df.columns and "path" in df.columns:
        df["audio_path"] = df["path"]

    # Load speaker-level predictions
    preds = smart_read_table(LABEL_TSV)
    if "client_id" not in preds.columns:
        raise ValueError("LABEL_TSV must contain 'client_id'")
    preds = preds.copy()

    # Infer german/dialect per speaker
    preds["is_german"] = preds.apply(infer_is_german, axis=1)
    # Keep one row per client_id (if duplicates, keep the latest if time available, else last)
    if "updated_at" in preds.columns:
        preds["_ts"] = pd.to_datetime(preds["updated_at"], errors="coerce")
        preds = preds.sort_values("_ts")
    preds = preds.drop_duplicates(subset=["client_id"], keep="last")

    # Map to label_id
    preds["label_id"] = preds["is_german"].map({1: GERMAN_ID, 0: VBG_ID})

    # Merge onto utterance-level rows
    out = df.merge(preds[["client_id", "label_id"]], on="client_id", how="left")

    # Report coverage
    n_total = len(out)
    n_labeled = out["label_id"].notna().sum()
    n_missing = n_total - n_labeled
    print(f"Rows total: {n_total:,} | with label_id: {n_labeled:,} | missing: {n_missing:,}")

    # Save
    os.makedirs(os.path.dirname(OUTPUT_TSV), exist_ok=True)
    out.to_csv(OUTPUT_TSV, sep="\t", index=False)
    print(f"✅ Saved with label_id → {OUTPUT_TSV}")

    # Quick label distribution
    print("Label counts (by label_id):")
    print(out["label_id"].value_counts(dropna=False).to_string())

if __name__ == "__main__":
    import os
    main()