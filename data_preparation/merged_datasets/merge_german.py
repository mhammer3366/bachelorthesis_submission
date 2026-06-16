
import os
from pathlib import Path
REPO_ROOT = Path(os.environ.get("THESIS_ROOT", Path(__file__).resolve().parents[2]))
# add_german_cv_to_splits.py
# Append German Common Voice (v22) samples to your existing merged Swiss splits.
# Target counts from German corpus: train=100_000, valid=4_000, test=4_000.
# Preference: pick rows with the MOST filled metadata (age, gender, accents, variant, locale);
# ties are broken randomly (seeded for reproducibility). Falls back to random if needed.

import os
import numpy as np
import pandas as pd

# --------------------- CONFIG ---------------------
# Existing Swiss splits (already merged & labeled with label_id)
SWISS_DIR = str(REPO_ROOT / "data_preparation/merged_datasets")
IN_TSV = {
    "train": os.path.join(SWISS_DIR, "merged_train.tsv"),
    "validation": os.path.join(SWISS_DIR, "merged_valid.tsv"),
    "test": os.path.join(SWISS_DIR, "merged_test.tsv"),
}

# Output with German appended
OUT_DIR = str(REPO_ROOT / "data_preparation/merged_datasets_plus_de")
os.makedirs(OUT_DIR, exist_ok=True)

# Common Voice v22 (German) base paths
CV_DE_DIR = "/home/ai/AI-DataPool/Datasets/audio/Deutschland/cv22-de/cv-corpus-22.0-2025-06-20/de"
CV_TSV = {
    "train": os.path.join(CV_DE_DIR, "train.tsv"),
    "validation": os.path.join(CV_DE_DIR, "validated.tsv"),
    "test": os.path.join(CV_DE_DIR, "test.tsv"),
}
CV_CLIPS_DIR = os.path.join(CV_DE_DIR, "clips")  # audio files live here

# Desired German additions per split
TARGET_ADD = {"train": 100_000, "validation": 4_000, "test": 4_000}

# Label to use for German (new class)
GERMAN_LABEL_NAME = "German"
GERMAN_LABEL_ID = 7  # keep Swiss 0..6; append German as 7

# Reproducibility
SEED = 42
np.random.seed(SEED)

# Columns we want in final TSV (Swiss schema)
FINAL_COLS = [
    "path", "sentence", "sentence_source", "client_id",
    "canton", "dialect_region", "zipcode", "age", "gender",
    "dataset", "audio_path", "label_id"
]

# Metadata columns to reward when present
INFO_COLS = ["age", "gender", "accents", "variant", "locale"]

# --------------------- HELPERS ---------------------
def read_tsv_safe(path: str) -> pd.DataFrame:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing TSV: {path}")
    return pd.read_csv(path, sep="\t")

def build_audio_path_cv(filename: str) -> str:
    return os.path.join(CV_CLIPS_DIR, filename)

def info_score(row: pd.Series) -> int:
    # Count non-empty metadata fields
    score = 0
    for c in INFO_COLS:
        val = row.get(c, "")
        if isinstance(val, str) and val.strip():
            score += 1
    return score

def quality_score(row: pd.Series) -> int:
    up = row.get("up_votes", 0)
    dn = row.get("down_votes", 0)
    try:
        up = int(up)
    except Exception:
        up = 0
    try:
        dn = int(dn)
    except Exception:
        dn = 0
    return up - dn

def prepare_cv_df(tsv_path: str) -> pd.DataFrame:
    df = read_tsv_safe(tsv_path).copy()

    # Ensure essential columns exist
    for col in ["path", "sentence", "client_id"]:
        if col not in df.columns:
            df[col] = ""

    # Absolute audio path
    df["audio_path"] = df["path"].apply(build_audio_path_cv)
    df = df[df["audio_path"].apply(lambda p: isinstance(p, str) and os.path.isfile(p))].reset_index(drop=True)

    # Scoring for selection
    df["__info_score"] = df.apply(info_score, axis=1)
    df["__qual_score"] = df.apply(quality_score, axis=1)

    # Add fields to match Swiss schema
    df["sentence_source"] = "CV22-DE"
    df["canton"] = ""
    df["dialect_region"] = GERMAN_LABEL_NAME
    df["zipcode"] = ""
    # Keep age/gender from CV if present; otherwise they’re already empty strings
    df["dataset"] = "CV22-DE"
    df["label_id"] = GERMAN_LABEL_ID

    # Keep only needed columns + scores for ranking
    keep = FINAL_COLS + ["__info_score", "__qual_score"]
    for col in keep:
        if col not in df.columns:
            df[col] = "" if col not in ["label_id"] else GERMAN_LABEL_ID
    return df[keep]

def select_cv_samples(df: pd.DataFrame, desired_n: int, used_paths: set) -> pd.DataFrame:
    # Exclude any files already used in another split
    if "audio_path" in df.columns:
        df = df[~df["audio_path"].isin(used_paths)].copy()

    if len(df) == 0:
        return df

    # Random tie-breaker, then prefer more metadata, then higher vote quality
    df = df.sample(frac=1.0, random_state=SEED)  # shuffle first
    df = df.sort_values(by=["__info_score", "__qual_score"], ascending=False)

    # Take top N (or all if fewer available)
    out = df.head(desired_n).copy()
    return out

def load_swiss_split(split: str) -> pd.DataFrame:
    df = read_tsv_safe(IN_TSV[split]).copy()

    # Normalize columns
    if "audio_path" not in df.columns and "path" in df.columns:
        # If only relative 'path', keep it, but we need absolute in 'audio_path'
        df["audio_path"] = df["path"]
    for col in FINAL_COLS:
        if col not in df.columns:
            df[col] = "" if col != "label_id" else -1

    # Ensure label_id is int
    df["label_id"] = pd.to_numeric(df["label_id"], errors="coerce").fillna(-1).astype(int)

    return df[FINAL_COLS]

# --------------------- MAIN ---------------------
def main():
    # Track used CV audio paths to avoid leakage across splits
    used_cv_paths = set()

    # Prepare CV dataframes once
    cv_df_train = prepare_cv_df(CV_TSV["train"])
    cv_df_valid = prepare_cv_df(CV_TSV["validation"])
    cv_df_test  = prepare_cv_df(CV_TSV["test"])

    # Order: pick TEST → VALID → TRAIN to avoid overlap
    additions = {}

    # TEST: 4k from CV test (fallback to validated if not enough)
    need = TARGET_ADD["test"]
    pick_test = select_cv_samples(cv_df_test, need, used_cv_paths)
    used_cv_paths.update(pick_test["audio_path"])
    if len(pick_test) < need:
        top_up = select_cv_samples(cv_df_valid, need - len(pick_test), used_cv_paths)
        used_cv_paths.update(top_up["audio_path"])
        pick_test = pd.concat([pick_test, top_up], ignore_index=True)
    additions["test"] = pick_test.drop(columns=["__info_score", "__qual_score"])

    # VALIDATION: 4k from CV validated (fallback to train if not enough)
    need = TARGET_ADD["validation"]
    pick_valid = select_cv_samples(cv_df_valid, need, used_cv_paths)
    used_cv_paths.update(pick_valid["audio_path"])
    if len(pick_valid) < need:
        top_up = select_cv_samples(cv_df_train, need - len(pick_valid), used_cv_paths)
        used_cv_paths.update(top_up["audio_path"])
        pick_valid = pd.concat([pick_valid, top_up], ignore_index=True)
    additions["validation"] = pick_valid.drop(columns=["__info_score", "__qual_score"])

    # TRAIN: 100k from CV train (fallback to validated if not enough)
    need = TARGET_ADD["train"]
    pick_train = select_cv_samples(cv_df_train, need, used_cv_paths)
    used_cv_paths.update(pick_train["audio_path"])
    if len(pick_train) < need:
        top_up = select_cv_samples(cv_df_valid, need - len(pick_train), used_cv_paths)
        used_cv_paths.update(top_up["audio_path"])
        pick_train = pd.concat([pick_train, top_up], ignore_index=True)
    additions["train"] = pick_train.drop(columns=["__info_score", "__qual_score"])

    # Append to Swiss splits and save
    for split in ["train", "validation", "test"]:
        swiss_df = load_swiss_split(split)
        out_df = pd.concat([swiss_df, additions[split]], ignore_index=True)
        out_path = os.path.join(OUT_DIR, f"merged_{split}.tsv")
        out_df.to_csv(out_path, sep="\t", index=False)
        print(f"[OK] {split}: Swiss {len(swiss_df):,} + German {len(additions[split]):,} = {len(out_df):,} → {out_path}")

if __name__ == "__main__":
    main()
