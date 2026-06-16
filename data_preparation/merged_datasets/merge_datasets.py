# merge_stt_sds_splits.py
import pandas as pd
from pathlib import Path

# ---------- config ----------
STT_PATHS = {
    "train": "/home/ai/AI-DataPool/Datasets/audio/Schweiz/STT4SG-350/train_all.tsv",
    "valid": "/home/ai/AI-DataPool/Datasets/audio/Schweiz/STT4SG-350/valid.tsv",
    "test":  "/home/ai/AI-DataPool/Datasets/audio/Schweiz/STT4SG-350/test.tsv",
}

SDS_PATHS = {
    "train": "/home/ai/AI-DataPool/Datasets/audio/Schweiz/SDS-200/SDS-200-Corpus/splits/train.tsv",
    "valid": "/home/ai/AI-DataPool/Datasets/audio/Schweiz/SDS-200/SDS-200-Corpus/splits/valid.tsv",
    "test":  "/home/ai/AI-DataPool/Datasets/audio/Schweiz/SDS-200/SDS-200-Corpus/splits/test.tsv",
}

OUT_DIR = Path("./merged_datasets_noNaN")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------- mapping ----------
CANTON_TO_DIALECT = {
    # Zürich
    "AG": "Zürich", "ZH": "Zürich", "ZG": "Zürich",
    # Innerschweiz
    "LU": "Innerschweiz", "SZ": "Innerschweiz", "UR": "Innerschweiz", "NW": "Innerschweiz",
    # Wallis
    "VS": "Wallis",
    # Graubünden
    "GR": "Graubünden",
    # Ostschweiz
    "TG": "Ostschweiz", "SG": "Ostschweiz", "AI": "Ostschweiz", "SH": "Ostschweiz",
    # Basel
    "BS": "Basel", "BL": "Basel",
    # Bern
    "BE": "Bern", "FR": "Bern",
}

# ---------- helpers ----------
COMMON_CLASSIF_COLS = [
    "path", "sentence", "sentence_source", "client_id",
    "canton", "dialect_region", "zipcode", "age", "gender"
]

def load_tsv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].str.strip()
    if "canton" in df.columns:
        df["canton"] = df["canton"].str.upper()
    return df

def normalize_stt(df: pd.DataFrame) -> pd.DataFrame:
    needed = {"path","sentence","sentence_source","client_id","canton","zipcode","age","gender"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"STT missing columns: {missing}")
    if "dialect_region" not in df.columns:
        df["dialect_region"] = pd.NA
    return df.reindex(columns=COMMON_CLASSIF_COLS, fill_value=pd.NA)

def normalize_sds(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={"clip_path": "path"})
    needed_any = {"path","sentence","sentence_source","client_id","canton","zipcode","age","gender"}
    missing = needed_any - set(df.columns)
    if missing:
        raise ValueError(f"SDS missing columns: {missing}")
    df["dialect_region"] = df["canton"].map(CANTON_TO_DIALECT)
    unmapped = df["canton"][df["dialect_region"].isna()].dropna().unique().tolist()
    if unmapped:
        print(f"[WARN] Unmapped cantons in SDS: {unmapped}")
    return df.reindex(columns=COMMON_CLASSIF_COLS, fill_value=pd.NA)

def merge_split(split_name: str, stt_path: str, sds_path: str) -> pd.DataFrame:
    stt = normalize_stt(load_tsv(stt_path)).assign(dataset="STT4SG-350")
    sds = normalize_sds(load_tsv(sds_path)).assign(dataset="SDS-200")
    merged = pd.concat([stt, sds], ignore_index=True)
    merged = merged.drop_duplicates(subset=["path"]).reset_index(drop=True)

    # ---- count NaNs before dropping ----
    nan_stt = stt["dialect_region"].isna().sum()
    nan_sds = sds["dialect_region"].isna().sum()
    total_before = len(merged)

    # ---- drop NaN dialect_region ----
    merged_clean = merged.dropna(subset=["dialect_region"]).reset_index(drop=True)
    dropped = total_before - len(merged_clean)

    # ---- save ----
    out_path = OUT_DIR / f"merged_{split_name}.tsv"
    merged_clean.to_csv(out_path.as_posix(), sep="\t", index=False)

    # ---- stats ----
    print("="*60)
    print(f"[{split_name.upper()}]")
    print(f" Total before drop: {total_before}")
    print(f"   NaN dialect_region in STT : {nan_stt}")
    print(f"   NaN dialect_region in SDS : {nan_sds}")
    print(f" Total dropped (NaN region) : {dropped}")
    print(f" Total after drop           : {len(merged_clean)}")
    print(f" dialect_region distribution:\n{merged_clean['dialect_region'].value_counts()}\n")

    return merged_clean

def main():
    for split in ["train", "valid", "test"]:
        merge_split(split, STT_PATHS[split], SDS_PATHS[split])
    print(f"✅ Merged datasets saved to {OUT_DIR}")

if __name__ == "__main__":
    main()
