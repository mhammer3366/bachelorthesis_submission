
import os
from pathlib import Path
REPO_ROOT = Path(os.environ.get("THESIS_ROOT", Path(__file__).resolve().parents[2]))
# make_audio_paths_absolute_by_split.py
import os
import pandas as pd

# Base dirs for each dataset & split
BASE_PATHS = {
    "STT4SG-350": {
        "train": "/home/ai/AI-DataPool/Datasets/audio/Schweiz/STT4SG-350/clips__train_valid",
        "valid": "/home/ai/AI-DataPool/Datasets/audio/Schweiz/STT4SG-350/clips__train_valid",
        "test": "/home/ai/AI-DataPool/Datasets/audio/Schweiz/STT4SG-350/clips__test",
    },
    "SDS-200": {
        "train": "/home/ai/AI-DataPool/Datasets/audio/Schweiz/SDS-200/Audio",
        "valid": "/home/ai/AI-DataPool/Datasets/audio/Schweiz/SDS-200/Audio",
        "test": "/home/ai/AI-DataPool/Datasets/audio/Schweiz/SDS-200/Audio",
    }
}

splits = ["train", "valid", "test"]

in_dir = str(REPO_ROOT / "data_preparation/merged_datasets")
out_dir = "merged_out_abs"
os.makedirs(out_dir, exist_ok=True)

for split in splits:
    in_tsv = os.path.join(in_dir, f"merged_{split}.tsv")
    out_tsv = os.path.join(out_dir, f"merged_{split}.tsv")

    df = pd.read_csv(in_tsv, sep="\t")

    # detect which column to use
    if "audio_path" in df.columns:
        col = "audio_path"
    elif "path" in df.columns:
        col = "path"
    else:
        raise ValueError(f"{in_tsv} has no 'audio_path' or 'path' column.")

    if "dataset" not in df.columns:
        raise ValueError(f"{in_tsv} must contain a 'dataset' column!")

    # build absolute paths
    def make_abs(row):
        dataset = row["dataset"]
        base = BASE_PATHS[dataset][split]  # pick correct base by dataset + split
        p = row[col]
        if not isinstance(p, str):
            return p
        return p if os.path.isabs(p) else os.path.join(base, p)

    df["audio_path"] = df.apply(make_abs, axis=1)

    df.to_csv(out_tsv, sep="\t", index=False)
    print(f"[OK] Wrote {out_tsv} with {len(df)} rows")
