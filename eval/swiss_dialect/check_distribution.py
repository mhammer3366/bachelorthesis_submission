
import os
from pathlib import Path
REPO_ROOT = Path(os.environ.get("THESIS_ROOT", Path(__file__).resolve().parents[2]))
import pandas as pd
from pathlib import Path

# Paths
out_dir = Path(str(REPO_ROOT / "eval/dialect_evaluation/swiss_srf_espeak_classification_two_stage_v3"))
clip_labels = out_dir / "clip_labels.csv"
tsv_path    = Path(str(REPO_ROOT / "eval/dialect_evaluation/swiss_srf_espeak_classification_two_stage_2/clip_labels.csv"))

# Load
clip_df = pd.read_csv(clip_labels)
tsv_df  = pd.read_csv(tsv_path, sep="\t", usecols=["clip_path","duration_sec","phoneme"])

# Normalize audio path column
if "clip_path" in tsv_df.columns:
    tsv_df = tsv_df.rename(columns={"clip_path":"audio_path"})

# Merge durations into clip labels
merged = clip_df.merge(tsv_df[["audio_path","duration_sec"]], on="audio_path", how="left")
merged["duration_sec"] = pd.to_numeric(merged["duration_sec"], errors="coerce").fillna(0.0)

# Dialect names
DIALECT_NAMES = {
    0: "Basel",
    1: "Bern",
    2: "Innerschweiz",
    3: "Ostschweiz",
    4: "Wallis",
    5: "Zürich",
    6: "Graubünden",
    7: "German"
}

# Group by final label and sum hours
hours = (
    merged.groupby("final_label_id")["duration_sec"]
    .sum()
    .div(3600)  # convert seconds → hours
    .reset_index()
)
hours["dialect"] = hours["final_label_id"].map(DIALECT_NAMES)

print(hours.sort_values("duration_sec", ascending=False))
