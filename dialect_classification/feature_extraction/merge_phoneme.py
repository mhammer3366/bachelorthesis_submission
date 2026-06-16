
import os
from pathlib import Path
REPO_ROOT = Path(os.environ.get("THESIS_ROOT", Path(__file__).resolve().parents[2]))
import pandas as pd

# Paths
orig_tsv = str(REPO_ROOT / "data_preparation/feature_extraction/vorarlberg/vorarlberg_with_phonemes.tsv")
fixed_tsv = str(REPO_ROOT / "data_preparation/feature_extraction/vorarlberg/vorarlberg_with_phonemes_fixed.tsv")
merged_tsv = str(REPO_ROOT / "data_preparation/feature_extraction/vorarlberg/vorarlberg_with_phonemes_merged.tsv")

# Load both
df_orig = pd.read_csv(orig_tsv, sep="\t", dtype=str, low_memory=False)
df_fixed = pd.read_csv(fixed_tsv, sep="\t", dtype=str, low_memory=False)

# Merge: use 'path' as the unique identifier
df_merged = df_orig.merge(
    df_fixed[["path", "phoneme"]],
    on="path",
    how="left",
    suffixes=("", "_fixed")
)

# Replace phoneme with fixed version if available
df_merged["phoneme"] = df_merged["phoneme_fixed"].combine_first(df_merged["phoneme"])

# Drop helper column
df_merged = df_merged.drop(columns=["phoneme_fixed"])

# Save merged file
df_merged.to_csv(merged_tsv, sep="\t", index=False)

print(f"✅ Merged file saved to {merged_tsv}")
