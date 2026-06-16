
import os
from pathlib import Path
REPO_ROOT = Path(os.environ.get("THESIS_ROOT", Path(__file__).resolve().parents[2]))
# test_rerun.py
# Quick test of the fixed rerun script

import os
import pandas as pd

# Create a small test TSV with just a few NO_PHONEME entries
INPUT_TSV = str(REPO_ROOT / "data_preparation/feature_extraction/vorarlberg/vorarlberg_with_phonemes.tsv")
TEST_TSV = str(REPO_ROOT / "data_preparation/feature_extraction/test_sample.tsv")

# Read original and filter for NO_PHONEME
df = pd.read_csv(INPUT_TSV, sep="\t", dtype=str, low_memory=False)
no_phoneme_df = df[df["phoneme"] == "NO_PHONEME"].head(5)  # Just 5 files for testing

# Save test sample
no_phoneme_df.to_csv(TEST_TSV, sep="\t", index=False)
print(f"Created test sample with {len(no_phoneme_df)} files")

# Show the test files
print("\nTest files:")
print(no_phoneme_df[["path", "text"]].to_string(index=False))
