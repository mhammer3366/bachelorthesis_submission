
import os
from pathlib import Path
REPO_ROOT = Path(os.environ.get("THESIS_ROOT", Path(__file__).resolve().parents[1]))
# add_label_id_column.py
import os
import pandas as pd

# Define your dialects in fixed order
dialects = ["Basel", "Bern", "Innerschweiz", "Ostschweiz", "Wallis", "Zürich", "Graubünden"]
dialect_to_id = {d: i for i, d in enumerate(dialects)}

in_dir = str(REPO_ROOT / "data_preparation/merged_datasets")
out_dir = str(REPO_ROOT / "data_preparation/merged_datasets_labeled")
os.makedirs(out_dir, exist_ok=True)

splits = ["train", "valid", "test"]

for split in splits:
    in_tsv = os.path.join(in_dir, f"merged_{split}.tsv")
    out_tsv = os.path.join(out_dir, f"merged_{split}.tsv")

    if not os.path.isfile(in_tsv):
        print(f"[WARN] {in_tsv} not found, skipping.")
        continue

    df = pd.read_csv(in_tsv, sep="\t")

    if "dialect_region" not in df.columns:
        print(f"[ERROR] {in_tsv} has no 'dialect_region' column!")
        continue

    # Map dialect_region to ID, drop unknowns
    df = df[df["dialect_region"].isin(dialects)].reset_index(drop=True)
    df["label_id"] = df["dialect_region"].map(dialect_to_id).astype(int)

    df.to_csv(out_tsv, sep="\t", index=False)
    print(f"[OK] wrote {out_tsv} with {len(df)} rows (label_id added)")
