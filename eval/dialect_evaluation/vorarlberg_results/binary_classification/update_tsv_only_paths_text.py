#!/usr/bin/env python3
"""
Filter a TSV to only rows with label_id == 8 and keep only the columns
`path` and `text`. The input file is rewritten in place atomically.

Usage:
  python update_tsv_only_paths_text.py
"""

from __future__ import annotations

import csv
import os
import sys
import tempfile
from pathlib import Path
from typing import List

REPO_ROOT = Path(os.environ.get("THESIS_ROOT", Path(__file__).resolve().parents[4]))

TSV_PATH = str(REPO_ROOT / "eval/dialect_evaluation/vorarlberg_results/binary_classification/vorarlberg_finalized.tsv")

def filter_tsv_in_place(tsv_path: str) -> None:
    if not os.path.isfile(tsv_path):
        raise FileNotFoundError(f"TSV not found: {tsv_path}")

    directory = os.path.dirname(tsv_path) or "."

    # Create temp file in same directory for atomic replace
    fd, tmp_path = tempfile.mkstemp(dir=directory, suffix=".tmp")
    os.close(fd)

    wrote_any = False

    try:
        with open(tsv_path, "r", encoding="utf-8", newline="") as fin, open(
            tmp_path, "w", encoding="utf-8", newline=""
        ) as fout:
            reader = csv.reader(fin, delimiter="\t")
            writer = csv.writer(fout, delimiter="\t")

            try:
                header: List[str] = next(reader)
            except StopIteration:
                # Empty file, just write minimal header and finish
                writer.writerow(["path", "text"])  # still produce a header
                os.replace(tmp_path, tsv_path)
                return

            # Locate required columns by name
            try:
                path_idx = header.index("path")
                text_idx = header.index("text")
                label_idx = header.index("label_id")
            except ValueError as err:
                raise ValueError(
                    "Input TSV must contain columns: 'path', 'text', 'label_id'"
                ) from err

            # Write new header with only path and text
            writer.writerow(["path", "text"])

            for row in reader:
                # Guard against short/irregular rows
                if len(row) <= max(path_idx, text_idx, label_idx):
                    continue
                if row[label_idx] == "8":
                    writer.writerow([row[path_idx], row[text_idx]])
                    wrote_any = True

        # Replace original with filtered version atomically
        os.replace(tmp_path, tsv_path)

        # Optional: warn if no rows matched
        if not wrote_any:
            print("No rows with label_id == 8 found. File now contains only header.", file=sys.stderr)
    finally:
        # Clean up temp file on error paths
        try:
            if os.path.exists(tmp_path):
                # If replace succeeded, tmp_path no longer exists; ignore errors
                os.remove(tmp_path)
        except OSError:
            pass


def main() -> None:
    filter_tsv_in_place(TSV_PATH)


if __name__ == "__main__":
    main()
