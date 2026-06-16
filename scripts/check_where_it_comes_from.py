#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Trace where each 'missing' clip_path comes from.

For every clip_path in:
  /home/ai/AI-DataPool/Datasets/audio/Schweiz/16000_mono_wav_splitted_1/missing_phonemes_full_rows.v3.tsv

…report membership in:
  - master_index.csv                               (base list)
  - master_index_with_phonemes.tsv                 (original phoneme dump)
      * and whether original row's phoneme was non-empty
  - master_index_with_phonemes.rebuilt.v3.tsv      (your rebuilt list)

Outputs:
  - missing_paths_origin_report.tsv  (one row per missing clip_path with flags)
  - also prints a short summary + a few examples per category
"""

import csv
import os
from collections import defaultdict
from tqdm import tqdm

BASE_CSV = "/home/ai/AI-DataPool/Datasets/audio/Schweiz/16000_mono_wav_splitted_1/master_index.csv"
ORIG_TSV = "/home/ai/AI-DataPool/Datasets/audio/Schweiz/16000_mono_wav_splitted_1/master_index_with_phonemes.tsv"
REBU_TSV = "/home/ai/AI-DataPool/Datasets/audio/Schweiz/16000_mono_wav_splitted_1/master_index_with_phonemes.rebuilt.v3.tsv"
MISS_TSV = "/home/ai/AI-DataPool/Datasets/audio/Schweiz/16000_mono_wav_splitted_1/missing_phonemes_full_rows.v3.tsv"

OUT_TSV  = "/home/ai/AI-DataPool/Datasets/audio/Schweiz/16000_mono_wav_splitted_1/missing_paths_origin_report.tsv"
SAMPLE_N = 5  # how many examples to print per category

def is_nonempty_phoneme(s: str) -> bool:
    if s is None:
        return False
    v = s.strip()
    if v == "":
        return False
    lv = v.lower()
    if lv in {"nan", "null", "none"}:
        return False
    return True

def read_missing_paths(miss_path: str) -> set:
    missing = set()
    with open(miss_path, encoding="utf-8", errors="replace", newline="") as f:
        r = csv.reader(f, delimiter="\t")
        header = next(r, None)
        # If header present and first col is literal "clip_path", skip it safely
        if header and header[0].strip() != "clip_path":
            # No header—treat the first row as data
            missing.add(header[0])
        for row in r:
            if not row: continue
            missing.add(row[0])
    return missing

def mark_presence_from_base(missing: set, report: dict):
    with open(BASE_CSV, encoding="utf-8", errors="replace", newline="") as f:
        r = csv.reader(f, delimiter=",")
        header = next(r, None)
        # If header starts with clip_path, ok; otherwise assume first column is clip_path
        for row in tqdm(r, desc="Scanning base CSV", unit="rows"):
            if not row: continue
            cp = row[0]
            if cp in missing:
                report[cp]["in_base"] = 1

def mark_presence_from_rebuilt(missing: set, report: dict):
    with open(REBU_TSV, encoding="utf-8", errors="replace", newline="") as f:
        r = csv.reader(f, delimiter="\t")
        header = next(r, None)  # assume header present
        # Find phoneme index if available
        ph_idx = None
        if header:
            try:
                ph_idx = header.index("phoneme")
            except ValueError:
                ph_idx = len(header) - 1
        for row in tqdm(r, desc="Scanning rebuilt TSV", unit="rows"):
            if not row: continue
            cp = row[0]
            if cp in missing:
                report[cp]["in_rebuilt"] = 1
                # (Optional) sanity: rebuilt phoneme should be empty here; we don't enforce

def mark_presence_from_original(missing: set, report: dict):
    with open(ORIG_TSV, encoding="utf-8", errors="replace", newline="") as f:
        # The original TSV has mixed row lengths (7/9/11). We treat last column as phoneme.
        r = csv.reader(f, delimiter="\t")
        header = next(r, None)  # assume header present
        for row in tqdm(r, desc="Scanning original phoneme TSV", unit="rows"):
            if not row: continue
            cp = row[0]
            if cp in missing:
                report[cp]["in_original"] = 1
                # phoneme presumed last field regardless of row length
                ph = row[-1] if row else ""
                if is_nonempty_phoneme(ph):
                    report[cp]["orig_phoneme_nonempty"] = 1

def write_report(report: dict):
    os.makedirs(os.path.dirname(OUT_TSV) or ".", exist_ok=True)
    fields = ["clip_path", "in_base", "in_original", "orig_phoneme_nonempty", "in_rebuilt", "category"]
    with open(OUT_TSV, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t", quoting=csv.QUOTE_NONE, escapechar="\\")
        w.writerow(fields)
        for cp, rec in report.items():
            # categorize for easier human scanning
            if rec["in_original"] == 0:
                cat = "ABSENT_IN_ORIGINAL"
            elif rec["orig_phoneme_nonempty"] == 0:
                cat = "PRESENT_IN_ORIGINAL_BUT_EMPTY_PHONEME"
            else:
                cat = "PRESENT_IN_ORIGINAL_WITH_PHONEME"  # rare/diagnostic

            w.writerow([
                cp,
                rec["in_base"],
                rec["in_original"],
                rec["orig_phoneme_nonempty"],
                rec["in_rebuilt"],
                cat
            ])

def print_summary(report: dict):
    total = len(report)
    c_base = sum(1 for r in report.values() if r["in_base"])
    c_orig = sum(1 for r in report.values() if r["in_original"])
    c_rebu = sum(1 for r in report.values() if r["in_rebuilt"])
    c_abs  = sum(1 for r in report.values() if r["in_original"] == 0)
    c_emp  = sum(1 for r in report.values() if r["in_original"] == 1 and r["orig_phoneme_nonempty"] == 0)
    c_ok   = sum(1 for r in report.values() if r["in_original"] == 1 and r["orig_phoneme_nonempty"] == 1)

    print("\n=== Summary over missing list ===")
    print(f"Total missing clip_paths          : {total:,}")
    print(f"Present in base CSV               : {c_base:,}")
    print(f"Present in original-phoneme TSV   : {c_orig:,}")
    print(f"Present in rebuilt TSV            : {c_rebu:,}")
    print(f"  ├─ Absent in original           : {c_abs:,}")
    print(f"  ├─ Present but empty phoneme    : {c_emp:,}")
    print(f"  └─ Present with non-empty phnm  : {c_ok:,}  (diagnostic; suggests path mismatch elsewhere)")

    # Show small samples for each category
    buckets = defaultdict(list)
    for cp, r in report.items():
        if r["in_original"] == 0:
            buckets["ABSENT_IN_ORIGINAL"].append(cp)
        elif r["orig_phoneme_nonempty"] == 0:
            buckets["EMPTY_IN_ORIGINAL"].append(cp)
        else:
            buckets["HAS_PHONEME_IN_ORIGINAL"].append(cp)

    def sample_print(name, arr):
        print(f"\nExamples: {name} ({min(len(arr), SAMPLE_N)} shown)")
        for cp in sorted(arr)[:SAMPLE_N]:
            print("  -", cp)

    if buckets["ABSENT_IN_ORIGINAL"]:
        sample_print("Absent in original", buckets["ABSENT_IN_ORIGINAL"])
    if buckets["EMPTY_IN_ORIGINAL"]:
        sample_print("Present but phoneme empty", buckets["EMPTY_IN_ORIGINAL"])
    if buckets["HAS_PHONEME_IN_ORIGINAL"]:
        sample_print("Present and phoneme non-empty (diagnostic)", buckets["HAS_PHONEME_IN_ORIGINAL"])

def main():
    print("Loading missing clip_paths…")
    missing = read_missing_paths(MISS_TSV)
    print(f"Missing set size: {len(missing):,}")

    # Initialize report dict only for missing paths
    report = {cp: {"in_base":0, "in_original":0, "orig_phoneme_nonempty":0, "in_rebuilt":0} for cp in missing}

    # Scan each corpus and mark memberships
    mark_presence_from_base(missing, report)
    mark_presence_from_original(missing, report)
    mark_presence_from_rebuilt(missing, report)

    # Output + summary
    write_report(report)
    print_summary(report)
    print(f"\nWrote detailed report to: {OUT_TSV}")

if __name__ == "__main__":
    main()
