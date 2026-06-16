#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Ultra-fast validator for rebuilt master TSV (streaming, low memory).

Checks:
  1) Row counts match base CSV (minus headers)
  2) Same set of clip_path in BASE (CSV) and REBUILT (TSV)
  3) No duplicates of clip_path in REBUILT
  4) Phoneme column has no empty/placeholder values
  5) Duration consistency: |(end-start) - duration_sec| <= tolerance
  6) Random sample of clip_paths exist on disk

Outputs small report files next to the rebuilt TSV:
  - only_in_base.txt
  - only_in_rebuilt.txt
  - duplicates.txt
  - duration_mismatch.sample.tsv
  - missing_audio.sample.txt
  - summary.json
"""

import argparse
import csv
import io
import json
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional, Tuple

# ---------- DEFAULT PATHS (edit if you like) ----------
DEF_BASE = "/home/ai/AI-DataPool/Datasets/audio/Schweiz/16000_mono_wav_splitted_1/master_index.csv"
DEF_REB  = "/home/ai/AI-DataPool/Datasets/audio/Schweiz/16000_mono_wav_splitted_1/master_index_with_phonemes.rebuilt.v3.tsv"
# ------------------------------------------------------


def fast_count_lines(path: str) -> int:
    """Count lines quickly (binary read), returns total lines."""
    n = 0
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            n += chunk.count(b"\n")
    return n


def read_header_and_index(path: str, sep: str, want: str) -> int:
    """Return column index of 'want' in header for given file."""
    with open(path, "r", encoding="utf-8", errors="replace", newline="") as f:
        r = csv.reader(f, delimiter=sep)
        header = next(r, None)
        if not header:
            raise RuntimeError(f"Empty file: {path}")
        try:
            return header.index(want)
        except ValueError:
            raise RuntimeError(f"Column '{want}' not found in header of {path}.\nHeader: {header}")


def extract_column_to_file(src_path: str, sep: str, colname: str, out_txt: Path) -> None:
    """Stream extract 'colname' to one-value-per-line TXT file."""
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    idx = read_header_and_index(src_path, sep, colname)
    with open(src_path, "r", encoding="utf-8", errors="replace", newline="") as fin, \
         open(out_txt, "w", encoding="utf-8", newline="") as fout:
        r = csv.reader(fin, delimiter=sep)
        next(r, None)  # skip header
        for row in r:
            if not row:
                continue
            val = row[idx] if idx < len(row) else ""
            fout.write(val + "\n")


def have_cmd(name: str) -> bool:
    return shutil.which(name) is not None


def external_sort_unique(in_path: Path, out_path: Path, mem: str = "50%") -> None:
    """Use system sort -u for speed; fallback to Python if missing (slower)."""
    if have_cmd("sort"):
        env = os.environ.copy()
        env.setdefault("LC_ALL", "C")
        cmd = ["sort", f"--parallel={os.cpu_count() or 1}", "-S", mem, "-u", str(in_path)]
        with open(out_path, "w", encoding="utf-8", newline="") as fout:
            subprocess.run(cmd, stdout=fout, check=True, env=env)
    else:
        # Python fallback (may be slow / memory heavy on very large files)
        with open(in_path, "r", encoding="utf-8") as f:
            data = sorted(set(line.rstrip("\n") for line in f))
        with open(out_path, "w", encoding="utf-8", newline="") as f:
            for s in data:
                f.write(s + "\n")


def external_sort_all(in_path: Path, out_path: Path, mem: str = "50%") -> None:
    """Full external sort (not unique)."""
    if have_cmd("sort"):
        env = os.environ.copy()
        env.setdefault("LC_ALL", "C")
        cmd = ["sort", f"--parallel={os.cpu_count() or 1}", "-S", mem, str(in_path)]
        with open(out_path, "w", encoding="utf-8", newline="") as fout:
            subprocess.run(cmd, stdout=fout, check=True, env=env)
    else:
        with open(in_path, "r", encoding="utf-8") as f:
            data = sorted(line.rstrip("\n") for line in f)
        with open(out_path, "w", encoding="utf-8", newline="") as f:
            for s in data:
                f.write(s + "\n")


def merge_diff(a_sorted: Path, b_sorted: Path, only_a: Path, only_b: Path) -> Tuple[int, int]:
    """Given two sorted unique lists, write differences and return counts."""
    only_a.parent.mkdir(parents=True, exist_ok=True)
    only_b.parent.mkdir(parents=True, exist_ok=True)
    c_a = c_b = 0
    with open(a_sorted, "r", encoding="utf-8") as fa, \
         open(b_sorted, "r", encoding="utf-8") as fb, \
         open(only_a, "w", encoding="utf-8", newline="") as oa, \
         open(only_b, "w", encoding="utf-8", newline="") as ob:

        la = fa.readline()
        lb = fb.readline()
        while la or lb:
            sa = la.rstrip("\n") if la else None
            sb = lb.rstrip("\n") if lb else None
            if sa is None:
                # remaining in b
                ob.write(sb + "\n")
                c_b += 1
                lb = fb.readline()
            elif sb is None:
                oa.write(sa + "\n")
                c_a += 1
                la = fa.readline()
            else:
                if sa == sb:
                    la = fa.readline()
                    lb = fb.readline()
                elif sa < sb:
                    oa.write(sa + "\n")
                    c_a += 1
                    la = fa.readline()
                else:
                    ob.write(sb + "\n")
                    c_b += 1
                    lb = fb.readline()
    return c_a, c_b


def count_duplicates_from_sorted(sorted_all: Path, out_dups: Path) -> int:
    """Scan a fully sorted list (with repeats) and write items that appear >1."""
    out_dups.parent.mkdir(parents=True, exist_ok=True)
    dups = 0
    prev = None
    count = 0
    with open(sorted_all, "r", encoding="utf-8") as f, \
         open(out_dups, "w", encoding="utf-8", newline="") as out:
        for line in f:
            s = line.rstrip("\n")
            if s == prev:
                count += 1
            else:
                if prev is not None and count > 1:
                    out.write(prev + "\n")
                    dups += 1
                prev = s
                count = 1
        # tail
        if prev is not None and count > 1:
            out.write(prev + "\n")
            dups += 1
    return dups


def check_phoneme_and_duration(
    reb_path: str,
    out_mismatch_sample: Path,
    phoneme_col: str = "phoneme",
    tol: float = 0.02,
    sample_limit: int = 2000
) -> Tuple[int, int, int, int]:
    """
    Stream rebuilt TSV and:
      - count empty/placeholder phonemes
      - duration consistency mismatches (write small sample)
    Returns: (rows_total, phon_empty, dur_bad, ipa_hits)
    """
    with open(reb_path, "r", encoding="utf-8", errors="replace", newline="") as f, \
         open(out_mismatch_sample, "w", encoding="utf-8", newline="") as fout:

        r = csv.reader(f, delimiter="\t")
        header = next(r, None)
        if not header:
            raise RuntimeError("Rebuilt TSV appears empty.")
        try:
            i_clip = header.index("clip_path")
            i_ph   = header.index(phoneme_col)
            i_st   = header.index("start")
            i_en   = header.index("end")
            i_du   = header.index("duration_sec")
        except ValueError as e:
            raise RuntimeError(f"Expected header columns missing: {e}\nHeader: {header}")

        w = csv.writer(fout, delimiter="\t", quoting=csv.QUOTE_NONE, escapechar="\\")
        w.writerow(header)

        total = empty = bad = ipa = 0
        ipa_chars = "ːʃʒŋɲɡɾʊəɔœɐʏθðçɕʑɽɸβɣʁʔ"
        for row in r:
            total += 1
            ph = row[i_ph].strip() if i_ph < len(row) else ""
            q = ph.lower()
            if (not ph) or q in {"nan", "null", "none", "no_phoneme", "error"}:
                empty += 1
            if any(ch in ph for ch in ipa_chars):
                ipa += 1

            # duration consistency
            try:
                st = float(row[i_st]) if i_st < len(row) else 0.0
                en = float(row[i_en]) if i_en < len(row) else 0.0
                du = float(row[i_du]) if i_du < len(row) else 0.0
                diff = abs((en - st) - du)
                if diff > tol:
                    if bad < sample_limit:
                        w.writerow(row)
                    bad += 1
            except Exception:
                # if parsing fails, count as bad and sample it
                if bad < sample_limit:
                    w.writerow(row)
                bad += 1

    return total, empty, bad, ipa


def reservoir_sample_paths(reb_path: str, k: int) -> list:
    """Reservoir-sample K clip_paths from rebuilt TSV."""
    sample = []
    with open(reb_path, "r", encoding="utf-8", errors="replace", newline="") as f:
        r = csv.reader(f, delimiter="\t")
        header = next(r, None)
        i_clip = header.index("clip_path")
        n = 0
        for row in r:
            cp = row[i_clip]
            n += 1
            if len(sample) < k:
                sample.append(cp)
            else:
                j = random.randint(1, n)
                if j <= k:
                    sample[j - 1] = cp
    return sample


def check_paths_exist(paths: list, out_txt: Path) -> int:
    """Check existence of sampled paths; write missing ones; return count missing."""
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    missing = 0
    with open(out_txt, "w", encoding="utf-8", newline="") as f:
        for p in paths:
            if not os.path.exists(p):
                f.write(p + "\n")
                missing += 1
    return missing


def main():
    ap = argparse.ArgumentParser(description="Fast validator for rebuilt TSV (streaming).")
    ap.add_argument("--base", default=DEF_BASE, help="Path to base CSV (master_index.csv)")
    ap.add_argument("--rebuilt", default=DEF_REB, help="Path to rebuilt TSV")
    ap.add_argument("--mem", default="50%", help="Sort memory (e.g., 4G or 50%%)")
    ap.add_argument("--tol", type=float, default=0.02, help="Duration tolerance in seconds")
    ap.add_argument("--sample", type=int, default=2000, help="Sample size for on-disk path check")
    args = ap.parse_args()

    base = Path(args.base)
    reb  = Path(args.rebuilt)
    outdir = reb.parent

    if not base.exists():
        print(f"Base CSV not found: {base}", file=sys.stderr); sys.exit(1)
    if not reb.exists():
        print(f"Rebuilt TSV not found: {reb}", file=sys.stderr); sys.exit(1)

    print("=== 0) Row count check ===")
    base_lines = fast_count_lines(str(base))
    reb_lines  = fast_count_lines(str(reb))
    base_rows = max(0, base_lines - 1)
    reb_rows  = max(0, reb_lines - 1)
    print(f"Base rows (no header): {base_rows:,}")
    print(f"Rebuilt rows (no header): {reb_rows:,}")
    rows_ok = (base_rows == reb_rows)
    print(f"Row-count match: {rows_ok}")

    print("\n=== 1) Coverage equality (clip_path set) ===")
    base_clip = outdir / "base.clip.txt"
    reb_clip  = outdir / "reb.clip.txt"
    base_uniq = outdir / "base.clip.u.txt"
    reb_uniq  = outdir / "reb.clip.u.txt"
    only_in_base = outdir / "only_in_base.txt"
    only_in_reb  = outdir / "only_in_rebuilt.txt"

    extract_column_to_file(str(base), ",", "clip_path", base_clip)
    extract_column_to_file(str(reb),  "\t", "clip_path", reb_clip)
    external_sort_unique(base_clip, base_uniq, args.mem)
    external_sort_unique(reb_clip,  reb_uniq,  args.mem)
    miss, extra = merge_diff(base_uniq, reb_uniq, only_in_base, only_in_reb)
    print(f"ONLY_IN_BASE     : {miss:,}  → {only_in_base}")
    print(f"ONLY_IN_REBUILT  : {extra:,}  → {only_in_reb}")
    coverage_ok = (miss == 0 and extra == 0)
    print(f"Coverage OK      : {coverage_ok}")

    print("\n=== 2) Duplicates in rebuilt (clip_path) ===")
    reb_sorted_all = outdir / "reb.clip.sorted.txt"
    dups_out = outdir / "duplicates.txt"
    external_sort_all(reb_clip, reb_sorted_all, args.mem)
    dup_count = count_duplicates_from_sorted(reb_sorted_all, dups_out)
    print(f"Duplicates found : {dup_count:,}  → {dups_out}")
    dups_ok = (dup_count == 0)
    print(f"No duplicates    : {dups_ok}")

    print("\n=== 3) Phoneme empties & 4) Duration consistency ===")
    dur_sample = outdir / "duration_mismatch.sample.tsv"
    total, empty, dur_bad, ipa_hits = check_phoneme_and_duration(
        str(reb), dur_sample, tol=args.tol
    )
    print(f"Rows total                : {total:,}")
    print(f"Empty/placeholder phoneme : {empty:,}  ({(empty/total*100):.5f}%)")
    print(f"Duration mismatches       : {dur_bad:,}  → sample: {dur_sample}")
    print(f"IPA-ish rows (rough)      : {ipa_hits:,}  ({(ipa_hits/total*100):.2f}%)")

    print("\n=== 5) Random sample on-disk path existence ===")
    sample_paths = reservoir_sample_paths(str(reb), args.sample)
    missing_audio_list = outdir / "missing_audio.sample.txt"
    miss_audio = check_paths_exist(sample_paths, missing_audio_list)
    print(f"Missing audio in sample   : {miss_audio:,} / {len(sample_paths)}  → {missing_audio_list}")

    # Summary JSON
    summary = {
        "rows_ok": rows_ok,
        "coverage_ok": coverage_ok,
        "dups_ok": dups_ok,
        "counts": {
            "base_rows": base_rows,
            "rebuilt_rows": reb_rows,
            "only_in_base": miss,
            "only_in_rebuilt": extra,
            "duplicates": dup_count,
            "total_rows": total,
            "empty_phoneme": empty,
            "duration_mismatch": dur_bad,
            "ipa_hits": ipa_hits,
            "sample_missing_audio": miss_audio,
            "sample_size": len(sample_paths),
        },
        "reports": {
            "only_in_base": str(only_in_base),
            "only_in_rebuilt": str(only_in_reb),
            "duplicates": str(dups_out),
            "duration_mismatch_sample": str(dur_sample),
            "missing_audio_sample": str(missing_audio_list),
        }
    }
    summary_path = outdir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n✓ Wrote summary → {summary_path}")

    # Exit non-zero if any major checks failed (useful for pipelines/CI)
    if not (rows_ok and coverage_ok and dups_ok and empty == 0):
        sys.exit(2)


if __name__ == "__main__":
    # Make sort/merge stable & fast
    os.environ.setdefault("LC_ALL", "C")
    random.seed(0xC0FFEE)
    main()
