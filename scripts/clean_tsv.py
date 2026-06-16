#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Rebuild a clean master TSV by harvesting phonemes from a mixed-format source TSV.

Understands these row shapes in the original TSV:
- 11 cols: [ ... id, flag, phoneme ]        → phoneme at [10]
- 9  cols: [ ... phoneme ]                   → phoneme at [8]
- 7  cols: [cp, start, end, dur, id, 0, ph ] → phoneme at [6]
- 6  cols: [cp, start, end, dur, id, ph ]    → phoneme at [5]
- Fallback: if last column looks like tokenized phonemes, use it.

Also processes missing_phonemes_full_rows.v3_with_phonemes.tsv to add additional
phonemes that were previously missing.

Then joins clip_path → phoneme onto the base CSV and writes a rebuilt TSV.
Also emits a TSV with full rows for entries still missing phonemes.

Paths are hardcoded; adjust if needed.
"""

import os, csv
from tqdm import tqdm

# ---------- PATHS ----------
ORIG_TSV = "/home/ai/AI-DataPool/Datasets/audio/Schweiz/16000_mono_wav_splitted_1/master_index_with_phonemes.tsv"
BASE_CSV = "/home/ai/AI-DataPool/Datasets/audio/Schweiz/16000_mono_wav_splitted_1/master_index.csv"
OUT_TSV  = "/home/ai/AI-DataPool/Datasets/audio/Schweiz/16000_mono_wav_splitted_1/master_index_with_phonemes.rebuilt.v3.tsv"
MISSING_FULL_ROWS = "/home/ai/AI-DataPool/Datasets/audio/Schweiz/16000_mono_wav_splitted_1/missing_phonemes_full_rows.v3.tsv"
MISSING_WITH_PHONEMES = "/home/ai/AI-DataPool/Datasets/audio/Schweiz/16000_mono_wav_splitted_1/missing_phonemes_full_rows.v3_with_phonemes.tsv"
# ---------------------------


def looks_number(s: str) -> bool:
    s = (s or "").strip()
    if not s:
        return False
    if s[0] in "+-":
        s = s[1:]
    # digits / one dot allowed
    return all(c.isdigit() or c == "." for c in s)


def looks_phoneme_like(s: str) -> bool:
    """Very light heuristic: has spaces and at least one non-ASCII letter or IPA-ish char."""
    if not s:
        return False
    t = s.strip()
    if not t or t.lower() in {"nan", "null", "none"}:
        return False
    if " " not in t:
        return False
    # if last token set is clearly not numeric and includes IPA-ish marks, good enough
    ipa_hints = "ːʃʒŋɲɡɾʊəɔœɐʏθðçɕʑɽɸβɣʁʔ"
    if any(ch in t for ch in ipa_hints):
        return True
    # otherwise, accept if it has many tokens and not all are numeric-ish
    tokens = t.split()
    nonnum = sum(not looks_number(tok) for tok in tokens)
    return nonnum >= max(3, len(tokens) // 2)


def choose_better(old, new):
    """
    Prefer richer source type (11/9 over 7/6 over M), else longer phoneme string.
    Returns (src_type, phoneme).
    """
    old_src, old_ph = old
    new_src, new_ph = new
    rank = {"11": 4, "9": 4, "7": 3, "6": 3, "M": 2, "F": 1}  # missing file medium priority
    if rank[new_src] != rank[old_src]:
        return new if rank[new_src] > rank[old_src] else old
    return new if len(new_ph) > len(old_ph) else old


def harvest_phonemes_from_missing_file(path, pmap):
    """Add phonemes from missing_phonemes_full_rows.v3_with_phonemes.tsv to existing pmap."""
    if not os.path.exists(path):
        print(f"Missing phonemes file not found: {path}")
        return pmap

    totals = {"total": 0, "nonempty": 0, "skipped": 0, "updated": 0, "new": 0}

    with open(path, "r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        
        for row in tqdm(reader, desc="Scanning missing phonemes file", unit="lines"):
            totals["total"] += 1
            if not row:
                totals["skipped"] += 1
                continue

            cp = row.get("clip_path", "").strip()
            ph = row.get("phoneme", "").strip()
            
            if not cp or not ph or ph.lower() in {"nan", "null", "none"}:
                totals["skipped"] += 1
                continue

            totals["nonempty"] += 1
            if cp in pmap:
                # Choose better phoneme if we already have one
                old_src, old_ph = pmap[cp]
                new_entry = ("M", ph)  # M for missing file
                pmap[cp] = choose_better((old_src, old_ph), new_entry)
                totals["updated"] += 1
            else:
                pmap[cp] = ("M", ph)  # M for missing file
                totals["new"] += 1

    print("\nMissing phonemes file stats:")
    for k in ("total", "nonempty", "skipped", "updated", "new"):
        print(f"{k:>10}: {totals[k]:,}")
    return pmap


def harvest_phonemes_from_original(path):
    """Build clip_path → (src_type, phoneme) from mixed-format original TSV."""
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    pmap = {}  # clip_path -> (src_type, phoneme)
    totals = {"total":0, "len11":0, "len9":0, "len7":0, "len6":0, "fallback":0, "skipped":0, "nonempty":0}

    with open(path, "r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        _ = next(reader, None)  # header (applies to 9/11-col blocks)

        for row in tqdm(reader, desc="Scanning original TSV for phonemes", unit="lines"):
            totals["total"] += 1
            if not row:
                totals["skipped"] += 1
                continue

            n = len(row)
            cp = row[0]

            src_type = None
            ph = ""

            if n == 11:
                totals["len11"] += 1
                ph = row[10]
                src_type = "11"
            elif n == 9:
                totals["len9"] += 1
                ph = row[8]
                src_type = "9"
            elif n == 7:
                # [cp, start, end, dur, id, flag, phoneme]
                totals["len7"] += 1
                ph = row[6]
                src_type = "7"
            elif n == 6:
                # [cp, start, end, dur, id, phoneme]
                totals["len6"] += 1
                ph = row[5]
                src_type = "6"
            else:
                # Fallback: last column sometimes is the phoneme blob
                last = row[-1]
                if looks_phoneme_like(last):
                    totals["fallback"] += 1
                    ph = last
                    src_type = "F"
                else:
                    totals["skipped"] += 1
                    continue

            phs = (ph or "").strip()
            if not phs or phs.lower() in {"nan", "null", "none"}:
                # empty phoneme → ignore
                continue

            totals["nonempty"] += 1
            if cp in pmap:
                pmap[cp] = choose_better(pmap[cp], (src_type, phs))
            else:
                pmap[cp] = (src_type, phs)

    print("\nPhoneme-map build stats:")
    for k in ("total","len11","len9","len7","len6","fallback","nonempty","skipped"):
        print(f"{k:>10}: {totals[k]:,}")
    print(f"unique clip_paths with phonemes: {len(pmap):,}")
    return pmap


def write_rebuilt(base_csv, pmap, out_tsv, missing_out):
    if not os.path.exists(base_csv):
        raise FileNotFoundError(base_csv)

    os.makedirs(os.path.dirname(out_tsv) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(missing_out) or ".", exist_ok=True)

    # Read base header to preserve order; ensure 'phoneme' exists (append if not)
    with open(base_csv, "r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.reader(f)
        base_header = next(reader)
        if "phoneme" not in base_header:
            base_header = base_header + ["phoneme"]

    written = 0
    have = 0

    with open(base_csv, "r", encoding="utf-8", errors="replace", newline="") as fin, \
         open(out_tsv,  "w", encoding="utf-8", newline="") as fout, \
         open(missing_out, "w", encoding="utf-8", newline="") as fmiss:

        src = csv.DictReader(fin)
        out = csv.writer(fout, delimiter="\t", quoting=csv.QUOTE_NONE, escapechar="\\")
        mis = csv.writer(fmiss, delimiter="\t", quoting=csv.QUOTE_NONE, escapechar="\\")

        out.writerow(base_header)
        mis.writerow(base_header)

        for row in tqdm(src, desc="Writing rebuilt TSV", unit="rows"):
            cp = row.get("clip_path") or row.get("clip") or row.get("path") or ""
            ph = pmap.get(cp, ("", ""))[1]

            # Materialize record in base_header order (phoneme last)
            record = [row.get(col, "") for col in base_header if col != "phoneme"] + [ph]
            out.writerow(record)

            written += 1
            if ph.strip():
                have += 1
            else:
                mis.writerow(record)

    print("\nDone.")
    print(f"  total rows written      : {written:,}")
    print(f"  rows with phonemes      : {have:,}")
    print(f"  rows without phonemes   : {written - have:,}")
    print(f"  output path             : {out_tsv}")
    print(f"  missing rows path       : {missing_out}")


def main():
    pmap = harvest_phonemes_from_original(ORIG_TSV)
    pmap = harvest_phonemes_from_missing_file(MISSING_WITH_PHONEMES, pmap)
    write_rebuilt(BASE_CSV, pmap, OUT_TSV, MISSING_FULL_ROWS)


if __name__ == "__main__":
    main()
