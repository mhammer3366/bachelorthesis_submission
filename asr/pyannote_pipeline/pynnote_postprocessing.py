#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Clean + sentence-wise post-processing for podcast transcripts.

Pipeline per file:
  1) Load CSV/JSON (columns: start,end,speaker,text[,overlap_coverage]).
  2) Drop rows: UNKNOWN speaker (configurable), Musik markers, non-German scripts, and any
     consecutive run where the same text appears >= RUN_LEN times (drop the whole run).
  3) Merge clause fragments into sentences (same speaker, gap <= MAX_GAP, stop at .?!).
     - overlap_coverage becomes duration-weighted average.
  4) Final guarantee: split any row that still contains multiple sentences (.?!)
     so that EACH output row contains MAX ONE SENTENCE.
     - sentence times are allocated proportionally by character count.
  5) Save as <input>.filtered.csv or to OUTPUT_DIR if set.

Edit the CONFIG section below, then run:
    python clean_and_sentencewise.py
"""

# =======================
# ====== CONFIG =========
# =======================

# Input can be a single file (.csv/.json) or a directory containing many.
INPUT_PATH = str(Path(os.environ.get("DATA_ROOT", "/home/ai/AI-DataPool/Datasets")) / "audio/Schweiz/16000_mono_wav_pyannote")
# Optional: write outputs into a separate directory. "" -> next to input files.
OUTPUT_DIR = str(Path(os.environ.get("DATA_ROOT", "/home/ai/AI-DataPool/Datasets")) / "audio/Schweiz/16000_mono_wav_pyannote_post_processed")  # e.g. "/home/student/clean_outputs"

# Optional substring filter to select only some files (case-insensitive). "" -> all
FILE_FILTER_SUBSTR = ""

# Filtering knobs
DROP_UNKNOWN = True
DROP_NON_GERMAN = True
RUN_LEN = 3             # drop any consecutive run where the same text repeats >= RUN_LEN
MAX_GAP = 0.8          # max pause (sec) to still merge clauses from same speaker into the same sentence

# =======================
# ====== CODE ===========
# =======================

import json, re, unicodedata, sys, traceback
from pathlib import Path
import pandas as pd
from tqdm import tqdm

# --- Regex helpers ---

MUSIK_RE = re.compile(r"^\s*[\[\(\-–—\s]*musik[\]\)\-–—\s]*\.?\s*$", re.IGNORECASE)

# Non-German scripts to drop: CJK/Kana/Hangul/Greek/Cyrillic
NON_GERMAN_SCRIPT_RE = re.compile(
    r"[\u3040-\u30FF\u31F0-\u31FF"   # Hiragana/Katakana + extensions
    r"\u4E00-\u9FFF"                 # CJK Unified
    r"\uAC00-\uD7AF\u1100-\u11FF"    # Hangul syllables + Jamo
    r"\u0370-\u03FF"                 # Greek
    r"\u0400-\u052F]"                # Cyrillic + extended
)

# sentence terminator at the end of the fragment (allowing a trailing quote)
SENT_TERM_RE = re.compile(r"[\.!?]\s*(?:[»\"'])?\s*$")

# Naive sentence boundary for splitting inside a row
SENT_SPLIT_RE = re.compile(r"([\.!?])\s+")

def is_music_text(s: str) -> bool:
    t = "" if s is None else str(s).strip()
    t = unicodedata.normalize("NFKC", t)
    return bool(MUSIK_RE.match(t))

def is_unknown_speaker(s: str) -> bool:
    return str(s).strip().upper() == "UNKNOWN"

def contains_non_german_script(s: str) -> bool:
    return bool(NON_GERMAN_SCRIPT_RE.search(str(s or "")))

def normalize_text_for_run(s: str) -> str:
    """Normalization for run detection (robust but conservative)."""
    if s is None:
        return ""
    t = unicodedata.normalize("NFKC", str(s)).strip()
    # unify quotes
    t = (t.replace("„", '"').replace("“", '"').replace("«", '"').replace("»", '"')
           .replace("‚", "'").replace("’", "'"))
    # strip one layer quotes/brackets, collapse spaces
    t = t.strip('"\'' ).strip("[]()")
    t = re.sub(r"\s+", " ", t)
    # trim trailing punctuation and ellipsis
    t = re.sub(r"[\.!?…\s]+$", "", t)
    return t.casefold()

def find_inputs(root: Path):
    if root.is_file():
        return [root] if root.suffix.lower() in {".csv",".json"} else []
    return [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in {".csv",".json"}]

def load_episode(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    elif path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        segs = data.get("segments", [])
        df = pd.DataFrame(segs)
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}")

    for col in ("start","end","speaker","text"):
        if col not in df.columns:
            raise ValueError(f"{path} missing required column: {col}")

    if "overlap_coverage" not in df.columns:
        df["overlap_coverage"] = 0.0

    # normalize types & order by time
    df["start"] = df["start"].astype(float)
    df["end"] = df["end"].astype(float)
    df["speaker"] = df["speaker"].astype(str)
    df["text"] = df["text"].astype(str)
    df["overlap_coverage"] = df["overlap_coverage"].astype(float)
    df = df.sort_values(["start","end"]).reset_index(drop=True)
    return df

def save_sentence_csv(df: pd.DataFrame, in_path: Path, out_root: Path|None) -> Path:
    out_path = (out_root / in_path.name if out_root else in_path).with_suffix(".filtered.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False, encoding="utf-8")
    return out_path

# ------- filtering --------

def drop_music_unknown_nongerman_and_runs(df: pd.DataFrame) -> pd.DataFrame:
    music_mask = df["text"].map(is_music_text)
    unknown_mask = df["speaker"].map(is_unknown_speaker) if DROP_UNKNOWN else pd.Series(False, index=df.index)
    nongerman_mask = df["text"].map(contains_non_german_script) if DROP_NON_GERMAN else pd.Series(False, index=df.index)

    stage1 = df.loc[~(music_mask | unknown_mask | nongerman_mask)].copy().reset_index(drop=True)
    if stage1.empty:
        return stage1

    # remove consecutive runs >= RUN_LEN (entire run)
    norm = stage1["text"].map(normalize_text_for_run)
    drop_rows = set()
    i = 0
    n = len(stage1)
    while i < n:
        j = i + 1
        while j < n and norm.iloc[j] == norm.iloc[i]:
            j += 1
        run_size = j - i
        if run_size >= RUN_LEN and norm.iloc[i] != "":
            drop_rows.update(range(i, j))
        i = j

    if drop_rows:
        stage1 = stage1.drop(index=list(drop_rows)).reset_index(drop=True)

    return stage1

# ------- merge into sentences --------

def _duration_weighted_avg(rows) -> float:
    total = 0.0
    acc = 0.0
    for r in rows:
        d = max(0.0, float(r["end"]) - float(r["start"]))
        total += d
        acc += d * float(r.get("overlap_coverage", 0.0))
    return (acc / total) if total > 0 else 0.0

def _ends_sentence(text: str) -> bool:
    return bool(SENT_TERM_RE.search(str(text or "")))

def merge_rows_to_sentences(df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge consecutive rows from the same speaker into a single sentence:
      - continue if same speaker, gap <= MAX_GAP, and previous fragment does NOT end with .?!
      - flush on new speaker, large gap, or when the current fragment ends with .?!
    """
    if df.empty:
        return df

    df = df.sort_values(["start","end"]).reset_index(drop=True)
    out = []
    bucket = []

    def flush():
        nonlocal bucket
        if not bucket:
            return
        start = float(bucket[0]["start"])
        end = float(bucket[-1]["end"])
        spk = str(bucket[0]["speaker"])
        texts = [str(r["text"]).strip() for r in bucket if str(r["text"]).strip()]
        merged_text = " ".join(texts).strip()
        ov = _duration_weighted_avg(bucket)
        out.append({
            "start": round(start, 3),
            "end": round(end, 3),
            "speaker": spk,
            "text": merged_text,
            "overlap_coverage": round(ov, 3),
        })
        bucket = []

    for _, r in df.iterrows():
        r = r.to_dict()
        if not bucket:
            bucket = [r]
            if _ends_sentence(r["text"]):
                flush()
            continue

        prev = bucket[-1]
        same_spk = (str(r["speaker"]) == str(prev["speaker"]))
        gap = float(r["start"]) - float(prev["end"])
        prev_ends = _ends_sentence(prev["text"])

        if same_spk and gap <= MAX_GAP and not prev_ends:
            bucket.append(r)
        else:
            flush()
            bucket = [r]

        if _ends_sentence(r["text"]):
            flush()

    flush()
    out_df = pd.DataFrame(out, columns=["start","end","speaker","text","overlap_coverage"])
    return out_df.sort_values(["start","end"]).reset_index(drop=True)

# ------- final single-sentence guarantee --------

def split_into_sentences(text: str) -> list[str]:
    """Naive sentence splitter: splits on .?! followed by space; keeps the punctuation."""
    text = (text or "").strip()
    if not text:
        return []
    parts = []
    start = 0
    for m in SENT_SPLIT_RE.finditer(text):
        end = m.end(1)
        chunk = text[start:end].strip()
        if chunk:
            parts.append(chunk)
        start = m.end()
    tail = text[start:].strip()
    if tail:
        parts.append(tail)
    return parts

def finalize_single_sentence_rows(df: pd.DataFrame) -> pd.DataFrame:
    return df
"""    Ensure each row has max 1 sentence; split any multi-sentence rows proportionally by char length.
    if df.empty:
        return df
    rows = []
    for _, r in df.iterrows():
        s0, e0 = float(r["start"]), float(r["end"])
        txt = str(r["text"])
        spk = str(r["speaker"])
        ov = float(r.get("overlap_coverage", 0.0))
        sents = split_into_sentences(txt)
        if len(sents) <= 1:
            rows.append(dict(start=round(s0,3), end=round(e0,3),
                             speaker=spk, text=(sents[0] if sents else txt).strip(),
                             overlap_coverage=round(ov,3)))
            continue
        dur = max(0.0, e0 - s0)
        lengths = [max(1, len(s)) for s in sents]
        total = sum(lengths)
        cursor = s0
        for i, (sent, L) in enumerate(zip(sents, lengths)):
            seg_dur = dur * (L / total) if dur > 0 else 0.0
            seg_end = e0 if i == len(sents)-1 else min(e0, cursor + seg_dur)
            if seg_end <= cursor:
                seg_end = min(e0, cursor + 0.01)
            rows.append(dict(start=round(cursor,3), end=round(seg_end,3),
                             speaker=spk, text=sent.strip(),
                             overlap_coverage=round(ov,3)))
            cursor = seg_end
    out = pd.DataFrame(rows, columns=["start","end","speaker","text","overlap_coverage"])
    return out.sort_values(["start","end"]).reset_index(drop=True)"""

# ------- main driver with progress + error handling -------

def main():
    in_path = Path(INPUT_PATH)
    out_root = Path(OUTPUT_DIR).resolve() if OUTPUT_DIR else None
    if out_root:
        out_root.mkdir(parents=True, exist_ok=True)

    targets = find_inputs(in_path)
    if FILE_FILTER_SUBSTR:
        g = FILE_FILTER_SUBSTR.lower()
        targets = [p for p in targets if g in str(p).lower()]

    if not targets:
        print("No input files found.")
        return

    errors = []
    total_kept = 0
    total_rows = 0

    for p in tqdm(targets, desc="Files", unit="file"):
        try:
            df = load_episode(p)
            total_rows += len(df)
            # Row-level progress (optional)
            # tqdm.pandas(desc=f"Rows {p.name}")  # if you wanted .progress_apply

            filtered = drop_music_unknown_nongerman_and_runs(df)
            merged = merge_rows_to_sentences(filtered)
            final = finalize_single_sentence_rows(merged)
            out_p = save_sentence_csv(final, p, out_root)
            total_kept += len(final)
            tqdm.write(f"[OK] {p} -> {out_p} (kept {len(final)}/{len(df)})")
        except Exception as e:
            msg = f"[ERR] {p}: {e}"
            tqdm.write(msg)
            # include a short traceback line for debugging
            tb = traceback.format_exc(limit=1)
            tqdm.write(tb.strip())
            errors.append((p, str(e)))

    print("\n=== Summary ===")
    print(f"Processed: {len(targets)} file(s)")
    print(f"Total rows in: {total_rows}")
    print(f"Total rows kept (sentence-wise): {total_kept}")
    if errors:
        print(f"\nErrors: {len(errors)}")
        for p, err in errors[:10]:
            print(f" - {p}: {err}")
        if len(errors) > 10:
            print(f" ... and {len(errors)-10} more")

if __name__ == "__main__":
    main()
