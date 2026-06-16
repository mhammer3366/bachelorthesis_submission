#!/usr/bin/env python3
"""
Tirol speaker-level binary NB evaluation (German vs Dialect).

Input TSV under $DATA_ROOT/audio/Österreich/; model under $THESIS_ROOT/models/.
Aggregates chunk predictions per speaker (median prob → label). See docs/PATH_AUDIT.md.
"""

import os
from pathlib import Path
REPO_ROOT = Path(os.environ.get("THESIS_ROOT", Path(__file__).resolve().parents[4]))
DATA_ROOT = Path(os.environ.get("DATA_ROOT", "/home/ai/AI-DataPool/Datasets"))
import numpy as np
import pandas as pd
from typing import List, Dict, Any
from joblib import load
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# ============== CONFIG ==============
INPUT_TSV = str(DATA_ROOT / "audio/Österreich/sliced_16000_mono/Tirol/master.cleaned_with_phonemes.tsv")

MODEL_DIR = str(REPO_ROOT / "models/5_nb_phoneme_binary")
VEC_PATH  = os.path.join(MODEL_DIR, "vectorizer.joblib")
NB_PATH   = os.path.join(MODEL_DIR, "nb_model.joblib")

SAVE_DIR  = str(DATA_ROOT / "audio/Österreich/sliced_16000_mono/Tirol/nb_phoneme_binary_60secs")

# Concat settings
CHUNK_SECS = 60.0       # 0 = disable (single mega-doc per speaker)
DEFAULT_UTT_SECS = 5.0  # used if no 'duration' column

CLASS_NAMES = ["dialect", "high_german"]  # 0 -> dialect, 1 -> high_german
MEDIAN_THRESH = 0.4     # decision threshold on prob_high_german_median
# ====================================


def load_data(tsv_path: str) -> pd.DataFrame:
    df = pd.read_csv(tsv_path, sep="\t", dtype=str, low_memory=False)

    # For Tirol: we expect 'speaker' instead of 'client_id'
    required = ["path", "text", "phoneme", "speaker"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"{tsv_path} must contain column '{c}' (has: {df.columns.tolist()})")

    df = df.copy()
    df["phoneme"] = df["phoneme"].astype(str).fillna("").str.strip()
    df = df[df["phoneme"].str.len() > 0].reset_index(drop=True)

    # Normalize speaker column into a canonical ID
    df["speaker_id"] = df["speaker"].astype(str)

    # Try to coerce duration if exists
    if "duration" in df.columns:
        df["duration"] = pd.to_numeric(df["duration"], errors="coerce").fillna(0.0).clip(lower=0.0)
    else:
        df["duration"] = np.nan  # mark missing so we can default later

    return df


def chunk_speaker_rows(rows: pd.DataFrame, target_secs: float, use_duration: bool) -> List[Dict[str, Any]]:
    """
    Create temporal chunks for a given speaker, concatenating phoneme strings.

    Returns a list of dicts:
      {"concat_phoneme": "...", "n_utts": int}
    """
    if target_secs <= 0:
        return [{"concat_phoneme": " ".join(map(str, rows["phoneme"].tolist())),
                 "n_utts": int(len(rows))}]

    chunks, buf, acc = [], [], 0.0
    for _, r in rows.sort_values("path").iterrows():
        p = str(r["phoneme"])
        if use_duration and pd.notna(r["duration"]) and float(r["duration"]) > 0:
            d = float(r["duration"])
        else:
            d = DEFAULT_UTT_SECS

        if acc + d > target_secs and buf:
            chunks.append({"concat_phoneme": " ".join(buf), "n_utts": len(buf)})
            buf, acc = [], 0.0

        buf.append(p)
        acc += d

    if buf:
        chunks.append({"concat_phoneme": " ".join(buf), "n_utts": len(buf)})

    return chunks


def predict_probs(texts: List[str], vec: CountVectorizer, nb: MultinomialNB) -> np.ndarray:
    X = vec.transform(texts)
    return nb.predict_proba(X)  # [:,0]=dialect, [:,1]=high_german


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    print("Loading model & vectorizer…")
    vec: CountVectorizer = load(VEC_PATH)
    nb: MultinomialNB = load(NB_PATH)

    print("Reading Tirol TSV…")
    df = load_data(INPUT_TSV)
    use_duration = df["duration"].gt(0).any()
    print("Using 'duration' for chunking."
          if use_duration else
          "No usable 'duration' found; defaulting to 5s per utterance.")
    print(f"Total utterances: {len(df):,}")
    n_speakers = df["speaker_id"].nunique()
    print(f"Unique speakers: {n_speakers:,}")

    speaker_rows = []
    chunk_rows = []

    for sid, g in df.groupby("speaker_id", dropna=False):
        chunks = chunk_speaker_rows(g, CHUNK_SECS, use_duration)
        texts = [c["concat_phoneme"] for c in chunks]

        if not texts:
            # skip empty speakers just in case
            continue

        probs = predict_probs(texts, vec, nb)  # shape [k, 2], k = n_chunks

        # ---- chunk-level rows (keep all chunks) ----
        for idx, (cinfo, pr) in enumerate(zip(chunks, probs)):
            chunk_rows.append({
                "speaker_id": sid,
                "chunk_idx": idx,
                "n_utts_in_chunk": cinfo["n_utts"],
                "prob_dialect": float(pr[0]),
                "prob_high_german": float(pr[1]),
                "pred_label_idx": int(np.argmax(pr)),
                "pred_label_name": CLASS_NAMES[int(np.argmax(pr))]
            })

        # ---- speaker-level aggregation using *all* chunks ----
        prob_hg = probs[:, 1]  # column 1 = high_german

        prob_hg_mean   = float(np.mean(prob_hg))
        prob_hg_median = float(np.median(prob_hg))
        prob_hg_max    = float(np.max(prob_hg))

        # Decision rule (as you requested):
        #   if median(prob_high_german) >= 0.4 -> dialect (idx 0)
        #   else -> high_german (idx 1)
        if prob_hg_median <= MEDIAN_THRESH:
            pred_idx = 0  # dialect
        else:
            pred_idx = 1  # high_german

        speaker_rows.append({
            "speaker_id": sid,
            "n_utterances": int(len(g)),
            "n_chunks": int(len(chunks)),
            "prob_high_german_mean": prob_hg_mean,
            "prob_high_german_median": prob_hg_median,
            "prob_high_german_max": prob_hg_max,
            "decision_stat": "median_prob_high_german",
            "decision_threshold": MEDIAN_THRESH,
            "pred_label_idx": pred_idx,
            "pred_label_name": CLASS_NAMES[pred_idx],
        })

    # Save outputs
    speaker_df = pd.DataFrame(speaker_rows).sort_values("speaker_id").reset_index(drop=True)
    chunk_df = pd.DataFrame(chunk_rows).sort_values(["speaker_id", "chunk_idx"]).reset_index(drop=True)

    spk_path = os.path.join(SAVE_DIR, "speaker_preds.csv")
    chk_path = os.path.join(SAVE_DIR, "chunk_preds.csv")

    speaker_df.to_csv(spk_path, index=False)
    chunk_df.to_csv(chk_path, index=False)

    print(f"✅ Saved speaker predictions → {spk_path}")
    print(f"✅ Saved chunk predictions   → {chk_path}")

    print("\nSample speakers:")
    print(speaker_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
