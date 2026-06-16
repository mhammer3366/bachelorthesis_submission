#!/usr/bin/env python3
"""
Wien speaker-level binary NB evaluation (German vs Dialect).

Paths resolve from THESIS_ROOT / DATA_ROOT; see docs/PATH_AUDIT.md.
"""

import os
from pathlib import Path
REPO_ROOT = Path(os.environ.get("THESIS_ROOT", Path(__file__).resolve().parents[4]))

import os
import json
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple
from joblib import load
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# ============== CONFIG ==============
INPUT_TSV = str(REPO_ROOT / "data_preparation/feature_extraction/vorarlberg/vorarlberg_with_phonemes_with_client.tsv")

MODEL_DIR = str(REPO_ROOT / "models/nb_phoneme_binary")
VEC_PATH  = os.path.join(MODEL_DIR, "vectorizer.joblib")
NB_PATH   = os.path.join(MODEL_DIR, "nb_model.joblib")

SAVE_DIR  = str(REPO_ROOT / "eval/dialect_evaluation/vorarlberg_results/binary_classification/concat_30")

# Concat settings
CHUNK_SECS = 90.0     # 0 = disable (single mega-doc per speaker)
DEFAULT_UTT_SECS = 5.0  # used if no 'duration' column

CLASS_NAMES = ["dialect", "high_german"]  # 0 -> dialect(non_german), 1 -> high_german(german)
# ====================================

def load_data(tsv_path: str) -> pd.DataFrame:
    df = pd.read_csv(tsv_path, sep="\t", dtype=str, low_memory=False)
    for c in ["path", "text", "phoneme", "client_id"]:
        if c not in df.columns:
            raise ValueError(f"{tsv_path} must contain column '{c}'")
    df = df.copy()
    df["phoneme"] = df["phoneme"].astype(str).fillna("").str.strip()
    df = df[df["phoneme"].str.len() > 0].reset_index(drop=True)

    # Try to coerce duration if exists
    if "duration" in df.columns:

        df["duration"] = pd.to_numeric(df["duration"], errors="coerce").fillna(0.0).clip(lower=0.0)
    else:
        df["duration"] = np.nan  # mark missing so we can default later
    return df

def chunk_speaker_rows(rows: pd.DataFrame, target_secs: float, use_duration: bool) -> List[Dict[str, Any]]:
    if target_secs <= 0:
        return [{"concat_phoneme": " ".join(map(str, rows["phoneme"].tolist())), "n_utts": int(len(rows))}]
    chunks, buf, acc = [], [], 0.0
    for _, r in rows.sort_values("path").iterrows():
        p = str(r["phoneme"])
        d = float(r["duration"]) if use_duration and pd.notna(r["duration"]) and r["duration"] > 0 else DEFAULT_UTT_SECS
        if acc + d > target_secs and buf:
            chunks.append({"concat_phoneme": " ".join(buf), "n_utts": len(buf)})
            buf, acc = [], 0.0
        buf.append(p); acc += d
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

    print("Reading Vorarlberg TSV…")
    df = load_data(INPUT_TSV)
    use_duration = "duration" in df.columns and pd.to_numeric(df["duration"], errors="coerce").fillna(0).gt(0).any()
    print("Using 'duration' for chunking."
        if use_duration else
        "No usable 'duration' found; defaulting to 5s per utterance.")
    print(f"Total utterances: {len(df):,}")
    n_speakers = df["client_id"].nunique()
    print(f"Unique speakers: {n_speakers:,}")

    speaker_rows = []
    chunk_rows = []

    for cid, g in df.groupby("client_id", dropna=False):
        chunks = chunk_speaker_rows(g, CHUNK_SECS, use_duration)
        texts = [c["concat_phoneme"] for c in chunks]
        probs = predict_probs(texts, vec, nb)  # shape [k,2]

        # store chunk-level
        for idx, (cinfo, pr) in enumerate(zip(chunks, probs)):
            chunk_rows.append({
                "client_id": cid,
                "chunk_idx": idx,
                "n_utts_in_chunk": cinfo["n_utts"],
                "prob_dialect": float(pr[0]),
                "prob_high_german": float(pr[1]),
                "pred_label_idx": int(np.argmax(pr)),
                "pred_label_name": CLASS_NAMES[int(np.argmax(pr))]
            })

        # aggregate per speaker
        prob_hg_mean = float(np.mean(probs[:,1]))     # average prob for high_german
        prob_hg_median = float(np.median(probs[:,1]))
        prob_hg_max = float(np.max(probs[:,1]))
        pred_idx = 1 if prob_hg_mean >= 0.5 else 0
        speaker_rows.append({
            "client_id": cid,
            "n_utterances": int(len(g)),
            "n_chunks": int(len(chunks)),
            "prob_high_german_mean": prob_hg_mean,
            "prob_high_german_median": prob_hg_median,
            "prob_high_german_max": prob_hg_max,
            "pred_label_idx": pred_idx,
            "pred_label_name": CLASS_NAMES[pred_idx]
        })

    # Save outputs
    speaker_df = pd.DataFrame(speaker_rows).sort_values("client_id").reset_index(drop=True)
    chunk_df = pd.DataFrame(chunk_rows).sort_values(["client_id","chunk_idx"]).reset_index(drop=True)

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
