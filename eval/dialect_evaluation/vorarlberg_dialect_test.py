#!/usr/bin/env python3
"""
Vorarlberg Dialect Classification (using precomputed phonemes)
--------------------------------------------------------------
Reads phonemes from a TSV (no audio processing), concatenates per speaker
into ~target-length chunks, and classifies with the trained NB model.
"""

import os
from pathlib import Path
REPO_ROOT = Path(os.environ.get("THESIS_ROOT", Path(__file__).resolve().parents[2]))

import os
from pathlib import Path
from typing import List, Tuple
import pandas as pd
import numpy as np
from joblib import load

# ====== HARD-CODED CONFIG ======
TSV_PATH   = str(REPO_ROOT / "data_preparation/feature_extraction/vorarlberg/vorarlberg_with_phonemes_merged.tsv")
MODEL_DIR  = str(REPO_ROOT / "models/nb_phoneme")
OUTPUT_DIR = str(REPO_ROOT / "eval/dialect_evaluation/vorarlberg_results")
MAX_FILES  = None         # e.g., 500 for testing; None = all
CONCAT_SECS = 30.0        # target duration per chunk (heuristic, using ~5s/row if no duration col)
MISSING_PHONEME = "NO_PHONEME"
# ===============================

def load_vorarlberg_data(tsv_path: str) -> pd.DataFrame:
    print(f"Loading data from {tsv_path} ...")
    df = pd.read_csv(tsv_path, sep="\t", dtype=str, low_memory=False)

    # Ensure columns exist
    required = {"path", "text", "phoneme"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required column(s) in TSV: {missing}")

    # Basic cleanup
    df = df.dropna(subset=["path", "phoneme"]).reset_index(drop=True)

    # Episode/speaker from path
    def part_or_unknown(p: str, idx_from_end: int) -> str:
        parts = Path(p).parts
        return parts[-idx_from_end] if len(parts) >= idx_from_end else "unknown"

    df["episode"] = df["path"].apply(lambda p: part_or_unknown(p, 3))
    df["speaker"] = df["path"].apply(lambda p: part_or_unknown(p, 2))

    # Duration: use column if present, else heuristic 5s
    if "duration" in df.columns:
        df["duration"] = pd.to_numeric(df["duration"], errors="coerce").fillna(5.0)
    else:
        df["duration"] = 5.0

    # Optional: cap to MAX_FILES for quick tests
    if MAX_FILES is not None and MAX_FILES < len(df):
        df = df.head(MAX_FILES).copy()
        print(f"Limited to first {MAX_FILES} rows")

    print(f"Loaded {len(df)} rows from {df['episode'].nunique()} episodes.")
    return df

def concat_by_speaker_phonemes(df: pd.DataFrame, target_secs: float = 30.0) -> pd.DataFrame:
    """
    Concatenate rows per (episode, speaker) into ~target_secs chunks.
    Phonemes are concatenated (space-separated). Text is optional, kept for reference.
    """
    if target_secs <= 0:
        # No concatenation; one row per file, keep phonemes as-is
        return df.assign(
            chunk_id=lambda x: x.index.map(lambda i: f"row_{i}"),
            audio_paths=lambda x: x["path"].apply(lambda p: [p]),
            num_files=1,
        )[["episode", "speaker", "chunk_id", "audio_paths", "phoneme", "text", "duration"]].rename(
            columns={"phoneme": "phonemes"}
        )

    print(f"Concatenating to ~{target_secs}s chunks per speaker...")
    chunks = []
    for (episode, speaker), g in df.groupby(["episode", "speaker"]):
        g = g.sort_values("path").reset_index(drop=True)
        buf_idx = []
        acc = 0.0
        chunk_idx = 0

        for i, row in g.iterrows():
            d = float(row["duration"]) if pd.notna(row["duration"]) else 5.0
            if acc + d > target_secs and buf_idx:
                sub = g.iloc[buf_idx]
                chunks.append({
                    "episode": episode,
                    "speaker": speaker,
                    "chunk_id": f"{episode}_{speaker}_{chunk_idx}",
                    "audio_paths": sub["path"].tolist(),
                    "phonemes": " ".join(p for p in sub["phoneme"].tolist() if p and p != MISSING_PHONEME),
                    "text": " ".join(sub["text"].astype(str).tolist()),
                    "duration": float(sub["duration"].sum()),
                    "num_files": int(len(sub)),
                })
                chunk_idx += 1
                buf_idx, acc = [], 0.0

            buf_idx.append(i)
            acc += d

        if buf_idx:
            sub = g.iloc[buf_idx]
            chunks.append({
                "episode": episode,
                "speaker": speaker,
                "chunk_id": f"{episode}_{speaker}_{chunk_idx}",
                "audio_paths": sub["path"].tolist(),
                "phonemes": " ".join(p for p in sub["phoneme"].tolist() if p and p != MISSING_PHONEME),
                "text": " ".join(sub["text"].astype(str).tolist()),
                "duration": float(sub["duration"].sum()),
                "num_files": int(len(sub)),
            })

    res = pd.DataFrame(chunks)
    # Drop empty-phoneme chunks (optional, but often helpful)
    before = len(res)
    res = res[res["phonemes"].str.strip().astype(bool)].reset_index(drop=True)
    dropped = before - len(res)
    if dropped:
        print(f"Dropped {dropped} empty-phoneme chunk(s).")

    print(f"Created {len(res)} chunks. Avg duration: {res['duration'].mean():.1f}s")
    return res

class DialectClassifier:
    """Loads NB model & vectorizer and predicts on phoneme strings."""
    def __init__(self, model_dir: str):
        print(f"Loading model from {model_dir} ...")
        model_path = os.path.join(model_dir, "nb_model.joblib")
        vec_path   = os.path.join(model_dir, "vectorizer.joblib")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Missing model: {model_path}")
        if not os.path.exists(vec_path):
            raise FileNotFoundError(f"Missing vectorizer: {vec_path}")
        self.model = load(model_path)
        self.vectorizer = load(vec_path)
        print("Model loaded successfully.")

    def predict_batch(self, phoneme_list: List[str]) -> List[Tuple[int, float]]:
        if not phoneme_list:
            return []
        X = self.vectorizer.transform(phoneme_list)
        preds = self.model.predict(X)
        probs = self.model.predict_proba(X)
        confs = probs.max(axis=1)
        return list(zip(preds, confs))

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = load_vorarlberg_data(TSV_PATH)
    df_chunks = concat_by_speaker_phonemes(df, target_secs=CONCAT_SECS)

    clf = DialectClassifier(MODEL_DIR)

    print(f"\nClassifying {len(df_chunks)} chunk(s)...")
    predictions = clf.predict_batch(df_chunks["phonemes"].tolist())
    df_chunks["predicted_dialect"] = [p for p, _ in predictions]
    df_chunks["confidence"] = [c for _, c in predictions]

    # Save per-chunk results
    per_chunk_path = os.path.join(OUTPUT_DIR, "vorarlberg_classification_results.csv")
    # Convert audio_paths (list) to pipe-separated string for CSV
    df_out = df_chunks.copy()
    df_out["audio_paths"] = df_out["audio_paths"].apply(lambda L: "|".join(L))
    df_out.to_csv(per_chunk_path, index=False)
    print(f"Per-chunk results saved to: {per_chunk_path}")

    # Episode-speaker summary
    summary_rows = []
    for (episode, speaker), g in df_chunks.groupby(["episode", "speaker"]):
        valid = g[g["phonemes"].str.strip().astype(bool)]
        if len(valid):
            most_common = valid["predicted_dialect"].mode().iloc[0]
            summary_rows.append({
                "episode": episode,
                "speaker": speaker,
                "num_chunks": int(len(valid)),
                "total_files": int(valid["num_files"].sum()),
                "total_duration": float(valid["duration"].sum()),
                "predicted_dialect": most_common,
                "avg_confidence": float(valid["confidence"].mean()),
                "confidence_std": float(valid["confidence"].std() if len(valid) > 1 else 0.0),
            })
        else:
            summary_rows.append({
                "episode": episode,
                "speaker": speaker,
                "num_chunks": 0,
                "total_files": 0,
                "total_duration": 0.0,
                "predicted_dialect": -1,
                "avg_confidence": 0.0,
                "confidence_std": 0.0,
            })

    df_summary = pd.DataFrame(summary_rows)
    summary_path = os.path.join(OUTPUT_DIR, "vorarlberg_episode_speaker_summary.csv")
    df_summary.to_csv(summary_path, index=False)
    print(f"Episode-speaker summary saved to: {summary_path}")

    # Quick terminal summary
    print("\n=== CLASSIFICATION SUMMARY ===")
    print(f"Total chunks: {len(df_chunks)}")
    print(f"Total files: {int(df_chunks['num_files'].sum())}")
    print(f"Total duration: {df_chunks['duration'].sum():.1f}s")
    if len(df_summary):
        print(f"Episode-speaker combos: {len(df_summary)}")
        print(f"Mean avg_confidence: {df_summary['avg_confidence'].mean():.3f}")
        vc = df_summary['predicted_dialect'].value_counts()
        print("Dialect counts (episode-speaker):")
        for k, v in vc.items():
            print(f"  {k}: {v}")

if __name__ == "__main__":
    main()
