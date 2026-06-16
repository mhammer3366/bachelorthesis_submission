#!/usr/bin/env python3
"""
Binary phoneme n-gram Naive Bayes: German (label_id=7) vs Non-German (label_id 0–6)
-------------------------------------------------------------------------------
- Uses phoneme TSVs (audio_path, label_id, phoneme) from mit_deutsch
- Optionally concatenates utterances per speaker to ~30s pseudo-docs
- Outputs model, vectorizer, metrics, predictions
"""

import os
from pathlib import Path
REPO_ROOT = Path(os.environ.get("THESIS_ROOT", Path(__file__).resolve().parents[2]))

import os
import json
import numpy as np
import pandas as pd
from typing import Optional, List
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from joblib import dump

# ============== CONFIG ==============
BASE = str(REPO_ROOT / "data_preparation/feature_extraction/mit_deutsch/saved_features_phoneme")
TRAIN_TSV = os.path.join(BASE, "train_phonemes.tsv")
VALID_TSV = os.path.join(BASE, "validation_phonemes.tsv")
TEST_TSV  = os.path.join(BASE, "test_phonemes.tsv")

# Optional meta TSVs for concat (set "" to disable concat)
META_BASE = str(REPO_ROOT / "data_preparation/merged_datasets_plus_de")
TRAIN_META = os.path.join(META_BASE, "merged_train.tsv")
VALID_META = os.path.join(META_BASE, "merged_valid.tsv")
TEST_META  = os.path.join(META_BASE, "merged_test.tsv")

SAVE_DIR  = str(REPO_ROOT / "models/5_nb_phoneme_binary")

# Vectorizer & NB params
NGRAM_LO  = 2
NGRAM_HI  = 4
MIN_DF    = 2
MAX_DF    = 1.0
ALPHA     = 0.1  # NB smoothing

# Concat by speaker (seconds). Set 0 to disable.
CONCAT_BY_CLIENT_SECS = 30.0

GERMAN_LABEL = 7
CLASS_NAMES = ["non_german", "german"]
# ====================================

def load_phoneme_tsv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", dtype=str, low_memory=False)
    for c in ["audio_path", "label_id", "phoneme"]:
        if c not in df.columns:
            raise ValueError(f"{path} must contain column '{c}'")
    df["phoneme"] = df["phoneme"].astype(str).fillna("").str.strip()
    df = df[df["phoneme"].str.len() > 0]
    df["label_id"] = pd.to_numeric(df["label_id"], errors="coerce")
    df = df.dropna(subset=["label_id"]).reset_index(drop=True)
    df["label_id"] = df["label_id"].astype(int)
    # Binary mapping
    df["binary_label"] = (df["label_id"] == GERMAN_LABEL).astype(int)
    return df[["audio_path", "label_id", "binary_label", "phoneme"]]

def load_meta_tsv(path: str) -> Optional[pd.DataFrame]:
    if not path or not os.path.isfile(path):
        return None
    df = pd.read_csv(path, sep="\t", dtype=str, low_memory=False)
    if "path" in df.columns and "audio_path" not in df.columns:
        df = df.rename(columns={"path": "audio_path"})
    keep = [c for c in ["audio_path", "client_id", "duration"] if c in df.columns]
    return df[keep] if keep else None

def concat_by_client(df_ph: pd.DataFrame, df_meta: Optional[pd.DataFrame], target_secs: float) -> pd.DataFrame:
    if target_secs <= 0:
        return df_ph.copy()
    if df_meta is None or "client_id" not in df_meta.columns:
        return df_ph.copy()

    df = df_ph.merge(df_meta, on="audio_path", how="left")
    if "duration" in df.columns:
        dur = pd.to_numeric(df["duration"], errors="coerce").fillna(0.0).clip(lower=0.0)
        df["duration"] = dur
    else:
        df["duration"] = 0.0

    chunks = []
    for (cid, lab), g in df.groupby(["client_id", "binary_label"], dropna=False):
        g = g.sort_values("audio_path")
        buf, acc = [], 0.0
        for _, row in g.iterrows():
            p = str(row["phoneme"])
            d = float(row["duration"]) if not pd.isna(row["duration"]) else 0.0
            if d <= 0.0:
                d = 5.0
            if acc + d > target_secs and buf:
                chunks.append({
                    "audio_path": f"concat_{cid}_{lab}_{len(chunks)}",
                    "binary_label": int(lab),
                    "phoneme": " ".join(buf)
                })
                buf, acc = [], 0.0
            buf.append(p)
            acc += d
        if buf:
            chunks.append({
                "audio_path": f"concat_{cid}_{lab}_{len(chunks)}",
                "binary_label": int(lab),
                "phoneme": " ".join(buf)
            })
    return pd.DataFrame(chunks)

def fit_vectorizer(texts: List[str]) -> CountVectorizer:
    vec = CountVectorizer(
        analyzer="word",
        token_pattern=r"[^ ]+",
        ngram_range=(NGRAM_LO, NGRAM_HI),
        min_df=MIN_DF,
        max_df=MAX_DF
    )
    vec.fit(texts)
    return vec

def train_and_eval(df_tr: pd.DataFrame, df_va: pd.DataFrame, df_te: pd.DataFrame):
    os.makedirs(SAVE_DIR, exist_ok=True)

    vec = fit_vectorizer(df_tr["phoneme"].tolist())
    X_tr = vec.transform(df_tr["phoneme"].tolist())
    y_tr = df_tr["binary_label"].to_numpy()

    nb = MultinomialNB(alpha=ALPHA)
    nb.fit(X_tr, y_tr)

    dump(nb, os.path.join(SAVE_DIR, "nb_model.joblib"))
    dump(vec, os.path.join(SAVE_DIR, "vectorizer.joblib"))

    def eval_split(df: pd.DataFrame, name: str):
        X = vec.transform(df["phoneme"].tolist())
        y = df["binary_label"].to_numpy()
        proba = nb.predict_proba(X)
        pred = proba.argmax(axis=1)
        acc = float(accuracy_score(y, pred))
        macro = float(f1_score(y, pred, average="macro"))
        f1_non = float(f1_score(y, pred, pos_label=0))
        f1_ger = float(f1_score(y, pred, pos_label=1))
        cm = confusion_matrix(y, pred, labels=[0,1]).tolist()
        metrics = {
            "classes": CLASS_NAMES,
            "accuracy": acc,
            "macro_f1": macro,
            "f1_non_german": f1_non,
            "f1_german": f1_ger,
            "confusion_matrix_labels": CLASS_NAMES,
            "confusion_matrix": cm
        }
        with open(os.path.join(SAVE_DIR, f"{name}_metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)
        out = df.copy()
        out["pred_binary"] = pred
        out["pred_label_name"] = [CLASS_NAMES[i] for i in pred]
        out["prob_non_german"] = proba[:,0].round(6)
        out["prob_german"] = proba[:,1].round(6)
        out.to_csv(os.path.join(SAVE_DIR, f"{name}_preds.csv"), index=False)
        print(f"{name}: acc={acc:.4f} macroF1={macro:.4f}  F1(non)={f1_non:.4f}  F1(ger)={f1_ger:.4f}")

    eval_split(df_tr, "train")
    eval_split(df_va, "valid")
    eval_split(df_te, "test")

def main():
    print("Loading TSVs…")
    df_tr = load_phoneme_tsv(TRAIN_TSV)
    df_va = load_phoneme_tsv(VALID_TSV)
    df_te = load_phoneme_tsv(TEST_TSV)
    m_tr = load_meta_tsv(TRAIN_META)
    m_va = load_meta_tsv(VALID_META)
    m_te = load_meta_tsv(TEST_META)

    if CONCAT_BY_CLIENT_SECS > 0:
        print(f"Concatenating to ~{CONCAT_BY_CLIENT_SECS}s chunks…")
        df_tr = concat_by_client(df_tr, m_tr, CONCAT_BY_CLIENT_SECS)
        df_va = concat_by_client(df_va, m_va, CONCAT_BY_CLIENT_SECS)
        df_te = concat_by_client(df_te, m_te, CONCAT_BY_CLIENT_SECS)

    print("Training binary linear classifier (German vs Non-German)…")
    train_and_eval(df_tr, df_va, df_te)

if __name__ == "__main__":
    main()
