#!/usr/bin/env python3
"""
Phoneme n-gram Naive Bayes dialect classifier (as used in Stucki et al. 2025)
----------------------------------------------------------------------------
Method: audio → phonemes (wav2vec2-espeak) → word n-grams over phoneme tokens →
Multinomial Naive Bayes. Trains on STT4SG phonemized data; optionally adds German CV
(or any extra German) if provided. Optionally concatenates utterances per speaker to
~30s pseudo-docs (recommended; matches paper setup).

Hardcoded paths version for consistent model saving.
"""

import os
from pathlib import Path
REPO_ROOT = Path(os.environ.get("THESIS_ROOT", Path(__file__).resolve().parents[2]))
import os
import json
import argparse
import numpy as np
import pandas as pd
from typing import Optional, Tuple, List
from collections import defaultdict

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from joblib import dump


def load_phoneme_tsv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep='\t', dtype=str, low_memory=False)
    for col in ['audio_path', 'label_id', 'phoneme']:
        if col not in df.columns:
            raise ValueError(f"{path} must contain column '{col}'")
    df = df[df['phoneme'].astype(str).str.len() > 0].copy()
    df['label_id'] = pd.to_numeric(df['label_id'], errors='coerce')
    df = df.dropna(subset=['label_id']).reset_index(drop=True)
    df['label_id'] = df['label_id'].astype(int)
    return df[['audio_path','label_id','phoneme']]


def load_meta_tsv(path: Optional[str]) -> Optional[pd.DataFrame]:
    if not path:
        return None
    df = pd.read_csv(path, sep='\t', dtype=str, low_memory=False)
    # not all columns guaranteed, but we try
    keep = [c for c in ['audio_path','client_id','duration'] if c in df.columns]
    if 'audio_path' not in keep and 'path' in df.columns:
        df = df.rename(columns={'path':'audio_path'})
        keep = [c for c in ['audio_path','client_id','duration'] if c in df.columns]
    return df[keep] if keep else None


def concat_by_client(df_ph: pd.DataFrame, df_meta: Optional[pd.DataFrame], target_secs: float) -> pd.DataFrame:
    """Concatenate phoneme strings per speaker into ~target_secs chunks.
    If df_meta is None or missing fields, falls back to concatenating all utterances per speaker.
    """
    if target_secs <= 0:
        return df_ph.copy()

    if df_meta is None or 'client_id' not in df_meta.columns:
        # simple: concat all by audio_path prefix speaker hash (not ideal)
        out = df_ph.groupby('label_id')['phoneme'].apply(list).reset_index()
        out['phoneme'] = out['phoneme'].apply(lambda lst: ' '.join(lst))
        # no per-sample ids anymore; replicate counts by class distribution
        out['audio_path'] = [f'concat_cls_{i}' for i in range(len(out))]
        return out[['audio_path','label_id','phoneme']]

    # merge
    df = df_ph.merge(df_meta, on='audio_path', how='left')
    # sanitize duration
    if 'duration' in df.columns:
        dur = pd.to_numeric(df['duration'], errors='coerce')
        dur = dur.fillna(0.0).clip(lower=0.0)
        df['duration'] = dur
    else:
        df['duration'] = 0.0

    # group by speaker (client_id) and dialect label
    chunks = []
    for (cid, lab), g in df.groupby(['client_id','label_id'], dropna=False):
        # keep a stable order by audio_path
        g = g.sort_values('audio_path')
        buf, acc = [], 0.0
        for _, row in g.iterrows():
            p = str(row['phoneme'])
            d = float(row['duration']) if not pd.isna(row['duration']) else 0.0
            if d <= 0.0:
                # approximate length by 5s per utterance if unknown
                d = 5.0
            if acc + d > target_secs and buf:
                chunks.append({
                    'audio_path': f'concat_{cid}_{lab}_{len(chunks)}',
                    'label_id': int(lab),
                    'phoneme': ' '.join(buf)
                })
                buf, acc = [], 0.0
            buf.append(p)
            acc += d
        if buf:
            chunks.append({
                'audio_path': f'concat_{cid}_{lab}_{len(chunks)}',
                'label_id': int(lab),
                'phoneme': ' '.join(buf)
            })
    return pd.DataFrame(chunks)


def fit_vectorizer(texts: List[str], min_df: int, max_df: float, ngram_lo: int, ngram_hi: int) -> CountVectorizer:
    # word-level n-grams where tokens are already space-delimited phonemes
    vec = CountVectorizer(analyzer='word', token_pattern=r"[^ ]+", ngram_range=(ngram_lo, ngram_hi), min_df=min_df, max_df=max_df)
    vec.fit(texts)
    return vec


def train_and_eval(df_tr: pd.DataFrame, df_va: pd.DataFrame, df_te: pd.DataFrame,
                   save_dir: str, alpha: float,
                   min_df: int, max_df: float, ngram_lo: int, ngram_hi: int):
    os.makedirs(save_dir, exist_ok=True)

    # vectorizer on train
    vec = fit_vectorizer(df_tr['phoneme'].tolist(), min_df, max_df, ngram_lo, ngram_hi)
    X_tr = vec.transform(df_tr['phoneme'].tolist())
    y_tr = df_tr['label_id'].to_numpy()

    # class priors via counts
    nb = MultinomialNB(alpha=alpha)
    nb.fit(X_tr, y_tr)

    dump(nb, os.path.join(save_dir, 'nb_model.joblib'))
    dump(vec, os.path.join(save_dir, 'vectorizer.joblib'))

    def eval_split(df: pd.DataFrame, name: str):
        X = vec.transform(df['phoneme'].tolist())
        y = df['label_id'].to_numpy()
        proba = nb.predict_proba(X)
        pred = proba.argmax(axis=1)
        acc = float(accuracy_score(y, pred))
        macro = float(f1_score(y, pred, average='macro'))
        cm = confusion_matrix(y, pred).tolist()
        with open(os.path.join(save_dir, f'{name}_metrics.json'), 'w') as f:
            json.dump({'acc': acc, 'macro_f1': macro, 'cm': cm}, f, indent=2)
        out = df.copy()
        out['pred'] = pred
        out['probs'] = [p.tolist() for p in proba]
        out.to_csv(os.path.join(save_dir, f'{name}_preds.csv'), index=False)
        print(f"{name}: acc={acc:.4f} macroF1={macro:.4f}")

    eval_split(df_tr, 'train')
    eval_split(df_va, 'valid')
    eval_split(df_te, 'test')


def main():
    # Hardcoded configuration
    train_phoneme_tsv = str(REPO_ROOT / "data_preparation/feature_extraction/ohne_deutsch/saved_features_phoneme/train_phonemes.tsv")
    valid_phoneme_tsv = str(REPO_ROOT / "data_preparation/feature_extraction/ohne_deutsch/saved_features_phoneme/validation_phonemes.tsv")
    test_phoneme_tsv = str(REPO_ROOT / "data_preparation/feature_extraction/ohne_deutsch/saved_features_phoneme/test_phonemes.tsv")
    train_meta_tsv = str(REPO_ROOT / "data_preparation/merged_datasets/merged_train.tsv")
    valid_meta_tsv = str(REPO_ROOT / "data_preparation/merged_datasets/merged_valid.tsv")
    test_meta_tsv = str(REPO_ROOT / "data_preparation/merged_datasets/merged_test.tsv")
    save_dir = str(REPO_ROOT / "models/3_nb_phoneme")
    concat_by_client_secs = 90.0
    alpha = 0.1
    min_df = 2
    max_df = 1.0
    ngram_lo = 2
    ngram_hi = 4

    df_tr = load_phoneme_tsv(train_phoneme_tsv)
    df_va = load_phoneme_tsv(valid_phoneme_tsv)
    df_te = load_phoneme_tsv(test_phoneme_tsv)

    m_tr = load_meta_tsv(train_meta_tsv)
    m_va = load_meta_tsv(valid_meta_tsv)
    m_te = load_meta_tsv(test_meta_tsv)

    if concat_by_client_secs > 0:
        print(f"Concatenating by client to ~{concat_by_client_secs}s chunks…")
        df_tr = concat_by_client(df_tr, m_tr, concat_by_client_secs)
        df_va = concat_by_client(df_va, m_va, concat_by_client_secs)
        df_te = concat_by_client(df_te, m_te, concat_by_client_secs)

    train_and_eval(df_tr, df_va, df_te, save_dir, alpha, min_df, max_df, ngram_lo, ngram_hi)

if __name__ == '__main__':
    main()
