#!/usr/bin/env python3
"""
Phoneme n-gram Linear Classifier (SGD-Logistic) for Swiss German dialects
Hardcoded paths version
"""

import os
from pathlib import Path
REPO_ROOT = Path(os.environ.get("THESIS_ROOT", Path(__file__).resolve().parents[2]))

import os
import json
import numpy as np
import pandas as pd
from typing import Optional, List, Tuple, Dict

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from joblib import dump

# ----------------------------- IO -----------------------------
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
    if 'audio_path' not in df.columns and 'path' in df.columns:
        df = df.rename(columns={'path': 'audio_path'})
    keep = [c for c in ['audio_path','client_id','duration'] if c in df.columns]
    return df[keep] if keep else None

def concat_by_client(df_ph: pd.DataFrame, df_meta: Optional[pd.DataFrame], target_secs: float) -> pd.DataFrame:
    if target_secs <= 0:
        return df_ph.copy()

    if df_meta is None or 'client_id' not in df_meta.columns:
        out = df_ph.groupby('label_id')['phoneme'].apply(list).reset_index()
        out['phoneme'] = out['phoneme'].apply(lambda lst: ' '.join(lst))
        out['audio_path'] = [f'concat_cls_{i}' for i in range(len(out))]
        return out[['audio_path','label_id','phoneme']]

    df = df_ph.merge(df_meta, on='audio_path', how='left')
    if 'duration' in df.columns:
        dur = pd.to_numeric(df['duration'], errors='coerce').fillna(0.0).clip(lower=0.0)
        df['duration'] = dur
    else:
        df['duration'] = 0.0

    chunks, gid = [], 0
    for (cid, lab), g in df.groupby(['client_id','label_id'], dropna=False):
        g = g.sort_values('audio_path')
        buf, acc = [], 0.0
        for _, row in g.iterrows():
            p = str(row['phoneme'])
            d = float(row['duration']) if pd.notna(row['duration']) else 0.0
            if d <= 0.0:
                d = 5.0
            if acc + d > target_secs and buf:
                chunks.append({
                    'audio_path': f'concat_{cid}_{lab}_{gid}',
                    'label_id': int(lab),
                    'phoneme': ' '.join(buf)
                })
                gid += 1
                buf, acc = [], 0.0
            buf.append(p)
            acc += d
        if buf:
            chunks.append({
                'audio_path': f'concat_{cid}_{lab}_{gid}',
                'label_id': int(lab),
                'phoneme': ' '.join(buf)
            })
            gid += 1
    return pd.DataFrame(chunks)

# ----------------------------- Vectorizer -----------------------------
def fit_vectorizer(texts: List[str], min_df: int, max_df: float, ngram_lo: int, ngram_hi: int) -> TfidfVectorizer:
    vec = TfidfVectorizer(
        analyzer='word',
        token_pattern=r"[^ ]+",
        ngram_range=(ngram_lo, ngram_hi),
        min_df=min_df,
        max_df=max_df,
        sublinear_tf=True,
        norm='l2',
        lowercase=False
    )
    vec.fit(texts)
    return vec

# ----------------------------- Model -----------------------------
def build_classifier(
    class_weight_mode: str = 'balanced',
    l2: float = 1e-4,
    early_stopping: bool = True,
    max_epochs_no_change: int = 5,
    random_state: int = 42
) -> SGDClassifier:
    return SGDClassifier(
        loss='log_loss',
        penalty='l2',
        alpha=l2,
        class_weight=class_weight_mode,
        early_stopping=early_stopping,
        n_iter_no_change=max_epochs_no_change,
        validation_fraction=0.1,
        learning_rate='optimal',
        fit_intercept=True,
        average=True,
        random_state=random_state,
        n_jobs=-1
    )

# ----------------------------- Train & Eval -----------------------------
def evaluate_split(X, y_true, clf, save_dir: str, name: str, ids: List[str]):
    proba = clf.predict_proba(X)
    y_pred = proba.argmax(axis=1)
    acc = float(np.mean(y_true == y_pred))
    macro = float(f1_score(y_true, y_pred, average='macro'))
    cm = confusion_matrix(y_true, y_pred).tolist()

    with open(os.path.join(save_dir, f'{name}_metrics.json'), 'w') as f:
        json.dump({'acc': acc, 'macro_f1': macro, 'cm': cm}, f, indent=2)

    pd.DataFrame({
        'audio_path': ids,
        'label_id': y_true,
        'pred': y_pred,
        'probs': [p.tolist() for p in proba]
    }).to_csv(os.path.join(save_dir, f'{name}_preds.csv'), index=False)

    print(f"{name}: acc={acc:.4f} macroF1={macro:.4f}")

def top_terms(coef: np.ndarray, vocab: List[str], topk: int = 20):
    out = {}
    for c in range(coef.shape[0]):
        w = coef[c]
        idx_top = np.argsort(-w)[:topk]
        idx_bot = np.argsort(w)[:topk]
        out[str(c)] = {
            'top': [(vocab[i], float(w[i])) for i in idx_top],
            'anti': [(vocab[i], float(w[i])) for i in idx_bot]
        }
    return out

def train_and_eval(df_tr, df_va, df_te, save_dir, min_df, max_df, ngram_lo, ngram_hi, l2, class_weight_mode, early_stopping, max_epochs_no_change, random_state):
    os.makedirs(save_dir, exist_ok=True)

    vec = fit_vectorizer(df_tr['phoneme'].tolist(), min_df, max_df, ngram_lo, ngram_hi)
    X_tr, y_tr = vec.transform(df_tr['phoneme']), df_tr['label_id'].to_numpy()
    X_va, y_va = vec.transform(df_va['phoneme']), df_va['label_id'].to_numpy()
    X_te, y_te = vec.transform(df_te['phoneme']), df_te['label_id'].to_numpy()

    clf = build_classifier(class_weight_mode, l2, early_stopping, max_epochs_no_change, random_state)
    clf.fit(X_tr, y_tr)

    dump(clf, os.path.join(save_dir, 'model.joblib'))
    dump(vec, os.path.join(save_dir, 'vectorizer.joblib'))

    evaluate_split(X_tr, y_tr, clf, save_dir, 'train', df_tr['audio_path'].tolist())
    evaluate_split(X_va, y_va, clf, save_dir, 'valid', df_va['audio_path'].tolist())
    evaluate_split(X_te, y_te, clf, save_dir, 'test',  df_te['audio_path'].tolist())

    if hasattr(clf, 'coef_'):
        vocab = np.array(sorted(vec.vocabulary_, key=lambda k: vec.vocabulary_[k]))
        coef_info = top_terms(clf.coef_, vocab, topk=30)
        with open(os.path.join(save_dir, 'coef_top_terms.json'), 'w') as f:
            json.dump(coef_info, f, indent=2)

# ----------------------------- Main -----------------------------
def main():
    # Hardcoded paths
    train_phoneme_tsv = str(REPO_ROOT / "data_preparation/feature_extraction/ohne_deutsch/saved_features_phoneme/train_phonemes.tsv")
    valid_phoneme_tsv = str(REPO_ROOT / "data_preparation/feature_extraction/ohne_deutsch/saved_features_phoneme/validation_phonemes.tsv")
    test_phoneme_tsv  = str(REPO_ROOT / "data_preparation/feature_extraction/ohne_deutsch/saved_features_phoneme/test_phonemes.tsv")

    train_meta_tsv = str(REPO_ROOT / "data_preparation/merged_datasets/merged_train.tsv")
    valid_meta_tsv = str(REPO_ROOT / "data_preparation/merged_datasets/merged_valid.tsv")
    test_meta_tsv  = str(REPO_ROOT / "data_preparation/merged_datasets/merged_test.tsv")

    save_dir = str(REPO_ROOT / "models/4_linear_phoneme")

    concat_by_client_secs = 90
    min_df, max_df = 2, 1.0
    ngram_lo, ngram_hi = 2, 4
    l2 = 1e-4
    class_weight_mode = "balanced"
    early_stopping, patience = True, 5
    seed = 42

    # Load
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

    train_and_eval(df_tr, df_va, df_te, save_dir, min_df, max_df, ngram_lo, ngram_hi, l2, class_weight_mode, early_stopping, patience, seed)

if __name__ == "__main__":
    main()
