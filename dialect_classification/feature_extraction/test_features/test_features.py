#!/usr/bin/env python3
import os, re, json, warnings, argparse, pathlib
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import mutual_info_classif, f_classif
from sklearn.manifold import TSNE
import umap
import matplotlib.pyplot as plt

# Optional XGBoost
try:
    from xgboost import XGBClassifier
    HAVE_XGB = True
except Exception:
    HAVE_XGB = False

# ------------------ helpers ------------------

def np_load_safe(path):
    arr = np.load(path, allow_pickle=True)
    return arr

def find_np_pair(dirpath, split_name):
    """
    Try common filename patterns for features/labels for a given split.
    Returns (features_path, labels_path) or (None, None) if not found.
    """
    if dirpath is None: return None, None
    cand = []
    for fn in os.listdir(dirpath):
        lf = fn.lower()
        if split_name in lf and lf.endswith(".npy"):
            cand.append(fn)
    # heuristics: prefer names with 'feature' vs 'label'
    feat = None; lab = None
    for fn in cand:
        if re.search(r"(feat|mel|spec)", fn.lower()):
            feat = os.path.join(dirpath, fn)
            break
    # fallbacks
    if feat is None:
        for fn in cand:
            if "label" not in fn.lower():
                feat = os.path.join(dirpath, fn); break
    for fn in cand:
        if "label" in fn.lower():
            lab = os.path.join(dirpath, fn)
            break
    return feat, lab

def load_mel_block(melspec_dir):
    """
    Load mel features & labels from splitwise .npy files in the mel dir.
    """
    if melspec_dir is None: return None, None
    Xs, ys = [], []
    for split in ["train", "validation", "test", "dev"]:
        fpath, lpath = find_np_pair(melspec_dir, split)
        if fpath and lpath and os.path.exists(fpath) and os.path.exists(lpath):
            X = np_load_safe(fpath)
            y = np_load_safe(lpath)
            # expect frame features; if 3D (N,T,F), pool by mean/std
            X = pool_maybe(X)
            ys.append(y.astype(int).ravel())
            Xs.append(X)
    if not Xs: return None, None
    return np.concatenate(Xs, axis=0), np.concatenate(ys, axis=0)

def pool_maybe(X):
    """
    If X is [N,T,D] -> pool to [N,3D] via mean/std/|z|-mean. If [N,D], pass through.
    """
    X = np.asarray(X)
    if X.ndim == 2:
        return X.astype(np.float32)
    if X.ndim == 3:
        N, T, D = X.shape
        mean = X.mean(axis=1)
        std  = X.std(axis=1) + 1e-9
        z = (X - mean[:, None, :]) / std[:, None, :]
        skewp = np.mean(np.abs(z), axis=1)
        return np.concatenate([mean, std, skewp], axis=1).astype(np.float32)
    raise ValueError(f"Unexpected feature shape {X.shape}")


def load_w2v6_block(w2v6_dir):
    if w2v6_dir is None: return None, None
    names = [
        ("train_features.npy","train_labels.npy"),
        ("validation_features.npy","validation_labels.npy"),
        ("dev_features.npy","dev_labels.npy"),
        ("test_features.npy","test_labels.npy"),
    ]
    Xs, ys = [], []
    for ffn, lfn in names:
        fp = os.path.join(w2v6_dir, ffn)
        lp = os.path.join(w2v6_dir, lfn)
        if os.path.exists(fp) and os.path.exists(lp):
            X = np_load_safe(fp)
            y = np_load_safe(lp)
            X = pool_maybe(X)  # in case they are sequences
            Xs.append(X)
            ys.append(y.astype(int).ravel())
    if not Xs: return None, None
    return np.concatenate(Xs, axis=0), np.concatenate(ys, axis=0)

def load_espeak_block(espeak_dir, max_features=4000, ngram_max=2):
    """
    Expect TSVs: train_phonemes.tsv, [validation_phonemes.tsv], test_phonemes.tsv
    with columns: phoneme (space-separated tokens), label_id (int).
    We join train+val for training pool and include test for analysis (no cheating:
    we do CV across the combined set anyway; this is a probe, not a final model).
    """
    if espeak_dir is None: return None, None, None
    tr = os.path.join(espeak_dir, "train_phonemes.tsv")
    va = os.path.join(espeak_dir, "validation_phonemes.tsv")
    te = os.path.join(espeak_dir, "test_phonemes.tsv")

    if not os.path.exists(tr) or not os.path.exists(te):
        return None, None, None

    train_df = pd.read_csv(tr, sep="\t")
    if os.path.exists(va):
        val_df = pd.read_csv(va, sep="\t")
        train_df = pd.concat([train_df, val_df], ignore_index=True)
    test_df = pd.read_csv(te, sep="\t")

    # Clean
    train_df = train_df.dropna(subset=["phoneme"])
    test_df = test_df.dropna(subset=["phoneme"])

    # Concat for a single probe pool
    all_df = pd.concat([train_df, test_df], ignore_index=True)
    # Ensure strings of tokens
    texts = all_df["phoneme"].astype(str).tolist()
    y = all_df["label_id"].astype(int).values

    # Vectorize phones (unigram+bigram)
    vect = CountVectorizer(
        analyzer="word",
        tokenizer=lambda s: s.split(),
        ngram_range=(1, ngram_max),
        max_features=max_features,
        min_df=2
    )
    X = vect.fit_transform(texts).astype(np.float32)
    return X, y, vect.get_feature_names_out().tolist()

def align_blocks(blocks_with_labels):
    """
    Given a list of tuples (X, y, feat_prefix, feat_names_opt),
    keep only the samples where labels agree; otherwise we concatenate
    by stacking datasets (dialect IDs assumed consistent across sets).
    """
    # Here we simply stack rows (since label spaces are consistent).
    X_concat = None
    y_concat = None
    featnames = []
    for (X, y, prefix, fns) in blocks_with_labels:
        if X is None or y is None: continue
        if hasattr(X, "toarray"):   # sparse
            Xd = X.toarray()
        else:
            Xd = np.asarray(X)
        if X_concat is None:
            X_concat = Xd
            y_concat = y
        else:
            X_concat = np.concatenate([X_concat, Xd], axis=0)
            y_concat = np.concatenate([y_concat, y], axis=0)
        if fns is None:
            featnames += [f"{prefix}_{i}" for i in range(Xd.shape[1])]
        else:
            featnames += [f"{prefix}:{n}" for n in fns]
    return X_concat, y_concat, featnames

def run_linear_probe(X, y, label_names, title, outdir):
    os.makedirs(outdir, exist_ok=True)
    scaler = StandardScaler(with_mean=True, with_std=True)
    Xn = scaler.fit_transform(X)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=13)
    y_true, y_pred = [], []
    for tr, va in skf.split(Xn, y):
        clf = LogisticRegression(
            max_iter=5000, n_jobs=8, class_weight="balanced", C=1.0
        )
        clf.fit(Xn[tr], y[tr])
        yp = clf.predict(Xn[va])
        y_true.append(y[va]); y_pred.append(yp)
    y_true = np.concatenate(y_true); y_pred = np.concatenate(y_pred)

    rep = classification_report(y_true, y_pred, target_names=label_names, digits=4)
    print(f"\n[{title}] 5-fold results:\n{rep}")

    cm = confusion_matrix(y_true, y_pred)
    fig = plt.figure(figsize=(6,5))
    plt.imshow(cm, interpolation='nearest')
    plt.title(f"{title} Confusion Matrix")
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.colorbar()
    ticks = np.arange(len(label_names))
    plt.xticks(ticks, label_names, rotation=45, ha="right")
    plt.yticks(ticks, label_names)
    plt.tight_layout()
    cm_path = os.path.join(outdir, f"{title}_confmat.png")
    plt.savefig(cm_path, dpi=180); plt.close(fig)
    return rep, cm_path

def run_xgb_probe(X, y, label_names, title, outdir):
    if not HAVE_XGB:
        print("XGBoost not installed; skipping.")
        return None, None
    os.makedirs(outdir, exist_ok=True)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=13)
    y_true, y_pred = [], []
    for tr, va in skf.split(X, y):
        clf = XGBClassifier(
            n_estimators=800, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
            objective="multi:softmax", num_class=len(np.unique(y)),
            tree_method="hist", n_jobs=8, random_state=13
        )
        clf.fit(X[tr], y[tr], eval_set=[(X[va], y[va])], verbose=False)
        yp = clf.predict(X[va])
        y_true.append(y[va]); y_pred.append(yp)
    y_true = np.concatenate(y_true); y_pred = np.concatenate(y_pred)
    rep = classification_report(y_true, y_pred, target_names=label_names, digits=4)
    print(f"\n[{title}] 5-fold results:\n{rep}")

    cm = confusion_matrix(y_true, y_pred)
    fig = plt.figure(figsize=(6,5))
    plt.imshow(cm, interpolation='nearest')
    plt.title(f"{title} Confusion Matrix")
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.colorbar()
    ticks = np.arange(len(label_names))
    plt.xticks(ticks, label_names, rotation=45, ha="right")
    plt.yticks(ticks, label_names)
    plt.tight_layout()
    cm_path = os.path.join(outdir, f"{title}_confmat.png")
    plt.savefig(cm_path, dpi=180); plt.close(fig)
    return rep, cm_path

def run_feature_scores(X, y, featnames, outdir, tag):
    os.makedirs(outdir, exist_ok=True)
    mi = mutual_info_classif(X, y, discrete_features=False, random_state=13)
    fvals, pvals = f_classif(X, y)
    df = pd.DataFrame({"feature": featnames, "MI": mi, "F": fvals, "p": pvals})
    df = df.sort_values("MI", ascending=False)
    out = os.path.join(outdir, f"{tag}_feature_scores.csv")
    df.to_csv(out, index=False)
    print(f"[Feature scores] -> {out}")
    return out

def run_maps(X, y, label_names, outdir, tag):
    os.makedirs(outdir, exist_ok=True)
    scaler = StandardScaler()
    Xn = scaler.fit_transform(X)

    um = umap.UMAP(n_neighbors=30, min_dist=0.1, random_state=13)
    U = um.fit_transform(Xn)
    fig = plt.figure(figsize=(6,5))
    for li, name in enumerate(label_names):
        idx = y == li
        plt.scatter(U[idx,0], U[idx,1], s=10, alpha=0.7, label=name)
    plt.title(f"{tag} UMAP"); plt.legend(markerscale=2, fontsize=8)
    upath = os.path.join(outdir, f"{tag}_UMAP.png")
    plt.tight_layout(); plt.savefig(upath, dpi=180); plt.close(fig)

    ts = TSNE(n_components=2, perplexity=30, init="pca", random_state=13)
    T2 = ts.fit_transform(Xn)
    fig = plt.figure(figsize=(6,5))
    for li, name in enumerate(label_names):
        idx = y == li
        plt.scatter(T2[idx,0], T2[idx,1], s=10, alpha=0.7, label=name)
    plt.title(f"{tag} t-SNE"); plt.legend(markerscale=2, fontsize=8)
    tpath = os.path.join(outdir, f"{tag}_TSNE.png")
    plt.tight_layout(); plt.savefig(tpath, dpi=180); plt.close(fig)
    print(f"[2D maps] -> {upath}, {tpath}")
    return upath, tpath

# ------------------ main ------------------

def main():
    ap = argparse.ArgumentParser(description="Swiss-German Dialect Feature Audit (split-wise)")
    ap.add_argument("--melspec_dir", type=str, default=None)
    ap.add_argument("--w2v6_dir", type=str, default=None)
    ap.add_argument("--espeak_dir", type=str, default=None)
    ap.add_argument("--outdir", type=str, default="feature_audit_out")
    ap.add_argument("--use", type=str, default="all",
                    choices=["mel","w2v6","espk","mel+w2v6","mel+espk","w2v6+espk","all"])
    ap.add_argument("--espk_max_features", type=int, default=4000)
    ap.add_argument("--espk_ngram_max", type=int, default=2)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    blocks = []  # list of (X, y, prefix, featnames_opt)

    # wav2vec layer 6
    if args.use in ["w2v6","mel+w2v6","w2v6+espk","all"] and args.w2v6_dir:
        X, y = load_w2v6_block(args.w2v6_dir)
        if X is not None:
            blocks.append((X, y, "w2v6", None))
            print(f"[w2v6] {X.shape} loaded")

    # mel
    if args.use in ["mel","mel+w2v6","mel+espk","all"] and args.melspec_dir:
        X, y = load_mel_block(args.melspec_dir)
        if X is not None:
            blocks.append((X, y, "mel", None))
            print(f"[mel] {X.shape} loaded")

    # espeak phoneme strings
    if args.use in ["espk","mel+espk","w2v6+espk","all"] and args.espeak_dir:
        X, y, names = load_espeak_block(args.espeak_dir,
                                        max_features=args.espk_max_features,
                                        ngram_max=args.espk_ngram_max)
        if X is not None:
            blocks.append((X, y, "espk", names))
            print(f"[espk] {X.shape} loaded (sparse -> dense for probes)")

    if not blocks:
        raise SystemExit("No features found. Check your paths and --use selection.")

    # Concatenate datasets (labels assumed to be same ID space across sets)
    X, y, featnames = align_blocks(blocks)

    # Label names as integers 0..K-1 unless you provide a mapping elsewhere
    classes = sorted(np.unique(y).tolist())
    # Ensure labels are 0..K-1
    label_to_idx = {lab:i for i, lab in enumerate(classes)}
    y_idx = np.array([label_to_idx[int(l)] for l in y], dtype=int)
    label_names = [str(c) for c in classes]

    # Save pooled/processed features
    np.save(os.path.join(args.outdir, "X.npy"), X)
    np.save(os.path.join(args.outdir, "y.npy"), y_idx)
    with open(os.path.join(args.outdir, "meta.json"), "w") as f:
        json.dump({"label_names": label_names,
                   "featnames": featnames}, f, indent=2)
    print(f"[Saved] X/y/meta to {args.outdir}")

    # Probes
    rep_lr, cm_lr = run_linear_probe(X, y_idx, label_names, f"LinearProbe_{args.use}", args.outdir)
    rep_xgb, cm_xgb = (None, None)
    if HAVE_XGB:
        rep_xgb, cm_xgb = run_xgb_probe(X, y_idx, label_names, f"XGB_{args.use}", args.outdir)

    # Feature scores (dense only)
    Xdense = X.toarray() if hasattr(X, "toarray") else X
    _ = run_feature_scores(Xdense, y_idx, featnames, args.outdir, tag=args.use.replace("+","_"))

    # 2D maps
    _ = run_maps(Xdense, y_idx, label_names, args.outdir, tag=args.use.replace("+","_"))

    # Write reports
    with open(os.path.join(args.outdir, "reports.txt"), "w") as f:
        f.write(rep_lr + "\n")
        if rep_xgb:
            f.write("\n" + rep_xgb + "\n")

    print("\nDone.")

if __name__ == "__main__":
    main()
