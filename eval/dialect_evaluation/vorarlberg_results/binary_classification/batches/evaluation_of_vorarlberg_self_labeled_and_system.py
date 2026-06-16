#!/usr/bin/env python3

import os
from pathlib import Path
REPO_ROOT = Path(os.environ.get("THESIS_ROOT", Path(__file__).resolve().parents[5]))
# merge_ls_tsv_vs_preds.py
import os, json, pandas as pd, numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

# paths
LABEL_STUDIO_TSV = str(REPO_ROOT / "eval/dialect_evaluation/vorarlberg_results/binary_classification/batches/self_labeled_vorarlberg.tsv")
SPEAKER_PREDS    = str(REPO_ROOT / "eval/dialect_evaluation/vorarlberg_results/binary_classification/concat_30/speaker_preds.csv")
OUT_DIR = str(REPO_ROOT / "labeling/vbg_binary")
OUT_MERGED_CSV   = os.path.join(OUT_DIR, "merged_eval.csv")
OUT_METRICS_JSON = os.path.join(OUT_DIR, "manual_eval.json")
os.makedirs(OUT_DIR, exist_ok=True)


LABEL_MAP = {"Dialect": 0, "High German": 1, "Unsure": -1}

def url_to_client_id(u: str) -> str:
    # Handles "/data/local-files/?d=.../audio/<cid__.wav>" and "local-files:/abs/.../<cid__.wav>"
    s = u or ""
    if "?d=" in s:
        s = s.split("?d=", 1)[1]
    if s.startswith("local-files:"):
        s = s[len("local-files:"):]
    base = os.path.splitext(os.path.basename(s))[0]
    return base.replace("__", "/")

def main():
    # --- read Label Studio export (TSV) ---
    ls = pd.read_csv(LABEL_STUDIO_TSV, sep="\t", dtype=str, engine="python", quotechar='"')
    for col in ["audio", "label", "updated_at"]:
        if col not in ls.columns:
            raise ValueError(f"TSV must contain '{col}' (got columns: {ls.columns.tolist()})")

    ls["client_id"]    = ls["audio"].apply(url_to_client_id)
    ls["label_manual"] = ls["label"].map(LABEL_MAP).fillna(-1).astype(int)
    ls["ts"]           = pd.to_datetime(ls["updated_at"], errors="coerce")

    # keep latest annotation per client
    ls_latest = ls.sort_values("ts").drop_duplicates("client_id", keep="last")

    print(f"Label Studio rows: {len(ls)}")
    print(f"Unique labeled client_id: {ls['client_id'].nunique()}  (latest kept: {len(ls_latest)})")
    print("Label distribution (latest):", ls_latest["label_manual"].value_counts(dropna=False).to_dict())

    # --- read predictions ---
    preds = pd.read_csv(SPEAKER_PREDS, dtype=str)
    if not {"client_id","pred_label_idx"}.issubset(preds.columns):
        raise ValueError("speaker_preds.csv must have columns: 'client_id', 'pred_label_idx'")
    preds["pred_label_idx"] = pd.to_numeric(preds["pred_label_idx"], errors="coerce")
    if "prob_high_german_mean" in preds.columns:
        preds["prob_high_german_mean"] = pd.to_numeric(preds["prob_high_german_mean"], errors="coerce")

    print(f"Preds unique client_id: {preds['client_id'].nunique()}  (rows={len(preds)})")

    # --- overlap diagnostics ---
    labeled_ids = set(ls_latest["client_id"])
    pred_ids    = set(preds["client_id"])
    intersect   = labeled_ids & pred_ids
    only_ls     = list(labeled_ids - pred_ids)
    only_pred   = list(pred_ids - labeled_ids)

    print(f"Overlap (unique client_id) = {len(intersect)}")
    if only_ls:
        print("Example labeled-but-not-in-preds (first 8):", only_ls[:8])
    if only_pred:
        print("Example in-preds-but-not-labeled (first 8):", only_pred[:8])

    # --- merge + save ---
    merged = preds.merge(ls_latest[["client_id","label_manual"]], on="client_id", how="inner")
    merged.to_csv(OUT_MERGED_CSV, index=False)
    print(f"✅ Saved merged → {OUT_MERGED_CSV}  (n_rows={len(merged)})")

    # --- metrics (only on certain labels 0/1) ---
    eval_df = merged[merged["label_manual"].isin([0,1])].copy()
    if len(eval_df) == 0:
        metrics = {"n": 0}
        print("⚠️ No certain manual labels (0/1) found after merge.")
    else:
        y_true = eval_df["label_manual"].astype(int).to_numpy()
        y_pred = eval_df["pred_label_idx"].astype(int).to_numpy()

        acc   = float(accuracy_score(y_true, y_pred))
        macro = float(f1_score(y_true, y_pred, average="macro"))
        f1_d  = float(f1_score(y_true, y_pred, pos_label=0))
        f1_hg = float(f1_score(y_true, y_pred, pos_label=1))
        cm    = confusion_matrix(y_true, y_pred, labels=[0,1]).tolist()

        metrics = {
            "n": int(len(eval_df)),
            "accuracy": acc,
            "macro_f1": macro,
            "f1_dialect": f1_d,
            "f1_high_german": f1_hg,
            "confusion_matrix_labels": ["dialect","high_german"],
            "confusion_matrix": cm,
            "support": {
                "dialect": int((y_true==0).sum()),
                "high_german": int((y_true==1).sum())
            }
        }
        print(f"n={metrics['n']}  acc={acc:.3f}  macroF1={macro:.3f}  "
              f"F1(dialect)={f1_d:.3f}  F1(high_german)={f1_hg:.3f}  cm={cm}")

    with open(OUT_METRICS_JSON, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"✅ Saved metrics → {OUT_METRICS_JSON}")

if __name__ == "__main__":
    main()