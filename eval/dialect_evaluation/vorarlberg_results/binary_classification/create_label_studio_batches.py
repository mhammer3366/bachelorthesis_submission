#!/usr/bin/env python3

import os
from pathlib import Path
REPO_ROOT = Path(os.environ.get("THESIS_ROOT", Path(__file__).resolve().parents[4]))
# build_label_pack.py
# Make a 100–200 speaker labeling pack: pick uncertain speakers, build ~30s WAV per speaker.

import os, math, random
import pandas as pd
import numpy as np

from tqdm import tqdm
try:
    import soundfile as sf
except Exception:
    sf = None
try:
    import librosa
except Exception:
    librosa = None

# ---------- config ----------
PHONEME_TSV = str(REPO_ROOT / "data_preparation/feature_extraction/vorarlberg/vorarlberg_with_phonemes_with_client.tsv")
SPEAKER_PREDS = str(REPO_ROOT / "eval/dialect_evaluation/vorarlberg_results/binary_classification/concat_30/speaker_preds.csv")

OUT_DIR = str(REPO_ROOT / "eval/dialect_evaluation/vorarlberg_results/binary_classification/batches")
AUDIO_DIR = os.path.join(OUT_DIR, "audio")
PACK_CSV  = os.path.join(OUT_DIR, "pack.csv")
LABELS_CSV= os.path.join(OUT_DIR, "labels.csv")

NUM_SPEAKERS = 2000         # choose 100–200
CHUNK_SECS   = 30.0
DEFAULT_UTT_SECS = 5.0     # if duration missing
SR_TARGET = 16000

SELECTION_STRATEGY = "random"  # "uncertain" or "random"
MIN_UTTS_PER_SPK = 3

# ----------------------------

def read_audio_mono_16k(path):
    if not os.path.isfile(path):
        return None
    # Prefer soundfile; fallback to librosa
    if sf is not None:
        try:
            y, sr = sf.read(path, always_2d=False)
            if y is None:
                return None
            if y.ndim > 1:
                y = y.mean(axis=1)
            if sr != SR_TARGET:
                if librosa is None:
                    return None
                y = librosa.resample(y.astype(np.float32), orig_sr=sr, target_sr=SR_TARGET)
                sr = SR_TARGET
            return y.astype(np.float32)
        except Exception:
            pass
    if librosa is not None:
        try:
            y, sr = librosa.load(path, sr=SR_TARGET, mono=True)
            return y.astype(np.float32)
        except Exception:
            pass
    return None

def main():
    os.makedirs(AUDIO_DIR, exist_ok=True)

    dfp = pd.read_csv(SPEAKER_PREDS, dtype=str)
    # numeric fields
    for c in ["prob_high_german_mean", "prob_high_german_median", "prob_high_german_max"]:
        if c in dfp.columns:
            dfp[c] = pd.to_numeric(dfp[c], errors="coerce")
    # uncertainty = distance to 0.5
    if "prob_high_german_mean" not in dfp.columns:
        raise ValueError("speaker_preds.csv must have 'prob_high_german_mean'")

    dfp["uncertainty"] = (dfp["prob_high_german_mean"] - 0.5).abs()

    # pick speakers
    if SELECTION_STRATEGY == "uncertain":
        pick = dfp.sort_values("uncertainty", ascending=True).head(NUM_SPEAKERS)["client_id"].tolist()
    else:
        pick = dfp["client_id"].dropna().unique().tolist()
        random.shuffle(pick)
        pick = pick[:NUM_SPEAKERS]

    # load utterances
    dfa = pd.read_csv(PHONEME_TSV, sep="\t", dtype=str, low_memory=False)
    if "duration" in dfa.columns:
        dfa["duration"] = pd.to_numeric(dfa["duration"], errors="coerce").fillna(0.0).clip(lower=0.0)
    else:
        dfa["duration"] = np.nan

    pack_rows = []
    for cid in tqdm(pick, desc="Building pack"):
        g = dfa[dfa["client_id"] == cid].copy()
        if len(g) < MIN_UTTS_PER_SPK:
            continue
        g = g.sort_values("path")
        # accumulate utterances until ~CHUNK_SECS
        buf_paths, acc = [], 0.0
        for _, r in g.iterrows():
            d = float(r["duration"]) if (isinstance(r["duration"], (int, float)) and r["duration"] > 0) else DEFAULT_UTT_SECS
            if acc + d > CHUNK_SECS and buf_paths:
                break
            buf_paths.append(r["path"])
            acc += d
        if not buf_paths:
            buf_paths = g["path"].tolist()[:MIN_UTTS_PER_SPK]

        # load & concat audio
        audio_parts = []
        for ap in buf_paths:
            y = read_audio_mono_16k(ap)
            if y is not None and y.size > 0:
                audio_parts.append(y)
        if not audio_parts:
            continue
        ycat = np.concatenate(audio_parts)
        out_wav = os.path.join(AUDIO_DIR, f"{cid.replace('/', '__')}.wav")

        # write wav
        if sf is None:
            raise RuntimeError("soundfile is required to write WAV files")
        sf.write(out_wav, ycat, SR_TARGET)

        row = {
            "client_id": cid,
            "audio_concat_path": out_wav,
            "n_utts_used": len(buf_paths),
            "approx_secs": round(len(ycat)/SR_TARGET, 2)
        }
        # attach model prob for context
        sp = dfp[dfp["client_id"] == cid].head(1)
        if len(sp):
            row["prob_high_german_mean"] = float(sp["prob_high_german_mean"].iloc[0])
            row["model_pred"] = str(sp["pred_label_name"].iloc[0]) if "pred_label_name" in sp.columns else ""
        pack_rows.append(row)

    pack_df = pd.DataFrame(pack_rows).sort_values("client_id").reset_index(drop=True)
    pack_df.to_csv(PACK_CSV, index=False)
    print(f"✅ Saved pack → {PACK_CSV}  (rows={len(pack_df)})")

    if not os.path.isfile(LABELS_CSV):
        pd.DataFrame(columns=["client_id","label_manual","label_name","annotator","notes"]).to_csv(LABELS_CSV, index=False)
        print(f"✅ Created empty labels file → {LABELS_CSV}")

if __name__ == "__main__":
    main()
