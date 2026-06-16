#!/usr/bin/env python3

import os
from pathlib import Path
REPO_ROOT = Path(os.environ.get("THESIS_ROOT", Path(__file__).resolve().parents[2]))
# -*- coding: utf-8 -*-

"""
Two-Stage Speaker Classification from Phonemes (clean rebuilt TSV)
------------------------------------------------------------------
Input TSV (rebuilt v3) must contain at least:
  clip_path, podcast, episode, speaker, duration_sec, phoneme
(If duration_sec is missing, start/end will be used.)

Stage A (binary): non-german(0) vs german(1)
Stage B (multiclass): dialect 0..6 (only if non-german)

Outputs (in OUTPUT_DIR):
  - speaker_assignments.csv
  - clip_labels.csv
  - manifests/dialect_{k}.tsv (k in 0..7)
  - distribution_overall.csv
  - distribution_by_podcast.csv
  - distribution_by_episode.csv
  - distribution_hours_full_master.csv
  - distribution_hours_full_master_by_podcast.csv
"""

import os
from pathlib import Path
from typing import List, Tuple
import numpy as np
import pandas as pd
from joblib import load
from tqdm import tqdm

# =================== CONFIG ===================
TSV_PATH = "/home/ai/AI-DataPool/Datasets/audio/Schweiz/16000_mono_wav_splitted_1/master_index_with_phonemes.rebuilt.v3.tsv"

# Stage A (binary) model dir: german vs non-german
BINARY_MODEL_DIR  = str(REPO_ROOT / "models/5_nb_phoneme_binary")
# Stage B (multiclass) model dir: dialect 0..6
DIALECT_MODEL_DIR = str(REPO_ROOT / "models/4_linear_phoneme")

OUTPUT_DIR = str(REPO_ROOT / "eval/dialect_evaluation/swiss_srf_espeak_classification_two_stage_2")

TARGET_CHUNK_SECS = 60.0     # target phoneme duration per doc
MAX_CYCLES        = 10_000   # safety for padding short speakers

DIALECT_NAMES = {
    0: "Basel",
    1: "Bern",
    2: "Innerschweiz",
    3: "Ostschweiz",
    4: "Wallis",
    5: "Zürich",
    6: "Graubünden",
    7: "German",
}
# =============================================


# ---------- helpers ----------

def ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize columns we need; compute duration if necessary."""
    # audio_path
    if "audio_path" not in df.columns:
        if "clip_path" in df.columns:
            df = df.rename(columns={"clip_path": "audio_path"})
        elif "path" in df.columns:
            df = df.rename(columns={"path": "audio_path"})
        else:
            raise ValueError("Missing 'clip_path' / 'audio_path' / 'path' in TSV.")

    # keys (assume present & clean in rebuilt TSV, but strip anyway)
    for c in ("podcast", "episode", "speaker"):
        if c not in df.columns:
            raise ValueError(f"Missing '{c}' in TSV.")
        df[c] = df[c].astype(str).str.strip()

    # durations
    if "duration_sec" in df.columns:
        df["duration_sec"] = pd.to_numeric(df["duration_sec"], errors="coerce").fillna(0.0).clip(lower=0.0)
    elif {"start", "end"}.issubset(df.columns):
        start = pd.to_numeric(df["start"], errors="coerce")
        end   = pd.to_numeric(df["end"], errors="coerce")
        df["duration_sec"] = (end - start).fillna(0.0).clip(lower=0.0)
    else:
        raise ValueError("Need 'duration_sec' or both 'start' and 'end'.")

    # phoneme
    if "phoneme" not in df.columns:
        raise ValueError("Missing 'phoneme' column.")
    df["phoneme"] = df["phoneme"].astype(str)

    return df


def build_docs_for_speaker(g: pd.DataFrame, target_secs: float) -> List[str]:
    """Concatenate phoneme strings to ~target_secs; cycle if short."""
    g = g.sort_values("audio_path")
    phonemes  = g["phoneme"].tolist()
    durations = g["duration_sec"].astype(float).tolist()
    total = float(np.sum(durations))
    docs: List[str] = []

    if total <= 0.0 and phonemes:
        docs.append(" ".join((phonemes * 12)[:12]))
        return docs

    if total >= target_secs:
        buf, acc = [], 0.0
        for p, d in zip(phonemes, durations):
            d = d if d > 0 else 5.0
            if acc + d > target_secs and buf:
                docs.append(" ".join(buf))
                buf, acc = [], 0.0
            buf.append(p); acc += d
        if buf:
            docs.append(" ".join(buf))
    else:
        if not phonemes:
            return []
        buf, acc = [], 0.0
        i, n = 0, len(phonemes)
        while acc < target_secs and i < MAX_CYCLES:
            j = i % n
            p = phonemes[j]
            d = durations[j] if durations[j] > 0 else 5.0
            buf.append(p); acc += d; i += 1
        docs.append(" ".join(buf))
    return docs


def load_models():
    bin_model = load(os.path.join(BINARY_MODEL_DIR, "nb_model.joblib"))
    bin_vec   = load(os.path.join(BINARY_MODEL_DIR, "vectorizer.joblib"))
    dia_model = load(os.path.join(DIALECT_MODEL_DIR, "model.joblib"))
    dia_vec   = load(os.path.join(DIALECT_MODEL_DIR, "vectorizer.joblib"))

    # Reorder helpers to enforce fixed class ordering
    def reorder_bin_probs(pb: np.ndarray) -> np.ndarray:
        # want columns [non_german(0), german(1)]
        if hasattr(bin_model, "classes_"):
            cls = list(bin_model.classes_)
            idx0 = cls.index(0) if 0 in cls else None
            idx1 = cls.index(1) if 1 in cls else None
            if idx0 is not None and idx1 is not None and (idx0, idx1) != (0, 1):
                pb = pb[:, [idx0, idx1]]
        return pb

    def reorder_dia_probs(pdias: np.ndarray) -> np.ndarray:
        # want columns [0,1,2,3,4,5,6]
        if hasattr(dia_model, "classes_"):
            cls = list(dia_model.classes_)
            if cls != list(range(7)):
                order = [cls.index(k) for k in range(7)]
                pdias = pdias[:, order]
        return pdias

    return bin_model, bin_vec, dia_model, dia_vec, reorder_bin_probs, reorder_dia_probs


def aggregate_probs_binary(probs: np.ndarray) -> Tuple[int, float, float]:
    m = probs.mean(axis=0)          # shape (2,)
    pred = int(np.argmax(m))        # 0=non, 1=german
    return pred, float(m[0]), float(m[1])


def aggregate_probs_multiclass(probs: np.ndarray) -> Tuple[int, List[float]]:
    m = probs.mean(axis=0)          # shape (7,)
    return int(np.argmax(m)), m.tolist()


def make_distribution(rows_df: pd.DataFrame, speakers_df: pd.DataFrame) -> pd.DataFrame:
    total_clips = len(rows_df)
    total_speakers = len(speakers_df)
    total_clips_excl_ger = int((rows_df["final_label_id"] != 7).sum())
    total_speakers_excl_ger = int((speakers_df["final_label_id"] != 7).sum())

    out = []
    for k in range(8):
        name   = DIALECT_NAMES[k]
        mask_k = (rows_df["final_label_id"] == k)
        clips_k = int(mask_k.sum())
        hours_k = float(rows_df.loc[mask_k, "duration_sec"].sum() / 3600.0)
        spk_k   = int((speakers_df["final_label_id"] == k).sum())

        pct_clips_all = (clips_k / total_clips * 100.0) if total_clips else 0.0
        pct_spk_all   = (spk_k / total_speakers * 100.0) if total_speakers else 0.0

        if k == 7:
            pct_clips_excl = 0.0
            pct_spk_excl   = 0.0
        else:
            pct_clips_excl = (clips_k / total_clips_excl_ger * 100.0) if total_clips_excl_ger else 0.0
            pct_spk_excl   = (spk_k / total_speakers_excl_ger * 100.0) if total_speakers_excl_ger else 0.0

        out.append({
            "final_label_id": k,
            "final_label_name": name,
            "clips": clips_k,
            "clips_pct_all": round(pct_clips_all, 4),
            "clips_pct_excl_german": round(pct_clips_excl, 4),
            "hours": round(hours_k, 4),
            "speakers": spk_k,
            "speakers_pct_all": round(pct_spk_all, 4),
            "speakers_pct_excl_german": round(pct_spk_excl, 4),
        })
    return pd.DataFrame(out)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    man_dir = Path(OUTPUT_DIR) / "manifests"
    man_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load TSV (clean, rebuilt v3) ----
    wanted = ["clip_path","podcast","episode","speaker","duration_sec","start","end","phoneme"]
    print("Loading TSV…")
    df_full = pd.read_csv(TSV_PATH, sep="\t", dtype=str, usecols=lambda c: c in wanted, low_memory=False)
    df_full = ensure_columns(df_full)
    df_full["phoneme"] = df_full["phoneme"].astype(str)

    # keep rows with non-empty phoneme for classification
    mask = df_full["phoneme"].str.strip().str.len() > 0
    df = df_full.loc[mask, ["audio_path","phoneme","duration_sec","podcast","episode","speaker"]].copy()
    df["duration_sec"] = pd.to_numeric(df["duration_sec"], errors="coerce").fillna(0.0)

    print(f"Total rows loaded: {len(df_full):,}")
    print(f"Rows with phonemes (used for classification): {len(df):,}")
    print(f"Unique speakers (full): {df_full[['podcast','episode','speaker']].drop_duplicates().shape[0]:,}")
    print(f"Unique speakers (with phonemes): {df[['podcast','episode','speaker']].drop_duplicates().shape[0]:,}")

    # ---- Load models ----
    print("Loading models…")
    bin_model, bin_vec, dia_model, dia_vec, reorder_bin_probs, reorder_dia_probs = load_models()

    # ---- Group by speaker & classify ----
    key_cols = ["podcast","episode","speaker"]
    groups = df.groupby(key_cols, sort=False)

    speaker_rows = []
    clip_rows    = []

    print("Classifying speakers…")
    for (podcast, episode, speaker), g in tqdm(groups, total=groups.ngroups, unit="spk"):
        docs = build_docs_for_speaker(g, TARGET_CHUNK_SECS)
        if not docs:
            # fallback: label German if we truly got nothing
            speaker_rows.append({
                "podcast": podcast, "episode": episode, "speaker": speaker,
                "is_german": 1, "prob_non_german": 0.0, "prob_german": 1.0,
                "dialect_pred": -1, "dialect_probs": "[]",
                "final_label_id": 7, "final_label_name": DIALECT_NAMES[7],
                "num_docs": 0, "num_clips": len(g)
            })
            for p in g["audio_path"].tolist():
                clip_rows.append({"audio_path": p, "final_label_id": 7,
                                  "podcast": podcast, "episode": episode, "speaker": speaker})
            continue

        # Stage A
        Xb = bin_vec.transform(docs)
        pb = bin_model.predict_proba(Xb)
        pb = reorder_bin_probs(pb)
        pred_bin, p_non, p_ger = aggregate_probs_binary(pb)

        if pred_bin == 1:
            # German
            speaker_rows.append({
                "podcast": podcast, "episode": episode, "speaker": speaker,
                "is_german": 1, "prob_non_german": p_non, "prob_german": p_ger,
                "dialect_pred": -1, "dialect_probs": "[]",
                "final_label_id": 7, "final_label_name": DIALECT_NAMES[7],
                "num_docs": len(docs), "num_clips": len(g)
            })
            for p in g["audio_path"].tolist():
                clip_rows.append({"audio_path": p, "final_label_id": 7,
                                  "podcast": podcast, "episode": episode, "speaker": speaker})
        else:
            # Dialect stage
            Xd = dia_vec.transform(docs)
            pdia = dia_model.predict_proba(Xd)
            pdia = reorder_dia_probs(pdia)
            pred_dia, mean_probs = aggregate_probs_multiclass(pdia)
            final_label = int(pred_dia)

            speaker_rows.append({
                "podcast": podcast, "episode": episode, "speaker": speaker,
                "is_german": 0, "prob_non_german": p_non, "prob_german": p_ger,
                "dialect_pred": pred_dia,
                "dialect_probs": ",".join(f"{x:.6f}" for x in mean_probs),
                "final_label_id": final_label,
                "final_label_name": DIALECT_NAMES.get(final_label, str(final_label)),
                "num_docs": len(docs), "num_clips": len(g)
            })
            for p in g["audio_path"].tolist():
                clip_rows.append({"audio_path": p, "final_label_id": final_label,
                                  "podcast": podcast, "episode": episode, "speaker": speaker})

    # ---- Save assignments & clip labels ----
    speaker_df = pd.DataFrame(speaker_rows)
    clip_df    = pd.DataFrame(clip_rows)

    out_dir = Path(OUTPUT_DIR)
    speaker_out = out_dir / "speaker_assignments.csv"
    clip_out    = out_dir / "clip_labels.csv"
    speaker_df.to_csv(speaker_out, index=False)
    clip_df.to_csv(clip_out, index=False)
    print(f"✓ Wrote {speaker_out} ({len(speaker_df):,} speakers)")
    print(f"✓ Wrote {clip_out} ({len(clip_df):,} clips)")

    # ---- Manifests by label ----
    man_dir = out_dir / "manifests"
    man_dir.mkdir(exist_ok=True, parents=True)
    for k in range(8):
        paths = clip_df.loc[clip_df["final_label_id"] == k, "audio_path"].tolist()
        man_p = man_dir / f"dialect_{k}.tsv"
        with open(man_p, "w", encoding="utf-8") as f:
            f.write("audio_path\n")
            for p in paths:
                f.write(f"{p}\n")
        print(f"✓ Manifest: {man_p}  ({len(paths):,} files)")

    # ---- Distributions over classified clips ----
    merged = clip_df.merge(df[["audio_path","duration_sec"]], on="audio_path", how="left")
    merged["duration_sec"] = pd.to_numeric(merged["duration_sec"], errors="coerce").fillna(0.0)

    overall = make_distribution(merged, speaker_df)
    overall.to_csv(out_dir / "distribution_overall.csv", index=False)
    print(f"✓ Wrote {out_dir / 'distribution_overall.csv'}")

    by_podcast_rows = []
    for podcast, g_clip in merged.groupby("podcast", dropna=False):
        g_spk = speaker_df.loc[speaker_df["podcast"] == podcast]
        dist = make_distribution(g_clip, g_spk)
        dist.insert(0, "podcast", podcast)
        by_podcast_rows.append(dist)
    if by_podcast_rows:
        by_podcast = pd.concat(by_podcast_rows, ignore_index=True)
        by_podcast.to_csv(out_dir / "distribution_by_podcast.csv", index=False)
        print(f"✓ Wrote {out_dir / 'distribution_by_podcast.csv'}")

    by_episode_rows = []
    for (podcast, episode), g_clip in merged.groupby(["podcast","episode"], dropna=False):
        g_spk = speaker_df.loc[(speaker_df["podcast"] == podcast) & (speaker_df["episode"] == episode)]
        dist = make_distribution(g_clip, g_spk)
        dist.insert(0, "episode", episode)
        dist.insert(0, "podcast", podcast)
        by_episode_rows.append(dist)
    if by_episode_rows:
        by_episode = pd.concat(by_episode_rows, ignore_index=True)
        by_episode.to_csv(out_dir / "distribution_by_episode.csv", index=False)
        print(f"✓ Wrote {out_dir / 'distribution_by_episode.csv'}")

    # ---- Hours per dialect over FULL master (project speaker labels to all rows) ----
    print("Computing hours per dialect over the FULL master TSV…")
    master_min = df_full[["audio_path","podcast","episode","speaker","duration_sec"]].copy()
    master_min["duration_sec"] = pd.to_numeric(master_min["duration_sec"], errors="coerce").fillna(0.0)
    total_master_hours = master_min["duration_sec"].sum() / 3600.0
    print(f"Total duration in master (all rows): {total_master_hours:.2f} h")

    speaker_df_norm = speaker_df.copy()
    for c in ("podcast","episode","speaker"):
        speaker_df_norm[c] = speaker_df_norm[c].astype(str).str.strip()

    full_merged = master_min.merge(
        speaker_df_norm[["podcast","episode","speaker","final_label_id"]],
        on=["podcast","episode","speaker"],
        how="left"
    ).dropna(subset=["final_label_id"]).copy()
    full_merged["final_label_id"] = full_merged["final_label_id"].astype(int)

    hours_overall = (
        full_merged.groupby("final_label_id", dropna=False)["duration_sec"]
        .sum()
        .div(3600.0)
        .reset_index(name="hours")
    )
    hours_overall["dialect"] = hours_overall["final_label_id"].map(DIALECT_NAMES)
    hours_overall = hours_overall.sort_values("hours", ascending=False)
    hours_overall.to_csv(out_dir / "distribution_hours_full_master.csv", index=False)
    print(f"✓ Wrote {out_dir / 'distribution_hours_full_master.csv'}")

    by_pod_list = []
    for pod, g in full_merged.groupby("podcast", dropna=False):
        h = (
            g.groupby("final_label_id", dropna=False)["duration_sec"]
            .sum()
            .div(3600.0)
            .reset_index(name="hours")
        )
        h["dialect"] = h["final_label_id"].map(DIALECT_NAMES)
        h.insert(0, "podcast", pod)
        by_pod_list.append(h)
    if by_pod_list:
        hours_by_podcast = pd.concat(by_pod_list, ignore_index=True).sort_values(
            ["podcast","hours"], ascending=[True, False]
        )
        hours_by_podcast.to_csv(out_dir / "distribution_hours_full_master_by_podcast.csv", index=False)
        print(f"✓ Wrote {out_dir / 'distribution_hours_full_master_by_podcast.csv'}")


if __name__ == "__main__":
    main()
