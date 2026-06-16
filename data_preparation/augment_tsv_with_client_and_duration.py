#!/usr/bin/env python3
"""
Augment a TSV of audio samples by adding 'client_id' and 'duration' columns.

Input TSV is expected to have at least column 'path'. Other columns are preserved.
 - client_id: derived from the parent directory name that starts with 'spk_'
 - duration: read from WAV header (seconds)

Usage:
  python augment_tsv_with_client_and_duration.py \
    --in "/home/ai/AI-DataPool/Datasets/audio/Österreich/sliced_16000_mono/Wien/master_with_phonemes.tsv" \
    --out "/home/ai/AI-DataPool/Datasets/audio/Österreich/sliced_16000_mono/Wien/master_with_phonemes.with_client_duration.tsv"
"""

import argparse
import os
import sys
import wave
from contextlib import closing
from typing import Optional

import pandas as pd


def infer_client_id_from_path(audio_path: str) -> Optional[str]:
    """Extract client_id as "<event_dir>/spk_*".

    Example:
      .../sliced_16000/<event_dir>/spk_speaker_0/sentence_001.wav
      -> client_id: "<event_dir>/spk_speaker_0"
    Returns None if pattern not found.
    """
    try:
        parts = os.path.normpath(audio_path).split(os.sep)
        # find index of the 'spk_*' folder
        spk_index = None
        for idx, part in enumerate(parts):
            if part.startswith("spk_"):
                spk_index = idx
        if spk_index is None:
            return None
        if spk_index - 1 >= 0:
            event_dir = parts[spk_index - 1]
            return f"{event_dir}/{parts[spk_index]}"
        return parts[spk_index]
    except Exception:
        return None


def wav_duration_seconds(audio_path: str) -> Optional[float]:
    """Return duration in seconds for a WAV file, or None if unavailable."""
    if not isinstance(audio_path, str) or not audio_path.lower().endswith(".wav"):
        return None
    if not os.path.exists(audio_path):
        return None
    try:
        with closing(wave.open(audio_path, 'rb')) as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            if rate and frames:
                return float(frames) / float(rate)
    except Exception:
        return None
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="inp", required=True, help="Input TSV path")
    parser.add_argument("--out", dest="out", required=True, help="Output TSV path")
    parser.add_argument("--limit", type=int, default=None, help="Optional row limit for testing")
    args = parser.parse_args()

    if not os.path.exists(args.inp):
        print(f"Input not found: {args.inp}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(args.inp, sep="\t", dtype=str, low_memory=False)
    if "path" not in df.columns:
        print("Input TSV must contain column 'path'", file=sys.stderr)
        sys.exit(1)

    if args.limit is not None:
        df = df.head(args.limit)

    # Create columns if missing
    if "client_id" not in df.columns:
        df["client_id"] = None
    if "duration" not in df.columns:
        df["duration"] = None

    # Fill values
    paths = df["path"].astype(str).tolist()
    client_ids = []
    durations = []

    for p in paths:
        cid = infer_client_id_from_path(p)
        dur = wav_duration_seconds(p)
        client_ids.append(cid)
        durations.append(dur)

    # Prefer not to overwrite existing non-null values
    if "client_id" in df.columns:
        df["client_id"] = df["client_id"].where(df["client_id"].notna() & (df["client_id"].astype(str).str.len() > 0), client_ids)
    else:
        df["client_id"] = client_ids

    if "duration" in df.columns:
        # keep existing where present, else use calculated
        df["duration"] = pd.to_numeric(df["duration"], errors="coerce")
        df["duration"] = df["duration"].where(df["duration"].notna() & (df["duration"] > 0), durations)
    else:
        df["duration"] = durations

    # Save
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, sep="\t", index=False)
    print(f"Wrote augmented TSV → {args.out}")


if __name__ == "__main__":
    main()


