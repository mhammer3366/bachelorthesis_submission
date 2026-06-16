#!/usr/bin/env python3

import os
from pathlib import Path
REPO_ROOT = Path(os.environ.get("THESIS_ROOT", Path(__file__).resolve().parents[3]))
# Adds client_id (speaker id) and duration (seconds) to Vorarlberg phoneme TSV.
# Input TSV columns: path, text, phoneme
# Output TSV columns: path, text, phoneme, client_id [, dataset_id], duration

import os
import re
import math
import pandas as pd
import numpy as np
from tqdm import tqdm

# Optional deps
try:
    import soundfile as sf  # fast header-based duration for many formats
except Exception:
    sf = None
try:
    import wave  # stdlib WAV reader (header-based)
except Exception:
    wave = None
try:
    import librosa  # last-resort fallback
except Exception:
    librosa = None

# ---------- hardcoded paths ----------
INPUT_TSV  = str(REPO_ROOT / "data_preparation/feature_extraction/vorarlberg/vorarlberg_with_phonemes.tsv")
OUTPUT_TSV = str(REPO_ROOT / "data_preparation/feature_extraction/vorarlberg/vorarlberg_with_phonemes_with_client_and_duration.tsv")

# Optional: also store the top-level dataset folder (e.g., "antenne_vorarlberg_nachrichten_im_dialekt")
ADD_DATASET_ID = False

def extract_client_id(p: str) -> str:
    """
    From a full path like:
      /.../Vorarlberg/<dataset>/sliced_16000/<hash>/speaker_0/sentence_001.wav
    return:
      <dataset>/sliced_16000/<hash>/speaker_0
    """
    if not isinstance(p, str) or not p:
        return "unknown_speaker"

    parts = os.path.normpath(p).split(os.sep)

    # find the "Vorarlberg" root folder
    try:
        i_vbg = next(i for i, part in enumerate(parts) if part == "Vorarlberg")
    except StopIteration:
        # fallback: nearest "speaker_*" and a few parents
        try:
            i_spk = max(i for i, part in enumerate(parts) if part.startswith("speaker_"))
            start = max(0, i_spk - 3)
            return "/".join(parts[start:i_spk+1])
        except ValueError:
            return "unknown_speaker"

    # find the 'speaker_*' directory after Vorarlberg
    try:
        i_spk = i_vbg + 1 + next(i for i, part in enumerate(parts[i_vbg+1:]) if part.startswith("speaker_"))
    except StopIteration:
        parent_dir = os.path.basename(os.path.dirname(p))
        grand_dir  = os.path.basename(os.path.dirname(os.path.dirname(p)))
        if parent_dir:
            return f"{grand_dir}/{parent_dir}" if grand_dir else parent_dir
        return "unknown_speaker"

    # join from the folder after "Vorarlberg" up to and including "speaker_*"
    speaker_parts = parts[i_vbg+1 : i_spk+1]
    if not speaker_parts:
        return "unknown_speaker"
    return "/".join(speaker_parts)

def extract_dataset_id_from_client(client_id: str) -> str:
    """
    From client_id = 'dataset/sliced_16000/<hash>/speaker_0' return 'dataset'.
    """
    if not isinstance(client_id, str) or not client_id:
        return "unknown_dataset"
    return client_id.split("/", 1)[0]

def _duration_soundfile(path: str) -> float:
    if sf is None:
        raise RuntimeError("soundfile unavailable")
    info = sf.info(path)
    if info.samplerate and info.frames:
        return float(info.frames) / float(info.samplerate)
    raise RuntimeError("soundfile info missing frames/samplerate")

def _duration_wave(path: str) -> float:
    if wave is None:
        raise RuntimeError("wave unavailable")
    with wave.open(path, "rb") as w:
        frames = w.getnframes()
        sr = w.getframerate()
    if sr > 0 and frames >= 0:
        return float(frames) / float(sr)
    raise RuntimeError("wave missing frames/samplerate")

def _duration_librosa(path: str) -> float:
    if librosa is None:
        raise RuntimeError("librosa unavailable")
    d = librosa.get_duration(filename=path)
    if d is None or not math.isfinite(d):
        raise RuntimeError("librosa returned invalid duration")
    return float(d)

def get_duration_sec(path: str) -> float:
    """
    Fast, robust duration in seconds. Returns np.nan if it cannot be determined.
    """
    try:
        if not isinstance(path, str) or not os.path.isfile(path):
            return np.nan
        # Try soundfile (fast header read)
        try:
            return _duration_soundfile(path)
        except Exception:
            pass
        # Try stdlib wave (WAV only)
        try:
            return _duration_wave(path)
        except Exception:
            pass
        # Last resort: librosa (opens data; slower)
        try:
            return _duration_librosa(path)
        except Exception:
            pass
        return np.nan
    except Exception:
        return np.nan

def main():
    print(f"Reading: {INPUT_TSV}")
    df = pd.read_csv(INPUT_TSV, sep="\t", dtype=str, low_memory=False)

    # sanity check
    for c in ["path", "text", "phoneme"]:
        if c not in df.columns:
            raise ValueError(f"Input TSV must contain column '{c}'")

    print("Extracting client_id from 'path'…")
    df["client_id"] = df["path"].apply(extract_client_id)

    if ADD_DATASET_ID:
        print("Adding dataset_id (top-level folder under Vorarlberg)…")
        df["dataset_id"] = df["client_id"].apply(extract_dataset_id_from_client)

    print("Computing durations (seconds)…")
    durations = []
    for p in tqdm(df["path"].tolist(), desc="durations", unit="file"):
        durations.append(get_duration_sec(p))
    df["duration"] = durations

    os.makedirs(os.path.dirname(OUTPUT_TSV), exist_ok=True)
    df.to_csv(OUTPUT_TSV, sep="\t", index=False)
    print(f"✅ Saved: {OUTPUT_TSV}")

    print("Sample rows:")
    print(df.head(5).to_string(index=False))

    # Quick stats
    n_total = len(df)
    n_nan = int(pd.isna(df["duration"]).sum())
    if n_nan > 0:
        print(f"⚠️  Durations missing for {n_nan}/{n_total} files "
              f"({(100.0*n_nan/n_total):.1f}%). Your training/inference scripts will default to 5s for those.")

if __name__ == "__main__":
    main()
