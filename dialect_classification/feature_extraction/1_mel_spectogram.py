
import os
from pathlib import Path
REPO_ROOT = Path(os.environ.get("THESIS_ROOT", Path(__file__).resolve().parents[2]))
# mel_spectogram.py
# Fast(ish) mel-spectrogram extraction from merged TSVs using multi-CPU parallelism.
# (librosa is CPU-bound; GPUs don't help here.)
# Reads TSVs with: audio_path, label_id  → writes chunked .npy files per split.

import os
import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm
from multiprocessing import Pool, cpu_count, get_context
from functools import partial

# --------------- config ---------------
SPLITS = {
    "train": str(REPO_ROOT / "data_preparation/merged_datasets/merged_train.tsv"),
    "validation": str(REPO_ROOT / "data_preparation/merged_datasets/merged_valid.tsv"),
    "test": str(REPO_ROOT / "data_preparation/merged_datasets/merged_test.tsv"),
}
OUT_DIR = str(REPO_ROOT / "data_preparation/feature_extraction/ohne_deutsch/saved_features_melspec")
os.makedirs(OUT_DIR, exist_ok=True)

# audio / mel params
SAMPLE_RATE = 16000
N_MELS = 128
HOP_LENGTH = 512
N_FFT = 2048
MAX_FRAMES = 400      # pad/truncate frames
CHUNK_SIZE = 10000    # examples per saved chunk

# parallelism
NUM_WORKERS = max(1, cpu_count() - 1)  # leave 1 core free
BATCH_SIZE_IO = 512                    # how many files to dispatch to workers at once

# to avoid thread oversubscription when using many processes
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

# --------------- helpers ---------------
def load_split_df(tsv_path: str) -> pd.DataFrame:
    df = pd.read_csv(tsv_path, sep="\t", dtype=str, low_memory=False)

    # choose path column
    if "audio_path" not in df.columns:
        if "path" in df.columns:
            df = df.rename(columns={"path": "audio_path"})
        else:
            raise ValueError(f"{tsv_path} must contain 'audio_path' or 'path' column.")

    # must have label_id already
    if "label_id" not in df.columns:
        raise ValueError(f"{tsv_path} must contain 'label_id' column.")

    # keep valid files only
    df = df[df["audio_path"].apply(lambda p: isinstance(p, str) and os.path.isfile(p))].reset_index(drop=True)
    df["label_id"] = pd.to_numeric(df["label_id"], errors="coerce")
    df = df.dropna(subset=["label_id"]).reset_index(drop=True)
    df["label_id"] = df["label_id"].astype(int)

    return df[["audio_path", "label_id"]]

def extract_log_mel_from_file(audio_path: str) -> np.ndarray:
    y, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)  # [n_mels, T]
    # pad/truncate to MAX_FRAMES
    T = log_mel.shape[1]
    if T >= MAX_FRAMES:
        log_mel = log_mel[:, :MAX_FRAMES]
    else:
        log_mel = np.pad(log_mel, ((0, 0), (0, MAX_FRAMES - T)), mode="constant")
    return log_mel.astype(np.float32)

def _worker_compute(path_label):
    """Worker wrapper: returns (mel, label) or (None, None) on failure."""
    path, label = path_label
    try:
        mel = extract_log_mel_from_file(path)
        return mel, label
    except Exception:
        return None, None

def save_chunk(split_name: str, chunk_id: int, mels: list, labels: list):
    np.save(os.path.join(OUT_DIR, f"{split_name}_melspec_chunk{chunk_id}.npy"),
            np.stack(mels, axis=0))
    np.save(os.path.join(OUT_DIR, f"{split_name}_labels_chunk{chunk_id}.npy"),
            np.array(labels, dtype=np.int64))

def process_split(split_name: str, tsv_path: str):
    df = load_split_df(tsv_path)
    n = len(df)
    if n == 0:
        print(f"[WARN] {split_name}: no valid rows after checks. Skipping.")
        return

    print(f"[INFO] {split_name}: {n} files to process "
          f"(workers={NUM_WORKERS}, batch_io={BATCH_SIZE_IO})")

    mels_buf, labels_buf = [], []
    chunk_id = 1

    # Use spawn context for safety
    with get_context("spawn").Pool(processes=NUM_WORKERS) as pool:
        for start in tqdm(range(0, n, BATCH_SIZE_IO), desc=split_name, unit="files"):
            end = min(start + BATCH_SIZE_IO, n)
            batch_paths = df["audio_path"].iloc[start:end].tolist()
            batch_labels = df["label_id"].iloc[start:end].tolist()

            # dispatch
            for mel, lab in pool.imap_unordered(_worker_compute, zip(batch_paths, batch_labels)):
                if mel is None:
                    continue
                mels_buf.append(mel)
                labels_buf.append(lab)

                if len(mels_buf) >= CHUNK_SIZE:
                    save_chunk(split_name, chunk_id, mels_buf, labels_buf)
                    chunk_id += 1
                    mels_buf, labels_buf = [], []

    # flush remainder
    if mels_buf:
        save_chunk(split_name, chunk_id, mels_buf, labels_buf)

    print(f"[OK] {split_name}: wrote {chunk_id} chunk(s) to {OUT_DIR} "
          f"(each mel shape {N_MELS}x{MAX_FRAMES})")

# --------------- run ---------------
if __name__ == "__main__":
    for split, tsv in SPLITS.items():
        process_split(split, tsv)
