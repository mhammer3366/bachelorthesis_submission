
import os
from pathlib import Path
REPO_ROOT = Path(os.environ.get("THESIS_ROOT", Path(__file__).resolve().parents[2]))
# wav2vec_base_layer6.py
# Multi-GPU (DataParallel) wav2vec2 layer-6 feature extraction, using label_id.

import os
import numpy as np
import pandas as pd
import torch
from datasets import Dataset, Audio
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from tqdm import tqdm

# --- config ---
splits = {
    "train": str(REPO_ROOT / "data_preparation/merged_datasets/merged_train.tsv"),
    "validation": str(REPO_ROOT / "data_preparation/merged_datasets/merged_valid.tsv"),
    "test": str(REPO_ROOT / "data_preparation/merged_datasets/merged_test.tsv"),
}
out_dir = str(REPO_ROOT / "data_preparation/feature_extraction/ohne_deutsch/wav2vec_base_layer6")
os.makedirs(out_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_gpus = torch.cuda.device_count()
print(f"Detected GPUs: {num_gpus}")

# tune these
batch_size = 32            # set >= num_gpus for best DataParallel throughput
sample_rate = 16000
max_length = 80000         # ~5s at 16kHz; increase if you want longer
target_layer = 6           # wav2vec2 hidden layer to pool (0..12 for base)

# --- load model/processor ---
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
base_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base", output_hidden_states=True)

# Wrap with DataParallel if we have >1 GPU
if num_gpus > 1:
    print("⚡ Using torch.nn.DataParallel")
    model = torch.nn.DataParallel(base_model).to(device)
else:
    model = base_model.to(device)

model.eval()

def load_split(path: str) -> Dataset:
    # robust read, no dtype warnings
    df = pd.read_csv(path, sep="\t", dtype=str, low_memory=False)

    # normalize path col
    if "audio_path" not in df.columns and "path" in df.columns:
        df = df.rename(columns={"path": "audio_path"})

    # keep only files that exist
    df = df[df["audio_path"].apply(lambda p: isinstance(p, str) and os.path.isfile(p))]

    # require label_id
    if "label_id" not in df.columns:
        raise ValueError(f"{path} must contain a 'label_id' column!")

    # cast label_id
    df["label_id"] = pd.to_numeric(df["label_id"], errors="coerce")
    df = df.dropna(subset=["label_id"]).reset_index(drop=True)
    df["label_id"] = df["label_id"].astype(int)

    ds = Dataset.from_pandas(df[["audio_path", "label_id"]], preserve_index=False)
    ds = ds.cast_column("audio_path", Audio(sampling_rate=sample_rate))
    return ds

@torch.no_grad()
def extract_batch(arrays):
    # arrays: list of 1D np arrays
    inputs = processor(
        arrays,
        sampling_rate=sample_rate,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    # move to GPU
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # forward (DataParallel will split on dim 0 automatically)
    outputs = model(**inputs)  # hidden_states: tuple(layer) of [B, T, 768]
    pooled = outputs.hidden_states[target_layer].mean(dim=1)  # [B, 768]
    return pooled.cpu().numpy()

for split, path in splits.items():
    ds = load_split(path)
    n = len(ds)
    feats, labels = [], []
    print(f"\nExtracting features for {split} ({n} samples) ... "
          f"(batch_size={batch_size}, target_layer={target_layer})")

    for i in tqdm(range(0, n, batch_size), desc=split, unit="batch"):
        batch = ds[i:i + batch_size]
        arrays = [a["array"] for a in batch["audio_path"]]

        # filter out failed decodes
        good = [(a, l) for a, l in zip(arrays, batch["label_id"])
                if isinstance(a, np.ndarray) and a.size > 0]
        if not good:
            continue
        arrays, batch_labels = zip(*good)

        # try; if OOM, automatically halve the sub-batch once and retry
        try:
            feats.append(extract_batch(list(arrays)))
            labels.extend(batch_labels)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            # fallback: process in two smaller chunks
            mid = len(arrays) // 2
            for sub in (list(arrays[:mid]), list(arrays[mid:])):
                if not sub:
                    continue
                feats.append(extract_batch(sub))
                # align labels
                if sub is arrays[:mid]:
                    labels.extend(batch_labels[:mid])
                else:
                    labels.extend(batch_labels[mid:])

    if not feats:
        print(f"⚠️ No valid samples found for {split}, skipping save.")
        continue

    X = np.vstack(feats).astype(np.float32)
    y = np.array(labels, dtype=np.int64)

    np.save(os.path.join(out_dir, f"{split}_features.npy"), X)
    np.save(os.path.join(out_dir, f"{split}_labels.npy"), y)
    print(f"✅ {split}: features {X.shape}, labels {y.shape} → {out_dir}")
