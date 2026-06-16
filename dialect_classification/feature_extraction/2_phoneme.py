
import os
from pathlib import Path
REPO_ROOT = Path(os.environ.get("THESIS_ROOT", Path(__file__).resolve().parents[2]))
# phoneme.py
# Multi-GPU (DataParallel) phoneme extraction with facebook/wav2vec2-xlsr-53-espeak-cv-ft
# Reads merged TSVs with columns: audio_path, label_id
# Writes per-split TSVs with: audio_path, label_id, phoneme

import os
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

# ---------- config ----------
SPLITS = {
    #"train": str(REPO_ROOT / "data_preparation/merged_datasets/merged_train.tsv"),
    #"validation": str(REPO_ROOT / "data_preparation/merged_datasets/merged_valid.tsv"),
    #"test": str(REPO_ROOT / "data_preparation/merged_datasets/merged_test.tsv"),
    "vorarlberg": "/home/ai/AI-DataPool/Datasets/audio/Vorarlberg/vorarlberger_daten_16000.tsv",
}
OUT_DIR = str(REPO_ROOT / "data_preparation/feature_extraction/Vorarlberg")
MODEL_ID = "facebook/wav2vec2-xlsr-53-espeak-cv-ft"

BATCH_SIZE = 4  # Reduced for testing - increase back to 32 once working
SAMPLE_RATE = 16000
MISSING_PHONEME = "NO_PHONEME"

os.makedirs(OUT_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_gpus = torch.cuda.device_count()
print(f"Detected GPUs: {num_gpus}")

# ---------- model / processor ----------
processor = Wav2Vec2Processor.from_pretrained(MODEL_ID)
base_model = Wav2Vec2ForCTC.from_pretrained(MODEL_ID)

# Use DataParallel if multiple GPUs
if num_gpus > 1:
    print("⚡ Using torch.nn.DataParallel for multi-GPU phoneme extraction")
    model = torch.nn.DataParallel(base_model).to(device)
else:
    model = base_model.to(device)

model.eval()
print("✅ Model loaded and set to eval mode")

# ---------- helpers ----------
def load_split_df(tsv_path: str) -> pd.DataFrame:
    df = pd.read_csv(tsv_path, sep="\t", dtype=str, low_memory=False)

    if "audio_path" not in df.columns:
        if "path" in df.columns:
            df = df.rename(columns={"path": "audio_path"})
        else:
            raise ValueError(f"{tsv_path} must contain 'audio_path' or 'path' column.")

    if "label_id" not in df.columns:
        raise ValueError(f"{tsv_path} must contain 'label_id' column.")

    # keep only existing files
    df = df[df["audio_path"].apply(lambda p: isinstance(p, str) and os.path.isfile(p))].reset_index(drop=True)

    # cast label_id
    df["label_id"] = pd.to_numeric(df["label_id"], errors="coerce")
    df = df.dropna(subset=["label_id"]).reset_index(drop=True)
    df["label_id"] = df["label_id"].astype(int)

    return df[["audio_path", "label_id"]]

@torch.no_grad()
def transcribe_arrays(arrays):
    """
    arrays: list of 1D numpy arrays (audio)
    returns: list of phoneme strings (using CTC decode)
    """
    inputs = processor(
        arrays,
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt",
        padding=True
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # forward pass (DataParallel will split across GPUs on batch dim)
    logits = model(**inputs).logits  # [B, T, vocab]
    pred_ids = torch.argmax(logits, dim=-1)  # [B, T]
    texts = processor.batch_decode(pred_ids)

    out = []
    for t in texts:
        t = (t or "").strip()
        out.append(t if t else MISSING_PHONEME)
    return out

# ---------- main ----------
print("🚀 Starting phoneme extraction...")
for split, tsv in SPLITS.items():
    print(f"📁 Processing split: {split}")
    print(f"📄 Reading TSV: {tsv}")
    
    # Check if file exists
    if not os.path.exists(tsv):
        print(f"❌ [ERROR] File not found: {tsv}")
        print(f"⏭️  Skipping {split} split...")
        continue
    
    print(f"✅ File exists, loading data...")
    df = load_split_df(tsv)
    n = len(df)
    print(f"📊 Loaded {n} valid rows from {split}")
    
    if n == 0:
        print(f"[WARN] {split}: no valid rows after checks. Skipping.")
        continue

    print(f"\n[INFO] {split}: transcribing {n} files → phoneme sequences "
          f"(batch_size={BATCH_SIZE})")

    phonemes = []
    # Enhanced progress bar with more details
    pbar = tqdm(range(0, n, BATCH_SIZE), 
                desc=f"Processing {split}", 
                unit="batch",
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} batches [{elapsed}<{remaining}, {rate_fmt}]')
    
    for i in pbar:
        batch_paths = df["audio_path"].iloc[i:i+BATCH_SIZE].tolist()
        current_batch_size = len(batch_paths)
        
        # Update progress bar description with current batch info
        pbar.set_postfix({
            'files': f"{i+current_batch_size}/{n}",
            'batch_size': current_batch_size
        })

        # load audio to arrays
        arrays = []
        for p in batch_paths:
            try:
                # lazy import librosa to keep deps minimal unless needed
                import librosa
                y, sr = librosa.load(p, sr=SAMPLE_RATE, mono=True)
                if isinstance(y, np.ndarray) and y.size > 0:
                    arrays.append(y)
                else:
                    arrays.append(np.zeros(1, dtype=np.float32))
            except Exception:
                arrays.append(np.zeros(1, dtype=np.float32))

        # try; if OOM, split once and retry in halves
        try:
            batch_phon = transcribe_arrays(arrays)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            half = len(arrays) // 2 or 1
            batch_phon = []
            for sub in (arrays[:half], arrays[half:]):
                if not sub:
                    continue
                try:
                    batch_phon.extend(transcribe_arrays(sub))
                except Exception:
                    batch_phon.extend([MISSING_PHONEME] * len(sub))
        except Exception:
            # fall back: per-file
            batch_phon = []
            for a in arrays:
                try:
                    batch_phon.extend(transcribe_arrays([a]))
                except Exception:
                    batch_phon.append(MISSING_PHONEME)

        phonemes.extend(batch_phon)
        
        # Update progress bar with completion info
        pbar.set_postfix({
            'files': f"{i+current_batch_size}/{n}",
            'batch_size': current_batch_size,
            'status': 'completed'
        })

    # trim to length and save
    phonemes = phonemes[:n]
    out_df = df.copy()
    out_df["phoneme"] = phonemes

    out_tsv = os.path.join(OUT_DIR, f"{split}_phonemes.tsv")
    out_df.to_csv(out_tsv, sep="\t", index=False)
    print(f"[OK] {split}: saved {len(out_df)} rows → {out_tsv}")
