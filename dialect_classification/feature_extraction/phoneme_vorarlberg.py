
import os
from pathlib import Path
REPO_ROOT = Path(os.environ.get("THESIS_ROOT", Path(__file__).resolve().parents[2]))
# phoneme_vorarlberg.py
# Multi-GPU (DataParallel) phoneme extraction for Vorarlberg dataset
# Reads vorarlberger_daten_16000.tsv with columns: path, text
# Writes new TSV with: path, text, phoneme

import os
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

# ---------- config ----------
INPUT_TSV = "/home/ai/AI-DataPool/Datasets/audio/Vorarlberg/vorarlberger_daten_16000.tsv"
OUTPUT_TSV = str(REPO_ROOT / "data_preparation/feature_extraction/vorarlberg/vorarlberg_with_phonemes.tsv")
MODEL_ID = "facebook/wav2vec2-xlsr-53-espeak-cv-ft"

BATCH_SIZE = 32  # Start with smaller batch size for testing
SAMPLE_RATE = 16000
MISSING_PHONEME = "NO_PHONEME"
SKIP_FILE_CHECK = True  # Set to False if you want to verify all files exist

# Create output directory
os.makedirs(os.path.dirname(OUTPUT_TSV), exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_gpus = torch.cuda.device_count()
print(f"Detected GPUs: {num_gpus}")

# ---------- model / processor ----------
print("🔄 Loading model and processor...")
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
def load_vorarlberg_df(tsv_path: str) -> pd.DataFrame:
    """Load and validate Vorarlberg TSV file"""
    print(f"📄 Reading TSV: {tsv_path}")
    df = pd.read_csv(tsv_path, sep="\t", dtype=str, low_memory=False)
    
    # Check required columns
    if "path" not in df.columns:
        raise ValueError(f"{tsv_path} must contain 'path' column.")
    if "text" not in df.columns:
        raise ValueError(f"{tsv_path} must contain 'text' column.")
    
    print(f"📊 Loaded {len(df)} total rows from TSV")
    
    if SKIP_FILE_CHECK:
        print("⚡ Skipping file existence check (SKIP_FILE_CHECK=True)")
        print(f"✅ Assuming all {len(df)} files exist")
    else:
        # Keep only existing audio files with progress bar
        print("🔍 Filtering existing audio files...")
        
        # Use tqdm for progress tracking
        valid_paths = []
        for path in tqdm(df["path"], desc="Checking files", unit="files"):
            if isinstance(path, str) and os.path.isfile(path):
                valid_paths.append(True)
            else:
                valid_paths.append(False)
        
        df = df[valid_paths].reset_index(drop=True)
        print(f"✅ Found {len(df)} valid audio files")
    return df[["path", "text"]]

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
print("🚀 Starting Vorarlberg phoneme extraction...")

# Load data
df = load_vorarlberg_df(INPUT_TSV)
n = len(df)

if n == 0:
    print("❌ No valid audio files found. Exiting.")
    exit(1)

print(f"\n[INFO] Transcribing {n} Vorarlberg audio files → phoneme sequences "
      f"(batch_size={BATCH_SIZE})")

phonemes = []
# Enhanced progress bar with more details
pbar = tqdm(range(0, n, BATCH_SIZE), 
            desc="Processing Vorarlberg", 
            unit="batch",
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} batches [{elapsed}<{remaining}, {rate_fmt}]')

for i in pbar:
    batch_paths = df["path"].iloc[i:i+BATCH_SIZE].tolist()
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
        except Exception as e:
            print(f"⚠️  Error loading {p}: {e}")
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
    except Exception as e:
        print(f"⚠️  Error in batch transcription: {e}")
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

# Save results
out_df.to_csv(OUTPUT_TSV, sep="\t", index=False)
print(f"\n✅ [SUCCESS] Saved {len(out_df)} rows with phonemes → {OUTPUT_TSV}")

# Show sample results
print("\n📋 Sample results:")
print(out_df.head(3).to_string(index=False))