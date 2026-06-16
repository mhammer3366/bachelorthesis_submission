
import os
from pathlib import Path
REPO_ROOT = Path(os.environ.get("THESIS_ROOT", Path(__file__).resolve().parents[2]))
# phoneme_vorarlberg_NO_PHONEME_rerun.py
# Rerun phoneme extraction specifically for files marked as NO_PHONEME
# Improved error handling and debugging for better results

import os
import numpy as np
import pandas as pd
import torch
import librosa
import soundfile as sf
from tqdm import tqdm
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import warnings
warnings.filterwarnings("ignore")

# ---------- config ----------
INPUT_TSV = str(REPO_ROOT / "data_preparation/feature_extraction/vorarlberg/vorarlberg_with_phonemes.tsv")
OUTPUT_TSV = str(REPO_ROOT / "data_preparation/feature_extraction/vorarlberg/vorarlberg_with_phonemes_fixed.tsv")
MODEL_ID = "facebook/wav2vec2-xlsr-53-espeak-cv-ft"

BATCH_SIZE = 8  # Even smaller batch size for problematic files
SAMPLE_RATE = 16000
MISSING_PHONEME = "NO_PHONEME"
MIN_AUDIO_LENGTH = 0.05  # Minimum audio length in seconds
MAX_AUDIO_LENGTH = 180.0  # Maximum audio length in seconds

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
def load_audio_robust(file_path: str, target_sr: int = 16000) -> np.ndarray:
    """
    Robust audio loading with multiple fallback methods
    """
    try:
        # Method 1: Try librosa first
        y, sr = librosa.load(file_path, sr=target_sr, mono=True)
        if isinstance(y, np.ndarray) and y.size > 0:
            return y
    except Exception as e1:
        print(f"⚠️  Librosa failed for {file_path}: {e1}")
        
        try:
            # Method 2: Try soundfile
            y, sr = sf.read(file_path)
            if sr != target_sr:
                y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
            if len(y.shape) > 1:
                y = librosa.to_mono(y)
            if isinstance(y, np.ndarray) and y.size > 0:
                return y
        except Exception as e2:
            print(f"⚠️  Soundfile failed for {file_path}: {e2}")
            
            try:
                # Method 3: Try with different librosa parameters
                y, sr = librosa.load(file_path, sr=target_sr, mono=True, 
                                   offset=0.0, duration=None, dtype=np.float32)
                if isinstance(y, np.ndarray) and y.size > 0:
                    return y
            except Exception as e3:
                print(f"⚠️  All audio loading methods failed for {file_path}: {e3}")
    
    # Return silence if all methods fail
    return np.zeros(int(target_sr * 0.1), dtype=np.float32)

def validate_audio(audio_array: np.ndarray) -> tuple[bool, str]:
    """
    Validate audio array and return (is_valid, reason)
    """
    if not isinstance(audio_array, np.ndarray):
        return False, "Not a numpy array"
    
    if audio_array.size == 0:
        return False, "Empty array"
    
    duration = len(audio_array) / SAMPLE_RATE
    
    if duration < MIN_AUDIO_LENGTH:
        return False, f"Too short: {duration:.3f}s < {MIN_AUDIO_LENGTH}s"
    
    if duration > MAX_AUDIO_LENGTH:
        return False, f"Too long: {duration:.3f}s > {MAX_AUDIO_LENGTH}s"
    
    # Check for silence
    if np.max(np.abs(audio_array)) < 1e-6:
        return False, "Silent audio"
    
    # Check for NaN or Inf values
    if not np.all(np.isfinite(audio_array)):
        return False, "Contains NaN or Inf values"
    
    return True, "Valid"

def load_no_phoneme_df(tsv_path: str) -> pd.DataFrame:
    """Load and filter for NO_PHONEME entries"""
    print(f"📄 Reading TSV: {tsv_path}")
    df = pd.read_csv(tsv_path, sep="\t", dtype=str, low_memory=False)
    
    # Filter for NO_PHONEME entries
    no_phoneme_mask = df["phoneme"] == MISSING_PHONEME
    no_phoneme_df = df[no_phoneme_mask].copy().reset_index(drop=True)
    
    print(f"📊 Found {len(no_phoneme_df)} files marked as {MISSING_PHONEME} out of {len(df)} total")
    
    return no_phoneme_df[["path", "text"]]

@torch.no_grad()
def transcribe_single_file(audio_array: np.ndarray, file_path: str) -> str:
    """
    Transcribe a single audio file with robust error handling
    """
    global model
    try:
        # Validate audio first
        is_valid, reason = validate_audio(audio_array)
        if not is_valid:
            print(f"⚠️  Invalid audio {file_path}: {reason}")
            return MISSING_PHONEME
        
        # Process single file
        inputs = processor(
            [audio_array],
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
            padding=True
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Forward pass
        logits = model(**inputs).logits
        pred_ids = torch.argmax(logits, dim=-1)
        texts = processor.batch_decode(pred_ids)
        
        # Extract result
        text = texts[0] if texts else ""
        text = (text or "").strip()
        result = text if text else MISSING_PHONEME
        
        if result == MISSING_PHONEME:
            print(f"⚠️  Empty transcription for {file_path}")
        
        return result
        
    except torch.cuda.OutOfMemoryError:
        print(f"⚠️  OOM for {file_path}, using CPU fallback")
        torch.cuda.empty_cache()
        
        try:
            # Move to CPU and try again
            inputs = processor(
                [audio_array],
                sampling_rate=SAMPLE_RATE,
                return_tensors="pt",
                padding=True
            )
            inputs = {k: v.cpu() for k, v in inputs.items()}
            
            # Use CPU model
            cpu_model = base_model.cpu()
            logits = cpu_model(**inputs).logits
            pred_ids = torch.argmax(logits, dim=-1)
            texts = processor.batch_decode(pred_ids)
            
            text = texts[0] if texts else ""
            text = (text or "").strip()
            result = text if text else MISSING_PHONEME
            
            # Move model back to GPU
            if num_gpus > 1:
                model = torch.nn.DataParallel(base_model).to(device)
            else:
                model = base_model.to(device)
            
            return result
                
        except Exception as e:
            print(f"⚠️  CPU fallback failed for {file_path}: {e}")
            return MISSING_PHONEME
            
    except Exception as e:
        print(f"⚠️  Transcription failed for {file_path}: {e}")
        return MISSING_PHONEME

# ---------- main ----------
print("🚀 Starting NO_PHONEME rerun with improved error handling...")

# Load data
df = load_no_phoneme_df(INPUT_TSV)
n = len(df)

if n == 0:
    print("❌ No NO_PHONEME files found. Exiting.")
    exit(1)

print(f"\n[INFO] Reprocessing {n} NO_PHONEME files with improved methods "
      f"(batch_size={BATCH_SIZE})")

phonemes = []
success_count = 0
error_count = 0

# Process files one by one for maximum robustness
pbar = tqdm(df.iterrows(), total=len(df), desc="Reprocessing NO_PHONEME", unit="files")

for idx, row in pbar:
    file_path = row["path"]
    
    # Update progress bar
    pbar.set_postfix({
        'file': f"{idx+1}/{n}",
        'success': success_count,
        'errors': error_count
    })
    
    try:
        # Load audio
        audio = load_audio_robust(file_path, SAMPLE_RATE)
        
        # Transcribe
        phoneme = transcribe_single_file(audio, file_path)
        phonemes.append(phoneme)
        
        # Count results
        if phoneme != MISSING_PHONEME:
            success_count += 1
        else:
            error_count += 1
            
    except Exception as e:
        print(f"⚠️  Failed to process {file_path}: {e}")
        phonemes.append(MISSING_PHONEME)
        error_count += 1

# Create output
out_df = df.copy()
out_df["phoneme"] = phonemes

# Save results
out_df.to_csv(OUTPUT_TSV, sep="\t", index=False)
print(f"\n✅ [SUCCESS] Saved {len(out_df)} rows with improved phonemes → {OUTPUT_TSV}")

# Show statistics
success_rate = (success_count / n) * 100 if n > 0 else 0
print(f"\n📊 Statistics:")
print(f"   Total files processed: {n}")
print(f"   Successful transcriptions: {success_count}")
print(f"   Failed transcriptions: {error_count}")
print(f"   Success rate: {success_rate:.1f}%")

# Show sample results
print("\n📋 Sample results:")
sample_df = out_df[out_df["phoneme"] != MISSING_PHONEME].head(5)
if len(sample_df) > 0:
    print(sample_df.to_string(index=False))
else:
    print("No successful transcriptions to show")

print("\n🔍 Files still with NO_PHONEME:")
no_phoneme_count = (out_df["phoneme"] == MISSING_PHONEME).sum()
print(f"   {no_phoneme_count} files still marked as NO_PHONEME")