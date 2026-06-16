
import os
from pathlib import Path
REPO_ROOT = Path(os.environ.get("THESIS_ROOT", Path(__file__).resolve().parents[2]))
# test_phoneme_rerun.py
# Quick test of the phoneme rerun functionality

import os
import numpy as np
import pandas as pd
import torch
import librosa
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import warnings
warnings.filterwarnings("ignore")

# ---------- config ----------
MODEL_ID = "facebook/wav2vec2-xlsr-53-espeak-cv-ft"
SAMPLE_RATE = 16000
MISSING_PHONEME = "NO_PHONEME"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ---------- model / processor ----------
print("🔄 Loading model and processor...")
processor = Wav2Vec2Processor.from_pretrained(MODEL_ID)
model = Wav2Vec2ForCTC.from_pretrained(MODEL_ID).to(device)
model.eval()
print("✅ Model loaded and set to eval mode")

# ---------- test function ----------
@torch.no_grad()
def test_transcribe(audio_path: str) -> str:
    """Test transcription of a single file"""
    try:
        # Load audio
        y, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
        
        if len(y) == 0:
            return MISSING_PHONEME
        
        # Process
        inputs = processor([y], sampling_rate=SAMPLE_RATE, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Forward pass
        logits = model(**inputs).logits
        pred_ids = torch.argmax(logits, dim=-1)
        texts = processor.batch_decode(pred_ids)
        
        # Extract result
        text = texts[0] if texts else ""
        text = (text or "").strip()
        return text if text else MISSING_PHONEME
        
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return MISSING_PHONEME

# ---------- test ----------
print("🧪 Testing with a few files...")

# Get a few NO_PHONEME files
df = pd.read_csv(str(REPO_ROOT / "data_preparation/feature_extraction/vorarlberg/vorarlberg_with_phonemes.tsv"), 
                 sep="\t", dtype=str, low_memory=False)
no_phoneme_df = df[df["phoneme"] == MISSING_PHONEME].head(3)

print(f"Testing {len(no_phoneme_df)} files...")

for idx, row in no_phoneme_df.iterrows():
    file_path = row["path"]
    text = row["text"]
    
    print(f"\n📁 File: {os.path.basename(file_path)}")
    print(f"📝 Text: {text}")
    
    if os.path.exists(file_path):
        result = test_transcribe(file_path)
        print(f"🎯 Result: {result}")
    else:
        print("❌ File not found")

print("\n✅ Test completed!")



