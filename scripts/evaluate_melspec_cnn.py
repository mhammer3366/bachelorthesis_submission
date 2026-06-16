#!/usr/bin/env python3
"""
Evaluate the trained MelSpec CNN model and generate metrics JSON files.
This script loads the saved model checkpoint and evaluates on train/val/test splits.
"""

import os
from pathlib import Path
REPO_ROOT = Path(os.environ.get("THESIS_ROOT", Path(__file__).resolve().parents[1]))

import json
import os
import re
from glob import glob
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
import os

# Add tqdm for progress bars
try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, **kwargs: x  # Dummy fallback if tqdm not installed

# Import model architecture from training script
import sys
sys.path.insert(0, str(REPO_ROOT / "data_preparation/train"))
# Direct import - we'll copy the necessary classes/functions here
import torch.nn as nn
import torch.nn.functional as F
from glob import glob
import re

# Copy ConvBlock and MelSpecCNN classes
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, p=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=k, padding=p)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.pool(x)
        return x

class MelSpecCNN(nn.Module):
    def __init__(self, n_classes: int = 7):
        super().__init__()
        self.backbone = nn.Sequential(
            ConvBlock(1, 32),   # [1,128,400] -> [32,64,200]
            ConvBlock(32, 64),  # -> [64,32,100]
            ConvBlock(64, 128), # -> [128,16,50]
            ConvBlock(128, 256) # -> [256,8,25]
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, n_classes)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        return x

# Copy dataset and evaluation functions
def find_chunks(feat_dir: str, split: str):
    """Return list of (mel_path, lbl_path, chunk_id) sorted by chunk id."""
    mel_files = sorted(glob(os.path.join(feat_dir, f"{split}_melspec_chunk*.npy")))
    out = []
    for m in mel_files:
        mname = os.path.basename(m)
        m = os.path.abspath(m)
        m_ = re.search(r"chunk(\d+)\.npy$", mname)
        if not m_:
            continue
        cid = int(m_.group(1))
        l = os.path.join(feat_dir, f"{split}_labels_chunk{cid}.npy")
        if os.path.isfile(l):
            out.append((m, os.path.abspath(l), cid))
    out.sort(key=lambda x: x[2])
    if not out:
        raise FileNotFoundError(f"No chunk pairs found for split '{split}' under {feat_dir}")
    return out

class ChunkedMelDataset(Dataset):
    def __init__(self, feat_dir: str, split: str, mean=None, std=None, train_mode: bool = False):
        super().__init__()
        self.pairs = find_chunks(feat_dir, split)
        self.mels = []
        self.lbls = []
        self.lengths = []
        for m, l, _ in self.pairs:
            mel = np.load(m, mmap_mode='r')
            lab = np.load(l, mmap_mode='r')
            if mel.shape[0] != lab.shape[0]:
                raise ValueError(f"Length mismatch: {m} vs {l}")
            self.mels.append(mel)
            self.lbls.append(lab)
            self.lengths.append(mel.shape[0])
        self.cum = np.cumsum([0] + self.lengths)
        self.total = self.cum[-1]
        self.mean = mean
        self.std = std
        self.train_mode = train_mode

    def __len__(self):
        return self.total

    def _locate(self, idx: int):
        c = np.searchsorted(self.cum, idx, side='right') - 1
        off = idx - self.cum[c]
        return c, off

    def __getitem__(self, idx: int):
        c, off = self._locate(idx)
        mel = self.mels[c][off]
        x = torch.from_numpy(mel.copy()).unsqueeze(0)
        y = int(self.lbls[c][off])
        if self.mean is not None and self.std is not None:
            mean = torch.from_numpy(self.mean).view(1, -1, 1)
            std = torch.from_numpy(self.std).view(1, -1, 1)
            x = (x - mean) / (std + 1e-6)
        return x, y

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_y, all_p = [], []
    # Add tqdm for loader
    for xb, yb in tqdm(loader, desc="Evaluating", leave=False):
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        logits = model(xb)
        probs = F.softmax(logits, dim=1)
        all_p.append(probs.detach().cpu().numpy())
        all_y.append(yb.detach().cpu().numpy())
    y_true = np.concatenate(all_y)
    y_prob = np.concatenate(all_p)
    y_pred = y_prob.argmax(axis=1)
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    cm = confusion_matrix(y_true, y_pred)
    return {
        'acc': float(acc),
        'macro_f1': float(macro_f1),
        'cm': cm.tolist(),
    }

# Configuration
feat_dir = str(REPO_ROOT / "data_preparation/feature_extraction/ohne_deutsch/saved_features_melspec")
save_dir = str(REPO_ROOT / "models/1_melspec_cnn")
batch_size = 64
num_workers = 2
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"Device: {device}")
print(f"Loading model from: {save_dir}")

# Load normalization stats
train_mean = np.load(os.path.join(save_dir, 'train_mean.npy'))
train_std = np.load(os.path.join(save_dir, 'train_std.npy'))
print(f"Loaded normalization stats: mean shape {train_mean.shape}, std shape {train_std.shape}")

# Load model checkpoint
best_path = os.path.join(save_dir, '1_melspec_cnn_best.pt')
if not os.path.exists(best_path):
    print(f"Error: Model checkpoint not found at {best_path}")
    sys.exit(1)

ckpt = torch.load(best_path, map_location=device, weights_only=False)
print(f"Loaded checkpoint from epoch {ckpt.get('epoch', 'unknown')}")

# Determine number of classes from checkpoint or dataset
# Try to infer from validation metrics if available
n_classes = 7  # Default, will be updated if found in checkpoint
if 'val_metrics' in ckpt and 'cm' in ckpt['val_metrics']:
    n_classes = len(ckpt['val_metrics']['cm'])

# Create model
model = MelSpecCNN(n_classes=n_classes).to(device)
model.load_state_dict(ckpt['model_state'])
model.eval()
print(f"Model loaded with {n_classes} classes")

# Create datasets
print("\nLoading datasets...")
ds_train = ChunkedMelDataset(feat_dir, 'train', mean=train_mean, std=train_std, train_mode=False)
ds_val = ChunkedMelDataset(feat_dir, 'validation', mean=train_mean, std=train_std, train_mode=False)
ds_test = ChunkedMelDataset(feat_dir, 'test', mean=train_mean, std=train_std, train_mode=False)

print(f"Train samples: {len(ds_train)}")
print(f"Validation samples: {len(ds_val)}")
print(f"Test samples: {len(ds_test)}")

# Create data loaders
train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
val_loader = DataLoader(ds_val, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
test_loader = DataLoader(ds_test, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

# Evaluate on all splits
print("\nEvaluating on train set...")
train_metrics = evaluate(model, train_loader, device)

print("Evaluating on validation set...")
val_metrics = evaluate(model, val_loader, device)

print("Evaluating on test set...")
test_metrics = evaluate(model, test_loader, device)

# Save metrics JSON files (using format expected by comparison script)
print("\nSaving metrics files...")

# Save train metrics
with open(os.path.join(save_dir, 'train_metrics.json'), 'w') as f:
    json.dump({
        'accuracy': train_metrics['acc'],
        'acc': train_metrics['acc'],
        'macro_f1': train_metrics['macro_f1'],
        'macro_f1_score': train_metrics['macro_f1'],
        'confusion_matrix': train_metrics['cm'],
        'cm': train_metrics['cm']
    }, f, indent=2)
print(f"  ✓ Saved train_metrics.json")

# Save validation metrics (save as both val and valid for compatibility)
with open(os.path.join(save_dir, 'val_metrics.json'), 'w') as f:
    json.dump({
        'accuracy': val_metrics['acc'],
        'acc': val_metrics['acc'],
        'macro_f1': val_metrics['macro_f1'],
        'macro_f1_score': val_metrics['macro_f1'],
        'confusion_matrix': val_metrics['cm'],
        'cm': val_metrics['cm']
    }, f, indent=2)
print(f"  ✓ Saved val_metrics.json")

with open(os.path.join(save_dir, 'valid_metrics.json'), 'w') as f:
    json.dump({
        'accuracy': val_metrics['acc'],
        'acc': val_metrics['acc'],
        'macro_f1': val_metrics['macro_f1'],
        'macro_f1_score': val_metrics['macro_f1'],
        'confusion_matrix': val_metrics['cm'],
        'cm': val_metrics['cm']
    }, f, indent=2)
print(f"  ✓ Saved valid_metrics.json")

# Save test metrics
with open(os.path.join(save_dir, 'test_metrics.json'), 'w') as f:
    json.dump({
        'accuracy': test_metrics['acc'],
        'acc': test_metrics['acc'],
        'macro_f1': test_metrics['macro_f1'],
        'macro_f1_score': test_metrics['macro_f1'],
        'confusion_matrix': test_metrics['cm'],
        'cm': test_metrics['cm'],
        'confusion_matrix_labels': [f"Class {i}" for i in range(n_classes)]
    }, f, indent=2)
print(f"  ✓ Saved test_metrics.json")

# Print summary
print("\n" + "="*80)
print("EVALUATION RESULTS")
print("="*80)
print(f"Train   - Accuracy: {train_metrics['acc']:.4f} | Macro-F1: {train_metrics['macro_f1']:.4f}")
print(f"Valid   - Accuracy: {val_metrics['acc']:.4f} | Macro-F1: {val_metrics['macro_f1']:.4f}")
print(f"Test    - Accuracy: {test_metrics['acc']:.4f} | Macro-F1: {test_metrics['macro_f1']:.4f}")
print("="*80)
print(f"\nMetrics files saved to: {save_dir}")
print("\nYou can now run comparison_prints_of_all_models.py to see the results!")

