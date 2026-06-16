#!/usr/bin/env python3
"""
Mel-Spectrogram CNN for 7-way Swiss German dialect classification.

Reads chunked .npy mel features produced by mel_spectogram.py:
  {split}_melspec_chunk{N}.npy  -> shape: [chunk_size, 128, 400]
  {split}_labels_chunk{N}.npy   -> shape: [chunk_size]

Trains a CNN with SpecAugment, class weighting, early stopping, and evaluates
Accuracy and Macro-F1 on validation/test. Saves best checkpoint + reports.

Hardcoded paths version for consistent model saving.
"""

import os
from pathlib import Path
REPO_ROOT = Path(os.environ.get("THESIS_ROOT", Path(__file__).resolve().parents[2]))
import argparse
import json
import math
import os
import re
import random
from glob import glob
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt

# -------------------- utils --------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def find_chunks(feat_dir: str, split: str) -> List[Tuple[str, str, int]]:
    """Return list of (mel_path, lbl_path, chunk_id) sorted by chunk id."""
    mel_files = sorted(glob(os.path.join(feat_dir, f"{split}_melspec_chunk*.npy")))
    out = []
    for m in mel_files:
        mname = os.path.basename(m)
        m = os.path.abspath(m)
        # extract chunk id
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
    """Map-style dataset over multiple memory-mapped chunks for a split.

    Optionally applies normalization and SpecAugment during __getitem__ (for train).
    """
    def __init__(self, feat_dir: str, split: str, mean=None, std=None, train_mode: bool = False,
                 time_mask_max: int = 32, freq_mask_max: int = 16, n_time_masks: int = 2, n_freq_masks: int = 2):
        super().__init__()
        self.pairs = find_chunks(feat_dir, split)
        self.mels = []  # list of memmaps
        self.lbls = []
        self.lengths = []
        for m, l, _ in self.pairs:
            mel = np.load(m, mmap_mode='r')  # [N, 128, 400]
            lab = np.load(l, mmap_mode='r')  # [N]
            if mel.shape[0] != lab.shape[0]:
                raise ValueError(f"Length mismatch: {m} vs {l}")
            self.mels.append(mel)
            self.lbls.append(lab)
            self.lengths.append(mel.shape[0])
        self.cum = np.cumsum([0] + self.lengths)  # prefix sums
        self.total = self.cum[-1]
        self.mean = mean  # shape [128] or None
        self.std = std    # shape [128] or None
        self.train_mode = train_mode
        # SpecAugment params
        self.time_mask_max = time_mask_max
        self.freq_mask_max = freq_mask_max
        self.n_time_masks = n_time_masks
        self.n_freq_masks = n_freq_masks

    def __len__(self):
        return self.total

    def _locate(self, idx: int) -> Tuple[int, int]:
        # find chunk index c s.t. cum[c] <= idx < cum[c+1]
        c = np.searchsorted(self.cum, idx, side='right') - 1
        off = idx - self.cum[c]
        return c, off

    @staticmethod
    def _spec_augment(x: torch.Tensor, n_time_masks: int, n_freq_masks: int, t_max: int, f_max: int) -> torch.Tensor:
        # x: [1, 128, 400]
        _, Fm, Tm = x.shape
        for _ in range(n_freq_masks):
            if f_max > 0:
                f = random.randint(0, f_max)
                f0 = random.randint(0, max(0, Fm - f))
                x[:, f0:f0+f, :] = 0
        for _ in range(n_time_masks):
            if t_max > 0:
                t = random.randint(0, t_max)
                t0 = random.randint(0, max(0, Tm - t))
                x[:, :, t0:t0+t] = 0
        return x

    def __getitem__(self, idx: int):
        c, off = self._locate(idx)
        mel = self.mels[c][off]  # np memmap [128, 400], read-only

        # Make it writable & contiguous before turning into a tensor:
        x = torch.from_numpy(mel.copy()).unsqueeze(0)  # [1, 128, 400]

        y = int(self.lbls[c][off])

        # per-frequency normalize if stats provided
        if self.mean is not None and self.std is not None:
            mean = torch.from_numpy(self.mean).view(1, -1, 1)
            std = torch.from_numpy(self.std).view(1, -1, 1)
            x = (x - mean) / (std + 1e-6)

        # SpecAugment only in train
        if self.train_mode:
            x = self._spec_augment(x, self.n_time_masks, self.n_freq_masks,
                                self.time_mask_max, self.freq_mask_max)
        return x, y



def compute_per_mel_stats(feat_dir: str, split: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute per-mel (128,) mean and std, and class counts for weighting."""
    pairs = find_chunks(feat_dir, split)
    sum_vec = np.zeros(128, dtype=np.float64)
    sumsq_vec = np.zeros(128, dtype=np.float64)
    count = 0
    # class counts
    class_counts = {}
    for m, l, _ in pairs:
        mel = np.load(m, mmap_mode='r')  # [N, 128, 400]
        lab = np.load(l, mmap_mode='r')
        # update class counts
        uniq, cnts = np.unique(lab, return_counts=True)
        for u, c in zip(uniq, cnts):
            class_counts[int(u)] = class_counts.get(int(u), 0) + int(c)
        # sum over time, then batch
        # reshape to [N, 128, 400] → sum over axis=(0,2) → [128]
        sum_vec += mel.sum(axis=(0, 2))
        sumsq_vec += (mel ** 2).sum(axis=(0, 2))
        count += mel.shape[0] * mel.shape[2]  # total frames across all examples
    mean = (sum_vec / count).astype(np.float32)
    var = (sumsq_vec / count) - (mean.astype(np.float64) ** 2)
    std = np.sqrt(np.maximum(var, 1e-8)).astype(np.float32)
    # convert class_counts to sorted vector
    max_label = max(class_counts.keys())
    counts_vec = np.zeros(max_label + 1, dtype=np.int64)
    for k, v in class_counts.items():
        counts_vec[k] = v
    return mean, std, counts_vec


# -------------------- model --------------------
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


# -------------------- training & eval --------------------
@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_y, all_p = [], []
    for xb, yb in loader:
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
    per_class_f1 = f1_score(y_true, y_pred, average=None)
    cm = confusion_matrix(y_true, y_pred)
    return {
        'acc': float(acc),
        'macro_f1': float(macro_f1),
        'per_class_f1': per_class_f1.tolist(),
        'cm': cm.tolist(),
        'y_true': y_true.tolist(),
        'y_pred': y_pred.tolist(),
        'y_prob': y_prob.tolist(),  # Add probabilities for CSV export
    }


def plot_confusion(cm: np.ndarray, save_path: str, title: str = "Confusion Matrix"):
    plt.figure(figsize=(6.5, 6))
    plt.imshow(cm, interpolation='nearest')
    plt.title(title)
    plt.colorbar()
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


# -------------------- main --------------------
def main():
    # Hardcoded configuration
    feat_dir = str(REPO_ROOT / "data_preparation/feature_extraction/ohne_deutsch/saved_features_melspec")
    save_dir = str(REPO_ROOT / "models/1_melspec_cnn")
    epochs = 20
    batch_size = 64  # Reduced from 128
    lr = 1e-3
    weight_decay = 1e-4
    seed = 42
    num_workers = 2  # Reduced from 4
    patience = 8

    os.makedirs(save_dir, exist_ok=True)
    set_seed(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # compute normalization + class counts on TRAIN
    print("Computing train stats (per-mel mean/std and class counts)…")
    train_mean, train_std, class_counts = compute_per_mel_stats(feat_dir, 'train')
    n_classes = int(class_counts.shape[0])
    print(f"Class counts: {class_counts.tolist()}")

    # class weights (inverse freq)
    weights = class_counts.astype(np.float64)
    weights = weights / weights.sum()
    weights = 1.0 / np.maximum(weights, 1e-6)
    weights = weights / weights.mean()
    class_weights = torch.tensor(weights, dtype=torch.float32, device=device)

    # save normalization for reuse
    np.save(os.path.join(save_dir, 'train_mean.npy'), train_mean)
    np.save(os.path.join(save_dir, 'train_std.npy'), train_std)

    # datasets (disable data augmentation for faster training)
    ds_train = ChunkedMelDataset(feat_dir, 'train', mean=train_mean, std=train_std, train_mode=False)
    ds_val = ChunkedMelDataset(feat_dir, 'validation', mean=train_mean, std=train_std, train_mode=False)
    ds_test = ChunkedMelDataset(feat_dir, 'test', mean=train_mean, std=train_std, train_mode=False)

    # loaders
    train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                              pin_memory=True, drop_last=True)
    val_loader = DataLoader(ds_val, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                            pin_memory=True)
    test_loader = DataLoader(ds_test, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                             pin_memory=True)

    # model
    model = MelSpecCNN(n_classes=n_classes).to(device)

    # optim
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    scaler = torch.amp.GradScaler('cuda', enabled=torch.cuda.is_available())

    best_val = -1.0
    best_path = os.path.join(save_dir, '1_melspec_cnn_best.pt')
    history = []
    patience_left = patience

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
                logits = model(xb)
                loss = criterion(logits, yb)
            scaler.scale(loss).backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item() * xb.size(0)
        scheduler.step()

        # eval on val
        val_metrics = evaluate(model, val_loader, device)
        train_loss = running_loss / (len(train_loader.dataset) if hasattr(train_loader.dataset, '__len__') else (len(train_loader) * batch_size))
        print(f"Epoch {epoch:02d} | train_loss={train_loss:.4f} | val_acc={val_metrics['acc']:.4f} | val_macroF1={val_metrics['macro_f1']:.4f}")

        history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_acc': val_metrics['acc'],
            'val_macro_f1': val_metrics['macro_f1']
        })

        # early stopping on macro-F1
        if val_metrics['macro_f1'] > best_val:
            best_val = val_metrics['macro_f1']
            torch.save({'epoch': epoch,
                        'model_state': model.state_dict(),
                        'optimizer_state': optimizer.state_dict(),
                        'val_metrics': val_metrics,
                        'mean': train_mean,
                        'std': train_std},
                       best_path)
            patience_left = patience
            print(f"  ✓ Saved new best to {best_path}")
        else:
            patience_left -= 1
            if patience_left <= 0:
                print("Early stopping triggered.")
                break

    # load best
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt['model_state'])

    # final evals on all splits
    train_metrics = evaluate(model, train_loader, device)
    val_metrics = evaluate(model, val_loader, device)
    test_metrics = evaluate(model, test_loader, device)

    # save metrics JSON files
    with open(os.path.join(save_dir, 'train_metrics.json'), 'w') as f:
        json.dump({'acc': train_metrics['acc'], 'macro_f1': train_metrics['macro_f1'], 'cm': train_metrics['cm']}, f, indent=2)
    with open(os.path.join(save_dir, 'val_metrics.json'), 'w') as f:
        json.dump({'acc': val_metrics['acc'], 'macro_f1': val_metrics['macro_f1'], 'cm': val_metrics['cm']}, f, indent=2)
    with open(os.path.join(save_dir, 'test_metrics.json'), 'w') as f:
        json.dump({'acc': test_metrics['acc'], 'macro_f1': test_metrics['macro_f1'], 'cm': test_metrics['cm']}, f, indent=2)

    # save prediction CSV files
    def save_preds_csv(metrics, split_name):
        df = pd.DataFrame({
            'label_id': metrics['y_true'],
            'pred': metrics['y_pred'],
            'probs': [p.tolist() for p in metrics['y_prob']]
        })
        df.to_csv(os.path.join(save_dir, f'{split_name}_preds.csv'), index=False)
    
    save_preds_csv(train_metrics, 'train')
    save_preds_csv(val_metrics, 'valid')
    save_preds_csv(test_metrics, 'test')

    # confusion matrices
    plot_confusion(np.array(val_metrics['cm']), os.path.join(save_dir, 'val_confusion.png'), 'Validation Confusion')
    plot_confusion(np.array(test_metrics['cm']), os.path.join(save_dir, 'test_confusion.png'), 'Test Confusion')

    print("\nBest VAL macro-F1:", best_val)
    print("Train Accuracy:", train_metrics['acc'], "| Train Macro-F1:", train_metrics['macro_f1'])
    print("Validation Accuracy:", val_metrics['acc'], "| Validation Macro-F1:", val_metrics['macro_f1'])
    print("Test Accuracy:", test_metrics['acc'], "| Test Macro-F1:", test_metrics['macro_f1'])
    print(f"Artifacts written to {save_dir}")


if __name__ == '__main__':
    main()
