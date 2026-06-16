#!/usr/bin/env python3
"""
Linear classifier on pre-extracted wav2vec2-base layer 6 features for 7-way Swiss German dialect classification.

- Reads pre-extracted features from wav2vec_base_layer6.py: {split}_features.npy, {split}_labels.npy
- Trains a simple MLP classifier on the frozen features
- Evaluates Accuracy and Macro-F1; early-stops on Macro-F1

Hardcoded paths version for consistent model saving.
"""

import os
from pathlib import Path
REPO_ROOT = Path(os.environ.get("THESIS_ROOT", Path(__file__).resolve().parents[2]))
import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from tqdm import tqdm


# ---------------------- utils ----------------------
def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------- dataset ----------------------
class Wav2Vec2FeatureDataset(Dataset):
    """Dataset for pre-extracted wav2vec2 features."""
    def __init__(self, features_path: str, labels_path: str):
        self.features = np.load(features_path).astype(np.float32)  # [N, 768]
        self.labels = np.load(labels_path).astype(np.int64)        # [N]
        
        if len(self.features) != len(self.labels):
            raise ValueError(f"Feature and label count mismatch: {len(self.features)} vs {len(self.labels)}")
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return torch.from_numpy(self.features[idx]), torch.tensor(self.labels[idx], dtype=torch.long)


# ---------------------- model ----------------------
class Wav2Vec2Classifier(nn.Module):
    """Simple MLP classifier for wav2vec2 features."""
    def __init__(self, input_dim: int = 768, n_classes: int = 7):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, n_classes)
        )
    
    def forward(self, x):
        return self.classifier(x)


# ---------------------- evaluation ----------------------
@torch.no_grad()
def eval_loop(model, loader, device):
    model.eval()
    y_true, y_pred, y_prob = [], [], []
    
    for features, labels in loader:
        features = features.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        logits = model(features)
        probs = F.softmax(logits, dim=1)
        
        y_true.append(labels.detach().cpu().numpy())
        y_pred.append(probs.argmax(dim=1).detach().cpu().numpy())
        y_prob.append(probs.detach().cpu().numpy())
    
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    y_prob = np.concatenate(y_prob)
    
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    cm = confusion_matrix(y_true, y_pred)
    
    return {'acc': float(acc), 'macro_f1': float(macro_f1), 'cm': cm.tolist()}


# ---------------------- training ----------------------
def train():
    # Hardcoded configuration
    feat_dir = str(REPO_ROOT / "data_preparation/feature_extraction/ohne_deutsch/saved_features_wav2vec_base_layer6")
    save_dir = str(REPO_ROOT / "models/3_wav2vec_base_layer6")
    
    batch_size = 128
    epochs = 50
    lr = 1e-3
    weight_decay = 1e-4
    seed = 42
    n_classes = 7
    patience = 10
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    set_seed(seed)
    
    # Load datasets
    print("Loading pre-extracted features...")
    train_ds = Wav2Vec2FeatureDataset(
        os.path.join(feat_dir, "train_features.npy"),
        os.path.join(feat_dir, "train_labels.npy")
    )
    val_ds = Wav2Vec2FeatureDataset(
        os.path.join(feat_dir, "validation_features.npy"),
        os.path.join(feat_dir, "validation_labels.npy")
    )
    test_ds = Wav2Vec2FeatureDataset(
        os.path.join(feat_dir, "test_features.npy"),
        os.path.join(feat_dir, "test_labels.npy")
    )
    
    print(f"Train: {len(train_ds)} samples")
    print(f"Validation: {len(val_ds)} samples")
    print(f"Test: {len(test_ds)} samples")
    
    # Data loaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    # Model
    model = Wav2Vec2Classifier(input_dim=768, n_classes=n_classes).to(device)
    
    if torch.cuda.device_count() > 1:
        print(f'Using DataParallel over {torch.cuda.device_count()} GPUs')
        model = nn.DataParallel(model)
    
    # Training setup
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
    
    # Training loop
    best_val = -1.0
    os.makedirs(save_dir, exist_ok=True)
    best_path = os.path.join(save_dir, '6_wav2vec_base_layer6_best.pt')
    patience_left = patience
    
    print(f"\nStarting training for {epochs} epochs...")
    
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        
        for features, labels in tqdm(train_loader, desc=f'Epoch {epoch}', unit='batch'):
            features = features.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)
            
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                logits = model(features)
                loss = criterion(logits, labels)
            
            scaler.scale(loss).backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
            
            running_loss += loss.item() * len(labels)
        
        scheduler.step()
        
        # Validation
        val_metrics = eval_loop(model, val_loader, device)
        train_loss = running_loss / len(train_ds)
        
        print(f"Epoch {epoch:02d} | train_loss={train_loss:.4f} | val_acc={val_metrics['acc']:.4f} | val_macroF1={val_metrics['macro_f1']:.4f}")
        
        # Early stopping and checkpointing
        if val_metrics['macro_f1'] > best_val:
            best_val = val_metrics['macro_f1']
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'val_metrics': val_metrics,
                'input_dim': 768,
                'n_classes': n_classes
            }, best_path)
            patience_left = patience
            print(f"  ✓ Saved new best to {best_path}")
        else:
            patience_left -= 1
            if patience_left <= 0:
                print("Early stopping triggered!")
                break
    
    # Load best model and evaluate
    print("\nLoading best model for final evaluation...")
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt['model_state'])
    
    val_metrics = eval_loop(model, val_loader, device)
    test_metrics = eval_loop(model, test_loader, device)
    
    # Save results
    with open(os.path.join(save_dir, 'val_metrics.json'), 'w') as f:
        json.dump(val_metrics, f, indent=2)
    with open(os.path.join(save_dir, 'test_metrics.json'), 'w') as f:
        json.dump(test_metrics, f, indent=2)
    
    print(f"\nBest VAL macro-F1: {best_val:.4f}")
    print(f"Test Accuracy: {test_metrics['acc']:.4f}")
    print(f"Test Macro-F1: {test_metrics['macro_f1']:.4f}")
    print(f"Artifacts saved to: {save_dir}")


if __name__ == '__main__':
    train()
