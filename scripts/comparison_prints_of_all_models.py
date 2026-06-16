#!/usr/bin/env python3
"""
Comprehensive Model Evaluation and Comparison Script
====================================================
Collects and compares evaluation metrics from all trained models.
Prints detailed results including accuracy, F1 scores, confusion matrices, etc.
"""

import os
from pathlib import Path
REPO_ROOT = Path(os.environ.get("THESIS_ROOT", Path(__file__).resolve().parents[1]))

import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict, Counter

try:
    import numpy as np
except ImportError:
    np = None

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich import box
    from rich.text import Text
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

try:
    from tabulate import tabulate
    TABULATE_AVAILABLE = True
except ImportError:
    TABULATE_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

# Initialize rich console if available
console = Console() if RICH_AVAILABLE else None

# Set matplotlib style for publication-quality figures
if MATPLOTLIB_AVAILABLE:
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_palette("husl")

# Base directory
BASE_DIR = REPO_ROOT
MODELS_DIR = BASE_DIR / "models"
META_BASE = BASE_DIR / "data_preparation" / "merged_datasets_plus_de"

# Model names mapping (for better display)
MODEL_NAMES = {
    "1_melspec_cnn": "MelSpec CNN",
    "3_nb_phoneme": "Naive Bayes Phoneme (Multi-class)",
    "3_wav2vec_base_layer6": "Wav2Vec2 Base Layer 6",
    "4_linear_phoneme": "Linear Phoneme (Multi-class)",
    "5_nb_phoneme_binary": "Naive Bayes Phoneme Binary",
    "7_linear_phoneme_binary": "Linear Phoneme Binary",
    "7_xlsr_300m_layer8": "XLSR 300M Layer 8",
}

# Class labels for multi-class models (if available)
# Based on actual label_id mapping from merged_datasets_plus_de
CLASS_LABELS_7 = {
    0: "Basel",
    1: "Bern",
    2: "Innerschweiz",
    3: "Ostschweiz",
    4: "Wallis",
    5: "Zürich",
    6: "Graubünden",
    7: "German"
}

# Neighbor relationships for Swiss dialects (bidirectional)
# Based on geographic proximity
NEIGHBORS = {
    0: [1],  # Basel neighbors: Bern
    1: [0, 2, 4],  # Bern neighbors: Basel, Innerschweiz (Central), Wallis (Valais)
    2: [1, 3, 4, 5, 6],  # Innerschweiz (Central) neighbors: Bern, Ostschweiz (Eastern), Wallis, Zürich, Graubünden
    3: [2, 5, 6],  # Ostschweiz (Eastern) neighbors: Innerschweiz, Zürich, Graubünden
    4: [1],  # Wallis (Valais) neighbors: Bern
    5: [2, 3],  # Zürich neighbors: Innerschweiz, Ostschweiz
    6: [2, 3],  # Graubünden neighbors: Innerschweiz, Ostschweiz
}

# Feature families for selective confusion matrix display
FEATURE_FAMILIES = {
    "phoneme_multiclass": ["3_nb_phoneme", "4_linear_phoneme"],
    "phoneme_binary": ["5_nb_phoneme_binary", "7_linear_phoneme_binary"],
    "wav2vec2": ["3_wav2vec_base_layer6"],
    "xlsr": ["7_xlsr_300m_layer8"],
}


def load_metrics(model_dir: Path) -> Dict[str, Dict]:
    """Load all metrics.json files from a model directory."""
    metrics = {}
    for split in ["train", "valid", "val", "test"]:
        for pattern in [f"{split}_metrics.json", f"{split}_metrics.json"]:
            metrics_file = model_dir / pattern
            if metrics_file.exists():
                try:
                    with open(metrics_file, "r") as f:
                        data = json.load(f)
                        metrics[split] = data
                except Exception as e:
                    print(f"Warning: Could not load {metrics_file}: {e}")
                break
    return metrics


def normalize_metric_key(data: Dict, key_variants: List[str]) -> Optional[float]:
    """Try different key names for the same metric."""
    for key in key_variants:
        if key in data:
            return float(data[key])
    return None


def print_confusion_matrix(cm: List[List[int]], labels: Optional[List[str]] = None, 
                          class_map: Optional[Dict[int, str]] = None):
    """Print confusion matrix in a readable format."""
    if np is not None:
        cm_array = np.array(cm)
    else:
        cm_array = cm
    n_classes = len(cm)
    
    # Determine labels
    if labels:
        display_labels = labels[:n_classes]
    elif class_map:
        display_labels = [class_map.get(i, f"Class {i}") for i in range(n_classes)]
    else:
        display_labels = [f"Class {i}" for i in range(n_classes)]
    
    # Use rich table if available
    if RICH_AVAILABLE and console:
        table = Table(title="Confusion Matrix", box=box.ROUNDED, show_header=True)
        table.add_column("True \\ Pred", style="cyan", no_wrap=True)
        for label in display_labels:
            table.add_column(label[:12], justify="right", style="magenta")
        table.add_column("Sum", justify="right", style="yellow")
        
        for i, row in enumerate(cm_array):
            row_vals = [str(val) for val in row]
            if np is not None:
                row_sum = cm_array[i, :].sum()
            else:
                row_sum = sum(row)
            table.add_row(display_labels[i][:12], *row_vals, str(row_sum))
        
        # Add totals row (column sums)
        if np is not None:
            col_sums = cm_array.sum(axis=0).tolist()
            total = cm_array.sum()
        else:
            col_sums = [sum(cm_array[j][i] for j in range(n_classes)) for i in range(n_classes)]
            total = sum(sum(row) for row in cm_array)
        
        # Add column sums row
        table.add_row("[bold]Total[/bold]", *[f"[bold]{s}[/bold]" for s in col_sums], f"[bold]{total}[/bold]", style="bold")
        console.print(table)
    else:
        # Fallback to plain text
        display_labels_short = [label[:15] for label in display_labels]
        print("    " + " ".join(f"{label:>12}" for label in display_labels_short))
        
        for i, row in enumerate(cm_array):
            label = display_labels[i][:12]
            if np is not None:
                row_vals = row
            else:
                row_vals = row
            print(f"{label:>3} " + " ".join(f"{val:>12}" for val in row_vals))
        
        if np is not None:
            row_sums = cm_array.sum(axis=1)
            col_sums = cm_array.sum(axis=0)
            total = cm_array.sum()
        else:
            row_sums = [sum(row) for row in cm_array]
            col_sums = [sum(cm_array[i][j] for i in range(n_classes)) for j in range(n_classes)]
            total = sum(sum(row) for row in cm_array)
        
        print(f"{'Sum':>3} " + " ".join(f"{val:>12}" for val in row_sums))
        print(f"{'Total':>3} " + f"{total:>12}")


def calculate_balanced_accuracy(cm: List[List[int]]) -> float:
    """Calculate balanced accuracy (average of per-class recall)."""
    if np is not None:
        cm_array = np.array(cm)
    else:
        cm_array = cm
    n_classes = len(cm_array)
    
    recalls = []
    for i in range(n_classes):
        if np is not None:
            tp = cm_array[i, i]
            fn = cm_array[i, :].sum() - tp
        else:
            tp = cm_array[i][i]
            fn = sum(cm_array[i][j] for j in range(n_classes)) - tp
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        recalls.append(recall)
    
    return sum(recalls) / len(recalls) if recalls else 0.0


def calculate_per_class_metrics(cm: List[List[int]], labels: Optional[List[str]] = None,
                                class_map: Optional[Dict[int, str]] = None) -> Dict[str, Dict]:
    """Calculate precision, recall, and F1 for each class."""
    if np is not None:
        cm_array = np.array(cm)
    else:
        cm_array = cm
    n_classes = len(cm_array)
    
    if labels:
        class_names = labels[:n_classes]
    elif class_map:
        class_names = [class_map.get(i, f"Class {i}") for i in range(n_classes)]
    else:
        class_names = [f"Class {i}" for i in range(n_classes)]
    
    metrics = {}
    for i in range(n_classes):
        if np is not None:
            tp = cm_array[i, i]
            fp = cm_array[:, i].sum() - tp
            fn = cm_array[i, :].sum() - tp
            total = cm_array.sum()
        else:
            tp = cm_array[i][i]
            fp = sum(cm_array[j][i] for j in range(n_classes)) - tp
            fn = sum(cm_array[i][j] for j in range(n_classes)) - tp
            total = sum(sum(row) for row in cm_array)
        tn = total - tp - fp - fn
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        metrics[class_names[i]] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": tp + fn
        }
    
    return metrics


def load_speaker_mapping(split: str) -> Dict[str, str]:
    """Load audio_path -> client_id mapping from metadata TSV."""
    mapping = {}
    meta_file = META_BASE / f"merged_{split}.tsv"
    if not meta_file.exists():
        return mapping
    
    try:
        with open(meta_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
            if len(lines) < 2:
                return mapping
            
            # Parse header
            header = lines[0].strip().split("\t")
            if "audio_path" not in header or "client_id" not in header:
                return mapping
            
            audio_idx = header.index("audio_path")
            client_idx = header.index("client_id")
            
            # Parse data
            for line in lines[1:]:
                parts = line.strip().split("\t")
                if len(parts) > max(audio_idx, client_idx):
                    audio_path = parts[audio_idx]
                    client_id = parts[client_idx]
                    mapping[audio_path] = client_id
    except Exception as e:
        print(f"Warning: Could not load speaker mapping from {meta_file}: {e}")
    
    return mapping


def compute_speaker_level_confusion_matrix(model_name: str, split: str, preds_file: Path) -> Optional[List[List[int]]]:
    """Compute speaker-level confusion matrix by majority voting."""
    if not preds_file.exists():
        return None
    
    # Check if this is a binary phoneme model
    is_binary_phoneme = model_name in ["5_nb_phoneme_binary", "7_linear_phoneme_binary"]
    if not is_binary_phoneme:
        return None
    
    try:
        import csv
        
        # Load speaker mapping
        speaker_map = load_speaker_mapping(split)
        if not speaker_map:
            return None
        
        # Load predictions and group by speaker
        speaker_predictions = defaultdict(lambda: {"preds": [], "true": None})
        
        with open(preds_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                audio_path = row.get("audio_path", "")
                # Try to find client_id from audio_path
                client_id = None
                for path, cid in speaker_map.items():
                    if path in audio_path or audio_path in path:
                        client_id = cid
                        break
                
                # Also try extracting from concat format: concat_{client_id}_{label}_{idx}
                if not client_id and audio_path.startswith("concat_"):
                    parts = audio_path.split("_")
                    if len(parts) >= 2:
                        potential_id = "_".join(parts[1:-2]) if len(parts) > 3 else parts[1]
                        if potential_id in speaker_map.values():
                            client_id = potential_id
                
                if not client_id:
                    continue
                
                # Get prediction and true label
                if "pred_binary" in row:
                    pred = int(row["pred_binary"])
                else:
                    continue
                
                if "binary_label" in row:
                    true_label = int(row["binary_label"])
                else:
                    continue
                
                if speaker_predictions[client_id]["true"] is None:
                    speaker_predictions[client_id]["true"] = true_label
                speaker_predictions[client_id]["preds"].append(pred)
        
        if not speaker_predictions:
            return None
        
        # Build confusion matrix (2x2 for binary)
        cm = [[0, 0], [0, 0]]
        
        for client_id, data in speaker_predictions.items():
            preds = data["preds"]
            true_label = data["true"]
            
            if not preds:
                continue
            
            # Majority vote
            majority_pred = Counter(preds).most_common(1)[0][0]
            cm[true_label][majority_pred] += 1
        
        return cm
    except Exception as e:
        print(f"Warning: Speaker-level confusion matrix computation failed: {e}")
        return None


def evaluate_speaker_level(model_name: str, split: str, preds_file: Path) -> Optional[Dict]:
    """Evaluate at speaker level by majority voting."""
    if not preds_file.exists():
        return None
    
    # Check if this is a phoneme-based model
    is_phoneme_model = model_name in ["3_nb_phoneme", "4_linear_phoneme", 
                                       "5_nb_phoneme_binary", "7_linear_phoneme_binary"]
    if not is_phoneme_model:
        return None
    
    try:
        import csv
        
        # Load speaker mapping
        speaker_map = load_speaker_mapping(split)
        if not speaker_map:
            return None
        
        # Load predictions
        predictions = {}
        with open(preds_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                audio_path = row.get("audio_path", "")
                # Try to find client_id from audio_path
                client_id = None
                for path, cid in speaker_map.items():
                    if path in audio_path or audio_path in path:
                        client_id = cid
                        break
                
                # Also try extracting from concat format: concat_{client_id}_{label}_{idx}
                if not client_id and audio_path.startswith("concat_"):
                    parts = audio_path.split("_")
                    if len(parts) >= 2:
                        # Try to match UUID pattern
                        potential_id = "_".join(parts[1:-2]) if len(parts) > 3 else parts[1]
                        if potential_id in speaker_map.values():
                            client_id = potential_id
                
                if not client_id:
                    continue
                
                # Get prediction and true label
                if "pred" in row:
                    pred = int(row["pred"])
                elif "pred_binary" in row:
                    pred = int(row["pred_binary"])
                else:
                    continue
                
                if "label_id" in row:
                    true_label = int(row["label_id"])
                elif "binary_label" in row:
                    true_label = int(row["binary_label"])
                else:
                    continue
                
                if client_id not in predictions:
                    predictions[client_id] = {"preds": [], "true": true_label}
                predictions[client_id]["preds"].append(pred)
        
        if not predictions:
            return None
        
        # Majority vote per speaker
        correct = 0
        total = 0
        for client_id, data in predictions.items():
            preds = data["preds"]
            true_label = data["true"]
            
            # Majority vote
            majority_pred = Counter(preds).most_common(1)[0][0]
            
            if majority_pred == true_label:
                correct += 1
            total += 1
        
        speaker_acc = correct / total if total > 0 else 0.0
        
        return {
            "speaker_accuracy": speaker_acc,
            "num_speakers": total
        }
    except Exception as e:
        print(f"Warning: Speaker-level evaluation failed for {model_name} ({split}): {e}")
        return None


def print_model_results(model_name: str, model_dir: Path, metrics: Dict[str, Dict], 
                       show_cm: bool = True):
    """Print detailed results for a single model."""
    display_name = MODEL_NAMES.get(model_name, model_name)
    
    if RICH_AVAILABLE and console:
        console.print()
        console.print(Panel.fit(
            f"[bold cyan]{display_name}[/bold cyan]\n[dim]{model_name}[/dim]",
            border_style="cyan"
        ))
    else:
        print("\n" + "=" * 80)
        print(f"MODEL: {display_name} ({model_name})")
        print("=" * 80)
    
    for split in ["train", "valid", "val", "test"]:
        if split not in metrics:
            continue
        
        data = metrics[split]
        
        if RICH_AVAILABLE and console:
            console.print(f"\n[bold yellow]--- {split.upper()} SET ---[/bold yellow]")
        else:
            print(f"\n--- {split.upper()} SET ---")
        
        # Collect metrics for display
        metrics_list = []
        
        # Accuracy
        acc = normalize_metric_key(data, ["accuracy", "acc"])
        if acc is not None:
            metrics_list.append(("Accuracy", f"{acc:.4f}", f"{acc*100:.2f}%"))
        
        # Macro F1
        macro_f1 = normalize_metric_key(data, ["macro_f1", "macro_f1_score"])
        if macro_f1 is not None:
            metrics_list.append(("Macro F1", f"{macro_f1:.4f}", ""))
        
        # Balanced Accuracy (for multi-class)
        cm = None
        if "confusion_matrix" in data:
            cm = data["confusion_matrix"]
        elif "cm" in data:
            cm = data["cm"]
        
        if cm and len(cm) > 2:
            balanced_acc = calculate_balanced_accuracy(cm)
            metrics_list.append(("Balanced Accuracy", f"{balanced_acc:.4f}", f"{balanced_acc*100:.2f}%"))
        
        # Per-class F1 scores (if available)
        if "f1_non_german" in data:
            metrics_list.append(("F1 (Non-German)", f"{data['f1_non_german']:.4f}", ""))
        if "f1_german" in data:
            metrics_list.append(("F1 (German)", f"{data['f1_german']:.4f}", ""))
        
        # Speaker-level evaluation (for phoneme models)
        if split == "test":
            preds_file = model_dir / f"{split}_preds.csv"
            speaker_metrics = evaluate_speaker_level(model_name, split, preds_file)
            if speaker_metrics:
                metrics_list.append(("Speaker-level Accuracy", 
                                   f"{speaker_metrics['speaker_accuracy']:.4f}",
                                   f"{speaker_metrics['speaker_accuracy']*100:.2f}% (n={speaker_metrics['num_speakers']})"))
        
        # Display metrics
        if RICH_AVAILABLE and console:
            metrics_table = Table(show_header=False, box=box.SIMPLE)
            metrics_table.add_column("Metric", style="cyan")
            metrics_table.add_column("Value", style="green", justify="right")
            metrics_table.add_column("Extra", style="dim", justify="right")
            for metric_name, value, extra in metrics_list:
                metrics_table.add_row(metric_name, value, extra)
            console.print(metrics_table)
        else:
            for metric_name, value, extra in metrics_list:
                if extra:
                    print(f"{metric_name}: {value} ({extra})")
                else:
                    print(f"{metric_name}: {value}")
        
        # Confusion matrix (only if show_cm is True)
        if cm and show_cm:
            # For binary models, also show speaker-level confusion matrix
            if model_name in ["5_nb_phoneme_binary", "7_linear_phoneme_binary"] and split == "test":
                preds_file = model_dir / f"{split}_preds.csv"
                cm_speaker = compute_speaker_level_confusion_matrix(model_name, split, preds_file)
                if cm_speaker:
                    print("\nConfusion Matrix (Sample-level - individual audio samples):")
                    labels = None
                    if "confusion_matrix_labels" in data:
                        labels = data["confusion_matrix_labels"]
                    elif "classes" in data:
                        labels = data["classes"]
                    print_confusion_matrix(cm, labels, None)
                    
                    print("\nConfusion Matrix (Speaker-level - aggregated by speaker, majority vote):")
                    print_confusion_matrix(cm_speaker, labels, None)
                    
                    # Verify both match their respective accuracies
                    total_samples = sum(sum(row) for row in cm)
                    correct_samples = cm[0][0] + cm[1][1]
                    acc_from_cm = correct_samples / total_samples if total_samples > 0 else 0.0
                    
                    total_speakers = sum(sum(row) for row in cm_speaker)
                    correct_speakers = cm_speaker[0][0] + cm_speaker[1][1]
                    acc_speaker_from_cm = correct_speakers / total_speakers if total_speakers > 0 else 0.0
                    
                    if abs(acc_from_cm - acc) > 0.001:
                        print(f"\n⚠️  WARNING: Sample-level CM accuracy ({acc_from_cm:.4f}) doesn't match stored accuracy ({acc:.4f})")
                    
                    # Note: Speaker-level accuracy is shown separately in metrics_list above
                    continue  # Skip the regular confusion matrix display since we already showed it
            
            # Verify confusion matrix matches accuracy for binary models
            if model_name in ["5_nb_phoneme_binary", "7_linear_phoneme_binary"] and acc is not None:
                total_from_cm = sum(sum(row) for row in cm)
                if len(cm) == 2:  # Binary classification
                    correct_from_cm = cm[0][0] + cm[1][1]
                    calc_acc = correct_from_cm / total_from_cm if total_from_cm > 0 else 0.0
                    if abs(calc_acc - acc) > 0.001:  # Allow small floating point differences
                        print(f"\n⚠️  WARNING: Confusion matrix accuracy ({calc_acc:.4f}) doesn't match stored accuracy ({acc:.4f})")
                        print(f"   Total from CM: {total_from_cm}, Correct: {correct_from_cm}")
            
            print("\nConfusion Matrix:")
            labels = None
            if "confusion_matrix_labels" in data:
                labels = data["confusion_matrix_labels"]
            elif "classes" in data:
                labels = data["classes"]
            
            class_map = None
            if len(cm) == 7:
                class_map = CLASS_LABELS_7
            
            print_confusion_matrix(cm, labels, class_map)
            
            # Calculate and print per-class metrics for multi-class
            if len(cm) > 2:
                per_class = calculate_per_class_metrics(cm, labels, class_map)
                
                if RICH_AVAILABLE and console:
                    per_class_table = Table(title="Per-Class Metrics", box=box.ROUNDED)
                    per_class_table.add_column("Class", style="cyan")
                    per_class_table.add_column("Precision", justify="right", style="green")
                    per_class_table.add_column("Recall", justify="right", style="green")
                    per_class_table.add_column("F1", justify="right", style="green")
                    per_class_table.add_column("Support", justify="right", style="yellow")
                    
                    for class_name, m in per_class.items():
                        per_class_table.add_row(
                            class_name,
                            f"{m['precision']:.4f}",
                            f"{m['recall']:.4f}",
                            f"{m['f1']:.4f}",
                            str(m['support'])
                        )
                    console.print(per_class_table)
                else:
                    print("\nPer-Class Metrics:")
                    print(f"{'Class':<20} {'Precision':>12} {'Recall':>12} {'F1':>12} {'Support':>12}")
                    print("-" * 72)
                    for class_name, m in per_class.items():
                        print(f"{class_name:<20} {m['precision']:>12.4f} {m['recall']:>12.4f} "
                              f"{m['f1']:>12.4f} {m['support']:>12}")


def create_comparison_table(all_results: Dict[str, Dict[str, Dict]]) -> List[Dict]:
    """Create a comparison table across all models."""
    rows = []
    
    for model_name, metrics in all_results.items():
        display_name = MODEL_NAMES.get(model_name, model_name)
        
        for split in ["train", "valid", "val", "test"]:
            if split not in metrics:
                continue
            
            data = metrics[split]
            acc = normalize_metric_key(data, ["accuracy", "acc"])
            macro_f1 = normalize_metric_key(data, ["macro_f1", "macro_f1_score"])
            
            # Calculate balanced accuracy for multi-class
            balanced_acc = None
            cm = None
            if "confusion_matrix" in data:
                cm = data["confusion_matrix"]
            elif "cm" in data:
                cm = data["cm"]
            
            if cm and len(cm) > 2:
                balanced_acc = calculate_balanced_accuracy(cm)
            
            row = {
                "Model": display_name,
                "Model_ID": model_name,
                "Split": split,
                "Accuracy": acc if acc is not None else None,
                "Macro_F1": macro_f1 if macro_f1 is not None else None,
                "Balanced_Accuracy": balanced_acc,
            }
            
            # Add per-class F1 for binary models
            if "f1_non_german" in data:
                row["F1_Non_German"] = data["f1_non_german"]
            if "f1_german" in data:
                row["F1_German"] = data["f1_german"]
            
            rows.append(row)
    
    return rows


def get_best_model_per_family(all_results: Dict[str, Dict[str, Dict]]) -> Dict[str, str]:
    """Determine best model per feature family based on test set macro F1."""
    best_models = {}
    
    for family_name, model_list in FEATURE_FAMILIES.items():
        best_model = None
        best_f1 = -1.0
        
        for model_name in model_list:
            if model_name not in all_results:
                continue
            
            metrics = all_results[model_name]
            # Try test, then val
            test_data = metrics.get("test") or metrics.get("val")
            if not test_data:
                continue
            
            macro_f1 = normalize_metric_key(test_data, ["macro_f1", "macro_f1_score"])
            if macro_f1 is not None and macro_f1 > best_f1:
                best_f1 = macro_f1
                best_model = model_name
        
        if best_model:
            best_models[family_name] = best_model
    
    return best_models


def print_curated_results_table(all_results: Dict[str, Dict[str, Dict]]):
    """Print curated results table for thesis (test set only)."""
    if RICH_AVAILABLE and console:
        console.print()
        console.print(Panel.fit(
            "[bold cyan]CURATED RESULTS TABLE (TEST SET ONLY)[/bold cyan]",
            border_style="cyan"
        ))
    else:
        print("\n" + "=" * 80)
        print("CURATED RESULTS TABLE (TEST SET ONLY)")
        print("=" * 80)
    
    rows = []
    for model_name, metrics in all_results.items():
        if "test" not in metrics:
            continue
        
        data = metrics["test"]
        display_name = MODEL_NAMES.get(model_name, model_name)
        
        acc = normalize_metric_key(data, ["accuracy", "acc"])
        macro_f1 = normalize_metric_key(data, ["macro_f1", "macro_f1_score"])
        
        # Balanced accuracy for multi-class
        balanced_acc = None
        cm = None
        if "confusion_matrix" in data:
            cm = data["confusion_matrix"]
        elif "cm" in data:
            cm = data["cm"]
        
        if cm and len(cm) > 2:
            balanced_acc = calculate_balanced_accuracy(cm)
        
        rows.append({
            "Model": display_name,
            "Accuracy": acc,
            "Macro_F1": macro_f1,
            "Balanced_Accuracy": balanced_acc,
        })
    
    # Sort by accuracy descending
    rows.sort(key=lambda x: x["Accuracy"] if x["Accuracy"] is not None else -1, reverse=True)
    
    if RICH_AVAILABLE and console:
        table = Table(title="Test Set Performance Comparison", box=box.ROUNDED, show_header=True)
        table.add_column("Model", style="cyan", no_wrap=False)
        table.add_column("Accuracy", justify="right", style="green")
        table.add_column("Macro F1", justify="right", style="green")
        table.add_column("Balanced Acc", justify="right", style="yellow")
        
        for row in rows:
            acc_str = f"{row['Accuracy']:.4f}" if row['Accuracy'] is not None else "N/A"
            f1_str = f"{row['Macro_F1']:.4f}" if row['Macro_F1'] is not None else "N/A"
            bal_str = f"{row['Balanced_Accuracy']:.4f}" if row['Balanced_Accuracy'] is not None else "N/A"
            table.add_row(row['Model'], acc_str, f1_str, bal_str)
        
        console.print(table)
    elif TABULATE_AVAILABLE:
        table_data = []
        headers = ["Model", "Accuracy", "Macro F1", "Balanced Acc"]
        for row in rows:
            acc_str = f"{row['Accuracy']:.4f}" if row['Accuracy'] is not None else "N/A"
            f1_str = f"{row['Macro_F1']:.4f}" if row['Macro_F1'] is not None else "N/A"
            bal_str = f"{row['Balanced_Accuracy']:.4f}" if row['Balanced_Accuracy'] is not None else "N/A"
            table_data.append([row['Model'], acc_str, f1_str, bal_str])
        print("\n" + tabulate(table_data, headers=headers, tablefmt="grid", floatfmt=".4f"))
    else:
        print(f"\n{'Model':<45} {'Accuracy':>12} {'Macro F1':>12} {'Balanced Acc':>15}")
        print("-" * 84)
        for row in rows:
            acc_str = f"{row['Accuracy']:.4f}" if row['Accuracy'] is not None else "N/A"
            f1_str = f"{row['Macro_F1']:.4f}" if row['Macro_F1'] is not None else "N/A"
            bal_str = f"{row['Balanced_Accuracy']:.4f}" if row['Balanced_Accuracy'] is not None else "N/A"
            print(f"{row['Model']:<45} {acc_str:>12} {f1_str:>12} {bal_str:>15}")


def print_summary_statistics(rows: List[Dict]):
    """Print summary statistics."""
    if RICH_AVAILABLE and console:
        console.print()
        console.print(Panel.fit(
            "[bold yellow]SUMMARY STATISTICS[/bold yellow]",
            border_style="yellow"
        ))
    else:
        print("\n" + "=" * 80)
        print("SUMMARY STATISTICS")
        print("=" * 80)
    
    # Test set only
    test_rows = [r for r in rows if r["Split"] == "test"]
    if len(test_rows) == 0:
        test_rows = [r for r in rows if r["Split"] == "val"]
    
    if len(test_rows) > 0:
        if RICH_AVAILABLE and console:
            table = Table(title="Test Set Performance", box=box.ROUNDED)
            table.add_column("Model", style="cyan")
            table.add_column("Accuracy", justify="right", style="green")
            table.add_column("Macro F1", justify="right", style="green")
            
            for row in test_rows:
                acc_str = f"{row['Accuracy']:.4f}" if row['Accuracy'] is not None else "N/A"
                f1_str = f"{row['Macro_F1']:.4f}" if row['Macro_F1'] is not None else "N/A"
                table.add_row(row['Model'], acc_str, f1_str)
            
            console.print(table)
        elif TABULATE_AVAILABLE:
            table_data = []
            for row in test_rows:
                acc_str = f"{row['Accuracy']:.4f}" if row['Accuracy'] is not None else "N/A"
                f1_str = f"{row['Macro_F1']:.4f}" if row['Macro_F1'] is not None else "N/A"
                table_data.append([row['Model'], acc_str, f1_str])
            print("\n--- Test Set Performance ---")
            print(tabulate(table_data, headers=["Model", "Accuracy", "Macro F1"], tablefmt="grid"))
        else:
            print("\n--- Test Set Performance ---")
            print(f"{'Model':<40} {'Accuracy':>12} {'Macro_F1':>12}")
            print("-" * 64)
            for row in test_rows:
                acc_str = f"{row['Accuracy']:.4f}" if row['Accuracy'] is not None else "N/A"
                f1_str = f"{row['Macro_F1']:.4f}" if row['Macro_F1'] is not None else "N/A"
                print(f"{row['Model']:<40} {acc_str:>12} {f1_str:>12}")
        
        # Best models
        valid_acc = [r for r in test_rows if r["Accuracy"] is not None]
        if valid_acc:
            best_acc = max(valid_acc, key=lambda x: x["Accuracy"])
            best_msg = f"Best Accuracy: {best_acc['Model']} ({best_acc['Accuracy']:.4f})"
            if RICH_AVAILABLE and console:
                console.print(f"\n[bold green]{best_msg}[/bold green]")
            else:
                print(f"\n{best_msg}")
        
        valid_f1 = [r for r in test_rows if r["Macro_F1"] is not None]
        if valid_f1:
            best_f1 = max(valid_f1, key=lambda x: x["Macro_F1"])
            best_msg = f"Best Macro F1: {best_f1['Model']} ({best_f1['Macro_F1']:.4f})"
            if RICH_AVAILABLE and console:
                console.print(f"[bold green]{best_msg}[/bold green]")
            else:
                print(best_msg)


def main():
    """Main evaluation function."""
    if RICH_AVAILABLE and console:
        console.print()
        console.print(Panel.fit(
            "[bold cyan]COMPREHENSIVE MODEL EVALUATION AND COMPARISON[/bold cyan]",
            border_style="cyan"
        ))
    else:
        print("=" * 80)
        print("COMPREHENSIVE MODEL EVALUATION AND COMPARISON")
        print("=" * 80)
    
    # Collect all results
    all_results = {}
    
    if not MODELS_DIR.exists():
        print(f"Error: Models directory not found: {MODELS_DIR}")
        return
    
    model_dirs = sorted([d for d in MODELS_DIR.iterdir() if d.is_dir()])
    
    if not model_dirs:
        print(f"No model directories found in {MODELS_DIR}")
        return
    
    print(f"\nFound {len(model_dirs)} model directories")
    
    # Load metrics for each model
    for model_dir in model_dirs:
        model_name = model_dir.name
        metrics = load_metrics(model_dir)
        if metrics:
            all_results[model_name] = metrics
    
    if not all_results:
        print("No metrics found in any model directory!")
        return
    
    # Determine best models per feature family
    best_models = get_best_model_per_family(all_results)
    
    # Print detailed results for each model (with selective confusion matrices)
    for model_name in sorted(all_results.keys()):
        # Only show confusion matrix for best model in each family
        show_cm = False
        for family_name, best_model in best_models.items():
            if model_name == best_model:
                show_cm = True
                break
        
        # Also show CM for binary models (they're in separate families)
        if model_name in ["5_nb_phoneme_binary", "7_linear_phoneme_binary"]:
            show_cm = True
        
        print_model_results(model_name, MODELS_DIR / model_name, all_results[model_name], 
                           show_cm=show_cm)
    
    # Print curated results table
    print_curated_results_table(all_results)
    
    # Create comparison table
    comparison_rows = create_comparison_table(all_results)
    
    # Print summary
    print_summary_statistics(comparison_rows)
    
    # Generate confusion matrix figures
    generate_figures(all_results)
    
    # Generate error breakdown bar chart
    plot_error_breakdown_bar_chart(all_results)
    
    # Save comparison table as CSV
    output_file = BASE_DIR / "model_comparison.csv"
    try:
        import csv
        if comparison_rows:
            # Get all possible fieldnames from all rows
            all_fieldnames = set()
            for row in comparison_rows:
                all_fieldnames.update(row.keys())
            fieldnames = sorted(all_fieldnames)
            
            with open(output_file, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(comparison_rows)
            print(f"\nComparison table saved to: {output_file}")
    except Exception as e:
        print(f"\nWarning: Could not save CSV file: {e}")
        print("Comparison data available in memory.")
    
    if RICH_AVAILABLE and console:
        console.print()
        console.print(Panel.fit(
            "[bold green]✓ EVALUATION COMPLETE[/bold green]",
            border_style="green"
        ))
    else:
        print("\n" + "=" * 80)
        print("EVALUATION COMPLETE")
        print("=" * 80)


def analyze_neighbor_confusion(cm: List[List[int]]) -> Dict:
    """
    Analyze confusion matrix to count correct, neighbor confusion, and non-neighbor confusion.
    Returns a dictionary with statistics.
    """
    if np is not None:
        cm_array = np.array(cm)
    else:
        cm_array = [[int(x) for x in row] for row in cm]
        import numpy as np_local
        cm_array = np_local.array(cm_array)
    
    n_classes = len(cm_array)
    total_correct = 0
    total_neighbor_confusion = 0
    total_non_neighbor_confusion = 0
    
    for true_label in range(n_classes):
        neighbors = NEIGHBORS.get(true_label, [])
        for pred_label in range(n_classes):
            value = int(cm_array[true_label, pred_label])
            if true_label == pred_label:
                total_correct += value
            elif pred_label in neighbors:
                total_neighbor_confusion += value
            else:
                total_non_neighbor_confusion += value
    
    total_samples = total_correct + total_neighbor_confusion + total_non_neighbor_confusion
    
    return {
        'correct': total_correct,
        'neighbor_confusion': total_neighbor_confusion,
        'non_neighbor_confusion': total_non_neighbor_confusion,
        'total': total_samples,
        'correct_pct': (total_correct / total_samples * 100) if total_samples > 0 else 0.0,
        'neighbor_pct': (total_neighbor_confusion / total_samples * 100) if total_samples > 0 else 0.0,
        'non_neighbor_pct': (total_non_neighbor_confusion / total_samples * 100) if total_samples > 0 else 0.0,
    }


def plot_error_breakdown_bar_chart(all_results: Dict[str, Dict[str, Dict]]):
    """Create bar chart showing correct vs neighbor vs non-neighbor confusion for all models."""
    if not MATPLOTLIB_AVAILABLE:
        print("Warning: matplotlib not available, cannot create error breakdown chart")
        return
    
    # Collect data for all multi-class models (7-way classification)
    model_data = []
    for model_name, metrics in all_results.items():
        # Only include 7-way classification models (exclude binary)
        if model_name in ["5_nb_phoneme_binary", "7_linear_phoneme_binary"]:
            continue
        
        if "test" not in metrics:
            continue
        
        data = metrics["test"]
        cm = data.get("cm") or data.get("confusion_matrix")
        if not cm or len(cm) != 7:  # Only 7-way classification
            continue
        
        stats = analyze_neighbor_confusion(cm)
        display_name = MODEL_NAMES.get(model_name, model_name)
        model_data.append({
            'model': display_name,
            'model_id': model_name,
            'correct': stats['correct'],
            'neighbor': stats['neighbor_confusion'],
            'non_neighbor': stats['non_neighbor_confusion'],
            'total': stats['total']
        })
    
    if not model_data:
        print("No multi-class models found for error breakdown chart")
        return
    
    # Sort by total accuracy (correct percentage)
    model_data.sort(key=lambda x: x['correct'] / x['total'] if x['total'] > 0 else 0, reverse=True)
    
    # Create bar chart
    fig, ax = plt.subplots(figsize=(12, 8))
    
    models = [d['model'] for d in model_data]
    correct_counts = [d['correct'] for d in model_data]
    neighbor_counts = [d['neighbor'] for d in model_data]
    non_neighbor_counts = [d['non_neighbor'] for d in model_data]
    
    x = np.arange(len(models))
    width = 0.6
    
    # Create stacked bars
    p1 = ax.bar(x, correct_counts, width, label='Correct', color='#2ecc71', alpha=0.8)
    p2 = ax.bar(x, neighbor_counts, width, bottom=correct_counts, label='Wrong (Neighbor)', color='#f39c12', alpha=0.8)
    p3 = ax.bar(x, non_neighbor_counts, width, 
                bottom=[c + n for c, n in zip(correct_counts, neighbor_counts)], 
                label='Wrong (Non-Neighbor)', color='#e74c3c', alpha=0.8)
    
    # Add value labels on bars
    for i, (c, n, nn) in enumerate(zip(correct_counts, neighbor_counts, non_neighbor_counts)):
        total = c + n + nn
        # Label for correct
        if c > total * 0.05:  # Only label if > 5% of bar
            ax.text(i, c/2, f'{c}', ha='center', va='center', fontweight='bold', fontsize=9)
        # Label for neighbor confusion
        if n > total * 0.05:
            ax.text(i, c + n/2, f'{n}', ha='center', va='center', fontsize=9)
        # Label for non-neighbor confusion
        if nn > total * 0.05:
            ax.text(i, c + n + nn/2, f'{nn}', ha='center', va='center', fontsize=9)
    
    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Samples', fontsize=12, fontweight='bold')
    ax.set_title('Error Breakdown: Correct vs Neighbor vs Non-Neighbor Confusion\n(Test Set)', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    output_path = BASE_DIR / "error_breakdown_bar_chart.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved error breakdown chart: {output_path}")
    plt.close()


def plot_neighbor_confusion_by_class(model_name: str, cm: List[List[int]], labels: List[str], 
                                     output_filename: str):
    """Create a bar chart showing per-class: correct %, wrong (neighbor) %, wrong (non-neighbor) %.
    Neighbor confusion is weighted by expected random chance to account for different numbers of neighbors."""
    if not MATPLOTLIB_AVAILABLE:
        print(f"Warning: matplotlib not available, cannot plot {output_filename}")
        return
    
    if np is not None:
        cm_array = np.array(cm)
    else:
        cm_array = [[int(x) for x in row] for row in cm]
        import numpy as np_local
        cm_array = np_local.array(cm_array)
    
    n_classes = len(cm_array)
    
    # Calculate percentages for each class (weighted by expected random chance)
    correct_pcts = []
    neighbor_wrong_pcts = []
    non_neighbor_wrong_pcts = []
    neighbor_ratios = []  # Actual neighbor confusion / Expected random neighbor confusion
    
    for true_label in range(n_classes):
        neighbors = NEIGHBORS.get(true_label, [])
        row_total = float(cm_array[true_label, :].sum())
        
        if row_total == 0:
            correct_pcts.append(0.0)
            neighbor_wrong_pcts.append(0.0)
            non_neighbor_wrong_pcts.append(0.0)
            neighbor_ratios.append(0.0)
            continue
        
        # Correct predictions (diagonal)
        correct_count = float(cm_array[true_label, true_label])
        correct_pct = (correct_count / row_total) * 100
        
        # Wrong predictions to neighbors
        neighbor_wrong_count = 0.0
        for pred_label in neighbors:
            if pred_label != true_label:  # Don't count correct as neighbor
                neighbor_wrong_count += float(cm_array[true_label, pred_label])
        
        # Total errors
        total_errors = row_total - correct_count
        
        if total_errors > 0:
            # Actual neighbor confusion percentage (of errors)
            actual_neighbor_pct_of_errors = (neighbor_wrong_count / total_errors) * 100
            
            # Expected neighbor confusion percentage if errors were random
            # = number_of_neighbors / (total_classes - 1) * 100
            num_neighbors = len(neighbors)
            num_possible_wrong_classes = n_classes - 1  # Exclude the correct class itself
            expected_neighbor_pct_of_errors = (num_neighbors / num_possible_wrong_classes) * 100 if num_possible_wrong_classes > 0 else 0.0
            
            # Weighted neighbor confusion: normalize by expected random chance
            # If expected is 0, use actual (shouldn't happen, but safety check)
            if expected_neighbor_pct_of_errors > 0:
                # Ratio: how much more/less than random chance
                neighbor_ratio = actual_neighbor_pct_of_errors / expected_neighbor_pct_of_errors
            else:
                neighbor_ratio = 0.0
            
            # For display: show weighted neighbor confusion as percentage of total samples
            # Weight by the ratio: if ratio > 1, model confuses neighbors more than random
            # We'll show the actual percentages but weight them visually or in calculation
            neighbor_wrong_pct = (neighbor_wrong_count / row_total) * 100
            neighbor_ratios.append(neighbor_ratio)
        else:
            neighbor_wrong_pct = 0.0
            neighbor_ratios.append(0.0)
        
        # Wrong predictions to non-neighbors
        non_neighbor_wrong_count = row_total - correct_count - neighbor_wrong_count
        non_neighbor_wrong_pct = (non_neighbor_wrong_count / row_total) * 100
        
        correct_pcts.append(correct_pct)
        neighbor_wrong_pcts.append(neighbor_wrong_pct)
        non_neighbor_wrong_pcts.append(non_neighbor_wrong_pct)
    
    # Create stacked bar chart
    fig, ax = plt.subplots(figsize=(14, 8))
    
    x = np.arange(n_classes)
    width = 0.7
    
    # Create stacked bars
    p1 = ax.bar(x, correct_pcts, width, label='Correct', color='#2ecc71', alpha=0.8)
    p2 = ax.bar(x, neighbor_wrong_pcts, width, bottom=correct_pcts, 
                label='Wrong (Neighbor)', color='#f39c12', alpha=0.8)
    p3 = ax.bar(x, non_neighbor_wrong_pcts, width, 
                bottom=[c + n for c, n in zip(correct_pcts, neighbor_wrong_pcts)], 
                label='Wrong (Non-Neighbor)', color='#e74c3c', alpha=0.8)
    
    # Add value labels on bars with neighbor ratio info
    for i, (c, n, nn, ratio) in enumerate(zip(correct_pcts, neighbor_wrong_pcts, non_neighbor_wrong_pcts, neighbor_ratios)):
        total = c + n + nn
        # Label for correct
        if c > 2:  # Only label if > 2%
            ax.text(i, c/2, f'{c:.1f}%', ha='center', va='center', 
                   fontweight='bold', fontsize=9, color='white')
        # Label for neighbor confusion (with ratio indicator)
        if n > 2:
            ratio_text = f'×{ratio:.2f}' if ratio > 0 else ''
            label_text = f'{n:.1f}%' + (f'\n{ratio_text}' if ratio_text else '')
            ax.text(i, c + n/2, label_text, ha='center', va='center', 
                   fontsize=8, color='white')
        # Label for non-neighbor confusion
        if nn > 2:
            ax.text(i, c + n + nn/2, f'{nn:.1f}%', ha='center', va='center', 
                   fontsize=9, color='white')
    
    ax.set_xlabel('Dialect Class', fontsize=12, fontweight='bold')
    ax.set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
    ax.set_title(f'{MODEL_NAMES.get(model_name, model_name)}\nPer-Class Error Breakdown: Correct vs Neighbor vs Non-Neighbor Confusion\n(Neighbor confusion weighted by expected random chance)', 
                 fontsize=13, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylim([0, 100])
    
    # Update legend to include ratio explanation
    legend = ax.legend(loc='upper left', fontsize=10)
    # Add text explaining the ratio
    ax.text(0.02, 0.98, 'Ratio (×) shows neighbor confusion vs random chance\n(×1.0 = random, >1.0 = more neighbor confusion)', 
            transform=ax.transAxes, fontsize=8, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    output_path = BASE_DIR / output_filename
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved neighbor confusion by class plot: {output_path}")
    plt.close()


def plot_confusion_matrix_figure(cm: List[List[int]], labels: List[str], 
                                title: str, filename: str, normalize: bool = False,
                                figsize: Tuple[int, int] = (10, 8)):
    """Plot a publication-quality confusion matrix figure."""
    if not MATPLOTLIB_AVAILABLE:
        print(f"Warning: matplotlib not available, cannot plot {filename}")
        return
    
    if np is not None:
        cm_array = np.array(cm)
    else:
        # Fallback if numpy not available
        cm_array = [[int(x) for x in row] for row in cm]
        import numpy as np_local
        cm_array = np_local.array(cm_array)
    
    if normalize:
        # Normalize by row (true labels)
        cm_array = cm_array.astype('float') / cm_array.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
        vmax = 1.0
    else:
        fmt = 'd'
        vmax = None
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create heatmap
    sns.heatmap(cm_array, annot=True, fmt=fmt, cmap='Blues', 
                xticklabels=labels, yticklabels=labels,
                cbar_kws={'label': 'Count' if not normalize else 'Proportion'},
                vmin=0, vmax=vmax, ax=ax, linewidths=0.5, linecolor='gray')
    
    ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    # Rotate labels for better readability
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    
    # Save figure
    output_path = BASE_DIR / filename
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved figure: {output_path}")
    plt.close()


def generate_figures(all_results: Dict[str, Dict[str, Dict]]):
    """Generate the three required confusion matrix figures."""
    if not MATPLOTLIB_AVAILABLE:
        print("Warning: matplotlib/seaborn not available. Install them to generate figures.")
        return
    
    print("\n" + "=" * 80)
    print("GENERATING CONFUSION MATRIX FIGURES")
    print("=" * 80)
    
    # Figure 1: Linear phoneme n-gram classifier (7-way classification)
    model_name = "4_linear_phoneme"
    if model_name in all_results and "test" in all_results[model_name]:
        data = all_results[model_name]["test"]
        cm = data.get("cm") or data.get("confusion_matrix")
        if cm:
            labels = [CLASS_LABELS_7[i] for i in range(7)]
            plot_confusion_matrix_figure(
                cm, labels,
                "Linear Phoneme N-gram Classifier\n(Seven-way Swiss Dialect Classification)",
                "figure1_linear_phoneme_confusion_matrix.png",
                normalize=False,
                figsize=(10, 8)
            )
    
    # Figure 2: Speaker-level binary classification confusion matrix
    model_name = "7_linear_phoneme_binary"  # Using linear phoneme binary
    model_dir = MODELS_DIR / model_name
    preds_file = model_dir / "test_preds.csv"
    
    cm_speaker = compute_speaker_level_confusion_matrix(model_name, "test", preds_file)
    if cm_speaker:
        labels = ["Non-German", "German"]
        plot_confusion_matrix_figure(
            cm_speaker, labels,
            "Speaker-level Binary Classification\n(German vs. Non-German, Phoneme N-grams)",
            "figure2_speaker_level_binary_confusion_matrix.png",
            normalize=False,
            figsize=(8, 6)
        )
    
    # Figure 3: XLS-R 300M embedding-based classifier
    model_name = "7_xlsr_300m_layer8"
    if model_name in all_results and "test" in all_results[model_name]:
        data = all_results[model_name]["test"]
        cm = data.get("cm") or data.get("confusion_matrix")
        if cm:
            labels = [CLASS_LABELS_7[i] for i in range(7)]
            plot_confusion_matrix_figure(
                cm, labels,
                "XLS-R 300M Embedding-based Classifier\n(Seven-way Swiss Dialect Classification)",
                "figure3_xlsr_confusion_matrix.png",
                normalize=False,
                figsize=(10, 8)
            )
            # Generate neighbor confusion by class plot for XLSR
            plot_neighbor_confusion_by_class(
                model_name, cm, labels,
                "neighbor_confusion_by_class_xlsr.png"
            )
    
    # Generate neighbor confusion by class plot for Wav2Vec2
    model_name = "3_wav2vec_base_layer6"
    if model_name in all_results and "test" in all_results[model_name]:
        data = all_results[model_name]["test"]
        cm = data.get("cm") or data.get("confusion_matrix")
        if cm and len(cm) == 7:
            labels = [CLASS_LABELS_7[i] for i in range(7)]
            plot_neighbor_confusion_by_class(
                model_name, cm, labels,
                "neighbor_confusion_by_class_wav2vec.png"
            )
    
    print("\nAll figures generated successfully!")


if __name__ == "__main__":
    main()

