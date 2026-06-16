#!/usr/bin/env python3
"""
Plot neighbor vs non-neighbor confusion for Swiss 7-way dialect classifiers.

Neighbor relationships follow geographic proximity (see comparison_prints_of_all_models.py).
Reads test-set confusion matrices from models/*/test_metrics.json and writes figures to
thesis_eval/figures/ for inclusion in REPORT.md.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
DEFAULT_OUTPUT_DIR = (
    Path(
        "/home/ai/AI-DataPool/Other/Backup/home_dirs/max_150/experiments/TTS/"
        "chatterbox/chatterbox-finetuning/thesis_eval/figures"
    )
)

CLASS_LABELS = [
    "Basel",
    "Bern",
    "Innerschweiz",
    "Ostschweiz",
    "Wallis",
    "Zürich",
    "Graubünden",
]

NEIGHBORS: Dict[int, List[int]] = {
    0: [1],
    1: [0, 2, 4],
    2: [1, 3, 4, 5, 6],
    3: [2, 5, 6],
    4: [1],
    5: [2, 3],
    6: [2, 3],
}

MODEL_NAMES = {
    "1_melspec_cnn": "MelSpec CNN",
    "3_nb_phoneme": "Naive Bayes Phoneme",
    "3_wav2vec_base_layer6": "Wav2Vec2 Base L6",
    "4_linear_phoneme": "Linear Phoneme",
    "7_xlsr_300m_layer8": "XLSR 300M L8",
}

MULTICLASS_MODELS = [
    "3_nb_phoneme",
    "4_linear_phoneme",
    "3_wav2vec_base_layer6",
    "7_xlsr_300m_layer8",
    "1_melspec_cnn",
]


def load_test_confusion(model_id: str) -> Optional[Tuple[List[List[int]], float]]:
    metrics_path = MODELS_DIR / model_id / "test_metrics.json"
    if not metrics_path.exists():
        return None
    data = json.loads(metrics_path.read_text())
    cm = data.get("cm") or data.get("confusion_matrix")
    if not cm or len(cm) != 7:
        return None
    acc = float(data.get("acc") or data.get("accuracy") or 0.0)
    return cm, acc


def analyze_neighbor_confusion(cm: List[List[int]]) -> Dict[str, float]:
    cm_array = np.array(cm, dtype=int)
    correct = neighbor = non_neighbor = 0
    for true_label in range(len(cm_array)):
        neighbors = NEIGHBORS.get(true_label, [])
        for pred_label in range(len(cm_array)):
            value = int(cm_array[true_label, pred_label])
            if true_label == pred_label:
                correct += value
            elif pred_label in neighbors:
                neighbor += value
            else:
                non_neighbor += value
    total = correct + neighbor + non_neighbor
    return {
        "correct": correct,
        "neighbor": neighbor,
        "non_neighbor": non_neighbor,
        "total": total,
        "correct_pct": 100.0 * correct / total if total else 0.0,
        "neighbor_pct": 100.0 * neighbor / total if total else 0.0,
        "non_neighbor_pct": 100.0 * non_neighbor / total if total else 0.0,
    }


def per_class_breakdown(cm: List[List[int]]) -> List[Dict[str, float]]:
    cm_array = np.array(cm, dtype=float)
    rows: List[Dict[str, float]] = []
    for true_label in range(len(cm_array)):
        neighbors = NEIGHBORS.get(true_label, [])
        row_total = float(cm_array[true_label, :].sum())
        if row_total == 0:
            rows.append(
                {
                    "label": CLASS_LABELS[true_label],
                    "correct_pct": 0.0,
                    "neighbor_pct": 0.0,
                    "non_neighbor_pct": 0.0,
                    "neighbor_ratio": 0.0,
                }
            )
            continue

        correct = float(cm_array[true_label, true_label])
        neighbor = sum(float(cm_array[true_label, p]) for p in neighbors if p != true_label)
        non_neighbor = row_total - correct - neighbor
        errors = row_total - correct
        num_neighbors = len(neighbors)
        expected = (num_neighbors / 6.0) * 100.0 if errors > 0 else 0.0
        actual = (neighbor / errors) * 100.0 if errors > 0 else 0.0
        ratio = actual / expected if expected > 0 else 0.0

        rows.append(
            {
                "label": CLASS_LABELS[true_label],
                "correct_pct": 100.0 * correct / row_total,
                "neighbor_pct": 100.0 * neighbor / row_total,
                "non_neighbor_pct": 100.0 * non_neighbor / row_total,
                "neighbor_ratio": ratio,
            }
        )
    return rows


def plot_model_overview(output_dir: Path) -> Optional[Path]:
    model_data = []
    for model_id in MULTICLASS_MODELS:
        loaded = load_test_confusion(model_id)
        if loaded is None:
            continue
        cm, acc = loaded
        stats = analyze_neighbor_confusion(cm)
        model_data.append(
            {
                "model_id": model_id,
                "model": MODEL_NAMES.get(model_id, model_id),
                "acc": acc,
                **stats,
            }
        )

    if not model_data:
        print("No 7-way test confusion matrices found.")
        return None

    model_data.sort(key=lambda d: d["correct_pct"], reverse=True)
    fig, ax = plt.subplots(figsize=(12, 7))
    x = np.arange(len(model_data))
    width = 0.6
    correct = [d["correct"] for d in model_data]
    neighbor = [d["neighbor"] for d in model_data]
    non_neighbor = [d["non_neighbor"] for d in model_data]
    labels = [d["model"] for d in model_data]

    ax.bar(x, correct, width, label="Correct", color="#2ecc71", alpha=0.85)
    ax.bar(x, neighbor, width, bottom=correct, label="Wrong (neighbor)", color="#f39c12", alpha=0.85)
    ax.bar(
        x,
        non_neighbor,
        width,
        bottom=[c + n for c, n in zip(correct, neighbor)],
        label="Wrong (non-neighbor)",
        color="#e74c3c",
        alpha=0.85,
    )

    for i, d in enumerate(model_data):
        ax.text(i, d["correct"] / 2, f"{d['correct_pct']:.1f}%", ha="center", va="center", fontsize=9, color="white")
        top = d["total"]
        ax.text(i, top + 8, f"acc={d['acc']:.3f}", ha="center", va="bottom", fontsize=9)

    ax.set_xlabel("Model")
    ax.set_ylabel("Test samples")
    ax.set_title("Swiss 7-way dialect classification: neighbor confusion breakdown")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    plt.tight_layout()

    out = output_dir / "neighbor_confusion_overview.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")
    return out


def plot_model_by_class(model_id: str, output_dir: Path) -> Optional[Path]:
    loaded = load_test_confusion(model_id)
    if loaded is None:
        print(f"Skipping {model_id}: no 7x7 test confusion matrix.")
        return None

    cm, acc = loaded
    rows = per_class_breakdown(cm)
    fig, ax = plt.subplots(figsize=(13, 7))
    x = np.arange(len(rows))
    width = 0.7
    correct = [r["correct_pct"] for r in rows]
    neighbor = [r["neighbor_pct"] for r in rows]
    non_neighbor = [r["non_neighbor_pct"] for r in rows]
    class_labels = [r["label"] for r in rows]

    ax.bar(x, correct, width, label="Correct", color="#2ecc71", alpha=0.85)
    ax.bar(x, neighbor, width, bottom=correct, label="Wrong (neighbor)", color="#f39c12", alpha=0.85)
    ax.bar(
        x,
        non_neighbor,
        width,
        bottom=[c + n for c, n in zip(correct, neighbor)],
        label="Wrong (non-neighbor)",
        color="#e74c3c",
        alpha=0.85,
    )

    for i, row in enumerate(rows):
        if row["neighbor_pct"] > 1.5:
            ax.text(
                i,
                correct[i] + row["neighbor_pct"] / 2,
                f"×{row['neighbor_ratio']:.2f}",
                ha="center",
                va="center",
                fontsize=8,
                color="white",
            )

    title = MODEL_NAMES.get(model_id, model_id)
    ax.set_title(f"{title} — per-class neighbor confusion (test acc={acc:.3f})")
    ax.set_xlabel("True dialect region")
    ax.set_ylabel("Share of test samples (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(class_labels, rotation=30, ha="right")
    ax.set_ylim(0, 100)
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    plt.tight_layout()

    out = output_dir / f"neighbor_confusion_{model_id}.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for PNG output (default: thesis_eval/figures)",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=["3_nb_phoneme", "4_linear_phoneme"],
        help="Model IDs for per-class plots",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    written = []
    overview = plot_model_overview(args.output_dir)
    if overview:
        written.append(overview)

    for model_id in args.models:
        path = plot_model_by_class(model_id, args.output_dir)
        if path:
            written.append(path)

    if not written:
        raise SystemExit("No figures were written.")
    print(f"Wrote {len(written)} figure(s) to {args.output_dir}")


if __name__ == "__main__":
    main()
