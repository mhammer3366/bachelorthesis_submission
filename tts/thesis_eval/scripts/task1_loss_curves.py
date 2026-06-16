#!/usr/bin/env python3
"""Task 1: Render training loss / LR / grad_norm curves from TensorBoard events + trainer_state."""
from __future__ import annotations

import os
from pathlib import Path

CHATTERBOX_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = Path(os.environ.get("THESIS_ROOT", CHATTERBOX_ROOT.parents[1]))
DATA_ROOT = Path(os.environ.get("DATA_ROOT", "/home/ai/AI-DataPool/Datasets"))
MODELS_ROOT = Path(os.environ.get("MODELS_ROOT", DATA_ROOT.parent / "Models"))
CHECKPOINTS_DIR = Path(os.environ.get("CHECKPOINTS_DIR", MODELS_ROOT / "TTS" / "chatterbox"))
BASE = CHATTERBOX_ROOT


import json
from pathlib import Path

import matplotlib.pyplot as plt
OUT = BASE / "thesis_eval"
FIG = OUT / "figures"

RUNS = {
    "vorarlberg_finetuned_10_epochs": CHECKPOINTS_DIR / "vorarlberg_finetuned_10_epochs",
    "chatterbox_finetuned_vorarlberg_binary": CHECKPOINTS_DIR / "chatterbox_finetuned_vorarlberg_binary",
    "chatterbox_finetuned_überleaba_more_epochs": CHECKPOINTS_DIR / "chatterbox_finetuned_überleaba_more_epochs",
}


def load_from_trainer_state(ckpt_dir: Path) -> pd.DataFrame:
    ts_path = ckpt_dir / "trainer_state.json"
    data = json.loads(ts_path.read_text())
    rows = []
    for entry in data.get("log_history", []):
        if "loss" in entry and "eval_loss" not in entry:
            rows.append(
                {
                    "step": entry.get("step"),
                    "epoch": entry.get("epoch"),
                    "loss": entry.get("loss"),
                    "learning_rate": entry.get("learning_rate"),
                    "grad_norm": entry.get("grad_norm"),
                }
            )
    df = pd.DataFrame(rows).dropna(subset=["step"]).sort_values("step")
    df["run"] = ckpt_dir.name
    return df


def load_from_tfevents(ckpt_dir: Path) -> pd.DataFrame:
    """Supplement with TensorBoard scalars when available."""
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        return pd.DataFrame()

    frames = []
    for ev in sorted((ckpt_dir / "runs").rglob("events.out.tfevents.*")):
        ea = EventAccumulator(str(ev.parent))
        ea.Reload()
        for tag in ea.Tags().get("scalars", []):
            if tag not in ("train/loss", "loss", "train/learning_rate", "learning_rate", "train/grad_norm", "grad_norm"):
                continue
            for e in ea.Scalars(tag):
                frames.append({"step": e.step, "value": e.value, "tag": tag, "run": ckpt_dir.name})
    if not frames:
        return pd.DataFrame()
    return pd.DataFrame(frames)


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    all_dfs = []
    for name, ckpt in RUNS.items():
        if not ckpt.exists():
            print(f"SKIP missing {name}")
            continue
        df = load_from_trainer_state(ckpt)
        all_dfs.append(df)
        print(f"{name}: {len(df)} train log points from trainer_state.json")

    if not all_dfs:
        raise SystemExit("No training logs found.")

    combined = pd.concat(all_dfs, ignore_index=True)
    combined.to_csv(OUT / "train_scalars_all_runs.csv", index=False)
    print(f"Wrote {OUT / 'train_scalars_all_runs.csv'} ({len(combined)} rows)")

    final = combined[combined["run"] == "vorarlberg_finetuned_10_epochs"].copy()
    if final.empty:
        final = all_dfs[0]

    # --- loss vs step (linear y, clipped caption note) ---
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(final["step"], final["loss"], linewidth=0.8, alpha=0.9)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Training loss (step loss)")
    ax.set_title("Vorarlberg 10-epoch run — training loss vs step\n(y clipped at 5; early spike ~20 at step 110)")
    ax.set_ylim(0, 5)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    p1 = FIG / "loss_vs_step_vorarlberg_10epochs.png"
    fig.savefig(p1, dpi=300)
    plt.close(fig)
    print(f"Wrote {p1}")

    # --- log-scale loss ---
    fig, ax = plt.subplots(figsize=(10, 5))
    y = final["loss"].clip(lower=1e-4)
    ax.plot(final["step"], y, linewidth=0.8)
    ax.set_yscale("log")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Training loss (log scale)")
    ax.set_title("Vorarlberg 10-epoch run — training loss (log scale)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    p1b = FIG / "loss_vs_step_vorarlberg_10epochs_log.png"
    fig.savefig(p1b, dpi=300)
    plt.close(fig)
    print(f"Wrote {p1b}")

    # --- learning rate ---
    lr = final.dropna(subset=["learning_rate"])
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(lr["step"], lr["learning_rate"], color="tab:orange")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Learning rate")
    ax.set_title("Vorarlberg 10-epoch run — cosine_with_restarts schedule")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    p2 = FIG / "learning_rate_vs_step_vorarlberg_10epochs.png"
    fig.savefig(p2, dpi=300)
    plt.close(fig)
    print(f"Wrote {p2}")

    # --- comparison overlay ---
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = {"vorarlberg_finetuned_10_epochs": "tab:blue", "chatterbox_finetuned_vorarlberg_binary": "tab:green", "chatterbox_finetuned_überleaba_more_epochs": "tab:red"}
    for run_name, df in zip(RUNS.keys(), all_dfs):
        d = df[df["loss"].notna()].copy()
        d = d[d["loss"] > 0]  # drop zero-loss masking artifacts for readability
        ax.plot(d["step"], d["loss"].clip(upper=5), label=run_name, alpha=0.7, linewidth=0.8, color=colors.get(run_name))
    ax.set_xlabel("Training step")
    ax.set_ylabel("Training loss (clipped at 5)")
    ax.set_title("Vorarlberg-related runs — training loss comparison")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    p3 = FIG / "loss_comparison_vorarlberg_runs.png"
    fig.savefig(p3, dpi=300)
    plt.close(fig)
    print(f"Wrote {p3}")

    print("\nTask 1 complete.")


if __name__ == "__main__":
    main()
