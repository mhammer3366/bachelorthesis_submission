#!/usr/bin/env python3
"""Task 2: Recover validation loss for final checkpoint (eval only, no retraining)."""
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
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

BASE = CHATTERBOX_ROOT
SRC = BASE / "src"
SCRIPTS = BASE / "thesis_eval" / "scripts"
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(SRC))

import eval_bootstrap  # noqa: F401, E402 — patch perth before chatterbox import

from chatterbox.tts import ChatterboxTTS
from finetune_t3 import DataArguments, SpeechFineTuningDataset, T3ForFineTuning
from eval_collator import EvalSpeechDataCollator

CKPT = CHECKPOINTS_DIR / "vorarlberg_finetuned_10_epochs"
TSV = DATA_ROOT / 'audio/Vorarlberg/vorarlberger_daten_16000.tsv'
CACHE = BASE / "thesis_eval" / "cache" / "valid_metadata_rows.json"
OUT = BASE / "thesis_eval" / "final_val_loss.json"

SEED = 42
EVAL_SPLIT = 0.01
BATCH_SIZE = 4
NUM_WORKERS = 0


def load_all_files_like_training(tsv_path: Path) -> list[dict]:
    """Replicate finetune_t3.py metadata loading (lines 492-505)."""
    if CACHE.is_file():
        data = json.loads(CACHE.read_text())
        print(f"Loaded {len(data)} cached rows from {CACHE}")
        return data

    dataset_root = tsv_path.parent
    pairs: list[tuple[str, str]] = []
    with tsv_path.open(encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("|")
            if len(parts) != 2:
                parts = line.strip().split("\t")
            if len(parts) == 2:
                pairs.append((parts[0], parts[1]))

    def resolve(pair: tuple[str, str]) -> dict | None:
        audio_file, text = pair
        audio_path = Path(audio_file) if Path(audio_file).is_absolute() else dataset_root / audio_file
        if audio_path.is_file():
            return {"audio": str(audio_path), "text": text}
        return None

    print(f"Checking {len(pairs)} TSV rows for existing audio …")
    all_files: list[dict] = []
    with ThreadPoolExecutor(max_workers=32) as ex:
        for item in ex.map(resolve, pairs, chunksize=512):
            if item is not None:
                all_files.append(item)

    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(all_files))
    print(f"Cached {len(all_files)} valid rows → {CACHE}")
    return all_files


def split_like_training(all_files: list[dict], seed: int, eval_frac: float) -> tuple[list, list]:
    """Replicate finetune_t3.py lines 515-521."""
    files = list(all_files)
    np.random.seed(seed)
    np.random.shuffle(files)
    split_idx = int(len(files) * (1 - eval_frac))
    if split_idx == 0:
        split_idx = 1
    if split_idx == len(files):
        split_idx = len(files) - 1
    return files[:split_idx], files[split_idx:]


@torch.no_grad()
def evaluate(model: T3ForFineTuning, loader: DataLoader, device: torch.device) -> dict:
    from chatterbox.models.t3.t3 import T3Cond

    model.eval()
    total_text = 0.0
    total_speech = 0.0
    n_batches = 0
    n_samples = 0
    empty_batches = 0

    for batch in loader:
        if not batch:
            empty_batches += 1
            continue
        n_samples += batch["text_tokens"].size(0)
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        t3_cond = T3Cond(
            speaker_emb=batch["t3_cond_speaker_emb"],
            cond_prompt_speech_tokens=batch["t3_cond_prompt_speech_tokens"],
            cond_prompt_speech_emb=None,
            emotion_adv=batch["t3_cond_emotion_adv"],
        ).to(device=device)
        lt, ls, _ = model.t3.loss(
            t3_cond=t3_cond,
            text_tokens=batch["text_tokens"],
            text_token_lens=batch["text_token_lens"],
            speech_tokens=batch["speech_tokens"],
            speech_token_lens=batch["speech_token_lens"],
            labels_text=batch["labels_text"],
            labels_speech=batch["labels_speech"],
        )
        total_text += float(lt.item())
        total_speech += float(ls.item())
        n_batches += 1

    return {
        "n_eval_batches": n_batches,
        "n_eval_samples_in_batches": n_samples,
        "empty_batches_all_masked": empty_batches,
        "mean_loss_text": total_text / max(n_batches, 1),
        "mean_loss_speech": total_speech / max(n_batches, 1),
        "mean_total_loss": (total_text + total_speech) / max(n_batches, 1),
    }


def count_skipped(dataset: SpeechFineTuningDataset, n: int, prompt_len: int) -> tuple[int, int]:
    ok, skip = 0, 0
    for i in range(n):
        item = dataset[i]
        if item is None:
            skip += 1
            continue
        sl = int(item["speech_token_lens"].item())
        if sl - 1 <= prompt_len:
            skip += 1
        else:
            ok += 1
        if (i + 1) % 200 == 0:
            print(f"  scanned {i+1}/{n} eval items …", flush=True)
    return ok, skip


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    all_files = load_all_files_like_training(TSV)
    train_rows, eval_rows = split_like_training(all_files, SEED, EVAL_SPLIT)
    print(f"Total valid rows: {len(all_files)}; train={len(train_rows)} eval={len(eval_rows)}", flush=True)

    data_args = DataArguments(
        metadata_file=str(TSV),
        text_column_name="text",
        audio_column_name="path",
        eval_split_size=EVAL_SPLIT,
    )

    print("Loading model from checkpoint …", flush=True)
    cb = ChatterboxTTS.from_local(ckpt_dir=str(CKPT), device=str(device))
    t3_cfg = cb.t3.hp
    hf_model = T3ForFineTuning(cb.t3, t3_cfg).to(device)
    hf_model.eval()

    eval_ds = SpeechFineTuningDataset(data_args, cb, t3_cfg, eval_rows, is_hf_format=False)
    prompt_len = t3_cfg.speech_cond_prompt_len
    print(f"Scanning {len(eval_ds)} eval items for all-masked labels …", flush=True)
    ok, skip = count_skipped(eval_ds, len(eval_ds), prompt_len)
    print(f"Eval: {ok} with valid speech labels; {skip} skipped (None or all-masked)", flush=True)

    collator = EvalSpeechDataCollator(t3_cfg, t3_cfg.stop_text_token, t3_cfg.stop_speech_token)
    loader = DataLoader(
        eval_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collator,
    )

    print("Running forward pass over validation split …", flush=True)
    metrics = evaluate(hf_model, loader, device)
    metrics.update(
        {
            "checkpoint": str(CKPT),
            "metadata_tsv": str(TSV),
            "seed": SEED,
            "eval_split_size": EVAL_SPLIT,
            "eval_rows_total": len(eval_rows),
            "eval_rows_valid_speech_labels": ok,
            "eval_rows_skipped_all_masked_or_none": skip,
            "note": "Validation loss of final checkpoint only; per-step eval curve during training was NaN and is not recovered.",
        }
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))
    print(f"\nWrote {OUT}")
    print("Task 2 complete.")


if __name__ == "__main__":
    main()
