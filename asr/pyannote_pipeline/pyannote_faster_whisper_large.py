#!/usr/bin/env python3
"""
Batch Audio → Sentence-level Speaker Attribution (parallel over multiple GPUs)
- Mirrors podcast folders under OUTPUT_ROOT.
- Saves per-episode transcripts (CSV + JSON).
- Assigns sentence segments to speakers, computes overlap_coverage.
- No resampling (expects input audio already 16kHz mono).
- Spawns one worker per GPU to process different episodes in parallel.
- Shows a single global progress bar with ETA.
"""

# =========================
# ======= SETTINGS ========
# =========================

# ---- paths ----
INPUT_ROOT  = "/home/ai/AI-DataPool/Datasets/audio/Schweiz/srf_audio_downloads"
OUTPUT_ROOT = "/home/ai/AI-DataPool/Datasets/audio/Schweiz/srf_audio_downloads_outputs"

# ---- model + decoding ----
HF_TOKEN       = "hf_xxxxx"   # replace with your HF token
WHISPER_MODEL  = "large-v3"
LANGUAGE       = "de"        # set None to auto-detect
USE_VAD        = False
WHISPER_COMPUTE_TYPE = "float16"

# ---- batching / parallelism inside faster-whisper ----
BATCH_SIZE  = 8
NUM_WORKERS = 4
CPU_THREADS = 4

# ---- diarization ----
MIN_SPEAKERS = 2
MAX_SPEAKERS = 6

# ---- overlap computation ----
OVERLAP_STEP_SEC = 0.01     # 10ms

# ---- audio discovery ----
AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".flac"}

# ---- multi-GPU parallelism ----
GPU_IDS = [0, 1, 2, 3, 4]   # use 5 GPUs
WORKERS_PER_GPU = 1
SHARDING_MODE = "roundrobin"  # or "chunk"

# ---- logging ----
LOG_LEVEL = "INFO"
LIB_LOG_LEVEL = "WARNING"

# =========================
# ========= CODE ==========
# =========================

import os, sys, json, time, logging
from math import ceil
from pathlib import Path
from datetime import datetime
from multiprocessing import Process, Manager

import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm

import torch

try:
    from faster_whisper import WhisperModel
    from pyannote.audio import Pipeline
    from pyannote.core import Annotation, Segment
except ImportError as e:
    print(f"Missing required library: {e}")
    sys.exit(1)


def setup_logging(log_path: Path):
    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL),
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.FileHandler(log_path, mode="w", encoding="utf-8"),
                  logging.StreamHandler(sys.stdout)],
    )
    for name in ("faster_whisper", "ctranslate2", "pyannote", "torch", "numba"):
        logging.getLogger(name).setLevel(getattr(logging, LIB_LOG_LEVEL))
    return logging.getLogger("audio_batch_pipeline")


logger = logging.getLogger("audio_batch_pipeline")


def ensure_dirs():
    Path(OUTPUT_ROOT).mkdir(parents=True, exist_ok=True)


def init_models(device_whisper: str, device_whisper_index: int, device_pyannote: str):
    if (device_whisper.startswith("cuda") or device_pyannote.startswith("cuda")) and not torch.cuda.is_available():
        logger.error("CUDA requested but not available.")
        sys.exit(1)

    logger.info("Loading Whisper model...")
    whisper_kwargs = dict(
        device=device_whisper,
        device_index=device_whisper_index,
        compute_type=WHISPER_COMPUTE_TYPE,
        num_workers=NUM_WORKERS,
        cpu_threads=CPU_THREADS,
    )
    whisper = WhisperModel(WHISPER_MODEL, max_batch_size=BATCH_SIZE, **whisper_kwargs)

    if not HF_TOKEN:
        logger.error("HF token required for pyannote.")
        sys.exit(1)

    logger.info("Loading pyannote diarization pipeline...")
    diar = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1",
                                    use_auth_token=HF_TOKEN)
    diar.to(torch.device(device_pyannote))
    return whisper, diar


def run_diarization(diar, wav_path: Path) -> Annotation:
    return diar(str(wav_path), min_speakers=MIN_SPEAKERS, max_speakers=MAX_SPEAKERS)


def run_transcription(whisper, wav_path: Path):
    segs, info = whisper.transcribe(
        str(wav_path),
        language=LANGUAGE,
        vad_filter=USE_VAD,
        beam_size=5,
    )
    out = []
    for s in segs:
        out.append({"start": float(s.start), "end": float(s.end), "text": s.text.strip()})
    return out, info


def segment_overlap_with_label(annotation: Annotation, segment: Segment, label: str) -> float:
    total = 0.0
    for (seg, _, lab) in annotation.itertracks(yield_label=True):
        if lab != label:
            continue
        inter = seg & segment
        if inter is not None:
            total += inter.duration
    return total


def assign_speaker_to_segment(annotation: Annotation, segment: Segment) -> str:
    label_durations = {}
    for lab in annotation.labels():
        dur = segment_overlap_with_label(annotation, segment, lab)
        if dur > 0:
            label_durations[lab] = label_durations.get(lab, 0.0) + dur
    if not label_durations:
        return "UNKNOWN"
    return max(label_durations.items(), key=lambda kv: kv[1])[0]


def overlap_coverage(annotation: Annotation, segment: Segment, step: float = OVERLAP_STEP_SEC) -> float:
    if segment.duration <= 0:
        return 0.0
    t = segment.start
    overlap_frames = 0
    total_frames = 0
    labeled = list(annotation.itertracks(yield_label=True))
    while t < segment.end:
        active = 0
        for (seg, _, _) in labeled:
            if seg.start <= t < seg.end:
                active += 1
                if active >= 2:
                    break
        if active >= 2:
            overlap_frames += 1
        total_frames += 1
        t += step
    return (overlap_frames / total_frames) if total_frames else 0.0


def discover_audio_files(root: Path):
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in AUDIO_EXTS:
            rel = path.relative_to(root)
            podcast = rel.parts[0] if len(rel.parts) > 0 else "unknown"
            yield podcast, path


def ensure_transcripts_dir(podcast: str) -> Path:
    pod_dir = Path(OUTPUT_ROOT) / podcast
    trans_dir = pod_dir / "transcripts"
    trans_dir.mkdir(parents=True, exist_ok=True)
    return trans_dir


def process_episode(whisper, diar, podcast: str, episode_path: Path, manifest_path: Path):
    try:
        trans_dir = ensure_transcripts_dir(podcast)
        ep_stem = episode_path.stem

        logger.info(f"[{podcast}] Diarization: {episode_path.name}")
        diarization = run_diarization(diar, episode_path)

        logger.info(f"[{podcast}] Transcription: {episode_path.name}")
        segments, info = run_transcription(whisper, episode_path)

        aligned_rows = []
        for seg in segments:
            seg_obj = Segment(seg["start"], seg["end"])
            spk = assign_speaker_to_segment(diarization, seg_obj)
            ov = overlap_coverage(diarization, seg_obj, step=OVERLAP_STEP_SEC)
            aligned_rows.append({
                "start": round(seg["start"], 3),
                "end": round(seg["end"], 3),
                "speaker": spk,
                "text": seg["text"],
                "overlap_coverage": round(ov, 3),
            })

        # save transcripts
        csv_path = (Path(trans_dir) / ep_stem).with_suffix(".csv")
        json_path = (Path(trans_dir) / ep_stem).with_suffix(".json")

        df = pd.DataFrame(aligned_rows, columns=["start", "end", "speaker", "text", "overlap_coverage"])
        df.to_csv(csv_path, index=False, encoding="utf-8")

        meta = {
            "podcast": podcast,
            "episode": ep_stem,
            "episode_file": str(episode_path.resolve()),
            "duration": librosa.get_duration(path=str(episode_path)),
            "num_speakers": len(diarization.labels()),
            "language": getattr(info, "language", None),
            "language_probability": getattr(info, "language_probability", None),
            "segments": aligned_rows,
            "timestamp": datetime.now().isoformat(),
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        # append to worker manifest
        row = {
            "podcast": podcast,
            "episode": ep_stem,
            "original_path": str(episode_path.resolve()),
            "csv": str(csv_path),
            "json": str(json_path),
            "duration_sec": round(meta["duration"], 3) if meta["duration"] else "",
            "num_speakers": meta["num_speakers"],
            "language": meta["language"],
            "language_probability": round(meta["language_probability"], 3) if meta["language_probability"] else "",
        }
        pd.DataFrame([row]).to_csv(manifest_path, mode="a", index=False, header=not manifest_path.exists())

        return row

    except Exception as e:
        logger.error(f"Failed to process {episode_path}: {e}", exc_info=False)
        return None


def shard_list(items, num_shards, mode="roundrobin"):
    if num_shards <= 1:
        return [items]
    if mode == "chunk":
        size = ceil(len(items) / num_shards)
        return [items[i*size:(i+1)*size] for i in range(num_shards)]
    shards = [[] for _ in range(num_shards)]
    for i, it in enumerate(items):
        shards[i % num_shards].append(it)
    return shards


def worker_run(gpu_id: int, file_tuples, shared):
    device_whisper = "cuda"
    device_whisper_index = gpu_id
    device_pyannote = f"cuda:{gpu_id}"

    try:
        torch.set_num_threads(1)
    except Exception:
        pass

    logger.info(f"[GPU{gpu_id}] starting worker with {len(file_tuples)} files")
    whisper, diar = init_models(device_whisper, device_whisper_index, device_pyannote)

    manifest_path = Path(OUTPUT_ROOT) / f"_manifest.gpu{gpu_id}.csv"

    for podcast, ep in file_tuples:
        t0 = time.time()
        _ = process_episode(whisper, diar, podcast, ep, manifest_path)
        dt = time.time() - t0
        with shared["lock"]:
            shared["done"].value += 1
            shared["secs"].value += dt

    logger.info(f"[GPU{gpu_id}] worker done.")


def main():
    ensure_dirs()

    log_file = Path(OUTPUT_ROOT) / f"batch_audio_processing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    global logger
    logger = setup_logging(log_file)

    input_root = Path(INPUT_ROOT)
    if not input_root.exists():
        logger.error(f"Input root not found: {INPUT_ROOT}")
        sys.exit(1)

    files = list(discover_audio_files(input_root))
    if not files:
        logger.warning("No audio files discovered.")
        return
    total = len(files)
    logger.info(f"Discovered {total} audio files.")

    num_workers = len(GPU_IDS) * WORKERS_PER_GPU
    shards = shard_list(files, num_workers, mode=SHARDING_MODE)

    manager = Manager()
    shared = {"done": manager.Value('i', 0),
              "secs": manager.Value('d', 0.0),
              "lock": manager.Lock()}

    procs = []
    try:
        for i in range(num_workers):
            gpu_id = GPU_IDS[i % len(GPU_IDS)]
            p = Process(target=worker_run, args=(gpu_id, shards[i], shared))
            p.start()
            procs.append(p)

        with tqdm(total=total, desc="Processing episodes", unit="ep") as bar:
            last_done = 0
            while any(p.is_alive() for p in procs):
                done = shared["done"].value
                delta = done - last_done
                if delta > 0:
                    bar.update(delta)
                    last_done = done
                done_now = max(done, 1)
                avg_sec = shared["secs"].value / done_now
                remaining = total - done
                eta_sec = max(0.0, avg_sec * remaining)
                bar.set_postfix_str(f"ETA ~ {eta_sec/3600:.1f} h (avg {avg_sec/60:.1f} min/ep)")
                time.sleep(2)

            done = shared["done"].value
            bar.update(max(0, done - last_done))
            done_now = max(done, 1)
            avg_sec = shared["secs"].value / done_now
            bar.set_postfix_str(f"ETA ~ 0.0 h (avg {avg_sec/60:.1f} min/ep)")

        for p in procs:
            p.join()

    except KeyboardInterrupt:
        for p in procs:
            p.terminate()
        for p in procs:
            p.join()
        raise

    manifest_path = Path(OUTPUT_ROOT) / "_manifest.csv"
    parts = sorted(Path(OUTPUT_ROOT).glob("_manifest.gpu*.csv"))
    if parts:
        mdfs = [pd.read_csv(p) for p in parts if p.exists()]
        if mdfs:
            pd.concat(mdfs, ignore_index=True).to_csv(manifest_path, index=False, encoding="utf-8")
            logger.info(f"Merged manifest: {manifest_path}")
        else:
            logger.warning("Partial manifests found but empty.")
    else:
        logger.warning("No partial manifests to merge.")

    logger.info("Batch processing complete.")


if __name__ == "__main__":
    main()
