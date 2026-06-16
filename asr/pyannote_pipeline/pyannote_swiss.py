#!/usr/bin/env python3
"""
Batch Audio → Sentence-level Speaker Attribution (parallel over multiple GPUs)
- Mirrors podcast folders under OUTPUT_ROOT.
- Saves per-episode transcripts (CSV + JSON).
- Assigns sentence segments to speakers, computes overlap_coverage.
- No resampling (expects input audio already 16kHz mono).
- Spawns multiple workers per GPU to process different episodes in parallel.
- Shows a single global progress bar with ETA.
- Optimized load balancing and performance improvements.
"""

# =========================
# ======= SETTINGS ========
# =========================

import warnings
warnings.filterwarnings("ignore", message=".*MPEG_LAYER_III.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*std\\(\\): degrees of freedom.*", category=UserWarning)
import argparse
import os
from itertools import islice
# ---- paths ----
INPUT_ROOT  = str(Path(os.environ.get("DATA_ROOT", "/home/ai/AI-DataPool/Datasets")) / "audio/Schweiz/16000_mono_wav")
OUTPUT_ROOT = str(Path(os.environ.get("DATA_ROOT", "/home/ai/AI-DataPool/Datasets")) / "audio/Schweiz/16000_mono_wav_pyannote")

# ---- model + decoding ----
HF_TOKEN = os.environ.get("HF_TOKEN", "")   # replace with your HF token
WHISPER_MODEL  = "large-v3"
LANGUAGE       = "de"        # set None to auto-detect
USE_VAD        = False
WHISPER_COMPUTE_TYPE = "float16"  # Changed from float16 for speed

# ---- batching / parallelism inside faster-whisper ----
BATCH_SIZE  = 24    # Increased from 8
NUM_WORKERS = 4
CPU_THREADS = 20     # Reduced to leave more resources for other workers

# ---- diarization ----
MIN_SPEAKERS = 2
MAX_SPEAKERS = 6

# ---- overlap computation ----
OVERLAP_STEP_SEC = 0.2     # 10ms

# ---- audio discovery ----
AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".flac"}

# ---- multi-GPU parallelism ----
GPU_IDS = [0, 1, 2, 3, 4]   # use 5 GPUs
WORKERS_PER_GPU = 2         # Increased from 1
SHARDING_MODE = "balanced"  # Changed from roundrobin

# ---- logging ----
LOG_LEVEL = "INFO"
LIB_LOG_LEVEL = "WARNING"

# ---- performance settings ----
ENABLE_CHECKPOINTING = True     # Resume from previous runs
CLEAR_GPU_MEMORY = True         # Clear memory after each episode
USE_FAST_DURATION = True        # Use mutagen instead of librosa for duration

# =========================
# ========= CODE ==========
# =========================

import os, sys, json, time, logging
from math import ceil
from pathlib import Path
from datetime import datetime
from multiprocessing import Process, Manager, Queue

import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm

import torch

try:
    from faster_whisper import WhisperModel
    from pyannote.audio import Pipeline
    from pyannote.core import Annotation, Segment
    import mutagen
except ImportError as e:
    print(f"Missing required library: {e}")
    print("Install missing libraries with: pip install mutagen")
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


def validate_setup():
    """Validate system requirements and configuration."""
    if not torch.cuda.is_available():
        logger.error("CUDA not available")
        sys.exit(1)
        
    available_gpus = torch.cuda.device_count()
    if max(GPU_IDS) >= available_gpus:
        logger.error(f"GPU {max(GPU_IDS)} not available. Only {available_gpus} GPUs detected")
        sys.exit(1)
        
    if not HF_TOKEN or HF_TOKEN == "your-token-here":
        logger.error("Valid HF_TOKEN required for pyannote")
        sys.exit(1)
        
    logger.info(f"Validated setup: {available_gpus} GPUs available, using GPUs {GPU_IDS}")


def ensure_dirs():
    Path(OUTPUT_ROOT).mkdir(parents=True, exist_ok=True)


def clear_gpu_memory():
    """Clear GPU memory cache."""
    if CLEAR_GPU_MEMORY and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def get_audio_duration_fast(path: Path) -> float:
    """Get audio duration quickly using mutagen, fallback to librosa."""
    if USE_FAST_DURATION:
        try:
            audio_file = mutagen.File(str(path))
            if audio_file and hasattr(audio_file, 'info') and hasattr(audio_file.info, 'length'):
                return audio_file.info.length
        except Exception:
            pass
    
    # Fallback to librosa
    try:
        return librosa.get_duration(path=str(path))
    except Exception:
        return 0.0


def init_models(device_whisper: str, device_whisper_index: int, device_pyannote: str):
    if (device_whisper.startswith("cuda") or device_pyannote.startswith("cuda")) and not torch.cuda.is_available():
        logger.error("CUDA requested but not available.")
        sys.exit(1)

    logger.info(f"Loading Whisper model on GPU {device_whisper_index}...")
    whisper_kwargs = dict(
        device=device_whisper,
        device_index=device_whisper_index,
        compute_type=WHISPER_COMPUTE_TYPE,
        num_workers=NUM_WORKERS,
        cpu_threads=CPU_THREADS,
    )
    whisper = WhisperModel(WHISPER_MODEL, **whisper_kwargs)

    logger.info(f"Loading pyannote diarization pipeline on {device_pyannote}...")
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
        beam_size=3,  # Reduced for speed
        #batch_size=BATCH_SIZE,
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


def discover_audio_files_with_duration(root: Path):
    """Discover audio files and get their durations for load balancing."""
    files = []
    logger.info("Discovering and analyzing audio files...")
    
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in AUDIO_EXTS:
            rel = path.relative_to(root)
            podcast = rel.parts[0] if len(rel.parts) > 0 else "unknown"
            duration = get_audio_duration_fast(path)
            files.append((podcast, path, duration))
    
    # Sort by duration descending for better load balancing
    files.sort(key=lambda x: x[2], reverse=True)
    logger.info(f"Discovered {len(files)} audio files, total duration: {sum(f[2] for f in files)/3600:.1f} hours")
    
    return [(podcast, path) for podcast, path, _ in files]


def get_processed_episodes() -> set:
    """Get set of already processed episodes for checkpointing."""
    if not ENABLE_CHECKPOINTING:
        return set()
        
    processed = set()
    manifest_files = list(Path(OUTPUT_ROOT).glob("_manifest*.csv"))
    
    for manifest_file in manifest_files:
        try:
            df = pd.read_csv(manifest_file)
            for _, row in df.iterrows():
                processed.add(row['original_path'])
        except Exception:
            continue
    
    if processed:
        logger.info(f"Found {len(processed)} previously processed episodes")
    
    return processed


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

        duration = get_audio_duration_fast(episode_path)
        meta = {
            "podcast": podcast,
            "episode": ep_stem,
            "episode_file": str(episode_path.resolve()),
            "duration": duration,
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
            "duration_sec": round(duration, 3) if duration else "",
            "num_speakers": meta["num_speakers"],
            "language": meta["language"],
            "language_probability": round(meta["language_probability"], 3) if meta["language_probability"] else "",
        }
        pd.DataFrame([row]).to_csv(manifest_path, mode="a", index=False, header=not manifest_path.exists())

        # Clear GPU memory after each episode
        clear_gpu_memory()
        
        return row

    except Exception as e:
        logger.error(f"Failed to process {episode_path}: {e}", exc_info=False)
        clear_gpu_memory()  # Clear memory even on failure
        return None


def shard_list_balanced(items, num_shards, mode="balanced"):
    """Improved load balancing using file sizes."""
    if num_shards <= 1:
        return [items]
        
    if mode == "chunk":
        size = ceil(len(items) / num_shards)
        return [items[i*size:(i+1)*size] for i in range(num_shards)]
    elif mode == "roundrobin":
        shards = [[] for _ in range(num_shards)]
        for i, item in enumerate(items):
            shards[i % num_shards].append(item)
        return shards
    elif mode == "balanced":
        # For balanced mode, we assume items are already sorted by duration (descending)
        # Use greedy assignment to minimize load imbalance
        shards = [[] for _ in range(num_shards)]
        shard_loads = [0] * num_shards
        
        for item in items:
            # Assign to shard with minimum current load
            min_shard = min(range(num_shards), key=lambda i: shard_loads[i])
            shards[min_shard].append(item)
            # Assume uniform load (could be improved with actual durations)
            shard_loads[min_shard] += 1
            
        return shards
    
    return [items]  # fallback
def discover_audio_files_fast(root: Path, limit: int = 0, pattern: str | None = None):
    """
    Fast: yields (podcast, path) without computing durations, stops at `limit` if > 0.
    Optional `pattern` substring filter on full path.
    """
    count = 0
    for path in root.rglob("*"):
        if not (path.is_file() and path.suffix.lower() in AUDIO_EXTS):
            continue
        if pattern and pattern.lower() not in str(path).lower():
            continue
        rel = path.relative_to(root)
        podcast = rel.parts[0] if len(rel.parts) > 0 else "unknown"
        yield (podcast, path)
        count += 1
        if limit and count >= limit:
            break


def worker_run(gpu_id: int, file_tuples, shared):
    device_whisper = "cuda"
    device_whisper_index = gpu_id
    device_pyannote = f"cuda:{gpu_id}"

    try:
        torch.set_num_threads(1)
    except Exception:
        pass

    logger.info(f"[GPU{gpu_id}] starting worker with {len(file_tuples)} files")
    
    try:
        whisper, diar = init_models(device_whisper, device_whisper_index, device_pyannote)
    except Exception as e:
        logger.error(f"[GPU{gpu_id}] Failed to initialize models: {e}")
        return

    manifest_path = Path(OUTPUT_ROOT) / f"_manifest.gpu{gpu_id}.csv"

    for podcast, ep in file_tuples:
        t0 = time.time()
        result = process_episode(whisper, diar, podcast, ep, manifest_path)
        dt = time.time() - t0
        
        with shared["lock"]:
            shared["done"].value += 1
            shared["secs"].value += dt
            if result:
                shared["success"].value += 1
            else:
                shared["failed"].value += 1

    #logger.info(f"[GPU{gpu_id}] worker done.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="Process at most N files (fast path, no duration scan).")
    parser.add_argument("--pattern", type=str, default="", help="Only process files whose path contains this substring.")
    args = parser.parse_args()

    validate_setup()
    ensure_dirs()

    log_file = Path(OUTPUT_ROOT) / f"batch_audio_processing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    global logger
    logger = setup_logging(log_file)

    input_root = Path(INPUT_ROOT)
    if not input_root.exists():
        logger.error(f"Input root not found: {INPUT_ROOT}")
        sys.exit(1)

    # Remove the [:50] limit for production use
    if args.limit > 0:
        logger.info(f"Fast discovery: grabbing up to {args.limit} files (no duration scan)...")
        files = list(discover_audio_files_fast(input_root, limit=args.limit, pattern=args.pattern or None))
    else:
        files = discover_audio_files_with_duration(input_root)
    
    # Filter out already processed files if checkpointing is enabled
    processed = get_processed_episodes()
    if processed:
        original_count = len(files)
        files = [(podcast, path) for podcast, path in files if str(path.resolve()) not in processed]
        logger.info(f"Filtered out {original_count - len(files)} already processed files")
    
    if not files:
        logger.warning("No audio files to process.")
        return
    
    total = len(files)
    logger.info(f"Processing {total} audio files with {WORKERS_PER_GPU} workers per GPU")

    num_workers = len(GPU_IDS) * WORKERS_PER_GPU
    shards = shard_list_balanced(files, num_workers, mode=SHARDING_MODE)

    manager = Manager()
    shared = {
        "done": manager.Value('i', 0),
        "success": manager.Value('i', 0),
        "failed": manager.Value('i', 0),
        "secs": manager.Value('d', 0.0),
        "lock": manager.Lock()
    }

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
                success = shared["success"].value
                failed = shared["failed"].value
                delta = done - last_done
                if delta > 0:
                    bar.update(delta)
                    last_done = done
                done_now = max(done, 1)
                avg_sec = shared["secs"].value / done_now
                remaining = total - done
                eta_sec = max(0.0, avg_sec * remaining)
                bar.set_postfix_str(f"Success: {success}, Failed: {failed}, ETA ~ {eta_sec/3600:.1f}h (avg {avg_sec/60:.1f}min/ep)")
                time.sleep(2)

            done = shared["done"].value
            success = shared["success"].value
            failed = shared["failed"].value
            bar.update(max(0, done - last_done))
            done_now = max(done, 1)
            avg_sec = shared["secs"].value / done_now
            bar.set_postfix_str(f"DONE - Success: {success}, Failed: {failed} (avg {avg_sec/60:.1f}min/ep)")

        for p in procs:
            p.join()

    except KeyboardInterrupt:
        logger.info("Interrupt received, terminating workers...")
        for p in procs:
            p.terminate()
        for p in procs:
            p.join()
        raise

    # Merge all manifests
    manifest_path = Path(OUTPUT_ROOT) / "_manifest.csv"
    parts = sorted(Path(OUTPUT_ROOT).glob("_manifest.gpu*.csv"))
    if parts:
        mdfs = []
        for part in parts:
            try:
                df = pd.read_csv(part)
                if not df.empty:
                    mdfs.append(df)
            except Exception as e:
                logger.warning(f"Could not read {part}: {e}")
        
        if mdfs:
            final_df = pd.concat(mdfs, ignore_index=True)
            final_df.to_csv(manifest_path, index=False, encoding="utf-8")
            logger.info(f"Merged manifest with {len(final_df)} episodes: {manifest_path}")
        else:
            logger.warning("All partial manifests were empty.")
    else:
        logger.warning("No partial manifests found to merge.")

    logger.info("Batch processing complete.")


if __name__ == "__main__":
    main()
