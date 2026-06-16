import os
import subprocess
import multiprocessing as mp
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging
import time
from typing import List, Tuple
import argparse

# =========================
# Defaults (CLI can override)
# =========================
INPUT_ROOT = Path(os.environ.get("DATA_ROOT", "/home/ai/AI-DataPool/Datasets")) / "audio/Schweiz/srf_audio_downloads"
OUTPUT_ROOT = Path(os.environ.get("DATA_ROOT", "/home/ai/AI-DataPool/Datasets")) / "audio/Schweiz/16000_mono_wav"

TARGET_SR = 16000
CHANNELS = 1
OUTPUT_EXT = ".wav"
AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".aac", ".wma", ".mkv", ".mp4", ".opus", ".webm"}

MAX_WORKERS = min(mp.cpu_count() * 2, 128)
BATCH_SIZE = 1000

# Timeout config
TIMEOUT_FACTOR = 2.0      # seconds of timeout per second of audio
MIN_TIMEOUT_SEC = 1800    # 30 minutes floor
RETRIES_ON_TIMEOUT = 1    # retry attempts on timeout (with backoff)

# ffmpeg threading: keep at 1 when running many workers
FFMPEG_THREADS = 1

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# =========================
# Helpers
# =========================
def get_audio_files(input_dir: Path) -> List[Path]:
    """Recursively find all audio files in the input directory."""
    audio_files = []
    for root, dirs, files in os.walk(input_dir):
        root_path = Path(root)
        for file in files:
            p = root_path / file
            if p.suffix.lower() in AUDIO_EXTS:
                audio_files.append(p)
    return sorted(audio_files)

def get_output_path(input_path: Path, input_root: Path, output_root: Path) -> Path:
    relative_path = input_path.relative_to(input_root)
    return output_root / relative_path.with_suffix(OUTPUT_EXT)

def get_file_duration(input_path: Path) -> float:
    """Return duration in seconds using ffprobe; 0.0 if unknown/error."""
    try:
        cmd = [
            'ffprobe', '-v', 'quiet',
            '-show_entries', 'format=duration',
            '-of', 'csv=p=0',
            str(input_path)
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if res.returncode == 0:
            return float(res.stdout.strip())
    except Exception:
        pass
    return 0.0

def _run_ffmpeg_once(cmd: list, timeout_sec: int) -> Tuple[bool, str]:
    """
    Run ffmpeg once. Returns (success, error_message_if_any).
    """
    try:
        res = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            check=False
        )
        if res.returncode == 0:
            return True, ""
        err = (res.stderr or "").strip()
        if "Invalid data found" in err:
            return False, "Corrupted audio file"
        if "No such file or directory" in err:
            return False, "Input file not found"
        return False, f"FFmpeg error: {err[:300]}"
    except subprocess.TimeoutExpired:
        mins = max(1, timeout_sec // 60)
        return False, f"Timeout ({mins} minutes)"
    except Exception as e:
        return False, str(e)

def _compute_timeout(duration: float, factor: float, min_timeout: int,
                     workers: int, ffmpeg_threads: int, cpu_cores: int) -> int:
    """
    Oversubscription-aware timeout.
    If duration==0 (ffprobe failed), use min_timeout as base.
    """
    oversub = max(1.0, (workers * max(1, ffmpeg_threads)) / max(1, cpu_cores))
    base = duration if duration > 0 else (min_timeout / max(factor, 0.1))
    return max(min_timeout, int(base * factor * oversub))


# =========================
# Worker
# =========================
def process_audio_file_ultra_fast(
    args: Tuple[Path, Path, int, int, str, bool, float, int, int, int, int, int]
) -> Tuple[bool, str, str]:
    """
    Process a single file with ffmpeg, adaptive timeout, retry on timeout.

    Returns: (success, input_path_str, error_message_if_any)
    """
    (input_path, output_path, target_sr, channels, output_ext, force_overwrite,
     timeout_factor, min_timeout_sec, ffmpeg_threads, workers, cpu_cores, retries_on_timeout) = args

    try:
        # Ensure out dir
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Skip if not forcing and output newer than input
        if not force_overwrite:
            if output_path.exists() and output_path.stat().st_mtime > input_path.stat().st_mtime:
                return True, str(input_path), "Already processed (up to date)"

        # Compute timeout
        duration = get_file_duration(input_path)
        timeout_sec = _compute_timeout(duration, timeout_factor, min_timeout_sec,
                                       workers, ffmpeg_threads, cpu_cores)

        # Build ffmpeg command
        cmd = [
            'ffmpeg',
            '-nostdin',
            '-i', str(input_path),
            '-vn',                 # ignore video
            '-map', 'a:0',         # ensure first audio stream
            '-ar', str(target_sr),
            '-ac', str(channels),
            '-c:a', 'pcm_s16le',
            '-y',
            '-loglevel', 'error',  # keep errors visible
            '-threads', str(ffmpeg_threads),
            '-fflags', '+genpts',
            '-avoid_negative_ts', 'make_zero',
            '-f', 'wav',
            str(output_path)
        ]

        # Try run + retries on timeout
        attempts = 0
        last_err = ""
        current_timeout = timeout_sec

        while True:
            ok, err = _run_ffmpeg_once(cmd, current_timeout)
            attempts += 1

            if ok:
                # sanity check output
                if output_path.exists() and output_path.stat().st_size > 1000:
                    return True, str(input_path), ""
                return False, str(input_path), "Output file too small or missing"

            last_err = err

            # Retry only on timeout
            if "Timeout (" in err and attempts <= retries_on_timeout:
                # Exponential backoff of timeout
                current_timeout = int(current_timeout * 2)
                continue

            # Not a timeout or retries exhausted
            return False, str(input_path), err

    except Exception as e:
        return False, str(input_path), str(e)


# =========================
# Orchestrator
# =========================
def process_files_ultra_fast(
    audio_files: List[Path],
    input_root: Path,
    output_root: Path,
    force_overwrite: bool,
    timeout_factor: float,
    min_timeout_sec: int,
    ffmpeg_threads: int,
    workers: int,
    batch_size: int,
    retries_on_timeout: int
) -> None:

    cpu_cores = mp.cpu_count() or 1

    # Prepare worker args
    work_items = []
    for in_p in audio_files:
        out_p = get_output_path(in_p, input_root, output_root)
        work_items.append((
            in_p, out_p, TARGET_SR, CHANNELS, OUTPUT_EXT, force_overwrite,
            timeout_factor, min_timeout_sec, ffmpeg_threads, workers, cpu_cores, retries_on_timeout
        ))

    logger.info(f"Starting ultra-fast processing of {len(audio_files)} files...")
    start = time.time()

    successful = failed = skipped = 0
    timeout_errors = corrupted_files = 0

    total_batches = (len(work_items) + batch_size - 1) // batch_size

    for b in range(total_batches):
        s = b * batch_size
        e = min(s + batch_size, len(work_items))
        batch = work_items[s:e]

        logger.info(f"Processing batch {b + 1}/{total_batches} ({len(batch)} files)")

        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(process_audio_file_ultra_fast, item): item for item in batch}
            with tqdm(total=len(batch), desc=f"Batch {b + 1}/{total_batches}", unit="file") as pbar:
                for fut in as_completed(futures):
                    ok, in_path_str, msg = fut.result()
                    if ok:
                        if "Already processed" in msg:
                            skipped += 1
                        else:
                            successful += 1
                    else:
                        failed += 1
                        if "Timeout" in msg:
                            timeout_errors += 1
                        elif "Corrupted" in msg:
                            corrupted_files += 1
                        if msg:
                            logger.error(f"Failed to process {in_path_str}: {msg}")

                    pbar.update(1)
                    pbar.set_postfix({
                        'Success': successful,
                        'Failed': failed,
                        'Skipped': skipped,
                        'Timeouts': timeout_errors,
                        'Corrupted': corrupted_files
                    })

    total = time.time() - start
    if audio_files:
        logger.info(f"Ultra-fast processing completed in {total:.2f} seconds")
        logger.info(f"Results: {successful} successful, {failed} failed, {skipped} skipped")
        logger.info(f"Error breakdown: {timeout_errors} timeouts, {corrupted_files} corrupted files")
        logger.info(f"Average time per file: {total/len(audio_files):.2f} seconds")
        logger.info(f"Processing speed: {len(audio_files)/total:.2f} files/second")


# =========================
# Main
# =========================
def main():
    global INPUT_ROOT, OUTPUT_ROOT, MAX_WORKERS, BATCH_SIZE
    global TIMEOUT_FACTOR, MIN_TIMEOUT_SEC, FFMPEG_THREADS, RETRIES_ON_TIMEOUT

    parser = argparse.ArgumentParser(description='Ultra-fast audio resampling with adaptive timeouts')
    parser.add_argument('--input', type=str, default=str(INPUT_ROOT), help='Input directory')
    parser.add_argument('--output', type=str, default=str(OUTPUT_ROOT), help='Output directory')
    parser.add_argument('--workers', type=int, default=MAX_WORKERS, help='Number of parallel workers (processes)')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE, help='Batch size')
    parser.add_argument('--force', action='store_true', help='Force reprocessing (ignore existing outputs)')
    parser.add_argument('--timeout-factor', type=float, default=TIMEOUT_FACTOR,
                        help='Timeout factor (seconds per second of audio). Default: 2.0')
    parser.add_argument('--min-timeout', type=int, default=MIN_TIMEOUT_SEC,
                        help='Minimum timeout per file in seconds. Default: 1800 (30 minutes)')
    parser.add_argument('--ffmpeg-threads', type=int, default=FFMPEG_THREADS,
                        help='Threads per ffmpeg process. Default: 1')
    parser.add_argument('--retries', type=int, default=RETRIES_ON_TIMEOUT,
                        help='Retries on timeout (exponential backoff). Default: 1')

    args = parser.parse_args()

    # Apply CLI overrides
    INPUT_ROOT = Path(args.input)
    OUTPUT_ROOT = Path(args.output)
    MAX_WORKERS = args.workers
    BATCH_SIZE = args.batch_size
    TIMEOUT_FACTOR = args.timeout_factor
    MIN_TIMEOUT_SEC = args.min_timeout
    FFMPEG_THREADS = args.ffmpeg_threads
    RETRIES_ON_TIMEOUT = args.retries

    logger.info(f"Using {MAX_WORKERS} workers, batch size {BATCH_SIZE}")
    logger.info(f"Force mode: {'ON' if args.force else 'OFF'}")
    logger.info(f"Timeouts: factor={TIMEOUT_FACTOR}, min={MIN_TIMEOUT_SEC}s, retries={RETRIES_ON_TIMEOUT}")
    logger.info(f"ffmpeg threads per process: {FFMPEG_THREADS}")

    # Validate input directory
    if not INPUT_ROOT.exists():
        logger.error(f"Input directory does not exist: {INPUT_ROOT}")
        return

    # Create output directory
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {OUTPUT_ROOT}")

    # Find all audio files
    logger.info("Scanning for audio files...")
    scan_start = time.time()
    audio_files = get_audio_files(INPUT_ROOT)
    scan_time = time.time() - scan_start

    if not audio_files:
        logger.warning("No audio files found!")
        return

    logger.info(f"Found {len(audio_files)} audio files in {scan_time:.2f} seconds")

    # Process
    process_files_ultra_fast(
        audio_files=audio_files,
        input_root=INPUT_ROOT,
        output_root=OUTPUT_ROOT,
        force_overwrite=args.force,
        timeout_factor=TIMEOUT_FACTOR,
        min_timeout_sec=MIN_TIMEOUT_SEC,
        ffmpeg_threads=FFMPEG_THREADS,
        workers=MAX_WORKERS,
        batch_size=BATCH_SIZE,
        retries_on_timeout=RETRIES_ON_TIMEOUT
    )

    logger.info("All ultra-fast processing completed!")


if __name__ == "__main__":
    main()
