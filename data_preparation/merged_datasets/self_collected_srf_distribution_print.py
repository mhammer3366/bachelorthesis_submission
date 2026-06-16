import os
import sys
from multiprocessing import Pool, cpu_count
import os
import sys
from multiprocessing import Pool, cpu_count
from functools import partial
from pathlib import Path

AUDIO_ROOT = str(Path(os.environ.get("DATA_ROOT", "/home/ai/AI-DataPool/Datasets")) / "audio/Vorarlberg")

# Audio file extensions to look for
AUDIO_EXTENSIONS = {'.wav', '.mp3', '.flac', '.m4a', '.ogg', '.opus'}

def get_audio_duration(audio_path):
    """Get duration of audio file in seconds"""
    # Quick existence check
    try:
        if not os.path.exists(audio_path):
            return None
    except (OSError, PermissionError):
        return None
    
    # Try ffprobe first (most reliable and fastest)
    try:
        import subprocess
        result = subprocess.run(
            ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
             '-of', 'default=noprint_wrappers=1:nokey=1', audio_path],
            capture_output=True,
            text=True,
            timeout=3  # Reduced timeout for faster processing
        )
        if result.returncode == 0 and result.stdout.strip():
            duration_str = result.stdout.strip()
            if duration_str and duration_str != "N/A" and duration_str:
                try:
                    duration = float(duration_str)
                    if duration > 0:
                        return duration
                except (ValueError, TypeError):
                    pass
    except subprocess.TimeoutExpired:
        # Timeout - file might be corrupted or very large, skip it
        return None
    except FileNotFoundError:
        # ffprobe not found
        pass
    except Exception:
        # Other errors - continue to next method
        pass
    
    # Try librosa (slower)
    try:
        import librosa
        duration = librosa.get_duration(path=audio_path)
        if duration and duration > 0:
            return duration
    except Exception:
        pass
    
    # Try soundfile (slower)
    try:
        import soundfile as sf
        info = sf.info(audio_path)
        if info.duration and info.duration > 0:
            return info.duration
    except Exception:
        pass
    
    return None

def main():
    if not os.path.exists(AUDIO_ROOT):
        print(f"❌ Error: Audio root directory not found: {AUDIO_ROOT}")
        sys.exit(1)
    
    if not os.access(AUDIO_ROOT, os.R_OK):
        print(f"❌ Error: No read permission for directory: {AUDIO_ROOT}")
        sys.exit(1)
    
    try:
        all_items = os.listdir(AUDIO_ROOT)
    except PermissionError:
        print(f"❌ Error: Permission denied accessing: {AUDIO_ROOT}")
        sys.exit(1)
    
    show_dirs = [d for d in all_items 
                 if os.path.isdir(os.path.join(AUDIO_ROOT, d))]
    
    if not show_dirs:
        print(f"⚠️  No show directories found in {AUDIO_ROOT}")
        sys.exit(0)
    
    print("="*60)
    print("📊 SRF Audio Download Distribution (Total hours/episodes per show)")
    print("="*60)
    print(f"Processing {len(show_dirs)} shows...\n")
    
    results = []
    total_files_processed = 0
    total_files_with_duration = 0
    
    for idx, show in enumerate(sorted(show_dirs), 1):
        show_path = os.path.join(AUDIO_ROOT, show)
        
        try:
            files_in_dir = os.listdir(show_path)
        except (PermissionError, OSError) as e:
            print(f"{show:40} | {'ERROR':>4} | {str(e)[:20]}")
            results.append({"show": show, "episodes": 0, "hours": 0.0})
            continue
        
        # Filter audio files more efficiently - check extension first, then verify it's a file
        audio_files = []
        for f in files_in_dir:
            ext = os.path.splitext(f)[1].lower()
            if ext in AUDIO_EXTENSIONS:
                file_path = os.path.join(show_path, f)
                # Only check if it's a file if extension matches (much faster)
                try:
                    if os.path.isfile(file_path):
                        audio_files.append(file_path)
                except (OSError, PermissionError):
                    # Skip files we can't access
                    continue
        
        total_seconds = 0
        num_episodes = 0
        
        # Process files with progress for large directories
        if len(audio_files) > 100:
            print(f"  Processing {show} ({len(audio_files)} files)...", end="", flush=True)
        
        # Use parallel processing for large directories
        if len(audio_files) > 50:
            # Use multiprocessing for faster processing
            num_workers = min(cpu_count(), 8)  # Limit to 8 workers to avoid overwhelming the system
            with Pool(processes=num_workers) as pool:
                durations = pool.map(get_audio_duration, sorted(audio_files))
            
            failed_count = 0
            for duration in durations:
                total_files_processed += 1
                if duration is not None and duration > 0:
                    total_seconds += duration
                    num_episodes += 1
                    total_files_with_duration += 1
                else:
                    failed_count += 1
        else:
            # Sequential processing for small directories
            failed_count = 0
            for file in sorted(audio_files):
                total_files_processed += 1
                try:
                    duration = get_audio_duration(file)
                    if duration is not None and duration > 0:
                        total_seconds += duration
                        num_episodes += 1
                        total_files_with_duration += 1
                    else:
                        failed_count += 1
                except (KeyboardInterrupt, SystemExit):
                    raise
                except Exception:
                    failed_count += 1
                    # Continue processing other files even if one fails
        
        if len(audio_files) > 100:
            status = "done"
            if failed_count > 0:
                status = f"done ({failed_count} failed)"
            print(f" {status}")
        
        hours = total_seconds / 3600
        print(f"{show:40} | {num_episodes:4d} episodes | {hours:8.2f} hours")
        results.append({"show": show, "episodes": num_episodes, "hours": hours})
    
    # Print Grand Total
    total_shows = len(results)
    total_episodes = sum(r["episodes"] for r in results)
    total_hours = sum(r["hours"] for r in results)
    print("-"*60)
    print(f"{'TOTAL':40} | {total_episodes:4d} episodes | {total_hours:8.2f} hours in {total_shows} shows")
    print(f"\nProcessed {total_files_processed} audio files, {total_files_with_duration} with valid durations")
    print("="*60)

if __name__ == "__main__":
    main()
