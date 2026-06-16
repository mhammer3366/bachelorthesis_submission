import pandas as pd
import os

def get_duration_from_audio(audio_path):
    """Get duration of audio file in seconds"""
    # Check if file exists and is readable
    if not os.path.exists(audio_path):
        return None
    if not os.access(audio_path, os.R_OK):
        return None
    
    # Try ffprobe first (usually fastest and most reliable)
    try:
        import subprocess
        result = subprocess.run(
            ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1', audio_path],
            capture_output=True,
            text=True,
            timeout=5,
            stderr=subprocess.DEVNULL  # Suppress stderr to avoid permission errors in output
        )
        if result.returncode == 0 and result.stdout.strip():
            duration_str = result.stdout.strip()
            if duration_str and duration_str != 'N/A':
                return float(duration_str)
    except (ValueError, subprocess.TimeoutExpired, FileNotFoundError):
        pass
    except Exception:
        pass
    
    # Try librosa
    try:
        import librosa
        duration = librosa.get_duration(path=audio_path)
        if duration and duration > 0:
            return duration
    except:
        pass
    
    # Try soundfile
    try:
        import soundfile as sf
        info = sf.info(audio_path)
        if info.duration and info.duration > 0:
            return info.duration
    except:
        pass
    
    return None

def add_duration_column(df, use_cache=True):
    """Add duration column by reading from audio files"""
    if "duration" in df.columns:
        return df
    
    if "audio_path" not in df.columns:
        print("Warning: No 'audio_path' column found. Cannot calculate duration.")
        return df
    
    # Check for cached duration file
    cache_file = "duration_cache.tsv"
    duration_cache = {}
    if use_cache and os.path.exists(cache_file):
        try:
            cache_df = pd.read_csv(cache_file, sep="\t")
            duration_cache = dict(zip(cache_df["audio_path"], cache_df["duration"]))
            print(f"Loaded {len(duration_cache)} cached durations from {cache_file}")
        except:
            pass
    
    print("Calculating duration from audio files...")
    durations = []
    new_cache_entries = []
    missing_files = 0
    unreadable_files = 0
    
    for idx, audio_path in enumerate(df["audio_path"]):
        if pd.isna(audio_path):
            durations.append(None)
            continue
            
        # Check cache first
        if audio_path in duration_cache:
            durations.append(duration_cache[audio_path])
        else:
            # Check if file exists
            if not os.path.exists(audio_path):
                missing_files += 1
                durations.append(None)
            elif not os.access(audio_path, os.R_OK):
                unreadable_files += 1
                durations.append(None)
            else:
                duration = get_duration_from_audio(audio_path)
                durations.append(duration)
                if duration is not None and duration > 0:
                    new_cache_entries.append({"audio_path": audio_path, "duration": duration})
        
        if (idx + 1) % 1000 == 0:
            valid_count = sum(1 for d in durations if d is not None and d > 0)
            print(f"  Processed {idx + 1}/{len(df)} files... ({valid_count} valid durations, {missing_files} missing, {unreadable_files} unreadable)")
    
    if missing_files > 0 or unreadable_files > 0:
        print(f"\n  ⚠️  File access issues: {missing_files} files not found, {unreadable_files} files not readable")
    
    df["duration"] = durations
    
    # Update cache file
    if new_cache_entries and use_cache:
        new_cache_df = pd.DataFrame(new_cache_entries)
        if os.path.exists(cache_file):
            existing_cache = pd.read_csv(cache_file, sep="\t")
            combined_cache = pd.concat([existing_cache, new_cache_df]).drop_duplicates(subset=["audio_path"])
            combined_cache.to_csv(cache_file, sep="\t", index=False)
        else:
            new_cache_df.to_csv(cache_file, sep="\t", index=False)
        print(f"Updated cache with {len(new_cache_entries)} new entries")
    
    return df


def print_distribution(df: pd.DataFrame, split: str):
    print("="*60)
    print(f"📊 Split: {split}")
    print("="*60)

    # Dialect distribution
    print("\nDialect region distribution:")
    dialect_counts = df["dialect_region"].value_counts(dropna=False)
    print(dialect_counts.to_string())
    if "duration" in df.columns:
        print("\nDialect region hours:")
        for region in dialect_counts.index:
            region_df = df[df["dialect_region"] == region]
            duration_sum = region_df["duration"].dropna().sum()
            hours = duration_sum / 3600 if duration_sum > 0 else 0
            print(f"{region}: {hours:.2f}h")

    # Canton distribution
    print("\nCanton distribution:")
    canton_counts = df["canton"].value_counts(dropna=False)
    print(canton_counts.to_string())
    if "duration" in df.columns:
        print("\nCanton hours:")
        for canton in canton_counts.index:
            canton_df = df[df["canton"] == canton]
            duration_sum = canton_df["duration"].dropna().sum()
            hours = duration_sum / 3600 if duration_sum > 0 else 0
            print(f"{canton}: {hours:.2f}h")

    # Gender distribution
    if "gender" in df.columns:
        print("\nGender distribution:")
        gender_counts = df["gender"].value_counts(dropna=False)
        print(gender_counts.to_string())
        if "duration" in df.columns:
            print("\nGender hours:")
            for gender in gender_counts.index:
                gender_df = df[df["gender"] == gender]
                duration_sum = gender_df["duration"].dropna().sum()
                hours = duration_sum / 3600 if duration_sum > 0 else 0
                print(f"{gender}: {hours:.2f}h")

    # Age distribution
    if "age" in df.columns:
        print("\nAge distribution:")
        age_counts = df["age"].value_counts(dropna=False)
        print(age_counts.to_string())
        if "duration" in df.columns:
            print("\nAge hours:")
            for age in age_counts.index:
                age_df = df[df["age"] == age]
                duration_sum = age_df["duration"].dropna().sum()
                hours = duration_sum / 3600 if duration_sum > 0 else 0
                print(f"{age}: {hours:.2f}h")

    # Dataset source
    if "dataset" in df.columns:
        print("\nDataset source distribution:")
        dataset_counts = df["dataset"].value_counts()
        print(dataset_counts.to_string())
        if "duration" in df.columns:
            print("\nDataset hours:")
            for dataset in dataset_counts.index:
                dataset_df = df[df["dataset"] == dataset]
                duration_sum = dataset_df["duration"].dropna().sum()
                hours = duration_sum / 3600 if duration_sum > 0 else 0
                print(f"{dataset}: {hours:.2f}h")

    # Rows missing dialect_region
    missing_dialect = df[df["dialect_region"].isna()]
    if not missing_dialect.empty:
        print(f"\nRows missing 'dialect_region' in {split} split (showing first 10):")
        print(missing_dialect.head(10))
        print(f"... total {len(missing_dialect)} rows")

        if "dataset" in df.columns:
            print("\nNaN 'dialect_region' counts per dataset:")
            print(missing_dialect["dataset"].value_counts())
    else:
        print(f"\nNo rows missing 'dialect_region' in {split} split.")

    # Count of NaN values per row
    nan_per_row_counts = df.isna().sum(axis=1).value_counts()
    print(f"\nDistribution of NaN counts per row in {split} split:")
    print(nan_per_row_counts.sort_index().to_string())

    # Column-level missing stats
    print("\nMissing values per column:")
    print(df.isna().sum())

    # Total duration summary
    if "duration" in df.columns:
        duration_sum = df["duration"].dropna().sum()
        total_hours = duration_sum / 3600 if duration_sum > 0 else 0
        valid_count = df["duration"].notna().sum()
        print(f"\nTotal hours in {split} split: {total_hours:.2f}h (from {valid_count}/{len(df)} valid durations)")

    print("\n\n")


# Example usage after your merge_split():
train_df = pd.read_csv("merged_train.tsv", sep="\t")
valid_df = pd.read_csv("merged_valid.tsv", sep="\t")
test_df  = pd.read_csv("merged_test.tsv", sep="\t")

# Add duration column if not present
if "duration" not in train_df.columns:
    print("Duration column not found. Calculating from audio files...")
    print("Note: This requires access to audio files. If files are not accessible,")
    print("      you may need to run with appropriate permissions or pre-compute durations.\n")
    train_df = add_duration_column(train_df)
    valid_df = add_duration_column(valid_df)
    test_df = add_duration_column(test_df)
    
    # Check if we got any valid durations
    total_valid = train_df["duration"].notna().sum() + valid_df["duration"].notna().sum() + test_df["duration"].notna().sum()
    if total_valid == 0:
        print("\n⚠️  WARNING: No valid durations were calculated!")
        print("   This could be due to:")
        print("   - Audio files not accessible (permission issues)")
        print("   - Audio files not found at specified paths")
        print("   - Missing audio processing libraries (librosa/soundfile)")
        print("   - Missing ffprobe utility")
        print("\n   Consider:")
        print("   - Adding a 'duration' column to your TSV files manually")
        print("   - Running with appropriate file permissions")
        print("   - Checking that audio file paths are correct\n")
    else:
        print(f"Duration calculation complete! ({total_valid:,} valid durations found)\n")

for split, df in [("train", train_df), ("valid", valid_df), ("test", test_df)]:
    print_distribution(df, split)
