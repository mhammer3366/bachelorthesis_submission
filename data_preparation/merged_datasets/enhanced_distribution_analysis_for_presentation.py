import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
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

    # Basic stats
    print(f"Total samples: {len(df):,}")
    print(f"Unique speakers: {df['client_id'].nunique() if 'client_id' in df.columns else 'N/A'}")
    
    # Dialect distribution
    print("\n🏛️ Dialect region distribution:")
    dialect_counts = df["dialect_region"].value_counts(dropna=False)
    print(dialect_counts.to_string())
    
    # Calculate percentages and hours
    print("\nDialect region breakdown (samples, percentage, hours):")
    dialect_pct = (dialect_counts / len(df) * 100).round(1)
    for region, count in dialect_counts.items():
        pct = dialect_pct[region]
        if "duration" in df.columns:
            region_df = df[df["dialect_region"] == region]
            duration_sum = region_df["duration"].dropna().sum()
            hours = duration_sum / 3600 if duration_sum > 0 else 0
            print(f"{region}: {count:,} samples ({pct}%) - {hours:.2f}h")
        else:
            print(f"{region}: {count:,} ({pct}%)")

    # Canton distribution
    print("\n🗺️ Canton distribution:")
    canton_counts = df["canton"].value_counts(dropna=False)
    print(canton_counts.to_string())
    
    # Canton hours
    if "duration" in df.columns:
        print("\nCanton hours:")
        for canton in canton_counts.index:
            canton_df = df[df["canton"] == canton]
            duration_sum = canton_df["duration"].dropna().sum()
            hours = duration_sum / 3600 if duration_sum > 0 else 0
            print(f"{canton}: {hours:.2f}h")

    # Gender distribution
    if "gender" in df.columns:
        print("\n👥 Gender distribution:")
        gender_counts = df["gender"].value_counts(dropna=False)
        print(gender_counts.to_string())
        
        # Gender percentages and hours
        gender_pct = (gender_counts / len(df) * 100).round(1)
        print("\nGender breakdown (samples, percentage, hours):")
        for gender, count in gender_counts.items():
            pct = gender_pct[gender]
            if "duration" in df.columns:
                gender_df = df[df["gender"] == gender]
                duration_sum = gender_df["duration"].dropna().sum()
                hours = duration_sum / 3600 if duration_sum > 0 else 0
                print(f"{gender}: {count:,} samples ({pct}%) - {hours:.2f}h")
            else:
                print(f"{gender}: {count:,} ({pct}%)")

    # Age distribution
    if "age" in df.columns:
        print("\n📅 Age distribution:")
        age_counts = df["age"].value_counts(dropna=False)
        print(age_counts.to_string())
        
        # Age hours
        if "duration" in df.columns:
            print("\nAge hours:")
            for age in age_counts.index:
                age_df = df[df["age"] == age]
                duration_sum = age_df["duration"].dropna().sum()
                hours = duration_sum / 3600 if duration_sum > 0 else 0
                print(f"{age}: {hours:.2f}h")

    # Dataset source
    if "dataset" in df.columns:
        print("\n📚 Dataset source distribution:")
        dataset_counts = df["dataset"].value_counts()
        print(dataset_counts.to_string())
        
        # Dataset percentages and hours
        dataset_pct = (dataset_counts / len(df) * 100).round(1)
        print("\nDataset breakdown (samples, percentage, hours):")
        for dataset, count in dataset_counts.items():
            pct = dataset_pct[dataset]
            if "duration" in df.columns:
                dataset_df = df[df["dataset"] == dataset]
                duration_sum = dataset_df["duration"].dropna().sum()
                hours = duration_sum / 3600 if duration_sum > 0 else 0
                print(f"{dataset}: {count:,} samples ({pct}%) - {hours:.2f}h")
            else:
                print(f"{dataset}: {count:,} ({pct}%)")

    # Audio duration analysis (if available)
    if "duration" in df.columns:
        # Filter out NaN values for statistics
        duration_valid = df["duration"].dropna()
        if len(duration_valid) > 0:
            print("\n⏱️ Audio duration statistics:")
            duration_stats = duration_valid.describe()
            if 'mean' in duration_stats:
                print(f"Mean: {duration_stats['mean']:.2f}s")
            if '50%' in duration_stats:
                print(f"Median: {duration_stats['50%']:.2f}s")
            if 'min' in duration_stats:
                print(f"Min: {duration_stats['min']:.2f}s")
            if 'max' in duration_stats:
                print(f"Max: {duration_stats['max']:.2f}s")
            total_hours = duration_valid.sum() / 3600
            print(f"Total hours: {total_hours:.1f}h")
            print(f"Valid duration entries: {len(duration_valid)}/{len(df)} ({len(duration_valid)/len(df)*100:.1f}%)")
        else:
            print("\n⏱️ Audio duration statistics:")
            print("⚠️  No valid duration values found (all NaN)")

    # Missing data analysis
    print("\n❌ Missing data analysis:")
    missing_dialect = df[df["dialect_region"].isna()]
    if not missing_dialect.empty:
        print(f"Rows missing 'dialect_region': {len(missing_dialect)} ({len(missing_dialect)/len(df)*100:.1f}%)")
        
        if "dataset" in df.columns:
            print("Missing 'dialect_region' by dataset:")
            print(missing_dialect["dataset"].value_counts())
    else:
        print("✅ No missing 'dialect_region' values")

    # Class balance analysis
    print("\n⚖️ Class balance analysis:")
    dialect_counts = df["dialect_region"].value_counts()
    min_count = dialect_counts.min()
    max_count = dialect_counts.max()
    balance_ratio = max_count / min_count if min_count > 0 else float('inf')
    print(f"Smallest class: {min_count:,} samples")
    print(f"Largest class: {max_count:,} samples")
    print(f"Balance ratio: {balance_ratio:.1f}:1")
    
    if balance_ratio > 5:
        print("⚠️  Warning: High class imbalance detected!")
    elif balance_ratio > 2:
        print("⚠️  Note: Moderate class imbalance")
    else:
        print("✅ Good class balance")

    print("\n" + "="*60 + "\n")

def create_distribution_plots(train_df, valid_df, test_df, save_dir="plots"):
    """Create visualization plots for the presentation"""
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    # Combine all splits for overall analysis
    all_data = []
    for split, df in [("Train", train_df), ("Validation", valid_df), ("Test", test_df)]:
        df_copy = df.copy()
        df_copy['split'] = split
        all_data.append(df_copy)
    
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # 1. Dialect distribution across splits
    plt.figure(figsize=(12, 8))
    dialect_split = pd.crosstab(combined_df['dialect_region'], combined_df['split'])
    dialect_split.plot(kind='bar', stacked=True)
    plt.title('Dialect Distribution Across Train/Validation/Test Splits')
    plt.xlabel('Dialect Region')
    plt.ylabel('Number of Samples')
    plt.xticks(rotation=45)
    plt.legend(title='Split')
    plt.tight_layout()
    plt.savefig(f'{save_dir}/dialect_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Dataset source contribution
    if 'dataset' in combined_df.columns:
        plt.figure(figsize=(10, 6))
        dataset_counts = combined_df['dataset'].value_counts()
        plt.pie(dataset_counts.values, labels=dataset_counts.index, autopct='%1.1f%%')
        plt.title('Dataset Source Contribution')
        plt.savefig(f'{save_dir}/dataset_contribution.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 3. Class balance visualization
    plt.figure(figsize=(12, 6))
    dialect_counts = combined_df['dialect_region'].value_counts()
    plt.bar(range(len(dialect_counts)), dialect_counts.values)
    plt.title('Class Distribution (All Splits Combined)')
    plt.xlabel('Dialect Region')
    plt.ylabel('Number of Samples')
    plt.xticks(range(len(dialect_counts)), dialect_counts.index, rotation=45)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/class_balance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Plots saved to {save_dir}/")

def print_summary_stats(train_df, valid_df, test_df):
    """Print summary statistics for presentation"""
    print("🎯 SUMMARY STATISTICS FOR PRESENTATION")
    print("="*50)
    
    total_samples = len(train_df) + len(valid_df) + len(test_df)
    total_speakers = len(set(train_df['client_id'].tolist() + valid_df['client_id'].tolist() + test_df['client_id'].tolist()))
    
    print(f"📊 Dataset Size:")
    print(f"  • Total samples: {total_samples:,}")
    print(f"  • Total speakers: {total_speakers:,}")
    print(f"  • Train: {len(train_df):,} ({len(train_df)/total_samples*100:.1f}%)")
    print(f"  • Validation: {len(valid_df):,} ({len(valid_df)/total_samples*100:.1f}%)")
    print(f"  • Test: {len(test_df):,} ({len(test_df)/total_samples*100:.1f}%)")
    
    # Dialect distribution
    all_dialects = pd.concat([train_df, valid_df, test_df])['dialect_region'].value_counts()
    print(f"\n🏛️ Dialect Classes: {len(all_dialects)}")
    for dialect, count in all_dialects.items():
        pct = count / total_samples * 100
        if 'duration' in train_df.columns:
            all_df = pd.concat([train_df, valid_df, test_df])
            dialect_df = all_df[all_df['dialect_region'] == dialect]
            duration_sum = dialect_df['duration'].dropna().sum()
            hours = duration_sum / 3600 if duration_sum > 0 else 0
            print(f"  • {dialect}: {count:,} samples ({pct:.1f}%) - {hours:.2f}h")
        else:
            print(f"  • {dialect}: {count:,} ({pct:.1f}%)")
    
    # Audio duration
    if 'duration' in train_df.columns:
        train_duration = train_df['duration'].dropna().sum()
        valid_duration = valid_df['duration'].dropna().sum()
        test_duration = test_df['duration'].dropna().sum()
        total_duration = train_duration + valid_duration + test_duration
        
        if total_duration > 0:
            train_hours = train_duration / 3600
            valid_hours = valid_duration / 3600
            test_hours = test_duration / 3600
            total_hours = total_duration / 3600
            print(f"\n⏱️ Audio Duration:")
            print(f"  • Total: {total_hours:.2f} hours")
            print(f"  • Train: {train_hours:.2f}h ({train_hours/total_hours*100:.1f}%)")
            print(f"  • Validation: {valid_hours:.2f}h ({valid_hours/total_hours*100:.1f}%)")
            print(f"  • Test: {test_hours:.2f}h ({test_hours/total_hours*100:.1f}%)")
        else:
            print(f"\n⏱️ Audio Duration: No valid duration values found")
    
    print("="*50)

# Load and analyze
if __name__ == "__main__":
    print("Loading datasets...")
    train_df = pd.read_csv("merged_train.tsv", sep="\t")
    valid_df = pd.read_csv("merged_valid.tsv", sep="\t")
    test_df = pd.read_csv("merged_test.tsv", sep="\t")
    
    # Add duration column if not present
    if "duration" not in train_df.columns:
        print("\nDuration column not found. Calculating from audio files...")
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
    
    # Print detailed distributions
    for split, df in [("TRAIN", train_df), ("VALIDATION", valid_df), ("TEST", test_df)]:
        print_distribution(df, split)
    
    # Print summary for presentation
    print_summary_stats(train_df, valid_df, test_df)
    
    # Create plots
    print("\nCreating visualization plots...")
    create_distribution_plots(train_df, valid_df, test_df)
    
    print("\n✅ Analysis complete!")
