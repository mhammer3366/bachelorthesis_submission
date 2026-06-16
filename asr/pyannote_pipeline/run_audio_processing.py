#!/usr/bin/env python3
"""
Simple script to run the audio processing with different optimization levels.
"""

import sys
import time
from pathlib import Path

def main():
    print("Audio Processing Scripts")
    print("=" * 50)
    print("1. Standard parallel processing (resample_mono_swiss.py)")
    print("2. Ultra-fast processing (resample_mono_swiss_ultra_fast.py)")
    print()
    
    choice = input("Choose processing mode (1 or 2): ").strip()
    
    if choice == "1":
        script_path = Path(__file__).parent / "resample_mono_swiss.py"
        print(f"Running standard parallel processing...")
    elif choice == "2":
        script_path = Path(__file__).parent / "resample_mono_swiss_ultra_fast.py"
        print(f"Running ultra-fast processing...")
    else:
        print("Invalid choice. Exiting.")
        return
    
    if not script_path.exists():
        print(f"Script not found: {script_path}")
        return
    
    # Run the selected script
    import subprocess
    start_time = time.time()
    
    try:
        result = subprocess.run([sys.executable, str(script_path)], check=True)
        end_time = time.time()
        print(f"\nProcessing completed successfully in {end_time - start_time:.2f} seconds")
    except subprocess.CalledProcessError as e:
        print(f"Processing failed with error code: {e.returncode}")
    except KeyboardInterrupt:
        print("\nProcessing interrupted by user")

if __name__ == "__main__":
    main()
