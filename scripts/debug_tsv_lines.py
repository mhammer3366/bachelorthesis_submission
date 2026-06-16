#!/usr/bin/env python3
import csv

def examine_tsv_lines(path, start_line, num_lines=5):
    """Examine specific lines in the TSV file to understand the formatting issue."""
    print(f"Examining lines around {start_line}...")
    
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        
        # Read header
        header = next(reader)
        print(f"Header ({len(header)} fields): {header}")
        print()
        
        # Skip to the target line
        for i in range(start_line - 1):
            try:
                next(reader)
            except StopIteration:
                print(f"Reached end of file before line {start_line}")
                return
        
        # Read and display the target lines
        for i in range(num_lines):
            try:
                line_num = start_line + i
                row = next(reader)
                print(f"Line {line_num} ({len(row)} fields) - ORIGINAL:")
                for j, field in enumerate(row):
                    print(f"  [{j:2d}]: '{field}'")
                
                # Apply the fix
                if len(row) == 11:
                    fixed_row = row[:8] + row[10:]  # Remove fields 8 and 9
                    print(f"Line {line_num} ({len(fixed_row)} fields) - FIXED:")
                    for j, field in enumerate(fixed_row):
                        print(f"  [{j:2d}]: '{field}'")
                print()
            except StopIteration:
                print(f"Reached end of file at line {start_line + i}")
                break

# Examine lines around the problematic area (305k)
print("=== Examining lines around 305,000 ===")
#examine_tsv_lines('/home/ai/AI-DataPool/Datasets/audio/Schweiz/16000_mono_wav_splitted_1/master_index_with_phonemes.clean.tsv', 315000, 3)
examine_tsv_lines('/home/ai/AI-DataPool/Datasets/audio/Schweiz/16000_mono_wav_splitted_1/master_index_with_phonemes.tsv', 450000, 3)

print("\n=== Examining lines around 1,000 ===")
#examine_tsv_lines('/home/ai/AI-DataPool/Datasets/audio/Schweiz/16000_mono_wav_splitted_1/master_index_with_phonemes.clean.tsv', 1000, 3)
examine_tsv_lines('/home/ai/AI-DataPool/Datasets/audio/Schweiz/16000_mono_wav_splitted_1/master_index_with_phonemes.tsv', 1000, 3)
