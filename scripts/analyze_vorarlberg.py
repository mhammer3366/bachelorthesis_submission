#!/usr/bin/env python3

import os
from pathlib import Path
REPO_ROOT = Path(os.environ.get("THESIS_ROOT", Path(__file__).resolve().parents[1]))
import pandas as pd
import numpy as np

print('📊 Vorarlberg Binary Classification Results Analysis')
print('='*60)

# Load the data
df = pd.read_csv(str(REPO_ROOT / "eval/dialect_evaluation/vorarlberg_results/binary_classification/vorarlberg_finalized.tsv"), sep='\t')

print(f'Total samples: {len(df):,}')
print()

# Label distribution
print('Label ID Distribution:')
label_counts = df['label_id'].value_counts().sort_index()
for label, count in label_counts.items():
    pct = (count / len(df)) * 100
    print(f'  Label {label}: {count:,} samples ({pct:.1f}%)')

print()

# Duration analysis
print('⏱️ Duration Analysis:')
df['duration'] = pd.to_numeric(df['duration'], errors='coerce')

# Duration by label
for label in sorted(df['label_id'].unique()):
    label_data = df[df['label_id'] == label]
    label_duration = label_data['duration'].sum()
    label_hours = label_duration / 3600
    label_count = len(label_data)
    avg_duration = label_duration / label_count if label_count > 0 else 0
    print(f'  Label {label}: {label_count:,} samples, {label_duration:,.1f}s total, {label_hours:.2f} hours, avg: {avg_duration:.1f}s per sample')

# Total duration
total_duration = df['duration'].sum()
total_hours = total_duration / 3600
print(f'\n  Total duration: {total_duration:,.1f} seconds ({total_hours:.2f} hours)')

print()
print('🎯 Summary for Labels 7 and 8:')
label7_count = label_counts.get(7, 0)
label8_count = label_counts.get(8, 0)
label7_duration = df[df['label_id'] == 7]['duration'].sum() / 3600
label8_duration = df[df['label_id'] == 8]['duration'].sum() / 3600

print(f'  Label 7 (German): {label7_count:,} samples, {label7_duration:.2f} hours')
print(f'  Label 8 (Vorarlberg): {label8_count:,} samples, {label8_duration:.2f} hours')
print(f'  Total: {label7_count + label8_count:,} samples, {label7_duration + label8_duration:.2f} hours')

