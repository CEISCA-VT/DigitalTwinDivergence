#!/usr/bin/env python3
"""Analyze all datasets in the project."""

from pathlib import Path
import json
import os

def analyze_datasets():
    # Paths
    analysis_path = Path("DigitalTwin/datasets/analysis")
    raw_logs_path = Path("raw_logs/telemetry")
    public_datasets_path = Path("public_datasets/im2nav")
    results_path = Path("results")
    
    # Count datasets
    dataset_dirs = sorted([d.name for d in analysis_path.iterdir() if d.is_dir()])
    
    categories = {
        'apriltag': [],
        'i2nav': [],
        'validation': [],
        'pairing': [],
        'analysis': [],
        'other': []
    }
    
    for name in dataset_dirs:
        lower = name.lower()
        if 'apriltag' in lower:
            categories['apriltag'].append(name)
        elif 'i2nav' in lower:
            categories['i2nav'].append(name)
        elif 'validation' in lower:
            categories['validation'].append(name)
        elif 'pairing' in lower:
            categories['pairing'].append(name)
        elif any(x in lower for x in ['accuracy', 'covariance', 'uncertainty', 'real_data', 'digital_twin']):
            categories['analysis'].append(name)
        else:
            categories['other'].append(name)
    
    print("=" * 70)
    print("DATASET STATISTICS")
    print("=" * 70)
    for cat, items in categories.items():
        print(f"{cat.upper():20} : {len(items):3} datasets")
    
    print(f"\n{'TOTAL DATASETS':20} : {len(dataset_dirs):3} datasets")
    
    # Count raw logs
    if raw_logs_path.exists():
        raw_logs = list(raw_logs_path.glob("*.csv"))
        print(f"{'RAW TELEMETRY LOGS':20} : {len(raw_logs):3} files")
    
    # Check public datasets
    if public_datasets_path.exists():
        im2nav_routes = sorted([d.name for d in public_datasets_path.iterdir() if d.is_dir()])
        total_public_files = 0
        print(f"\n{'PUBLIC DATASETS':20} : {len(im2nav_routes)} routes")
        for route in im2nav_routes:
            files = list((public_datasets_path / route).glob("*"))
            total_public_files += len(files)
            print(f"  - {route:40} : {len(files):3} files")
        print(f"  Total public files: {total_public_files}")
    
    # Check results
    if results_path.exists():
        result_files = list(results_path.glob("*"))
        print(f"\n{'RESULT FILES':20} : {len(result_files):3} files")
        for f in sorted(result_files)[:10]:
            if f.is_file():
                size_kb = f.stat().st_size / 1024
                print(f"  - {f.name:50} : {size_kb:8.1f} KB")
    
    print("\n" + "=" * 70)
    print("CATEGORY DETAILS")
    print("=" * 70)
    
    if categories['apriltag']:
        print(f"\nAPRILTAG DATASETS ({len(categories['apriltag'])}):")
        for name in sorted(categories['apriltag'])[:15]:
            print(f"  - {name}")
        if len(categories['apriltag']) > 15:
            print(f"  ... and {len(categories['apriltag']) - 15} more")
    
    if categories['i2nav']:
        print(f"\ni2NAV CONVERSIONS ({len(categories['i2nav'])}):")
        for name in sorted(categories['i2nav']):
            print(f"  - {name}")
    
    if categories['validation']:
        print(f"\nVALIDATION DATASETS ({len(categories['validation'])}):")
        for name in sorted(categories['validation'])[:10]:
            print(f"  - {name}")
        if len(categories['validation']) > 10:
            print(f"  ... and {len(categories['validation']) - 10} more")

if __name__ == "__main__":
    analyze_datasets()
