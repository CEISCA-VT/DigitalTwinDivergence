#!/usr/bin/env python3
"""
Create apriltag_independent_runs.csv by analyzing dataset metadata and directory names.
Map each dataset to its source rover telemetry file if available.
"""

from pathlib import Path
import json
import csv
from datetime import datetime

def extract_run_info(dataset_name):
    """
    Extract information from dataset name.
    Naming patterns:
    - validation_carpet_142023_* : validation carpet run from July 14, ~20:23
    - validation_carpet_142907_* : validation carpet run from July 14, ~29:07
    - ugv01_apriltag_carpet_142023 : UGV01 AprilTag carpet run 142023
    - apriltag_* : various development/calibration runs
    """
    
    name_lower = dataset_name.lower()
    
    # Initialize defaults
    info = {
        'physical_recording': None,
        'surface': None,
        'speed': None,
        'route': None,
        'dataset_category': None,
        'has_aligned_gt': False,
        'notes': ''
    }
    
    # Determine dataset category and extract info
    if 'validation_carpet' in name_lower:
        info['dataset_category'] = 'VALIDATION_CARPET'
        info['surface'] = 'carpet'
        info['route'] = 'square'
        
        if '142023' in name_lower:
            info['physical_recording'] = 'validation_carpet_142023'
            info['notes'] = 'Validation run 1'
        elif '142907' in name_lower:
            info['physical_recording'] = 'validation_carpet_142907'
            info['notes'] = 'Validation run 2 (alternative timing)'
        elif '142023' in name_lower and 'baseline' in name_lower:
            info['physical_recording'] = 'validation_carpet_142023_baseline'
            info['notes'] = 'Baseline comparison'
        elif '142907' in name_lower and 'baseline' in name_lower:
            info['physical_recording'] = 'validation_carpet_142907_baseline'
            info['notes'] = 'Baseline comparison variant'
        
        if 'four_tag' in name_lower:
            info['notes'] += ' (four-tag coverage)'
        if 'full' in name_lower:
            info['notes'] += ' (full processing)'
        if 'elevation' in name_lower:
            info['notes'] += ' (elevation corrected)'
            
    elif 'ugv01_apriltag_carpet' in name_lower:
        info['dataset_category'] = 'UGV01_APRILTAG'
        info['surface'] = 'carpet'
        info['route'] = 'square'
        info['physical_recording'] = 'UGV01_carpet_AprilTag'
        
        if 'finetuned' in name_lower:
            info['notes'] = 'Refined tracker'
        else:
            info['notes'] = 'Initial tracking'
            
        if '142023' in name_lower:
            info['notes'] += ' from July 14'
            
    elif 'apriltag_trapezoid' in name_lower:
        info['dataset_category'] = 'APRILTAG_DEV'
        info['surface'] = 'unknown'
        info['route'] = 'trapezoid'
        info['notes'] = 'Route calibration/development'
        
    elif 'apriltag_trial1' in name_lower:
        info['dataset_category'] = 'APRILTAG_DEV'
        info['surface'] = 'unknown'
        info['route'] = 'square_1.5m'
        if 'frozen_validation' in name_lower:
            info['dataset_category'] = 'VALIDATION_CANDIDATE'
            info['notes'] = 'Validation candidate'
            
    elif 'apriltag_carpet_2x1' in name_lower:
        info['dataset_category'] = 'APRILTAG_DEV'
        info['surface'] = 'carpet'
        info['route'] = 'square_2x1'
        info['notes'] = 'Larger route development'
        
    else:
        info['dataset_category'] = 'OTHER_DEV'
        info['notes'] = 'Development/calibration'
    
    return info

def main():
    analysis_path = Path("DigitalTwin/datasets/analysis")
    raw_logs_path = Path("raw_logs/telemetry")
    
    # Find all datasets
    all_datasets = sorted([d for d in analysis_path.iterdir() if d.is_dir()])
    
    # Find all raw rover logs with timestamp
    raw_logs = sorted(raw_logs_path.glob("speed-*.csv"))
    
    print(f"Found {len(all_datasets)} analysis datasets")
    print(f"Found {len(raw_logs)} raw rover telemetry logs")
    print()
    
    # Create the audit table
    audit_rows = []
    
    # Track which physical recordings we've seen
    physical_recordings = {}
    
    for dataset in all_datasets:
        # Check if it has aligned ground truth
        has_aligned_csv = (dataset / "aligned_samples.csv").exists()
        has_aligned_npz = (dataset / "aligned_samples.npz").exists()
        has_aligned = has_aligned_csv or has_aligned_npz
        
        # Extract information
        info = extract_run_info(dataset.name)
        info['has_aligned_gt'] = has_aligned
        
        # Try to read metadata
        prep_json = dataset / "preparation_summary.json"
        metadata = {}
        if prep_json.exists():
            try:
                with open(prep_json, 'r') as f:
                    metadata = json.load(f)
            except:
                pass
        
        # Count files in directory
        files = list(dataset.glob("*"))
        num_files = len(files)
        
        # Determine data availability
        data_status = []
        if has_aligned_csv:
            data_status.append("CSV-GT")
        if has_aligned_npz:
            data_status.append("NPZ-GT")
        if any(dataset.glob("*.mp4")) or any(dataset.glob("*.avi")):
            data_status.append("VIDEO")
        if not data_status:
            data_status.append("METADATA_ONLY")
        
        # Create audit row
        row = {
            'run_id': dataset.name[:20],  # Short ID
            'full_dataset_name': dataset.name,
            'category': info['dataset_category'],
            'physical_recording': info['physical_recording'] or 'N/A',
            'surface': info['surface'] or 'unknown',
            'speed': 'unknown',
            'route': info['route'] or 'unknown',
            'has_aligned_gt': 'YES' if has_aligned else 'NO',
            'data_available': ';'.join(data_status),
            'duration_s': metadata.get('duration_s', 'N/A'),
            'rows': metadata.get('rows', 'N/A'),
            'used_for_calibration': 'UNKNOWN',
            'notes': info['notes']
        }
        
        audit_rows.append(row)
        
        # Track unique physical recordings
        if info['physical_recording']:
            if info['physical_recording'] not in physical_recordings:
                physical_recordings[info['physical_recording']] = {
                    'category': info['dataset_category'],
                    'count': 0,
                    'with_gt': 0
                }
            physical_recordings[info['physical_recording']]['count'] += 1
            if has_aligned:
                physical_recordings[info['physical_recording']]['with_gt'] += 1
    
    # Write CSV
    output_file = Path("apriltag_independent_runs.csv")
    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'run_id', 'full_dataset_name', 'category', 'physical_recording',
            'surface', 'speed', 'route', 'has_aligned_gt', 'data_available',
            'duration_s', 'rows', 'used_for_calibration', 'notes'
        ])
        writer.writeheader()
        writer.writerows(audit_rows)
    
    print(f"✓ Written apriltag_independent_runs.csv ({len(audit_rows)} rows)")
    print()
    
    # Print summary
    print("=" * 80)
    print("PHYSICAL RECORDINGS SUMMARY")
    print("=" * 80)
    print(f"\nUnique physical recordings identified: {len(physical_recordings)}\n")
    
    for rec_name in sorted(physical_recordings.keys()):
        info = physical_recordings[rec_name]
        print(f"{rec_name}:")
        print(f"  Category: {info['category']}")
        print(f"  Processing variants: {info['count']}")
        print(f"  With aligned ground truth: {info['with_gt']}")
        print()
    
    # Summary statistics
    print("=" * 80)
    print("DATASET STATISTICS")
    print("=" * 80)
    
    total = len(audit_rows)
    with_gt = sum(1 for r in audit_rows if r['has_aligned_gt'] == 'YES')
    validation = sum(1 for r in audit_rows if r['category'] == 'VALIDATION_CANDIDATE')
    ugv01 = sum(1 for r in audit_rows if r['category'] == 'UGV01_APRILTAG')
    dev = sum(1 for r in audit_rows if r['category'].endswith('_DEV'))
    
    print(f"Total datasets: {total}")
    print(f"  - With aligned ground truth: {with_gt}")
    print(f"  - Validation candidates: {validation}")
    print(f"  - UGV01 AprilTag runs: {ugv01}")
    print(f"  - Development/calibration: {dev}")
    print(f"  - Other: {total - validation - ugv01 - dev}")
    
    print()
    print("✓ Audit complete!")

if __name__ == "__main__":
    main()
