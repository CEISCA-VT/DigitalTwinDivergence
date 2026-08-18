#!/usr/bin/env python3
"""
Audit AprilTag datasets to identify independent physical runs vs. processing variants.
"""

from pathlib import Path
import json
import os

def audit_apriltag_datasets():
    analysis_path = Path("DigitalTwin/datasets/analysis")
    
    # Find all apriltag directories
    apriltag_dirs = sorted([d for d in analysis_path.iterdir() 
                           if d.is_dir() and 'apriltag' in d.name.lower()])
    
    print(f"\n{'=' * 100}")
    print(f"APRILTAG DATASET AUDIT")
    print(f"{'=' * 100}")
    print(f"Total AprilTag directories: {len(apriltag_dirs)}\n")
    
    # Analyze each one
    independent_runs = []
    processing_variants = []
    
    for dataset_dir in apriltag_dirs:
        name = dataset_dir.name
        
        # Check what files exist
        has_aligned = (dataset_dir / "aligned_samples.csv").exists()
        has_npz = (dataset_dir / "aligned_samples.npz").exists()
        has_prep_json = (dataset_dir / "preparation_summary.json").exists()
        has_video = any(dataset_dir.glob("*.mp4")) or any(dataset_dir.glob("*.avi"))
        
        # Read metadata if available
        metadata = None
        if has_prep_json:
            try:
                with open(dataset_dir / "preparation_summary.json", 'r') as f:
                    metadata = json.load(f)
            except:
                metadata = None
        
        # Classify the dataset
        is_ground_truth = has_aligned or has_npz
        
        print(f"\n{'─' * 100}")
        print(f"Dataset: {name}")
        print(f"  Ground Truth Available: {is_ground_truth}")
        print(f"    - aligned_samples.csv: {has_aligned}")
        print(f"    - aligned_samples.npz: {has_npz}")
        print(f"    - preparation_summary.json: {has_prep_json}")
        print(f"    - Video files: {has_video}")
        
        if metadata:
            print(f"  Metadata:")
            print(f"    - Source: {metadata.get('source_directory', 'N/A')}")
            print(f"    - Duration: {metadata.get('duration_s', 'N/A')} s")
            print(f"    - Sample rate: {metadata.get('rate_hz', 'N/A')} Hz")
            print(f"    - Rows: {metadata.get('rows', 'N/A')}")
        
        # Classification
        if 'validation' in name.lower():
            category = "VALIDATION_CANDIDATE"
        elif 'candidate' in name.lower():
            category = "VALIDATION_CANDIDATE"
        elif 'full' in name.lower():
            category = "FULL_PROCESSING"
        elif 'fidelity' in name.lower():
            category = "FIDELITY_ANALYSIS"
        elif 'tracking' in name.lower() or 'tracked' in name.lower():
            category = "TRACKING_DEVELOPMENT"
        elif 'motion' in name.lower() or 'calibration' in name.lower():
            category = "CALIBRATION"
        elif 'smoke' in name.lower() or 'test' in name.lower():
            category = "SMOKE_TEST"
        else:
            category = "OTHER"
        
        print(f"  Classification: {category}")
        
        if is_ground_truth and category in ['VALIDATION_CANDIDATE', 'FIDELITY_ANALYSIS']:
            independent_runs.append({
                'name': name,
                'category': category,
                'has_aligned': has_aligned,
                'metadata': metadata
            })
        else:
            processing_variants.append({
                'name': name,
                'category': category,
                'has_aligned': has_aligned
            })
    
    print(f"\n{'=' * 100}")
    print(f"SUMMARY")
    print(f"{'=' * 100}")
    print(f"Independent runs with ground truth: {len(independent_runs)}")
    print(f"Processing variants/calibration: {len(processing_variants)}")
    print(f"\nIndependent runs:")
    for run in independent_runs:
        print(f"  - {run['name']}")
    
    print(f"\nProcessing variants (not independent runs):")
    for variant in processing_variants[:10]:
        print(f"  - {variant['name']} ({variant['category']})")
    if len(processing_variants) > 10:
        print(f"  ... and {len(processing_variants) - 10} more")

if __name__ == "__main__":
    audit_apriltag_datasets()
