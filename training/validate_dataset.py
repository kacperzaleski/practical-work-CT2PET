"""
Dataset validation and statistics utilities.
Check data integrity and provide statistics about the processed dataset.
"""

import sys as _sys, pathlib as _pathlib  # repo-root bootstrap (script moved into a subfolder)
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))
import os
import json
import numpy as np
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
import argparse


def get_dataset_statistics(data_root):
    """
    Compute statistics about the processed dataset.
    
    Args:
        data_root: Path to processed data root directory
    
    Returns:
        Dictionary containing dataset statistics
    """
    
    data_root = Path(data_root)
    stats = {
        'splits': {},
        'ct_stats': {'min': None, 'max': None, 'mean': None, 'std': None},
        'pet_stats': {'min': None, 'max': None, 'mean': None, 'std': None},
        'attention_coverage': {}
    }
    
    ct_values = []
    pet_values = []
    attention_coverage_per_split = defaultdict(list)
    
    for split in ['train', 'val', 'test']:
        split_dir = data_root / split
        
        ct_dir = split_dir / 'CT'
        pet_dir = split_dir / 'PET'
        attention_dir = split_dir / 'AttentionMaps'
        
        if not ct_dir.exists():
            continue
        
        ct_files = sorted(list(ct_dir.glob('*.npy')))
        pet_files = sorted(list(pet_dir.glob('*.npy')))
        
        num_slices = len(ct_files)
        
        stats['splits'][split] = {
            'num_slices': num_slices,
            'ct_files': num_slices,
            'pet_files': len(pet_files)
        }
        
        print(f"\nProcessing {split} split...")
        
        for ct_file, pet_file in tqdm(zip(ct_files, pet_files), total=num_slices):
            ct_data = np.load(ct_file)
            pet_data = np.load(pet_file)
            
            ct_values.append(ct_data)
            pet_values.append(pet_data)
            
            # Check attention map coverage
            if attention_dir.exists():
                attention_file = attention_dir / ct_file.name
                if attention_file.exists():
                    attention_data = np.load(attention_file)
                    coverage = np.mean(attention_data)
                    attention_coverage_per_split[split].append(coverage)
    
    # Compute global statistics
    if ct_values:
        ct_array = np.concatenate(ct_values)
        stats['ct_stats'] = {
            'min': float(np.min(ct_array)),
            'max': float(np.max(ct_array)),
            'mean': float(np.mean(ct_array)),
            'std': float(np.std(ct_array)),
            'percentile_25': float(np.percentile(ct_array, 25)),
            'percentile_50': float(np.percentile(ct_array, 50)),
            'percentile_75': float(np.percentile(ct_array, 75))
        }
    
    if pet_values:
        pet_array = np.concatenate(pet_values)
        stats['pet_stats'] = {
            'min': float(np.min(pet_array)),
            'max': float(np.max(pet_array)),
            'mean': float(np.mean(pet_array)),
            'std': float(np.std(pet_array)),
            'percentile_25': float(np.percentile(pet_array, 25)),
            'percentile_50': float(np.percentile(pet_array, 50)),
            'percentile_75': float(np.percentile(pet_array, 75))
        }
    
    # Compute attention coverage statistics
    for split, coverages in attention_coverage_per_split.items():
        if coverages:
            stats['attention_coverage'][split] = {
                'mean_coverage': float(np.mean(coverages)),
                'std_coverage': float(np.std(coverages)),
                'min_coverage': float(np.min(coverages)),
                'max_coverage': float(np.max(coverages))
            }
    
    stats['total_slices'] = sum(s['num_slices'] for s in stats['splits'].values())
    
    return stats


def validate_dataset(data_root):
    """
    Validate dataset integrity.
    Check that all required files exist and have correct shapes.
    
    Args:
        data_root: Path to processed data root directory
    
    Returns:
        List of validation issues (empty if valid)
    """
    
    data_root = Path(data_root)
    issues = []
    
    for split in ['train', 'val', 'test']:
        split_dir = data_root / split
        
        if not split_dir.exists():
            issues.append(f"Split directory missing: {split_dir}")
            continue
        
        ct_dir = split_dir / 'CT'
        pet_dir = split_dir / 'PET'
        
        if not ct_dir.exists():
            issues.append(f"CT directory missing: {ct_dir}")
            continue
        
        if not pet_dir.exists():
            issues.append(f"PET directory missing: {pet_dir}")
            continue
        
        ct_files = sorted(list(ct_dir.glob('*.npy')))
        pet_files = sorted(list(pet_dir.glob('*.npy')))
        
        # Check file counts match
        if len(ct_files) != len(pet_files):
            issues.append(
                f"Mismatched file counts in {split}: "
                f"CT={len(ct_files)}, PET={len(pet_files)}"
            )
        
        # Check file shapes
        print(f"Validating {split} split...")
        for ct_file in tqdm(ct_files):
            try:
                ct_data = np.load(ct_file)
                pet_file = pet_dir / ct_file.name
                pet_data = np.load(pet_file)
                
                # Check shapes
                if ct_data.shape != (64, 64):
                    issues.append(
                        f"Invalid CT shape in {ct_file}: {ct_data.shape}"
                    )
                
                if pet_data.shape != (64, 64):
                    issues.append(
                        f"Invalid PET shape in {pet_file}: {pet_data.shape}"
                    )
                
                # Check value ranges (should be normalized to [-1, 1])
                if ct_data.min() < -1.1 or ct_data.max() > 1.1:
                    issues.append(
                        f"CT values out of range in {ct_file}: "
                        f"[{ct_data.min()}, {ct_data.max()}]"
                    )
                
                if pet_data.min() < -1.1 or pet_data.max() > 1.1:
                    issues.append(
                        f"PET values out of range in {pet_file}: "
                        f"[{pet_data.min()}, {pet_data.max()}]"
                    )
                
            except Exception as e:
                issues.append(f"Error loading {ct_file}: {e}")
    
    return issues


def print_statistics(stats):
    """Pretty print dataset statistics."""
    
    print("\n" + "="*70)
    print("DATASET STATISTICS")
    print("="*70)
    
    print("\nSplit Information:")
    print("-" * 70)
    total_slices = 0
    for split, info in stats['splits'].items():
        num_slices = info['num_slices']
        total_slices += num_slices
        print(f"  {split:10s}: {num_slices:7d} slices")
    print(f"  {'TOTAL':10s}: {total_slices:7d} slices")
    
    print("\nCT Statistics (Normalized to [-1, 1]):")
    print("-" * 70)
    ct_stats = stats['ct_stats']
    for key in ['min', 'max', 'mean', 'std', 'percentile_25', 'percentile_50', 'percentile_75']:
        if key in ct_stats:
            print(f"  {key:20s}: {ct_stats[key]:10.6f}")
    
    print("\nPET Statistics (Normalized to [-1, 1]):")
    print("-" * 70)
    pet_stats = stats['pet_stats']
    for key in ['min', 'max', 'mean', 'std', 'percentile_25', 'percentile_50', 'percentile_75']:
        if key in pet_stats:
            print(f"  {key:20s}: {pet_stats[key]:10.6f}")
    
    print("\nAttention Map Coverage:")
    print("-" * 70)
    for split, coverage in stats['attention_coverage'].items():
        print(f"  {split}:")
        for key, value in coverage.items():
            print(f"    {key:20s}: {value:10.6f}")
    
    print("\n" + "="*70)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Validate and analyze dataset')
    parser.add_argument('--data-root', type=str, required=True,
                       help='Path to processed data root directory')
    parser.add_argument('--validate', action='store_true',
                       help='Validate dataset integrity')
    parser.add_argument('--stats', action='store_true', default=True,
                       help='Compute dataset statistics')
    parser.add_argument('--save-stats', type=str, default=None,
                       help='Save statistics to JSON file')
    parser.add_argument('--max-samples', type=int, default=None,
                       help='Maximum samples per split (for testing)')
    
    args = parser.parse_args()
    
    print(f"Processing dataset: {args.data_root}")
    
    if args.validate:
        print("\nValidating dataset...")
        issues = validate_dataset(args.data_root)
        
        if issues:
            print("\n⚠️  Validation Issues Found:")
            for issue in issues:
                print(f"  - {issue}")
        else:
            print("\n✅ Dataset validation passed!")
    
    if args.stats:
        print("\nComputing statistics...")
        stats = get_dataset_statistics(args.data_root)
        print_statistics(stats)
        
        if args.save_stats:
            with open(args.save_stats, 'w') as f:
                json.dump(stats, f, indent=2)
            print(f"\nStatistics saved to: {args.save_stats}")
