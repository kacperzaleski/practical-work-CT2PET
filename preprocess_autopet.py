"""
Preprocess autoPET dataset from raw NIfTI files to organized 2D brain slices.
Extracts 64x64 slices from brain region (last 20% of full-body scans).
Includes normalization to [-1, 1]. Attention maps are generated on-the-fly during training.
"""

import os
import json
import numpy as np
import nibabel as nib
from pathlib import Path
from tqdm import tqdm
import argparse
from scipy.ndimage import zoom, binary_closing, generate_binary_structure


def load_split_file(split_file):
    """Load the train/val/test split information."""
    with open(split_file, 'r') as f:
        return json.load(f)[0]


def get_study_name(filename):
    """Extract study name from filename."""
    # Remove _0000.nii.gz or _0001.nii.gz suffix
    return filename.replace('_0000.nii.gz', '').replace('_0001.nii.gz', '')


def load_nifti_volume(filepath):
    """Load a NIfTI file and return the data."""
    img = nib.load(filepath)
    data = img.get_fdata()
    return data


def normalize_ct(ct_data, clip_min=-1000, clip_max=3071):
    """Normalize CT data to [-1, 1] range using HU units."""
    ct_clipped = np.clip(ct_data, clip_min, clip_max)
    ct_normalized = (ct_clipped - clip_min) / (clip_max - clip_min) * 2 - 1
    return ct_normalized.astype(np.float32)


def normalize_pet(pet_data, pet_max=32):
    """Normalize PET data to [-1, 1] range using SUV."""
    pet_clipped = np.clip(pet_data, 0, pet_max)
    pet_normalized = (pet_clipped / pet_max) * 2 - 1
    return pet_normalized.astype(np.float32)

# might be useful later
def create_attention_map(pet_slice, threshold=None):
    """
    Create attention map from PET slice.
    Highlights high-uptake regions (areas of interest).
    """
    if threshold is None:
        threshold = np.percentile(pet_slice, 75)
    
    attention_map = (pet_slice > threshold).astype(np.uint8)
    
    # Apply morphological closing to fill small holes
    attention_map = binary_closing(attention_map, structure=generate_binary_structure(2, 2))
    
    return attention_map.astype(np.uint8)


def process_autopet_dataset(
    raw_dir,
    output_dir,
    split_file,
    target_size=64,
    brain_percent=0.2,
    max_patients=None
):
    """
    Process autoPET dataset from raw NIfTI files.
    Extracts brain slices (last brain_percent of volume) and resizes to target_size.
    Data is normalized to [-1, 1] range.
    Attention maps are generated on-the-fly during training.
    
    Args:
        raw_dir: Path to raw autoPET directory
        output_dir: Path to output directory where processed data will be saved
        split_file: Path to splits_80_10_10.json file
        target_size: Target size for 2D slices (default 64)
        brain_percent: Fraction of slices from end = brain region (default 0.2 = 20%)
        max_patients: Maximum patients per split for testing (default None = all)
    """
    
    # Create output directories
    for split in ['train', 'val', 'test']:
        split_dir = Path(output_dir) / split
        for modality in ['CT', 'PET', 'Labels']:
            (split_dir / modality).mkdir(parents=True, exist_ok=True)
    
    # Load splits
    splits = load_split_file(split_file)
    
    # Set up paths
    images_dir = Path(raw_dir) / 'imagesTr'
    labels_dir = Path(raw_dir) / 'labelsTr'
    
    # Process each split
    slice_counts = {'train': 0, 'val': 0, 'test': 0}
    patient_counts = {'train': 0, 'val': 0, 'test': 0}
    
    for split_name, split_patients in splits.items():
        if max_patients:
            split_patients = split_patients[:max_patients]
        
        print(f"\nProcessing {split_name} split ({len(split_patients)} patients)...")
        
        output_split_dir = Path(output_dir) / split_name
        
        for patient_id in tqdm(split_patients):
            # Build file paths
            ct_file = f"{patient_id}_0000.nii.gz"
            pet_file = f"{patient_id}_0001.nii.gz"
            label_file = f"{patient_id}.nii.gz"
            
            ct_path = images_dir / ct_file
            pet_path = images_dir / pet_file
            label_path = labels_dir / label_file
            
            # Check all files exist
            if not all([ct_path.exists(), pet_path.exists(), label_path.exists()]):
                continue
            
            try:
                # Load volumes
                ct_data = load_nifti_volume(ct_path)
                pet_data = load_nifti_volume(pet_path)
                label_data = load_nifti_volume(label_path)
                
            except Exception as e:
                print(f"Error loading {patient_id}: {e}")
                continue
            
            # Verify shapes match
            if not (ct_data.shape == pet_data.shape == label_data.shape):
                print(f"Shape mismatch for {patient_id}")
                continue
            
            # Extract brain slices: last brain_percent of volume
            num_slices = ct_data.shape[2]
            num_brain_slices = int(num_slices * brain_percent)
            start_slice = num_slices - num_brain_slices
            
            # Clean patient_id for filename
            safe_patient_id = patient_id.replace(' ', '_')
            
            # Process each brain slice
            for i in range(start_slice, num_slices):
                slice_ct = ct_data[:, :, i]
                slice_pet = pet_data[:, :, i]
                slice_label = label_data[:, :, i]
                
                # Skip empty slices
                if slice_ct.max() < 1e-6 and slice_pet.max() < 1e-6:
                    continue
                
                # Resize to target size
                ct_resized = zoom(slice_ct, target_size / slice_ct.shape[0])
                pet_resized = zoom(slice_pet, target_size / slice_pet.shape[0])
                label_resized = zoom(slice_label, target_size / slice_label.shape[0], order=0)
                
                # Normalize
                ct_normalized = normalize_ct(ct_resized)
                pet_normalized = normalize_pet(pet_resized)
                
                # Create attention map
                #attention_map = create_attention_map(pet_normalized)
                
                # Build filename
                slice_filename = f"{safe_patient_id}_s{i:03d}.npy"
                
                # Save
                np.save(output_split_dir / 'CT' / slice_filename, ct_normalized)
                np.save(output_split_dir / 'PET' / slice_filename, pet_normalized)
                np.save(output_split_dir / 'Labels' / slice_filename, label_resized.astype(np.uint8))
                #np.save(output_split_dir / 'AttentionMaps' / slice_filename, attention_map)
                
                slice_counts[split_name] += 1
            
            patient_counts[split_name] += 1
    
    # Print summary
    print("\n" + "="*70)
    print("Dataset Preprocessing Complete!")
    print("="*70)
    print("\nPatient counts:")
    for split, count in patient_counts.items():
        print(f"  {split:6s}: {count:5d} patients")
    print("\nSlice counts (64x64, brain region only):")
    for split, count in slice_counts.items():
        print(f"  {split:6s}: {count:7d} slices")
    print(f"\nTotal slices: {sum(slice_counts.values()):7d}")
    print(f"Output directory: {output_dir}")
    print("="*70)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess autoPET dataset to brain slices')
    parser.add_argument('--raw-dir', type=str, 
                       default='data/raw/autoPET',
                       help='Path to raw autoPET directory')
    parser.add_argument('--output-dir', type=str,
                       default='data/processed',
                       help='Path to output directory')
    parser.add_argument('--split-file', type=str,
                       default=None,
                       help='Path to splits file (auto-detected if not provided)')
    parser.add_argument('--target-size', type=int, default=64,
                       help='Target size for 2D slices')
    parser.add_argument('--brain-percent', type=float, default=0.2,
                       help='Fraction of slices from end = brain region')
    parser.add_argument('--max-patients', type=int, default=None,
                       help='Maximum patients per split (for testing)')
    
    args = parser.parse_args()
    
    # Auto-detect split file if not provided
    if args.split_file is None:
        args.split_file = os.path.join(args.raw_dir, 'splits_80_10_10.json')
    
    process_autopet_dataset(
        raw_dir=args.raw_dir,
        output_dir=args.output_dir,
        split_file=args.split_file,
        target_size=args.target_size,
        brain_percent=args.brain_percent,
        max_patients=args.max_patients
    )
