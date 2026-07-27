"""
Preprocess the autoPET dataset from raw NIfTI files to organized 2D .npy slices.

Three extraction modes (``--body-region``):
  * ``brain``      — legacy: keep only the last ``--brain-percent`` of the axial
    stack (the head). Used by the original CPDM experiments.
  * ``head_torso`` — keep the top ``--keep-top-fraction`` of the *foreground*
    slices (vertex down through the pelvis), dropping the legs. This is the
    default for the flow-based (PMRF) thesis experiments: it keeps the brain
    plus the FDG-rich thorax/abdomen/pelvis while staying tractable CPU-only.
  * ``full``       — whole-body, foreground-filtered: every slice with real
    tissue. Largest / most diverse; heaviest to train.

In all foreground modes a slice is kept only if its body-voxel fraction
(raw CT HU > -500) >= ``--min-foreground``, which drops air-only slices.

By default each volume is cropped to its body bounding box (background removal)
before resizing, so anatomy fills the frame instead of air — disable with
``--no-crop-body``. Slices are then resized to ``--target-size`` (default
128x128) and normalized to [-1, 1] (CT via HU window, PET via SUV window). The
CT/PET normalization is identical in all modes so models stay comparable.

Typical use:
  # legacy brain-only set (CPDM), 64x64
  python preprocess_autopet.py --output-dir data/processed \
      --body-region brain --target-size 64

  # head+torso set for the flow experiments, 128x128
  python preprocess_autopet.py --output-dir data/processed_fullbody \
      --body-region head_torso --target-size 128
"""

import sys as _sys, pathlib as _pathlib  # repo-root bootstrap (script moved into a subfolder)
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))
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


def is_foreground(ct_slice, body_hu=-500.0, min_fraction=0.02):
    """
    Decide whether a raw CT axial slice contains enough body tissue to keep.

    Air is ~-1000 HU and the scanner bed/table is sparse, so a slice whose body
    fraction (voxels above ``body_hu``) exceeds ``min_fraction`` is considered
    real anatomy. Operates on raw (un-normalized) HU values.
    """
    body = ct_slice > body_hu
    return body.mean() >= min_fraction


def compute_body_bbox(ct_volume, slice_indices, body_hu=-500.0, margin=0.05):
    """
    Per-volume body bounding box (background removal).

    The autoPET CT field-of-view is mostly air: the patient occupies a centred
    blob and the rest is ~-1000 HU. A full-frame resize therefore spends most of
    its pixels on background, which (a) wastes the limited 64x64 grid and (b)
    lets a model score well by trivially predicting the constant background.

    We threshold the body (HU > ``body_hu``) on every KEPT slice of a patient and
    take the UNION over the volume, so a single crop window is shared by all of
    that patient's slices (consistent zoom across the stack — a per-slice box
    would zoom each slice differently and break anatomical scale). A relative
    ``margin`` is added so skin/edges are not clipped. Returns ``(rmin, rmax,
    cmin, cmax)`` inclusive on the in-plane grid, or ``None`` if no body is found.
    """
    H, W = ct_volume.shape[0], ct_volume.shape[1]
    mask = np.zeros((H, W), dtype=bool)
    for i in slice_indices:
        mask |= (ct_volume[:, :, i] > body_hu)
    if not mask.any():
        return None
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    rmin, rmax = int(rows[0]), int(rows[-1])
    cmin, cmax = int(cols[0]), int(cols[-1])
    mr = int(round((rmax - rmin + 1) * margin))
    mc = int(round((cmax - cmin + 1) * margin))
    rmin = max(0, rmin - mr); rmax = min(H - 1, rmax + mr)
    cmin = max(0, cmin - mc); cmax = min(W - 1, cmax + mc)
    return rmin, rmax, cmin, cmax


def crop_to_square(slice2d, bbox, pad_value):
    """
    Crop ``slice2d`` to ``bbox`` then pad to a centred square with ``pad_value``.

    Padding (rather than a non-uniform resize) keeps the anatomical aspect ratio
    so the body is not stretched. The same ``bbox`` is used for CT/PET/Labels of
    a patient, so the modalities stay pixel-aligned; only the constant fill
    differs (air = raw clip_min for CT, 0 for PET/Labels).
    """
    rmin, rmax, cmin, cmax = bbox
    crop = slice2d[rmin:rmax + 1, cmin:cmax + 1]
    h, w = crop.shape
    side = max(h, w)
    out = np.full((side, side), pad_value, dtype=slice2d.dtype)
    r0 = (side - h) // 2
    c0 = (side - w) // 2
    out[r0:r0 + h, c0:c0 + w] = crop
    return out


def process_autopet_dataset(
    raw_dir,
    output_dir,
    split_file,
    target_size=128,
    body_region='head_torso',
    brain_percent=0.2,
    keep_top_fraction=0.6,
    min_foreground=0.02,
    crop_body=True,
    crop_margin=0.05,
    save_labels=True,
    skip_existing=False,
    max_patients=None,
):
    """
    Process the autoPET dataset from raw NIfTI files to 2D .npy slices.

    Args:
        raw_dir: Path to raw autoPET directory (expects imagesTr/, labelsTr/).
        output_dir: Output directory for processed data.
        split_file: Path to the splits_*.json file.
        target_size: Target square size for 2D slices (default 128).
        body_region: 'head_torso' (top keep_top_fraction of foreground slices),
            'full' (all foreground slices), or 'brain' (legacy: last
            brain_percent of the stack).
        brain_percent: Fraction of trailing slices kept in 'brain' mode.
        keep_top_fraction: In 'head_torso' mode, fraction of foreground slices
            kept from the head end (drops the legs at the low-index end).
        min_foreground: Min body-voxel fraction to keep a slice (foreground modes).
        crop_body: Crop each volume to its per-volume body bounding box (square,
            padded) before resizing — removes the air background so anatomy fills
            the frame. See ``compute_body_bbox`` / ``crop_to_square``.
        crop_margin: Relative margin added around the body bbox (crop_body mode).
        save_labels: Also write the autoPET segmentation label slices.
        skip_existing: Skip patients whose output slices already exist (resume).
        max_patients: Cap patients per split (smoke tests); None = all.
    """

    # Create output directories
    modalities = ['CT', 'PET'] + (['Labels'] if save_labels else [])
    for split in ['train', 'val', 'test']:
        split_dir = Path(output_dir) / split
        for modality in modalities:
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
            safe_patient_id = patient_id.replace(' ', '_')

            # Resume support: skip a patient whose slices are already on disk.
            if skip_existing and next(
                (output_split_dir / 'CT').glob(f"{safe_patient_id}_s*.npy"), None
            ) is not None:
                patient_counts[split_name] += 1
                continue

            # Build file paths
            ct_file = f"{patient_id}_0000.nii.gz"
            pet_file = f"{patient_id}_0001.nii.gz"
            label_file = f"{patient_id}.nii.gz"

            ct_path = images_dir / ct_file
            pet_path = images_dir / pet_file
            label_path = labels_dir / label_file

            # Check files exist (label only required if we're saving it)
            need = [ct_path.exists(), pet_path.exists()]
            if save_labels:
                need.append(label_path.exists())
            if not all(need):
                continue

            try:
                ct_data = load_nifti_volume(ct_path)
                pet_data = load_nifti_volume(pet_path)
                label_data = load_nifti_volume(label_path) if save_labels else None
            except Exception as e:
                print(f"Error loading {patient_id}: {e}")
                continue

            # Verify shapes match
            shapes_ok = ct_data.shape == pet_data.shape
            if save_labels:
                shapes_ok = shapes_ok and ct_data.shape == label_data.shape
            if not shapes_ok:
                print(f"Shape mismatch for {patient_id}")
                continue

            num_slices = ct_data.shape[2]

            # Choose which axial indices to keep. The head is at the HIGH-index
            # end of axis 2 (brain mode historically took the trailing slices).
            if body_region == 'brain':
                num_brain_slices = int(num_slices * brain_percent)
                slice_indices = list(range(num_slices - num_brain_slices, num_slices))
            else:
                # Foreground slices (real tissue), then optionally drop the legs.
                fg = [i for i in range(num_slices)
                      if is_foreground(ct_data[:, :, i], min_fraction=min_foreground)]
                if body_region == 'head_torso' and fg:
                    n_keep = max(1, int(round(len(fg) * keep_top_fraction)))
                    slice_indices = fg[len(fg) - n_keep:]   # top = head-ward
                else:  # 'full'
                    slice_indices = fg

            # One body bbox per volume (shared by every kept slice) so the crop
            # zoom is consistent across the stack. None -> fall back to full frame.
            bbox = compute_body_bbox(ct_data, slice_indices, margin=crop_margin) \
                if crop_body else None

            for i in slice_indices:
                slice_ct = ct_data[:, :, i]
                slice_pet = pet_data[:, :, i]

                if body_region == 'brain':
                    # Legacy skip: both modalities essentially empty.
                    if slice_ct.max() < 1e-6 and slice_pet.max() < 1e-6:
                        continue

                # Background removal: crop to the body bbox, pad to a square with
                # air (CT clip_min / PET 0) so resizing keeps the aspect ratio.
                if bbox is not None:
                    slice_ct = crop_to_square(slice_ct, bbox, pad_value=-1000.0)
                    slice_pet = crop_to_square(slice_pet, bbox, pad_value=0.0)

                # Resize to target size
                ct_resized = zoom(slice_ct, target_size / slice_ct.shape[0])
                pet_resized = zoom(slice_pet, target_size / slice_pet.shape[0])

                # Normalize
                ct_normalized = normalize_ct(ct_resized)
                pet_normalized = normalize_pet(pet_resized)

                slice_filename = f"{safe_patient_id}_s{i:03d}.npy"

                np.save(output_split_dir / 'CT' / slice_filename, ct_normalized)
                np.save(output_split_dir / 'PET' / slice_filename, pet_normalized)
                if save_labels:
                    slice_label = label_data[:, :, i]
                    if bbox is not None:
                        slice_label = crop_to_square(slice_label, bbox, pad_value=0.0)
                    label_resized = zoom(slice_label, target_size / slice_label.shape[0], order=0)
                    np.save(output_split_dir / 'Labels' / slice_filename, label_resized.astype(np.uint8))

                slice_counts[split_name] += 1

            patient_counts[split_name] += 1

    # Print summary
    print("\n" + "=" * 70)
    print("Dataset Preprocessing Complete!")
    print("=" * 70)
    if body_region == 'brain':
        mode_detail = f" (last {brain_percent:.0%})"
    elif body_region == 'head_torso':
        mode_detail = f" (top {keep_top_fraction:.0%} of foreground, min_fg={min_foreground})"
    else:
        mode_detail = f" (min_foreground={min_foreground})"
    crop_detail = f"body-bbox crop (margin={crop_margin:.0%})" if crop_body else "full-frame (no crop)"
    print(f"\nMode: body_region='{body_region}'{mode_detail}, target_size={target_size}, {crop_detail}")
    print("\nPatient counts:")
    for split, count in patient_counts.items():
        print(f"  {split:6s}: {count:5d} patients")
    print(f"\nSlice counts ({target_size}x{target_size}):")
    for split, count in slice_counts.items():
        print(f"  {split:6s}: {count:7d} slices")
    print(f"\nTotal slices: {sum(slice_counts.values()):7d}")
    print(f"Output directory: {output_dir}")
    print("=" * 70)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess autoPET dataset to 2D slices')
    parser.add_argument('--raw-dir', type=str,
                        default='data/raw/autoPET',
                        help='Path to raw autoPET directory')
    parser.add_argument('--output-dir', type=str,
                        default='data/processed_fullbody',
                        help='Path to output directory')
    parser.add_argument('--split-file', type=str,
                        default=None,
                        help='Path to splits file (auto-detected if not provided)')
    parser.add_argument('--target-size', type=int, default=128,
                        help='Target size for 2D slices')
    parser.add_argument('--body-region', type=str, default='head_torso',
                        choices=['head_torso', 'full', 'brain'],
                        help="'head_torso' = top --keep-top-fraction of foreground "
                             "(drops legs); 'full' = whole-body foreground-filtered; "
                             "'brain' = legacy last --brain-percent of stack")
    parser.add_argument('--brain-percent', type=float, default=0.2,
                        help='Fraction of trailing slices kept in brain mode')
    parser.add_argument('--keep-top-fraction', type=float, default=0.6,
                        help='Fraction of foreground slices kept (head end) in head_torso mode')
    parser.add_argument('--min-foreground', type=float, default=0.02,
                        help='Min body-voxel fraction to keep a slice (foreground modes)')
    parser.add_argument('--no-crop-body', action='store_true',
                        help='Disable per-volume body-bbox crop (keep full-frame resize). '
                             'Crop is ON by default: it removes the air background.')
    parser.add_argument('--crop-margin', type=float, default=0.05,
                        help='Relative margin around the body bbox when cropping')
    parser.add_argument('--no-labels', action='store_true',
                        help='Do not write segmentation label slices')
    parser.add_argument('--skip-existing', action='store_true',
                        help='Skip patients already on disk (resume an interrupted run)')
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
        body_region=args.body_region,
        brain_percent=args.brain_percent,
        keep_top_fraction=args.keep_top_fraction,
        min_foreground=args.min_foreground,
        crop_body=not args.no_crop_body,
        crop_margin=args.crop_margin,
        save_labels=not args.no_labels,
        skip_existing=args.skip_existing,
        max_patients=args.max_patients,
    )
