import os
import nibabel as nib
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm

def preprocess_brain_slices(raw_data_dir, output_dir, split_file, brain_percent=0.2, max_patients=None):
    images_tr_dir = os.path.join(raw_data_dir, 'imagesTr')
    labels_tr_dir = os.path.join(raw_data_dir, 'labelsTr')
    
    with open(split_file, 'r') as f:
        splits = json.load(f)[0] # Take the first split
        
    for split_name in ['train', 'val', 'test']:
        if split_name not in splits:
            continue
        patient_ids = splits[split_name]
        if max_patients:
            patient_ids = patient_ids[:max_patients]
            
        print(f"Processing {len(patient_ids)} patients for {split_name} split...")
        
        split_out_dir = os.path.join(output_dir, split_name)
        out_dirs = {
            'CT': os.path.join(split_out_dir, 'CT'),
            'PET': os.path.join(split_out_dir, 'PET'),
            'Labels': os.path.join(split_out_dir, 'Labels')
        }
        
        for d in out_dirs.values():
            os.makedirs(d, exist_ok=True)
            
        for patient_id in tqdm(patient_ids):
            ct_file = f"{patient_id}_0000.nii.gz"
            pet_file = f"{patient_id}_0001.nii.gz"
            label_file = f"{patient_id}.nii.gz"
            
            ct_path = os.path.join(images_tr_dir, ct_file)
            pet_path = os.path.join(images_tr_dir, pet_file)
            label_path = os.path.join(labels_tr_dir, label_file)
            
            if not all(os.path.exists(p) for p in [ct_path, pet_path, label_path]):
                # print(f"Missing files for patient {patient_id}, skipping.")
                continue
                
            try:
                # Use nibabel to load. We use proxy loading to save memory if possible, 
                # but get_fdata() loads everything.
                ct_obj = nib.load(ct_path)
                pet_obj = nib.load(pet_path)
                label_obj = nib.load(label_path)
                
                ct_data = ct_obj.get_fdata()
                pet_data = pet_obj.get_fdata()
                label_data = label_obj.get_fdata()
            except Exception as e:
                print(f"Error loading {patient_id}: {e}")
                continue
                
            if not (ct_data.shape == pet_data.shape == label_data.shape):
                print(f"Shape mismatch for {patient_id}: {ct_data.shape}, {pet_data.shape}, {label_data.shape}")
                continue
                
            num_slices = ct_data.shape[2]
            num_brain_slices = int(num_slices * brain_percent)
            start_slice = num_slices - num_brain_slices
            
            # Clean patient_id for filename (remove spaces etc if any)
            safe_patient_id = patient_id.replace(' ', '_')
            
            for i in range(start_slice, num_slices):
                slice_ct = ct_data[:, :, i]
                slice_pet = pet_data[:, :, i]
                slice_label = label_data[:, :, i]
                
                slice_filename = f"{safe_patient_id}_s{i:03d}.npy"
                
                np.save(os.path.join(out_dirs['CT'], slice_filename), slice_ct.astype(np.float32))
                np.save(os.path.join(out_dirs['PET'], slice_filename), slice_pet.astype(np.float32))
                np.save(os.path.join(out_dirs['Labels'], slice_filename), slice_label.astype(np.uint8))

if __name__ == "__main__":
    RAW_DIR = 'data/raw/autoPET'
    OUT_DIR = 'data/processed/brain_slices'
    SPLIT_FILE = os.path.join(RAW_DIR, 'splits_80_10_10.json')
    
    # Set max_patients to None to process all patients, or a number for testing
    MAX_PATIENTS = None 
    preprocess_brain_slices(RAW_DIR, OUT_DIR, SPLIT_FILE, max_patients=MAX_PATIENTS)
