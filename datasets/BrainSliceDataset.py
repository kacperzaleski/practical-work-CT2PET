import os
import torch
import random
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
import torchvision.transforms as transforms
from PIL import Image

class BrainSliceDataset(Dataset):
    """
    Dataset for brain slices extracted from CT and PET volumes.
    Organized as:
    root_dir/
        split/
            CT/
                patient_id_sXXX.npy
            PET/
                patient_id_sXXX.npy
            Labels/
                patient_id_sXXX.npy
    """
    def __init__(self, root_dir, split='train', image_size=(256, 256), 
                 ct_max=2047, pet_max=5.0, to_normal=False, flip=False):
        self.root_dir = Path(root_dir) / split
        self.image_size = image_size
        self.ct_max = float(ct_max)
        self.pet_max = float(pet_max)
        self.to_normal = to_normal
        self.flip = flip
        
        self.ct_dir = self.root_dir / 'CT'
        self.pet_dir = self.root_dir / 'PET'
        self.label_dir = self.root_dir / 'Labels'
        
        if not self.ct_dir.exists():
            raise FileNotFoundError(f"CT directory not found: {self.ct_dir}")
            
        self.slice_names = sorted([f.name for f in self.ct_dir.glob('*.npy')])
        
    def __len__(self):
        return len(self.slice_names)
        
    def __getitem__(self, index):
        slice_name = self.slice_names[index]
        
        ct_path = self.ct_dir / slice_name
        pet_path = self.pet_dir / slice_name
        label_path = self.label_dir / slice_name
        
        # Load .npy files
        ct_slice = np.load(ct_path)
        pet_slice = np.load(pet_path)
        label_slice = np.load(label_path)
        
        # Clip CT values (optional but common for medical imaging)
        # Standard CT range is -1000 to 1000 or more.
        # We can normalize to [0, 1] using ct_max
        ct_slice = (ct_slice + 1000) / (self.ct_max + 1000)
        ct_slice = np.clip(ct_slice, 0, 1)
        
        # PET normalization (SUV values)
        pet_slice = pet_slice / self.pet_max
        pet_slice = np.clip(pet_slice, 0, 1)
        
        # To PIL for easy transforms
        # Values are now in [0, 1], so we multiply by 255 for PIL if needed, 
        # but transforms.ToTensor() handles [0, 1] floats if we convert carefully.
        # Actually PIL Image.fromarray likes [0, 255] uint8 for L mode or float32 for F mode.
        ct_img = Image.fromarray(ct_slice.astype(np.float32), mode='F')
        pet_img = Image.fromarray(pet_slice.astype(np.float32), mode='F')
        label_img = Image.fromarray(label_slice.astype(np.uint8), mode='L')
        
        if self.flip and random.random() > 0.5:
            ct_img = ct_img.transpose(Image.FLIP_LEFT_RIGHT)
            pet_img = pet_img.transpose(Image.FLIP_LEFT_RIGHT)
            label_img = label_img.transpose(Image.FLIP_LEFT_RIGHT)
            
        transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor()
        ])
        
        # Label needs nearest neighbor interpolation to preserve classes
        label_transform = transforms.Compose([
            transforms.Resize(self.image_size, interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor()
        ])
        
        ct_tensor = transform(ct_img)
        pet_tensor = transform(pet_img)
        label_tensor = label_transform(label_img)
        
        if self.to_normal:
            ct_tensor = (ct_tensor - 0.5) * 2.
            pet_tensor = (pet_tensor - 0.5) * 2.
            
        return ct_tensor, pet_tensor, label_tensor, slice_name
