"""
Dataset for attention map training.
Generates binary attention maps from PET images on-the-fly.
"""

import torch
import random
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
from scipy.ndimage import zoom, binary_closing, generate_binary_structure


class AttentionMapDataset(Dataset):
    """
    Dataset for training attention map generation from CT.

    Target = **focal high-uptake** mask: PET above an *absolute* SUV threshold
    (default SUV>2.0). This replaces the old "PET > 75th percentile" rule, which
    computed the percentile over the *whole* slice — and since only ~25% of the
    background-cropped frame is body, the 75th percentile landed at the body/air
    boundary, so the target was effectively the entire body. The UNet learned that
    trivially (IoU≈0.9) but it carried no localization, so CPDM, conditioned on a
    body-shaped blob, collapsed to a diffuse central cloud. An absolute SUV cut
    yields a genuinely focal target (~3% of the frame) that localizes the high-uptake
    organs/lesions the diffusion must place. See NOTES_BA.md.

    PET on disk is normalized SUV[0,32]→[-1,1], so SUV s ↔ normalized (s/32)*2 − 1.
    """

    PET_SUV_MAX = 32.0  # matches preprocess_autopet PET clip ceiling

    def __init__(self, root_dir, split='train', image_size=64, flip=False, suv_threshold=2.0):
        """
        Args:
            root_dir: Root directory containing processed data (CT/PET/Labels)
            split: One of 'train', 'val', 'test'
            image_size: Target image size (assumes square)
            flip: Apply random horizontal flips
            suv_threshold: absolute SUV above which a voxel is "high uptake"
        """
        self.root_dir = Path(root_dir) / split
        self.image_size = (image_size, image_size) if isinstance(image_size, int) else image_size
        self.flip = flip and (split == 'train')  # Only flip during training
        self.suv_threshold = suv_threshold
        
        # Set up directory paths
        self.ct_dir = self.root_dir / 'CT'
        self.pet_dir = self.root_dir / 'PET'
        
        if not self.ct_dir.exists() or not self.pet_dir.exists():
            raise FileNotFoundError(f"CT or PET directory not found in {self.root_dir}")
        
        # Get list of slices
        self.slice_names = sorted([f.name for f in self.ct_dir.glob('*.npy')])
        
        if len(self.slice_names) == 0:
            raise ValueError(f"No slices found in {self.ct_dir}")
    
    def __len__(self):
        return len(self.slice_names)
    
    def _generate_attention_map(self, pet_slice, threshold=None):
        """
        Generate a focal high-uptake attention map from a PET slice.
        Threshold is an absolute SUV (converted to the [-1,1] normalized scale).
        """
        if threshold is None:
            # SUV -> normalized: (suv / SUV_MAX) * 2 - 1
            threshold = (self.suv_threshold / self.PET_SUV_MAX) * 2.0 - 1.0

        attention_map = (pet_slice > threshold).astype(np.float32)
        
        # Apply morphological closing to fill small holes
        if attention_map.sum() > 0:  # Only if there's content
            attention_map = binary_closing(
                attention_map, 
                structure=generate_binary_structure(2, 2)
            ).astype(np.float32)
        
        return attention_map
    
    def __getitem__(self, index):
        slice_name = self.slice_names[index]
        
        ct_path = self.ct_dir / slice_name
        pet_path = self.pet_dir / slice_name
        
        # Load data (already normalized to [-1, 1])
        ct_data = np.load(ct_path)  # Shape: (64, 64), range: [-1, 1]
        pet_data = np.load(pet_path)  # Shape: (64, 64), range: [-1, 1]
        
        # Generate attention map from PET
        attention_map = self._generate_attention_map(pet_data)
        
        # Apply random flip
        if self.flip and random.random() > 0.5:
            # .copy() because np.fliplr returns a negative-stride view that torch.from_numpy can't ingest
            ct_data = np.fliplr(ct_data).copy()
            attention_map = np.fliplr(attention_map).copy()
        
        # Convert to tensors with channel dimension
        ct_tensor = torch.from_numpy(ct_data[np.newaxis, :, :]).float()  # (1, 64, 64)
        attention_tensor = torch.from_numpy(attention_map[np.newaxis, :, :]).float()  # (1, 64, 64)
        
        # Resize if needed (usually already 64x64)
        if ct_tensor.shape[1:] != self.image_size:
            ct_tensor = torch.nn.functional.interpolate(
                ct_tensor.unsqueeze(0), size=self.image_size, mode='bilinear', align_corners=False
            ).squeeze(0)
            attention_tensor = torch.nn.functional.interpolate(
                attention_tensor.unsqueeze(0), size=self.image_size, mode='nearest'
            ).squeeze(0)
        
        return {
            'ct': ct_tensor,
            'attention_map': attention_tensor,
            'slice_name': slice_name
        }


if __name__ == '__main__':
    # Test dataset
    dataset = AttentionMapDataset(
        root_dir='/home/kacperzaleski/Projects/practical-work-CT2PET/data/processed',
        split='train',
        image_size=64,
        flip=True
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # Get sample
    sample = dataset[0]
    print(f"CT shape: {sample['ct'].shape}")
    print(f"Attention map shape: {sample['attention_map'].shape}")
    print(f"Attention map unique values: {torch.unique(sample['attention_map'])}")
