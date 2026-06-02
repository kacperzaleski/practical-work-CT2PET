"""
Registered dataset for paired CT/PET .npy slices produced by preprocess_autopet.py.

On-disk layout:
    {data_root}/{stage}/CT/<slice>.npy   - float32, shape (H, W), values in [-1, 1]
    {data_root}/{stage}/PET/<slice>.npy  - float32, shape (H, W), values in [-1, 1]

Returned batch element matches the CPDM runner's expectation
    ((x, x_name), (x_cond, x_cond_name)) = batch
where x = PET (target), x_cond = CT (conditioning). x_name is the slice stem,
which the CPDM model uses to look up pre-exported attention maps by filename.
"""

import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from Register import Registers


@Registers.datasets.register_with_name('ct2pet_aligned')
class CT2PETAlignedDataset(Dataset):
    def __init__(self, dataset_config, stage='train'):
        super().__init__()
        self.data_root = Path(dataset_config.data_root)
        self.stage = stage
        self.image_size = int(dataset_config.image_size)
        self.flip = bool(getattr(dataset_config, 'flip', False)) if stage == 'train' else False

        ct_dir = self.data_root / stage / 'CT'
        pet_dir = self.data_root / stage / 'PET'
        if not ct_dir.is_dir() or not pet_dir.is_dir():
            raise FileNotFoundError(f'Missing CT or PET dir under {self.data_root}/{stage}')

        ct_names = sorted(f.name for f in ct_dir.glob('*.npy'))
        pet_set = {f.name for f in pet_dir.glob('*.npy')}
        self.names = [n for n in ct_names if n in pet_set]

        # Optional cap on dataset size — useful for CPU training to keep an "epoch" bounded.
        # Configured via dataset_config.max_samples (per-stage dict or single int).
        max_samples = getattr(dataset_config, 'max_samples', None)
        if isinstance(max_samples, dict):
            max_samples = max_samples.get(stage)
        elif hasattr(max_samples, stage):
            max_samples = getattr(max_samples, stage)
        if max_samples is not None and len(self.names) > int(max_samples):
            self.names = self.names[:int(max_samples)]

        self.ct_dir = ct_dir
        self.pet_dir = pet_dir

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        name = self.names[idx]
        ct = np.load(self.ct_dir / name)
        pet = np.load(self.pet_dir / name)

        if self.flip and random.random() < 0.5:
            ct = np.fliplr(ct).copy()
            pet = np.fliplr(pet).copy()

        ct_t = torch.from_numpy(ct[np.newaxis, :, :]).float()
        pet_t = torch.from_numpy(pet[np.newaxis, :, :]).float()

        if ct_t.shape[-2:] != (self.image_size, self.image_size):
            ct_t = F.interpolate(ct_t.unsqueeze(0), size=self.image_size,
                                 mode='bilinear', align_corners=False).squeeze(0)
            pet_t = F.interpolate(pet_t.unsqueeze(0), size=self.image_size,
                                  mode='bilinear', align_corners=False).squeeze(0)

        stem = Path(name).stem
        return (pet_t, stem), (ct_t, stem)
