"""
Run the trained attention-map UNet over every CT slice and save the predicted
attention maps into data/processed/{split}/AttentionMaps/<slice_name>.npy.

Saved as float32 sigmoid probabilities in [0, 1], shape (64, 64), matching
the layout that BrainSliceDataset / the CPDM pipeline expect.
"""

import sys as _sys, pathlib as _pathlib  # repo-root bootstrap (script moved into a subfolder)
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))
import argparse
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from training.train_attention_map import UNetAttentionMapModel


class CTSliceDataset(Dataset):
    def __init__(self, ct_dir: Path):
        self.ct_dir = Path(ct_dir)
        self.names = sorted(f.name for f in self.ct_dir.glob('*.npy'))

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        name = self.names[idx]
        ct = np.load(self.ct_dir / name)
        ct = torch.from_numpy(ct[np.newaxis, :, :]).float()  # (1, 64, 64)
        return ct, name


def export_split(model, data_root: Path, split: str, batch_size: int, num_workers: int,
                 out_dir_name: str = 'AttentionMaps'):
    ct_dir = data_root / split / 'CT'
    out_dir = data_root / split / out_dir_name
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = CTSliceDataset(ct_dir)
    if len(dataset) == 0:
        print(f'[{split}] no CT slices found — skipping')
        return

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=False, drop_last=False,
    )

    print(f'[{split}] exporting {len(dataset)} attention maps -> {out_dir}')
    saved = 0
    with torch.no_grad():
        for ct_batch, names in tqdm(loader, total=len(loader), desc=split):
            logits = model(ct_batch)
            probs = torch.sigmoid(logits).squeeze(1).cpu().numpy().astype(np.float32)
            for prob, name in zip(probs, names):
                np.save(out_dir / name, prob)
                saved += 1
    print(f'[{split}] saved {saved} maps')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, required=True, help='Lightning .ckpt checkpoint')
    parser.add_argument('--data-root', type=str, default='data/processed')
    parser.add_argument('--splits', type=str, nargs='+', default=['train', 'val', 'test'])
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--out-dir-name', type=str, default='AttentionMaps',
                        help="subfolder under each split to write maps into "
                             "(use a new name to preserve existing maps)")
    args = parser.parse_args()

    torch.set_num_threads(max(1, (os.cpu_count() or 1) - args.num_workers))

    print(f'Loading checkpoint: {args.ckpt}')
    model = UNetAttentionMapModel.load_from_checkpoint(args.ckpt, map_location='cpu')
    model.eval()
    model.freeze()

    data_root = Path(args.data_root)
    for split in args.splits:
        export_split(model, data_root, split, args.batch_size, args.num_workers, args.out_dir_name)

    print('Done.')


if __name__ == '__main__':
    main()
