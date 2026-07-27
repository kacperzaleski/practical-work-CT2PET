"""
Sample the concatenation-conditioned latent diffusion (Diffusion 2) on the test
set and dump [0,1] PET .npy for eval_masked.py (same layout as
sample_pmrf.py --save-npy), so every model is scored by the identical ruler.

  python sample_concat_diffusion.py --ckpt checkpoints/ConcatDiff/last.ckpt \
      --num-steps 200 --save-npy results/ConcatDiff/eval

DDIM-samples the PET latent from noise conditioned on the CT latent, undoes the
LDM scale_factor, decodes with the frozen VQGAN, and maps [-1,1] -> [0,1].
"""

import sys as _sys, pathlib as _pathlib  # repo-root bootstrap (script moved into a subfolder)
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))
import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from datasets.CT2PETAlignedDataset import CT2PETAlignedDataset
from training.train_concat_diffusion import ConcatDiffusionModule
import pmrf_common as C


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=str, default='config/ConcatDiff-autoPET-fb64.yaml')
    p.add_argument('--data-root', type=str, default='data/processed_fullbody')
    p.add_argument('--ckpt', type=str, required=True)
    p.add_argument('--num-steps', type=int, default=None,
                   help='DDIM steps (default: config sample_step).')
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--num-workers', type=int, default=2)
    p.add_argument('--max-samples', type=int, default=None)
    p.add_argument('--names-file', type=str, default=None,
                   help='Restrict the test set to the slice stems in this manifest '
                        '(shared multi-patient set; see build_test_manifest.py).')
    p.add_argument('--save-npy', type=str, default=None,
                   help='Dump pred/gt PET as [0,1] .npy under <dir>/{pred,gt}.')
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    torch.manual_seed(args.seed)
    device = 'cpu'

    cfg = C.load_config(args.config)
    data_cfg = dict(cfg['data']['dataset_config'])
    data_cfg['data_root'] = args.data_root
    if args.max_samples is not None:
        data_cfg['max_samples'] = args.max_samples
    if args.names_file is not None:
        data_cfg['test_names_file'] = args.names_file
    ds = CT2PETAlignedDataset(argparse.Namespace(**data_cfg), stage='test')
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers)

    model = ConcatDiffusionModule.load_from_checkpoint(args.ckpt, map_location=device).eval()
    steps = args.num_steps or model.sample_step
    print(f'test samples: {len(ds)}  DDIM steps: {steps}  '
          f'scale_factor={model.scale_factor.item():.4f}', flush=True)

    pred_dir = gt_dir = None
    if args.save_npy:
        pred_dir = os.path.join(args.save_npy, 'pred')
        gt_dir = os.path.join(args.save_npy, 'gt')
        os.makedirs(pred_dir, exist_ok=True)
        os.makedirs(gt_dir, exist_ok=True)

    with torch.no_grad():
        for (pet, names), (ct, _) in loader:
            ct, pet = ct.to(device), pet.to(device)
            cond = model.encode(ct) * model.scale_factor
            shape = (ct.size(0), cond.size(1), cond.size(2), cond.size(3))
            lat = model.diffusion.ddim_sample(model.unet, shape, cond, steps, device)
            pred = model.decode(lat / model.scale_factor).clamp(-1, 1)
            if args.save_npy:
                pred01 = (pred * 0.5 + 0.5).clamp(0, 1).cpu().numpy()
                gt01 = (pet * 0.5 + 0.5).clamp(0, 1).cpu().numpy()
                for i, nm in enumerate(names):
                    np.save(os.path.join(pred_dir, f'{nm}.npy'), pred01[i, 0])
                    np.save(os.path.join(gt_dir, f'{nm}.npy'), gt01[i, 0])
    print('done', flush=True)


if __name__ == '__main__':
    main()
