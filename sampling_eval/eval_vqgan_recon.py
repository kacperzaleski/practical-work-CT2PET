#!/usr/bin/env python
"""
VQGAN reconstruction evaluation + qualitative figure.

Loads one or more VQModel checkpoints (the `state_dict` key written by
training/train_vqgan_gan.py), reconstructs a fixed set of validation CT and PET slices,
and reports per-modality L1 and VGG-LPIPS. For each checkpoint it also writes a
qualitative grid (input vs reconstruction) suitable for the thesis.

Run from the repo root:
  python sampling_eval/eval_vqgan_recon.py \
      --config config/VQGAN-autoPET-fb64.yaml --data-root data/processed_fullbody \
      --ckpt checkpoints/VQGAN_gan_fb64/best.ckpt checkpoints/VQGAN_gan_fb64/last.ckpt \
      --n 6 --out-dir results/vqgan_recon
"""
import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from datasets.CT2PETAlignedDataset import CT2PETAlignedDataset
from model.VQGAN.vqgan import VQModel


def build_dataset_config(cfg, data_root, max_val):
    dc = dict(cfg['model']['VQGAN'].get('dataset_config', {}))
    dc.setdefault('image_size', 64)
    dc.setdefault('channels', 1)
    dc.setdefault('to_normal', True)
    dc['data_root'] = data_root
    dc['flip'] = False
    dc['max_samples'] = {'train': 1, 'val': max_val, 'test': 1}
    return argparse.Namespace(**dc)


@torch.no_grad()
def reconstruct(model, x):
    xrec, _ = model(x)
    return xrec.clamp(-1, 1)


def to01(t):
    return ((t + 1) / 2).clamp(0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='config/VQGAN-autoPET-fb64.yaml')
    ap.add_argument('--data-root', default='data/processed_fullbody')
    ap.add_argument('--ckpt', nargs='+', required=True)
    ap.add_argument('--n', type=int, default=6, help='number of val slices per modality')
    ap.add_argument('--out-dir', default='results/vqgan_recon')
    args = ap.parse_args()

    device = torch.device('cpu')
    os.makedirs(args.out_dir, exist_ok=True)

    with open(args.config) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    vq = cfg['model']['VQGAN']['params']

    ds_cfg = build_dataset_config(cfg, args.data_root, args.n)
    val_ds = CT2PETAlignedDataset(ds_cfg, stage='val')
    # Fixed slices: first n (dataset is deterministic with flip off).
    cts, pets = [], []
    for i in range(min(args.n, len(val_ds))):
        (pet, _), (ct, _) = val_ds[i]
        cts.append(ct)
        pets.append(pet)
    ct = torch.stack(cts).to(device)   # (n,1,64,64) in [-1,1]
    pet = torch.stack(pets).to(device)

    import lpips as lpips_lib
    percept = lpips_lib.LPIPS(net='vgg').to(device).eval()
    for p in percept.parameters():
        p.requires_grad_(False)

    def lpips_mean(a, b):
        # inputs in [-1,1], tile grayscale to 3 channels
        a3 = a.repeat(1, 3, 1, 1)
        b3 = b.repeat(1, 3, 1, 1)
        return float(percept(a3, b3).mean())

    ddconfig = argparse.Namespace(**dict(vq['ddconfig']))
    print(f'{"checkpoint":<45} {"CT L1":>8} {"CT LPIPS":>9} {"PET L1":>8} {"PET LPIPS":>10}')
    for ckpt_path in args.ckpt:
        model = VQModel(ddconfig=ddconfig,
                        lossconfig=argparse.Namespace(target='torch.nn.Identity'),
                        n_embed=int(vq['n_embed']), embed_dim=int(vq['embed_dim'])).to(device)
        sd = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(sd['state_dict'], strict=False)
        model.eval()

        ct_rec = reconstruct(model, ct)
        pet_rec = reconstruct(model, pet)
        ct_l1 = float((ct_rec - ct).abs().mean())
        pet_l1 = float((pet_rec - pet).abs().mean())
        ct_lp = lpips_mean(ct_rec, ct)
        pet_lp = lpips_mean(pet_rec, pet)
        tag = os.path.basename(os.path.dirname(ckpt_path)) + '/' + os.path.basename(ckpt_path)
        print(f'{tag:<45} {ct_l1:>8.4f} {ct_lp:>9.4f} {pet_l1:>8.4f} {pet_lp:>10.4f}', flush=True)

        # qualitative grid: rows = samples; cols = CT in | CT rec | PET in | PET rec
        n = ct.shape[0]
        fig, axes = plt.subplots(n, 4, figsize=(4 * 2.0, n * 2.0))
        if n == 1:
            axes = axes[None, :]
        col_titles = ['CT input', 'CT recon', 'PET input', 'PET recon']
        for r in range(n):
            imgs = [to01(ct[r, 0]), to01(ct_rec[r, 0]), to01(pet[r, 0]), to01(pet_rec[r, 0])]
            cmaps = ['gray', 'gray', 'hot', 'hot']
            for c in range(4):
                axes[r, c].imshow(imgs[c].cpu().numpy(), cmap=cmaps[c], vmin=0, vmax=1)
                axes[r, c].axis('off')
                if r == 0:
                    axes[r, c].set_title(col_titles[c], fontsize=11)
        fig.suptitle(tag, fontsize=10)
        fig.tight_layout()
        name = os.path.basename(os.path.dirname(ckpt_path)) + '_' + os.path.splitext(os.path.basename(ckpt_path))[0]
        out = os.path.join(args.out_dir, f'recon_{name}.png')
        fig.savefig(out, dpi=130, bbox_inches='tight')
        plt.close(fig)
        print(f'  figure -> {out}', flush=True)


if __name__ == '__main__':
    main()
