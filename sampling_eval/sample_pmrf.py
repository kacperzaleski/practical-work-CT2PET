"""
Inference + evaluation for the flow-based CT->PET models.

Runs one of the three models on the test set, computes distortion metrics
(MSE / PSNR / SSIM) and perceptual metrics (LPIPS, optionally FID / KID), and
optionally dumps qualitative CT | pred | GT panels.

  --model pm    : Stage-1 posterior mean (no flow).            needs --pm-ckpt
  --model rf    : RF baseline, K Euler steps from noised CT.   needs --flow-ckpt
  --model pmrf  : PMRF, K Euler steps from noised PM output.    needs --pm-ckpt + --flow-ckpt

Examples:
  python sample_pmrf.py --model pm   --pm-ckpt checkpoints/PMRF_stage1/last.ckpt
  python sample_pmrf.py --model rf   --flow-ckpt checkpoints/PMRF_rf/last.ckpt   --num-steps 200
  python sample_pmrf.py --model pmrf --pm-ckpt checkpoints/PMRF_stage1/last.ckpt \
      --flow-ckpt checkpoints/PMRF_pmrf/last.ckpt --num-steps 200 --fid --save-images 8
"""

import sys as _sys, pathlib as _pathlib  # repo-root bootstrap (script moved into a subfolder)
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))
import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchmetrics.image import (
    PeakSignalNoiseRatio,
    StructuralSimilarityIndexMeasure,
)
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from datasets.CT2PETAlignedDataset import CT2PETAlignedDataset
from model.PMRF.flow import RectifiedFlow
from training.train_pmrf_stage1 import PosteriorMeanModule
from training.train_flow import FlowModule
import pmrf_common as C


def to01_3ch(x):
    """[-1,1] single-channel -> [0,1] 3-channel for perceptual nets."""
    x = (x.clamp(-1, 1) + 1) / 2
    return x.repeat(1, 3, 1, 1)


def save_panels(ct, pred, gt, out_dir, names, n):
    import matplotlib.pyplot as plt
    os.makedirs(out_dir, exist_ok=True)
    n = min(n, ct.size(0))
    for i in range(n):
        fig, axes = plt.subplots(1, 3, figsize=(9, 3))
        for ax, img, title in zip(axes, [ct[i, 0], pred[i, 0], gt[i, 0]],
                                  ['CT (input)', 'Prediction', 'PET (GT)']):
            ax.imshow(img.cpu().numpy(), cmap='gray', vmin=-1, vmax=1)
            ax.set_title(title); ax.axis('off')
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f'{names[i]}.png'), dpi=110)
        plt.close(fig)


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=str, default='config/PMRF-autoPET.yaml')
    p.add_argument('--data-root', type=str, default='data/processed_fullbody')
    p.add_argument('--model', type=str, required=True, choices=['pm', 'rf', 'pmrf', 'cond'])
    p.add_argument('--pm-ckpt', type=str, default=None)
    p.add_argument('--flow-ckpt', type=str, default=None)
    p.add_argument('--num-steps', type=int, default=200, help='K Euler steps (rf/pmrf).')
    p.add_argument('--sigma-s', type=float, default=0.1)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--num-workers', type=int, default=4)
    p.add_argument('--max-samples', type=int, default=None)
    p.add_argument('--names-file', type=str, default=None,
                   help='Restrict the test set to the slice stems in this manifest '
                        '(shared multi-patient set; see build_test_manifest.py).')
    p.add_argument('--fid', action='store_true', help='Also compute FID/KID (slow).')
    p.add_argument('--save-images', type=int, default=0)
    p.add_argument('--save-npy', type=str, default=None,
                   help='Dump pred/gt PET as [0,1] .npy under <dir>/{pred,gt} for eval_masked.py.')
    p.add_argument('--out-dir', type=str, default='results/PMRF/samples')
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
    print(f'test samples: {len(ds)}  model={args.model}  K={args.num_steps}', flush=True)

    pm_net = flow_net = None
    if args.model in ('pm', 'pmrf'):
        assert args.pm_ckpt, 'pm/pmrf needs --pm-ckpt'
        pm_net = PosteriorMeanModule.load_from_checkpoint(args.pm_ckpt, map_location=device).net.eval()
    if args.model in ('rf', 'pmrf', 'cond'):
        assert args.flow_ckpt, f'{args.model} needs --flow-ckpt'
        flow_net = FlowModule.load_from_checkpoint(args.flow_ckpt, map_location=device).net.eval()
    flow = RectifiedFlow(sigma_s=args.sigma_s)

    psnr = PeakSignalNoiseRatio(data_range=2.0)
    ssim = StructuralSimilarityIndexMeasure(data_range=2.0)
    lpips = LearnedPerceptualImagePatchSimilarity(net_type='vgg', normalize=False)
    se_sum, n_pix = 0.0, 0
    fid = kid = None
    if args.fid:
        from torchmetrics.image.fid import FrechetInceptionDistance
        from torchmetrics.image.kid import KernelInceptionDistance
        fid = FrechetInceptionDistance(feature=2048, normalize=True)
        kid_subset = min(50, max(2, len(ds) // 2))
        kid = KernelInceptionDistance(feature=2048, normalize=True, subset_size=kid_subset)

    pred_npy_dir = gt_npy_dir = None
    if args.save_npy:
        pred_npy_dir = os.path.join(args.save_npy, 'pred')
        gt_npy_dir = os.path.join(args.save_npy, 'gt')
        os.makedirs(pred_npy_dir, exist_ok=True)
        os.makedirs(gt_npy_dir, exist_ok=True)

    saved = False
    for (pet, names), (ct, _) in loader:
        ct, pet = ct.to(device), pet.to(device)
        if args.model == 'pm':
            pred = pm_net(ct)
        elif args.model == 'cond':
            # concat-conditioned flow: from pure noise, CT enters via the channel
            z0 = torch.randn_like(pet)
            pred = flow.euler_sample(flow_net, z0, args.num_steps, cond=ct)
        else:
            base = pm_net(ct) if args.model == 'pmrf' else ct
            z0 = flow.perturb(base)
            pred = flow.euler_sample(flow_net, z0, args.num_steps)
        pred = pred.clamp(-1, 1)

        se_sum += torch.sum((pred - pet) ** 2).item()
        n_pix += pet.numel()
        psnr.update(pred, pet)
        ssim.update(pred, pet)
        lpips.update(pred.clamp(-1, 1).repeat(1, 3, 1, 1), pet.repeat(1, 3, 1, 1))
        if args.fid:
            fid.update(to01_3ch(pet), real=True)
            fid.update(to01_3ch(pred), real=False)
            kid.update(to01_3ch(pet), real=True)
            kid.update(to01_3ch(pred), real=False)

        if args.save_npy:
            # map [-1,1] -> [0,1] to match CPDM's saved space exactly
            pred01 = (pred * 0.5 + 0.5).clamp(0, 1).cpu().numpy()
            gt01 = (pet * 0.5 + 0.5).clamp(0, 1).cpu().numpy()
            for i, nm in enumerate(names):
                np.save(os.path.join(pred_npy_dir, f'{nm}.npy'), pred01[i, 0])
                np.save(os.path.join(gt_npy_dir, f'{nm}.npy'), gt01[i, 0])

        if args.save_images and not saved:
            save_panels(ct, pred, pet, args.out_dir, list(names), args.save_images)
            saved = True

    results = {
        'MSE': se_sum / n_pix,
        'PSNR': float(psnr.compute()),
        'SSIM': float(ssim.compute()),
        'LPIPS': float(lpips.compute()),
    }
    if args.fid:
        results['FID'] = float(fid.compute())
        kid_mean, _ = kid.compute()
        results['KID'] = float(kid_mean)

    print('\n=== Test metrics ===')
    for k, v in results.items():
        print(f'  {k:6s}: {v:.5f}')
    print(f'(model={args.model}, K={args.num_steps}, n={len(ds)})')


if __name__ == '__main__':
    main()
