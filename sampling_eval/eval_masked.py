"""Fair global-vs-masked PET evaluation shared by CPDM and PMRF.

Reads predicted & ground-truth PET ``.npy`` (single-channel, ``[0, 1]``) from two
directories, matched by filename, and reports MAE/PSNR/SSIM globally and inside
GT-derived ROI masks (active tissue + high-uptake/lesion).  ONE identical code
path for every model -> apples-to-apples.  The gap between the global column and
the lesion column is the honest signal that the headline numbers hide: PET is
~89% near-zero, so global metrics are dominated by background.

CPDM dumps the needed .npy via ``sample_to_eval`` already; PMRF dumps them with
``sample_pmrf.py --save-npy <dir>``.  Both write [0, 1] single-channel arrays.

Examples
--------
  # CPDM (sample_step subdir holds the predictions; 'ground_truth' holds GT)
  python eval_masked.py \
    --pred-dir results/CT2PET_autoPET_fullbody/CPDM/sample_to_eval/200 \
    --gt-dir   results/CT2PET_autoPET_fullbody/CPDM/sample_to_eval/ground_truth

  # PMRF
  python eval_masked.py --pred-dir results/PMRF/pmrf/pred --gt-dir results/PMRF/pmrf/gt
"""
import sys as _sys, pathlib as _pathlib  # repo-root bootstrap (script moved into a subfolder)
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))
import argparse
import glob
import os

import numpy as np

from metrics_common import image_metrics, uptake_mask
from stats_common import cluster_bootstrap_ci, fmt_ci, patient_of


def load01(path):
    a = np.load(path).astype(np.float64).squeeze()
    return np.clip(a, 0.0, 1.0)


def compute_fid_kid(pred_paths, gt_paths, batch=50):
    """Global FID/KID over the matched set. Distributional perception metrics:
    they compare InceptionV3 feature distributions of all generated vs. all real
    PETs, so they are set-level (not per-image) and not maskable. [0,1]
    single-channel arrays are tiled to 3 channels; torchmetrics resizes to the
    Inception input internally (normalize=True expects [0,1] floats)."""
    import torch
    from torchmetrics.image.fid import FrechetInceptionDistance
    from torchmetrics.image.kid import KernelInceptionDistance

    n = len(pred_paths)
    if n < 2:
        return None
    subset = max(2, min(50, n // 2))
    fid = FrechetInceptionDistance(feature=2048, normalize=True)
    kid = KernelInceptionDistance(feature=2048, normalize=True, subset_size=subset)

    def feed(paths, real):
        for i in range(0, len(paths), batch):
            chunk = paths[i:i + batch]
            arr = np.stack([load01(p).astype(np.float32) for p in chunk])  # (b,H,W) in [0,1]
            t = torch.from_numpy(arr).unsqueeze(1).repeat(1, 3, 1, 1)       # (b,3,H,W)
            fid.update(t, real=real)
            kid.update(t, real=real)

    with torch.no_grad():
        feed(gt_paths, real=True)
        feed(pred_paths, real=False)
        kid_mean, kid_std = kid.compute()
        return {'fid': float(fid.compute()),
                'kid_mean': float(kid_mean), 'kid_std': float(kid_std),
                'kid_subset': subset}


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--pred-dir', required=True)
    p.add_argument('--gt-dir', required=True)
    p.add_argument('--active-suv', type=float, default=0.5,
                   help='active-tissue SUV threshold (default 0.5)')
    p.add_argument('--lesion-suv', type=float, default=2.5,
                   help='high-uptake/lesion SUV threshold (default 2.5)')
    p.add_argument('--limit', type=int, default=None,
                   help='evaluate at most N images (smoke test)')
    p.add_argument('--fid', action='store_true',
                   help='also compute global FID/KID (slow; loads InceptionV3).')
    p.add_argument('--ci', action='store_true',
                   help='report patient-clustered bootstrap 95%% CIs next to each mean.')
    args = p.parse_args()

    gts = {os.path.basename(f): f for f in glob.glob(os.path.join(args.gt_dir, '*.npy'))}
    preds = {os.path.basename(f): f for f in glob.glob(os.path.join(args.pred_dir, '*.npy'))}
    names = sorted(set(gts) & set(preds))
    if args.limit:
        names = names[:args.limit]
    if not names:
        raise SystemExit(
            f'No matching .npy filenames between\n  {args.pred_dir} ({len(preds)} files)'
            f'\n  {args.gt_dir} ({len(gts)} files)')

    tiers = {
        'global': None,
        f'active (SUV>{args.active_suv:g})': args.active_suv,
        f'lesion (SUV>{args.lesion_suv:g})': args.lesion_suv,
    }
    acc = {t: {'mae': [], 'psnr': [], 'ssim': []} for t in tiers}
    frac = {t: [] for t, thr in tiers.items() if thr is not None}

    for nm in names:
        pred = load01(preds[nm])
        gt = load01(gts[nm])
        for t, thr in tiers.items():
            mask = None if thr is None else uptake_mask(gt, thr)
            if mask is not None:
                frac[t].append(float(mask.mean()))
            m = image_metrics(pred, gt, mask)
            for k in acc[t]:
                acc[t][k].append(m[k])

    patients = np.array([patient_of(nm) for nm in names])
    n_pat = len(set(patients.tolist()))
    print(f'\nMatched images: {len(names)}  from {n_pat} patients '
          f'(pred dir {len(preds)}, gt dir {len(gts)})')

    if args.ci:
        # patient-clustered bootstrap 95% CI per tier/metric
        print(f'{"tier":22s}  {"MAE [95% CI]":>24s}  {"PSNR [95% CI]":>26s}  '
              f'{"SSIM [95% CI]":>24s}  {"%pix":>6s}')
        print('-' * 112)
        for t in tiers:
            fp = (np.nanmean(frac[t]) * 100) if t in frac else 100.0
            cells = []
            for k, prec in (('mae', 4), ('psnr', 3), ('ssim', 4)):
                pt, lo, hi = cluster_bootstrap_ci(acc[t][k], patients)
                cells.append(fmt_ci(pt, lo, hi, prec))
            print(f'{t:22s}  {cells[0]:>24s}  {cells[1]:>26s}  {cells[2]:>24s}  {fp:6.2f}')
    else:
        print(f'{"tier":22s}  {"MAE":>8s}  {"PSNR":>8s}  {"SSIM":>8s}  {"%pixels":>8s}')
        print('-' * 64)
        for t in tiers:
            mae = np.nanmean(acc[t]['mae'])
            psnr = np.nanmean(acc[t]['psnr'])
            ss = np.nanmean(acc[t]['ssim'])
            fp = (np.nanmean(frac[t]) * 100) if t in frac else 100.0
            print(f'{t:22s}  {mae:8.4f}  {psnr:8.3f}  {ss:8.4f}  {fp:8.2f}')

    if args.fid:
        print('\nComputing global FID/KID (InceptionV3)...', flush=True)
        fk = compute_fid_kid([preds[nm] for nm in names], [gts[nm] for nm in names])
        if fk is None:
            print('FID/KID skipped: need >=2 images.')
        else:
            print(f'FID  = {fk["fid"]:.3f}   (global, lower=better)')
            print(f'KID  = {fk["kid_mean"]:.5f} +/- {fk["kid_std"]:.5f}'
                  f'   (subset_size={fk["kid_subset"]}, lower=better)')


if __name__ == '__main__':
    main()
