"""Global LPIPS (and PSNR, for the text) per model on the shared manifest.

The main table (compare_models.py) reports MAE/SSIM at the global/active/lesion tiers plus
FID/KID. This companion adds the two metrics discussed in the thesis but not produced by the
shared ruler eval_masked.py:
  * LPIPS  -- per-image, full-reference perceptual distance (VGG), tabulated in the thesis.
  * PSNR   -- a monotone function of MSE; reported here only for the text (it duplicates MAE).
Both are global (LPIPS is patch-based and not cleanly maskable, like FID/KID). Point estimates
come with patient-clustered bootstrap 95% CIs, the same honest interval used everywhere else.

Usage (same --model NAME:PRED_DIR:GT_DIR interface as compare_models.py):
  python sampling_eval/extra_metrics.py \
    --model pm:results/eval/pm/pred:results/eval/pm/gt \
    --model pmrf:results/eval/pmrf/pred:results/eval/pmrf/gt \
    --model cond:results/eval/cond/pred:results/eval/cond/gt \
    --model concatdiff:results/eval/concatdiff/pred:results/eval/concatdiff/gt \
    --model cpdm_focal:results/CT2PET_autoPET_fullbody/CPDM_focal/sample_to_eval/200:\
results/CT2PET_autoPET_fullbody/CPDM_focal/sample_to_eval/ground_truth
"""
import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))
import argparse
import glob
import os

import numpy as np

from metrics_common import image_metrics
from stats_common import cluster_bootstrap_ci, patient_of, fmt_ci


def load01(path):
    return np.clip(np.load(path).astype(np.float64).squeeze(), 0.0, 1.0)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--model', action='append', required=True,
                   help='NAME:PRED_DIR:GT_DIR (repeatable)')
    p.add_argument('--n-boot', type=int, default=2000)
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()

    models = {}
    for spec in args.model:
        name, pred_dir, gt_dir = spec.split(':')
        models[name] = (pred_dir, gt_dir)

    # shared slice set across all models (by stem), exactly like compare_models.py
    stem_sets = []
    for pred_dir, gt_dir in models.values():
        preds = {os.path.basename(f)[:-4] for f in glob.glob(os.path.join(pred_dir, '*.npy'))}
        gts = {os.path.basename(f)[:-4] for f in glob.glob(os.path.join(gt_dir, '*.npy'))}
        stem_sets.append(preds & gts)
    common = sorted(set.intersection(*stem_sets))
    if not common:
        raise SystemExit('No shared slice stems across the given models.')
    patients = np.array([patient_of(s) for s in common])
    print(f'shared slices: {len(common)}  patients: {len(set(patients.tolist()))}  '
          f'models: {list(models)}\n')

    # LPIPS (VGG). normalize=True expects [0,1]; single-channel tiled to 3.
    import torch
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
    lpips = LearnedPerceptualImagePatchSimilarity(net_type='vgg', normalize=True).eval()

    def lpips_one(pred, gt):
        def t(x):
            return torch.from_numpy(x.astype(np.float32))[None, None].repeat(1, 3, 1, 1).clamp(0, 1)
        with torch.no_grad():
            return float(lpips(t(pred), t(gt)))

    print(f'{"model":12s}  {"MAE":>22s}  {"PSNR":>22s}  {"SSIM":>22s}  {"LPIPS":>22s}')
    print('-' * 104)
    for name, (pred_dir, gt_dir) in models.items():
        mae, psnr, ssim, lp = [], [], [], []
        for s in common:
            pr = load01(os.path.join(pred_dir, f'{s}.npy'))
            gt = load01(os.path.join(gt_dir, f'{s}.npy'))
            m = image_metrics(pr, gt, None)
            mae.append(m['mae']); psnr.append(m['psnr']); ssim.append(m['ssim'])
            lp.append(lpips_one(pr, gt))
        row = f'{name:12s}'
        for arr, prec in ((mae, 4), (psnr, 2), (ssim, 4), (lp, 4)):
            pt, lo, hi = cluster_bootstrap_ci(np.array(arr), patients,
                                              n_boot=args.n_boot, seed=args.seed)
            row += f'  {fmt_ci(pt, lo, hi, prec):>22s}'
        print(row, flush=True)


if __name__ == '__main__':
    main()
