"""Report per-slice (unclustered) vs patient-clustered 95% CIs side by side, so the
thesis can show concretely how much the confidence interval widens once we stop
pretending every slice is an independent sample.

For each model on the shared manifest it recomputes per-slice SSIM at the three tiers
(global / active SUV>0.5 / lesion SUV>2.5), derives patient ids from filenames, and
prints, per tier:
  point   [naive per-slice CI]   [patient-clustered bootstrap CI]   [per-patient t CI]
The last two agree; both are wider than the naive one. Pixel-only (no Inception), so
it is fast and needs no GPU.
"""
import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))
import glob
import os

import numpy as np

from metrics_common import image_metrics, uptake_mask
from stats_common import (patient_of, naive_ci, patient_ci, cluster_bootstrap_ci)

D = 'results/CT2PET_autoPET_fullbody'
MODELS = {
    'PM':          ('results/eval/pm/pred', 'results/eval/pm/gt'),
    'CPDM(focal)': (f'{D}/CPDM_focal/sample_to_eval/200', f'{D}/CPDM_focal/sample_to_eval/ground_truth'),
    'concat-diff': ('results/eval/concatdiff/pred', 'results/eval/concatdiff/gt'),
    'concat-flow': ('results/eval/cond/pred', 'results/eval/cond/gt'),
    'PMRF':        ('results/eval/pmrf/pred', 'results/eval/pmrf/gt'),
}
TIERS = [('global', None), ('active', 0.5), ('lesion', 2.5)]


def load01(p):
    return np.clip(np.load(p).astype(np.float64).squeeze(), 0, 1)


def per_slice_ssim(pred_dir, gt_dir):
    names = sorted(set(os.path.basename(f) for f in glob.glob(f'{pred_dir}/*.npy'))
                   & set(os.path.basename(f) for f in glob.glob(f'{gt_dir}/*.npy')))
    out = {t: [] for t, _ in TIERS}
    pats = []
    for nm in names:
        pred, gt = load01(f'{pred_dir}/{nm}'), load01(f'{gt_dir}/{nm}')
        pats.append(patient_of(nm))
        for t, thr in TIERS:
            mask = None if thr is None else uptake_mask(gt, thr)
            out[t].append(image_metrics(pred, gt, mask)['ssim'])
    return {t: np.array(v) for t, v in out.items()}, np.array(pats)


def main():
    print(f'{"model":12s} {"tier":7s} {"point":>7s}   {"naive per-slice":>22s}   '
          f'{"patient bootstrap":>22s}   {"per-patient t":>22s}')
    print('-' * 100)
    for name, (pd, gd) in MODELS.items():
        ss, pats = per_slice_ssim(pd, gd)
        npat = len(set(pats))
        for t, _ in TIERS:
            v = ss[t]
            p0, nlo, nhi = naive_ci(v)
            _, blo, bhi = cluster_bootstrap_ci(v, pats)
            _, tlo, thi = patient_ci(v, pats)
            wn, wb = nhi - nlo, bhi - blo
            print(f'{name:12s} {t:7s} {p0:7.3f}   [{nlo:6.3f}, {nhi:6.3f}] w={wn:.3f}   '
                  f'[{blo:6.3f}, {bhi:6.3f}] w={wb:.3f}   [{tlo:6.3f}, {thi:6.3f}]   '
                  f'(x{wb/wn:.1f} wider, {npat} pat)')
        print()


if __name__ == '__main__':
    main()
