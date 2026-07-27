"""Pre-registered, statistically-tested model comparison for the CT->PET study.

Reads several models' saved [0,1] pred/gt .npy (the same shared multi-patient test
manifest for every model) and answers the thesis hypotheses with proper statistics:

  * Per-model metric table with patient-clustered bootstrap 95% CIs
    (MAE/SSIM at global/active/lesion tiers, plus FID point + KID CI).
  * Pre-registered pairwise contrasts (paired patient-clustered bootstrap + Wilcoxon
    signed-rank on the per-slice pixel metrics; paired patient bootstrap on KID for the
    distributional axis), with Holm-Bonferroni correction across the contrast family.

Distributional metric choice: at ~300 test images a 2048-d FID covariance is
rank-deficient and its matrix-sqrt is too slow to bootstrap, so KID (unbiased
polynomial-kernel MMD) is used for the tested distributional contrasts; FID is reported
as a point estimate only, for continuity with earlier tables.

Models are given as NAME:PRED_DIR:GT_DIR (repeatable). Known model keys used by the
pre-registered contrasts: pm, pmrf, cond (concat-flow), concatdiff, cpdm (headline),
cpdm_blob (ablation).

Example:
  python sampling_eval/compare_models.py \
    --model pm:results/eval/pm/pred:results/eval/pm/gt \
    --model pmrf:results/eval/pmrf/pred:results/eval/pmrf/gt \
    --model cond:results/eval/cond/pred:results/eval/cond/gt \
    --model concatdiff:results/eval/concatdiff/pred:results/eval/concatdiff/gt \
    --model cpdm:results/eval/cpdm_focal/pred:results/eval/cpdm_focal/gt \
    --model cpdm_blob:results/eval/cpdm_blob/pred:results/eval/cpdm_blob/gt --kid
"""
import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))
import argparse
import glob
import os

import numpy as np

from metrics_common import image_metrics, uptake_mask
from stats_common import (cluster_bootstrap_ci, paired_cluster_bootstrap,
                          wilcoxon_signed_rank, holm_correction, patient_of, fmt_ci)

TIERS = {'global': None, 'active': 0.5, 'lesion': 2.5}

# Pre-registered contrasts: (label, hypothesis, model_a, model_b, tier, metric, better)
# 'better' = the direction the hypothesis predicts A beats B. Tests are two-sided; the
# direction only documents the prediction. The posterior-mean anchor (pm) is deliberately
# NOT contrasted on SSIM — it is the distortion extreme and is expected to win there; the
# hypotheses concern the *generators* and the perception axis.
CONTRASTS = [
    # H2 — paradigm-native vs concatenation, WITHIN each family:
    ('H2-flow: PMRF vs concat-flow (lesion SSIM)', 'H2', 'pmrf', 'cond',       'lesion', 'ssim', 'higher'),
    ('H2-flow: PMRF vs concat-flow (KID)',         'H2', 'pmrf', 'cond',       'global', 'kid',  'lower'),
    ('H2-diff: CPDM vs concat-diff (lesion SSIM)', 'H2', 'cpdm', 'concatdiff', 'lesion', 'ssim', 'higher'),
    ('H2-diff: CPDM vs concat-diff (KID)',         'H2', 'cpdm', 'concatdiff', 'global', 'kid',  'lower'),
    # H3 — PMRF is best knee: best perception AND best distortion AMONG THE GENERATORS:
    ('H3: PMRF vs concat-flow (KID)',              'H3', 'pmrf', 'cond',       'global', 'kid',  'lower'),
    ('H3: PMRF vs concat-diff (KID)',              'H3', 'pmrf', 'concatdiff', 'global', 'kid',  'lower'),
    ('H3: PMRF vs CPDM (KID)',                     'H3', 'pmrf', 'cpdm',       'global', 'kid',  'lower'),
    ('H3: PMRF vs concat-flow (active SSIM)',      'H3', 'pmrf', 'cond',       'active', 'ssim', 'higher'),
    ('H3: PMRF vs concat-diff (active SSIM)',      'H3', 'pmrf', 'concatdiff', 'active', 'ssim', 'higher'),
    # H5 — conditioning quality: focal-corrected CPDM vs uninformative blob:
    ('H5: CPDM-focal vs blob (active SSIM)',       'H5', 'cpdm', 'cpdm_blob',  'active', 'ssim', 'higher'),
    ('H5: CPDM-focal vs blob (KID)',               'H5', 'cpdm', 'cpdm_blob',  'global', 'kid',  'lower'),
]


def load01(path):
    return np.clip(np.load(path).astype(np.float64).squeeze(), 0.0, 1.0)


# ----------------------------- distributional (KID) --------------------------
def inception_features(images01):
    """(N,H,W) [0,1] -> (N,2048) InceptionV3 pool3 features (torchmetrics extractor)."""
    import torch
    from torchmetrics.image.fid import NoTrainInceptionV3
    net = NoTrainInceptionV3(name='inception-v3-compat', features_list=['2048']).eval()
    feats = []
    with torch.no_grad():
        for i in range(0, len(images01), 50):
            arr = np.stack(images01[i:i + 50]).astype(np.float32)      # (b,H,W) [0,1]
            t = torch.from_numpy(arr).unsqueeze(1).repeat(1, 3, 1, 1)  # (b,3,H,W)
            t = (t * 255).clamp(0, 255).to(torch.uint8)               # extractor wants uint8
            feats.append(net(t).cpu().numpy())
    return np.concatenate(feats, axis=0)


def _poly_kernel(x, y):
    d = x.shape[1]
    return (x @ y.T / d + 1.0) ** 3


def kid_from_features(fr, ff):
    """Unbiased KID (squared polynomial-kernel MMD) between real/fake feature sets."""
    m, n = len(fr), len(ff)
    if m < 2 or n < 2:
        return np.nan
    kxx, kyy, kxy = _poly_kernel(fr, fr), _poly_kernel(ff, ff), _poly_kernel(fr, ff)
    np.fill_diagonal(kxx, 0.0); np.fill_diagonal(kyy, 0.0)
    return (kxx.sum() / (m * (m - 1)) + kyy.sum() / (n * (n - 1))
            - 2.0 * kxy.mean())


def fid_from_features(fr, ff):
    from scipy.linalg import sqrtm
    mu1, mu2 = fr.mean(0), ff.mean(0)
    c1, c2 = np.cov(fr, rowvar=False), np.cov(ff, rowvar=False)
    covmean = sqrtm(c1 @ c2)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    diff = mu1 - mu2
    return float(diff @ diff + np.trace(c1 + c2 - 2 * covmean))


def _patient_groups(patients):
    groups = {}
    for i, p in enumerate(patients):
        groups.setdefault(p, []).append(i)
    return [np.asarray(ix) for ix in groups.values()]


def kid_ci(fr, ff, patients, alpha=0.05):
    """Leave-one-patient-out JACKKNIFE CI for KID. A cluster bootstrap is invalid here:
    KID is a quadratic U-statistic, so duplicating whole patients (identical feature rows)
    self-matches in the polynomial kernel and biases the estimate upward. The delete-1
    jackknife over patients avoids duplication and gives a valid variance for a smooth
    statistic. CI = point +/- z * jackknife-SE (normal approx)."""
    groups = _patient_groups(patients)
    n = len(groups)
    point = kid_from_features(fr, ff)
    if n < 3:
        return point, np.nan, np.nan
    loo = np.array([kid_from_features(np.delete(fr, groups[j], axis=0),
                                      np.delete(ff, groups[j], axis=0)) for j in range(n)])
    var = (n - 1) / n * np.sum((loo - loo.mean()) ** 2)
    se = float(np.sqrt(max(var, 0.0)))
    z = 1.959964
    return point, point - z * se, point + z * se


def paired_kid_bootstrap(frA, ffA, frB, ffB, patients, alpha=0.05):
    """Leave-one-patient-out jackknife on the paired difference KID_A - KID_B (same real
    GT features; A/B fakes on the same slices). Returns delta, its normal CI, and a
    two-sided z-test p-value. Name kept for call-site stability."""
    groups = _patient_groups(patients)
    n = len(groups)
    delta = kid_from_features(frA, ffA) - kid_from_features(frB, ffB)
    if n < 3:
        return {'delta': delta, 'lo': np.nan, 'hi': np.nan, 'p': np.nan}
    loo = np.array([
        kid_from_features(np.delete(frA, groups[j], axis=0), np.delete(ffA, groups[j], axis=0))
        - kid_from_features(np.delete(frB, groups[j], axis=0), np.delete(ffB, groups[j], axis=0))
        for j in range(n)])
    var = (n - 1) / n * np.sum((loo - loo.mean()) ** 2)
    se = float(np.sqrt(max(var, 1e-30)))
    z = 1.959964
    from scipy.stats import norm
    p = float(2 * norm.sf(abs(delta) / se)) if se > 0 else 1.0
    return {'delta': delta, 'lo': delta - z * se, 'hi': delta + z * se, 'p': p}


# --------------------------------- main --------------------------------------
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--model', action='append', required=True,
                   help='NAME:PRED_DIR:GT_DIR (repeatable)')
    p.add_argument('--kid', action='store_true', help='compute KID/FID (loads InceptionV3).')
    p.add_argument('--n-boot', type=int, default=2000)
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()

    models = {}
    for spec in args.model:
        name, pred_dir, gt_dir = spec.split(':')
        models[name] = (pred_dir, gt_dir)

    # common slice set across all models (by stem)
    stem_sets = []
    for name, (pred_dir, gt_dir) in models.items():
        preds = {os.path.basename(f)[:-4] for f in glob.glob(os.path.join(pred_dir, '*.npy'))}
        gts = {os.path.basename(f)[:-4] for f in glob.glob(os.path.join(gt_dir, '*.npy'))}
        stem_sets.append(preds & gts)
    common = sorted(set.intersection(*stem_sets))
    if not common:
        raise SystemExit('No shared slice stems across the given models.')
    patients = np.array([patient_of(s) for s in common])
    n_pat = len(set(patients.tolist()))
    print(f'shared slices: {len(common)}  patients: {n_pat}  models: {list(models)}\n')

    # per-slice pixel metrics per model + GT masks (masks come from GT, shared per slice)
    # GT is identical across models (same manifest) -> use the first model's GT dir.
    ref_gt_dir = next(iter(models.values()))[1]
    masks = {}
    for s in common:
        gt = load01(os.path.join(ref_gt_dir, f'{s}.npy'))
        masks[s] = {t: (None if thr is None else uptake_mask(gt, thr)) for t, thr in TIERS.items()}

    per = {}  # per[name][tier][metric] -> np.array aligned to `common`
    for name, (pred_dir, gt_dir) in models.items():
        per[name] = {t: {'mae': [], 'ssim': []} for t in TIERS}
        for s in common:
            pred = load01(os.path.join(pred_dir, f'{s}.npy'))
            gt = load01(os.path.join(gt_dir, f'{s}.npy'))
            for t in TIERS:
                m = image_metrics(pred, gt, masks[s][t])
                per[name][t]['mae'].append(m['mae'])
                per[name][t]['ssim'].append(m['ssim'])
        for t in TIERS:
            for k in ('mae', 'ssim'):
                per[name][t][k] = np.array(per[name][t][k], dtype=np.float64)

    # KID/FID features per model
    feats = {}
    gt_feat = None
    if args.kid:
        print('extracting InceptionV3 features...', flush=True)
        gt_imgs = [load01(os.path.join(ref_gt_dir, f'{s}.npy')) for s in common]
        gt_feat = inception_features(gt_imgs)
        for name, (pred_dir, _gt) in models.items():
            imgs = [load01(os.path.join(pred_dir, f'{s}.npy')) for s in common]
            feats[name] = inception_features(imgs)

    # ---- per-model table (CIs) ----
    print('=== Per-model metrics (point [95% CI], patient-clustered bootstrap) ===')
    hdr = f'{"model":11s}  {"global SSIM":>22s}  {"active SSIM":>22s}  {"lesion SSIM":>22s}'
    if args.kid:
        hdr += f'  {"KID x1e3":>20s}  {"FID":>7s}'
    print(hdr); print('-' * len(hdr))
    for name in models:
        row = f'{name:11s}'
        for t in ('global', 'active', 'lesion'):
            pt, lo, hi = cluster_bootstrap_ci(per[name][t]['ssim'], patients,
                                              n_boot=args.n_boot, seed=args.seed)
            row += f'  {fmt_ci(pt, lo, hi, 4):>22s}'
        if args.kid:
            kpt, klo, khi = kid_ci(gt_feat, feats[name], patients)
            row += f'  {fmt_ci(kpt*1e3, klo*1e3, khi*1e3, 2):>20s}'
            row += f'  {fid_from_features(gt_feat, feats[name]):7.1f}'
        print(row)

    # ---- pre-registered contrasts ----
    print('\n=== Pre-registered contrasts (paired patient bootstrap; Holm-corrected) ===')
    rows, pvals = [], []
    for label, hyp, a, b, tier, metric, better in CONTRASTS:
        if a not in models or b not in models:
            continue
        if metric == 'kid':
            if not args.kid:
                continue
            r = paired_kid_bootstrap(gt_feat, feats[a], gt_feat, feats[b], patients)
            delta, lo, hi, pp = r['delta'] * 1e3, r['lo'] * 1e3, r['hi'] * 1e3, r['p']
            wp = np.nan
            scale = 'x1e3'
        else:
            va, vb = per[a][tier][metric], per[b][tier][metric]
            r = paired_cluster_bootstrap(va, vb, patients, n_boot=args.n_boot, seed=args.seed)
            delta, lo, hi, pp = r['delta'], r['lo'], r['hi'], r['p']
            _, wp = wilcoxon_signed_rank(va, vb)
            scale = ''
        rows.append((label, hyp, delta, lo, hi, pp, wp, scale))
        pvals.append(pp)
    p_holm = holm_correction(pvals) if pvals else []

    print(f'{"contrast":38s} {"H":3s} {"delta":>9s} {"95% CI":>20s} {"p_boot":>8s} '
          f'{"p_holm":>8s} {"p_wilcox":>9s}')
    print('-' * 100)
    for (label, hyp, delta, lo, hi, pp, wp, scale), ph in zip(rows, p_holm):
        sig = '*' if ph < 0.05 else ' '
        wtxt = f'{wp:.1e}' if not np.isnan(wp) else '   -   '
        print(f'{label:38s} {hyp:3s} {delta:9.4f} [{lo:8.4f},{hi:8.4f}] '
              f'{pp:8.4f} {ph:7.4f}{sig} {wtxt:>9s}  {scale}')
    print('\n(* Holm-adjusted p < 0.05.  KID deltas are x1e3.  '
          'SSIM higher=better, KID/MAE lower=better.)')


if __name__ == '__main__':
    main()
