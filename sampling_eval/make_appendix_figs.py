"""
Build the thesis *appendix* qualitative figures (written to thesis/figures/):

  appendix_pmrf.png — the PMRF two-stage story, columns:
      CT | Posterior mean (stage 1) | PMRF (stage 2) | Ground truth
  appendix_cpdm.png — the CPDM conditioning + blob-vs-focal story, columns:
      CT | Attenuation map | Blob attention | Focal attention | Blob CPDM | Focal CPDM | GT

Rows are test slices ranked by GT lesion fraction (SUV>2.5), spread across the (few)
test patients, split into high- and medium-uptake bands — same selection logic as
make_qualitative.py. PET columns use a per-row hot-colormap stretched to the GT 99.5th
percentile so uptake renders bright. The attenuation map is the CT-derived 511 keV factor
exp(-mu) that CPDM consumes (replicated from CT2PETDiffusionModel.attenuationCT_to_511,
KVP=140); it is identical for the blob and focal runs, hence one shared column.
"""
import sys as _sys, pathlib as _pathlib  # repo-root bootstrap (script moved into a subfolder)
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))
import glob
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

ROOT = 'data/processed_fullbody/test'
CT_DIR = f'{ROOT}/CT'
ATT_BLOB = f'{ROOT}/AttentionMaps'
ATT_FOCAL = f'{ROOT}/AttentionMaps_focal'
PM_DIR = 'results/eval/pm/pred'
PMRF_DIR = 'results/eval/pmrf/pred'
GT_DIR = 'results/eval/pm/gt'
CPDM_BLOB = 'results/CT2PET_autoPET_fullbody/CPDM/sample_to_eval/200'
CPDM_FOCAL = 'results/CT2PET_autoPET_fullbody/CPDM_focal/sample_to_eval/200'
OUTDIR = 'thesis/figures'
SUV_MAX = 32.0
N_HIGH, N_MED = 4, 2


def load01(p):
    return np.clip(np.load(p).astype(np.float32).squeeze(), 0, 1)


def patient(nm):
    m = nm.split('_')
    return '_'.join(m[:2]) if len(m) >= 2 else nm


def pick_spread(cands, k):
    by_pat = {}
    for nm in cands:
        by_pat.setdefault(patient(nm), []).append(nm)
    order = sorted(by_pat, key=lambda p: -len(by_pat[p]))
    out = []
    while len(out) < k and any(by_pat[p] for p in order):
        for p in order:
            if by_pat[p]:
                out.append(by_pat[p].pop(0))
                if len(out) == k:
                    break
    return out


def attenuation_map(ct_norm):
    """CT[-1,1] -> 511 keV attenuation factor exp(-mu), KVP=140 (faithful to CPDM)."""
    HU_MIN, HU_MAX = -1000.0, 3071.0
    HU = (np.clip(ct_norm, -1, 1) + 1.0) * 0.5 * (HU_MAX - HU_MIN) + HU_MIN
    a = [9.3e-5, 5.59e-5, 0.698e-5]
    b = [0.093, 0.093, 0.142]
    z = np.array([[-1000, b[0] - 1000 * a[0]], [0, b[1]],
                  [1000, b[1] + a[1] * 1000], [3000, b[1] + a[1] * 1000 + a[2] * 2000]])
    vali = np.arange(-1000, 3000 + 0.1, 0.1)
    inter = interp1d(z[:, 0], z[:, 1], kind='linear', fill_value='extrapolate')(vali)
    mu = np.interp(HU.flatten(), vali, inter).reshape(HU.shape)
    return np.exp(-mu)


def select_rows(common):
    frac = [(nm, float((load01(os.path.join(GT_DIR, nm)) * SUV_MAX > 2.5).mean())) for nm in common]
    frac.sort(key=lambda x: x[1], reverse=True)
    high = pick_spread([nm for nm, _ in frac], N_HIGH)
    nz = [nm for nm, f in frac if f > 0 and nm not in high]
    mid = len(nz) // 2
    med = pick_spread(nz[max(0, mid - N_MED): mid + N_MED + 1], N_MED)
    return high + med, ['high'] * len(high) + ['med'] * len(med)


def names_in(*dirs):
    s = None
    for d in dirs:
        cur = set(os.path.basename(f) for f in glob.glob(os.path.join(d, '*.npy')))
        s = cur if s is None else (s & cur)
    return s


def panel(out, cols, rows, kinds, render):
    fig, axes = plt.subplots(len(rows), len(cols), figsize=(1.95 * len(cols), 2.0 * len(rows)))
    if len(rows) == 1:
        axes = axes[None, :]
    for r, (nm, kind) in enumerate(zip(rows, kinds)):
        for c, (img, cmap, vmin, vmax) in enumerate(render(nm)):
            ax = axes[r, c]
            ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(cols[c], fontsize=9)
        axes[r, 0].set_ylabel(f'{kind}-uptake', fontsize=9)
    plt.tight_layout()
    os.makedirs(OUTDIR, exist_ok=True)
    fig.savefig(os.path.join(OUTDIR, out), dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f'wrote {OUTDIR}/{out}  ({len(rows)} rows x {len(cols)} cols)')


def ct01(nm):
    return (np.clip(np.load(os.path.join(CT_DIR, nm)).astype(np.float32).squeeze(), -1, 1) + 1) / 2


def main():
    # ---- appendix_pmrf.png ----
    common = names_in(GT_DIR, CT_DIR, PM_DIR, PMRF_DIR)
    rows, kinds = select_rows(sorted(common))
    print('pmrf rows:', [patient(n) for n in rows])

    def render_pmrf(nm):
        gt = load01(os.path.join(GT_DIR, nm)); vmax = max(float(np.percentile(gt, 99.5)), 0.12)
        return [(ct01(nm), 'gray', 0, 1),
                (load01(os.path.join(PM_DIR, nm)), 'hot', 0, vmax),
                (load01(os.path.join(PMRF_DIR, nm)), 'hot', 0, vmax),
                (gt, 'hot', 0, vmax)]
    panel('appendix_pmrf.png', ['CT', 'Posterior mean', 'PMRF', 'Ground truth'],
          rows, kinds, render_pmrf)
    # single representative row for inline use in the body (highest-uptake slice)
    panel('appendix_pmrf_row.png', ['CT', 'Posterior mean', 'PMRF', 'Ground truth'],
          rows[:1], kinds[:1], render_pmrf)

    # ---- appendix_cpdm.png ----
    common = names_in(GT_DIR, CT_DIR, ATT_BLOB, ATT_FOCAL, CPDM_BLOB, CPDM_FOCAL)
    rows, kinds = select_rows(sorted(common))
    print('cpdm rows:', [patient(n) for n in rows])

    def render_cpdm(nm):
        gt = load01(os.path.join(GT_DIR, nm)); vmax = max(float(np.percentile(gt, 99.5)), 0.12)
        att_b = np.load(os.path.join(ATT_BLOB, nm)).astype(np.float32).squeeze()
        att_f = np.load(os.path.join(ATT_FOCAL, nm)).astype(np.float32).squeeze()
        atten = attenuation_map(np.load(os.path.join(CT_DIR, nm)).astype(np.float32).squeeze())
        return [(ct01(nm), 'gray', 0, 1),
                (atten, 'bone', float(atten.min()), float(atten.max())),
                (att_b, 'viridis', 0, 1),
                (att_f, 'viridis', 0, 1),
                (load01(os.path.join(CPDM_BLOB, nm)), 'hot', 0, vmax),
                (load01(os.path.join(CPDM_FOCAL, nm)), 'hot', 0, vmax),
                (gt, 'hot', 0, vmax)]
    cpdm_cols = ['CT', 'Attenuation', 'Blob attention', 'Focal attention',
                 'Blob CPDM', 'Focal CPDM', 'Ground truth']
    panel('appendix_cpdm.png', cpdm_cols, rows, kinds, render_cpdm)
    # single representative row for inline use in the body (highest-uptake slice)
    panel('appendix_cpdm_row.png', cpdm_cols, rows[:1], kinds[:1], render_cpdm)


if __name__ == '__main__':
    main()
