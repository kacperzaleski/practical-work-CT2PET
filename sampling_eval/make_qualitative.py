"""
Build the qualitative comparison figure for the thesis: rows are test slices,
columns are CT | <each model> | Ground truth, all on the shared 256-slice test
dump. Models are matched by filename across their [0,1] prediction dumps.

Rows are split into HIGH-uptake and MEDIUM-uptake slices (ranked by GT lesion
fraction, SUV>2.5) so the panel shows behaviour both where there is a strong
focal signal and where uptake is moderate/diffuse. PET columns use a per-row
`hot` colormap stretched to the 99th percentile of that slice's GT, so brain/
lesion uptake renders bright instead of being crushed by the SUV=32 ceiling
(same stretch applied to every model + GT in a row -> visually comparable).

Writes thesis/figures/qualitative.png.
"""
import sys as _sys, pathlib as _pathlib  # repo-root bootstrap (script moved into a subfolder)
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))
import glob
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

CT_DIR = 'data/processed_fullbody/test/CT'
# (column label -> dir of [0,1] preds), left-to-right. PM is the distortion
# anchor; then the 2x2 (2 diffusion, 2 flow); GT last.
MODELS = [
    ('Posterior mean', 'results/eval/pm/pred'),
    # CPDM = the focal-attention fine-tune (the current better model; FID 79.7 vs the
    # blob run's 87.1). The blob-vs-focal contrast lives in thesis/figures/appendix_cpdm.png.
    ('CPDM', 'results/CT2PET_autoPET_fullbody/CPDM_focal/sample_to_eval/200'),
    ('Concat-diffusion', 'results/eval/concatdiff/pred'),
    ('Concat-flow', 'results/eval/cond/pred'),
    ('PMRF', 'results/eval/pmrf/pred'),
]
GT_DIR = 'results/eval/pm/gt'
OUT = 'thesis/figures/qualitative.png'
OUT_SHORT = 'thesis/figures/qualitative_short.png'  # 1 high + 1 med row, for inline use
N_HIGH = 3       # highest GT lesion fraction
N_MED = 3        # around the median of slices that still have some uptake
SUV_MAX = 32.0


def load01(p):
    return np.clip(np.load(p).astype(np.float32).squeeze(), 0, 1)


def main():
    # filenames present for CT, GT, and *every* model column
    names = set(os.path.basename(f) for f in glob.glob(os.path.join(GT_DIR, '*.npy')))
    names &= set(os.path.basename(f) for f in glob.glob(os.path.join(CT_DIR, '*.npy')))
    missing = []
    for lbl, d in MODELS:
        s = set(os.path.basename(f) for f in glob.glob(os.path.join(d, '*.npy')))
        if not s:
            missing.append(lbl)
        names &= s
    if missing:
        print(f'WARNING: no dumps for {missing} — those columns will be empty/dropped')
    names = sorted(names)
    if not names:
        raise SystemExit('No common prediction filenames across all models — run the samplers first.')
    print(f'{len(names)} slices common to all {len(MODELS)} models + CT + GT')

    # patient id = the leading "fdg_<hash>" token (the eval set spans only a couple
    # of patients; spread the rows across them so the figure isn't one patient).
    def patient(nm):
        m = nm.split('_')
        return '_'.join(m[:2]) if len(m) >= 2 else nm

    # rank by GT lesion fraction (SUV>2.5)
    frac = [(nm, float((load01(os.path.join(GT_DIR, nm)) * SUV_MAX > 2.5).mean())) for nm in names]
    frac.sort(key=lambda x: x[1], reverse=True)

    def pick_spread(cands, k):
        """Take k names, round-robin across patients so no single patient dominates."""
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

    # high-uptake: highest GT lesion fraction, spread across patients
    high = pick_spread([nm for nm, _ in frac], N_HIGH)
    # medium: slices with nonzero uptake around the median, spread across patients,
    # excluding the ones already chosen for the high rows
    nz = [nm for nm, f in frac if f > 0 and nm not in high]
    mid = len(nz) // 2
    med_window = nz[max(0, mid - N_MED): mid + N_MED + 1]
    med = pick_spread(med_window, N_MED)
    chosen = high + med
    row_kind = ['high'] * len(high) + ['med'] * len(med)
    print('high:', [(patient(n)) for n in high], high)
    print('med :', [(patient(n)) for n in med], med)

    cols = ['CT'] + [lbl for lbl, _ in MODELS] + ['Ground truth']

    def render(rows, kinds, out):
        nrows = len(rows)
        fig, axes = plt.subplots(nrows, len(cols), figsize=(1.9 * len(cols), 2.0 * nrows))
        if nrows == 1:
            axes = axes[None, :]
        for r, (nm, kind) in enumerate(zip(rows, kinds)):
            ct = np.load(os.path.join(CT_DIR, nm)).astype(np.float32).squeeze()
            ct01 = (np.clip(ct, -1, 1) + 1) / 2
            gt = load01(os.path.join(GT_DIR, nm))
            vmax = max(float(np.percentile(gt, 99.5)), 0.12)  # stretch hot map to GT uptake
            preds = []
            for _, d in MODELS:
                p = os.path.join(d, nm)
                preds.append(load01(p) if os.path.exists(p) else None)
            imgs = [ct01] + preds + [gt]
            for c, img in enumerate(imgs):
                ax = axes[r, c]
                if img is None:
                    ax.text(0.5, 0.5, 'n/a', ha='center', va='center', transform=ax.transAxes, fontsize=8)
                elif c == 0:
                    ax.imshow(img, cmap='gray', vmin=0, vmax=1)
                else:
                    ax.imshow(img, cmap='hot', vmin=0, vmax=vmax)
                ax.set_xticks([]); ax.set_yticks([])
                if r == 0:
                    ax.set_title(cols[c], fontsize=10)
            axes[r, 0].set_ylabel(f'{kind}-uptake', fontsize=9)
        plt.tight_layout()
        os.makedirs(os.path.dirname(out), exist_ok=True)
        fig.savefig(out, dpi=130, bbox_inches='tight')
        plt.close(fig)
        print(f'wrote {out}  ({nrows} slices x {len(cols)} cols)')

    render(chosen, row_kind, OUT)
    # short version for inline use: one high-uptake + one medium-uptake row
    if high and med:
        render([high[0], med[0]], ['high', 'med'], OUT_SHORT)


if __name__ == '__main__':
    main()
