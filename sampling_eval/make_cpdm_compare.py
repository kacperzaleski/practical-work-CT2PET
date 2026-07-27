"""Qualitative comparison of the two CPDM variants (fine-tuned vs from-scratch focal)
on the shared manifest, so the visual difference can be judged before deciding which
becomes the thesis's CPDM.

Columns: CT input | CPDM (fine-tuned focal) | CPDM (from-scratch focal) | ground truth.
Rows: manifest test slices with the highest GT uptake (most informative). PET in a hot
colormap with a per-row [0,1] stretch to the GT 99.5th percentile; CT in grayscale.

Writes thesis/figures/cpdm_scratch_vs_finetune.png.
"""
import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))
import glob
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

D = 'results/CT2PET_autoPET_fullbody'
FT = f'{D}/CPDM_focal/sample_to_eval'
SC = f'{D}/CPDM_focal_scratch/sample_to_eval'
CT_DIR = 'data/processed_fullbody/test/CT'
OUT = 'thesis/figures/cpdm_scratch_vs_finetune.png'
NROWS = 6


def load01(p):
    return np.clip(np.load(p).astype(np.float64).squeeze(), 0, 1)


def main():
    # slices common to both variants' dumps
    ft_pred = {os.path.basename(f)[:-4] for f in glob.glob(f'{FT}/200/*.npy')}
    sc_pred = {os.path.basename(f)[:-4] for f in glob.glob(f'{SC}/200/*.npy')}
    names = sorted(ft_pred & sc_pred)
    # rank by GT uptake (mean over active voxels) to pick the informative slices
    gts = {n: load01(f'{SC}/ground_truth/{n}.npy') for n in names}
    names = sorted(names, key=lambda n: -gts[n].mean())[:NROWS]

    fig, ax = plt.subplots(NROWS, 4, figsize=(9, 2.2 * NROWS))
    cols = ['CT (input)', 'CPDM (fine-tuned)', 'CPDM (from scratch)', 'PET (ground truth)']
    for r, n in enumerate(names):
        ct = np.load(f'{CT_DIR}/{n}.npy')  # [-1,1]
        gt = gts[n]
        ft = load01(f'{FT}/200/{n}.npy')
        sc = load01(f'{SC}/200/{n}.npy')
        vmax = max(1e-6, np.percentile(gt, 99.5))
        ax[r, 0].imshow(ct, cmap='gray', vmin=-1, vmax=1)
        for c, img in zip((1, 2, 3), (ft, sc, gt)):
            ax[r, c].imshow(img, cmap='hot', vmin=0, vmax=vmax)
        for c in range(4):
            ax[r, c].axis('off')
            if r == 0:
                ax[r, c].set_title(cols[c], fontsize=11)
    fig.suptitle('CPDM: fine-tuned vs from-scratch focal-attention (highest-uptake test slices)',
                 fontsize=12)
    fig.tight_layout()
    os.makedirs('thesis/figures', exist_ok=True)
    fig.savefig(OUT, dpi=130, bbox_inches='tight')
    print('wrote', OUT, '| slices:', names)


if __name__ == '__main__':
    main()
