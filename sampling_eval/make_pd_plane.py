"""The perception-distortion plane: every model as a point, best = bottom-left.

A 2x2 grid so we can show BOTH choices of each axis and let the reader see they tell
the same story:
  columns = distortion axis:  (1 - lesion SSIM)   |   global MAE
  rows    = perception axis:   KID (x10^3)         |   FID
Lower is better on every axis, so the ideal corner is bottom-left in all four panels.

Point values and patient-clustered 95% CIs are taken from the shared 12-patient /
296-slice manifest evaluation (logs/contrasts_ft.log for lesion SSIM / KID / FID,
logs/eval_manifest.log for global MAE). Hardcoded here so the figure is reproducible
offline and needs no Inception pass. Writes thesis/figures/pd_plane.png.
"""
import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# model -> dict of point + 95% CI (lo, hi). KID is x10^3. FID point only (no CI).
#   les_ssim / kid  : shared-296 per-model (compare_models, patient-clustered)
#   mae             : per-model global MAE + patient-clustered CI (eval_masked --ci)
M = {
    'Posterior mean': dict(c='#555555', mk='s', anchor=True,
        les=(0.536, 0.437, 0.629), kid=(36.6, 29.5, 43.6), fid=77.4,
        mae=(0.0061, 0.0050, 0.0071)),
    'CPDM':           dict(c='#d62728', mk='o',
        les=(0.430, 0.356, 0.502), kid=(25.3, 17.8, 32.7), fid=62.3,
        mae=(0.0070, 0.0061, 0.0080)),
    'Concat-diff':    dict(c='#ff7f0e', mk='o',
        les=(0.409, 0.339, 0.475), kid=(14.6, 10.9, 18.3), fid=50.6,
        mae=(0.0074, 0.0065, 0.0084)),
    'Concat-flow':    dict(c='#1f77b4', mk='o',
        les=(0.466, 0.386, 0.542), kid=(11.2, 7.5, 15.0), fid=47.3,
        mae=(0.0081, 0.0071, 0.0092)),
    'PMRF':           dict(c='#2ca02c', mk='*',
        les=(0.457, 0.377, 0.533), kid=(7.5, 3.4, 11.7), fid=41.8,
        mae=(0.0065, 0.0056, 0.0073)),
}
OUT = 'thesis/figures/pd_plane.png'


def xerr_from_ssim(les):
    """lesion SSIM (pt,lo,hi) -> x=(1-pt) with error bar half-widths for 1-SSIM."""
    pt, lo, hi = les
    x = 1 - pt
    return x, np.array([[x - (1 - hi)], [(1 - lo) - x]])


def xerr_plain(triple):
    pt, lo, hi = triple
    return pt, np.array([[pt - lo], [hi - pt]])


def draw(ax, xkind, ykind, xlabel, ylabel):
    for name, d in M.items():
        # x = distortion
        if xkind == 'les':
            x, xe = xerr_from_ssim(d['les'])
        else:
            x, xe = xerr_plain(d['mae'])
        # y = perception
        if ykind == 'kid':
            y, ye = xerr_plain(d['kid'])
        else:
            y, ye = d['fid'], None
        ax.errorbar(x, y, xerr=xe, yerr=ye, fmt=d['mk'], ms=13 if d['mk'] == '*' else 8,
                    color=d['c'], ecolor=d['c'], elinewidth=1, capsize=2.5,
                    alpha=0.9, label=name, zorder=3)
        ax.annotate(name, (x, y), textcoords='offset points', xytext=(7, 5),
                    fontsize=8, color=d['c'])
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.grid(alpha=0.3)
    # "better" arrow toward bottom-left
    ax.annotate('better', xy=(0.06, 0.06), xytext=(0.28, 0.28),
                xycoords='axes fraction', textcoords='axes fraction', fontsize=8,
                color='green', arrowprops=dict(arrowstyle='->', color='green', lw=1.2))


def main():
    fig, ax = plt.subplots(2, 2, figsize=(11, 9))
    draw(ax[0, 0], 'les', 'kid', r'distortion:  $1-$ lesion SSIM', r'perception:  KID $\times10^{3}$')
    draw(ax[0, 1], 'mae', 'kid', 'distortion:  global MAE', r'perception:  KID $\times10^{3}$')
    draw(ax[1, 0], 'les', 'fid', r'distortion:  $1-$ lesion SSIM', 'perception:  FID')
    draw(ax[1, 1], 'mae', 'fid', 'distortion:  global MAE', 'perception:  FID')
    ax[0, 0].set_title('Masked distortion (lesion) vs KID', fontsize=10)
    ax[0, 1].set_title('Global distortion (MAE) vs KID', fontsize=10)
    ax[1, 0].set_title('Masked distortion (lesion) vs FID', fontsize=10)
    ax[1, 1].set_title('Global distortion (MAE) vs FID', fontsize=10)
    fig.suptitle('The perception-distortion plane: each model as a point (lower-left is better)\n'
                 'error bars = patient-clustered 95% CI; FID has no error bar (point estimate only)',
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs('thesis/figures', exist_ok=True)
    fig.savefig(OUT, dpi=130, bbox_inches='tight')
    print('wrote', OUT)


if __name__ == '__main__':
    main()
