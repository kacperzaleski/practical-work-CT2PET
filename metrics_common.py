"""Shared image-quality metrics for a fair CPDM vs PMRF comparison.

All inputs are single-channel 2D arrays in ``[0, 1]`` — the on-disk PET space
(``[0, 32]`` SUV -> ``[-1, 1]`` by ``preprocess_autopet.normalize_pet`` ->
``[0, 1]`` by the savers).  SUV = value * ``SUV_MAX``.

Metrics are computed globally and within ROI masks derived from the GT PET, so
the SAME ruler applies to every model (the masks come from the ground truth,
never from a model's own output, so no model can game them).

``image_metrics`` returns MAE/PSNR/SSIM.  LPIPS/FID are intentionally NOT here:
they are patch/feature based and cannot be cleanly restricted to an ROI, so the
per-model scripts keep reporting those globally.
"""
import numpy as np
from skimage.metrics import structural_similarity as _ssim

# PET clip ceiling used in preprocess_autopet.normalize_pet ([0, 32] SUV).
SUV_MAX = 32.0


def suv(v01):
    """Map normalized [0, 1] PET back to SUV units."""
    return np.asarray(v01, dtype=np.float64) * SUV_MAX


def uptake_mask(gt01, suv_thresh):
    """Boolean ROI: GT voxels whose SUV exceeds ``suv_thresh``."""
    return suv(gt01) > suv_thresh


def _mae(pred, gt, mask):
    if mask is None:
        return float(np.mean(np.abs(pred - gt)))
    if not mask.any():
        return np.nan
    return float(np.mean(np.abs(pred[mask] - gt[mask])))


def _psnr(pred, gt, mask, data_range):
    if mask is None:
        mse = float(np.mean((pred - gt) ** 2))
    elif not mask.any():
        return np.nan
    else:
        mse = float(np.mean((pred[mask] - gt[mask]) ** 2))
    if mse <= 0:
        return np.nan
    return float(10.0 * np.log10((data_range ** 2) / mse))


def _ssim_masked(pred, gt, mask, data_range):
    # Full SSIM map, then average within the ROI (standard ROI-SSIM). The map is
    # computed on the whole image so the Gaussian window has proper context;
    # only the averaging is restricted to the mask.
    score, smap = _ssim(gt, pred, data_range=data_range, full=True)
    if mask is None:
        return float(score)
    if not mask.any():
        return np.nan
    return float(smap[mask].mean())


def image_metrics(pred01, gt01, mask=None, data_range=1.0):
    """MAE/PSNR/SSIM between two [0, 1] images, globally (mask=None) or in ROI."""
    pred = np.asarray(pred01, dtype=np.float64).squeeze()
    gt = np.asarray(gt01, dtype=np.float64).squeeze()
    return {
        'mae': _mae(pred, gt, mask),
        'psnr': _psnr(pred, gt, mask, data_range),
        'ssim': _ssim_masked(pred, gt, mask, data_range),
    }
