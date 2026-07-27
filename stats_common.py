"""Statistical helpers for the CT->PET comparison: patient-clustered bootstrap
confidence intervals and paired significance tests.

Why patient-clustered. The test set has many slices but few patients, and slices
from the same patient are strongly correlated (same anatomy, scanner, tracer). A
naive per-slice bootstrap would treat 25 slices of one patient as 25 independent
samples and badly understate the uncertainty. So every resample here is over
*patients* (the independent unit): we draw patients with replacement and pool all
their slices. This is the standard cluster/block bootstrap.

Pure-numpy (+ scipy.stats for Wilcoxon) so it imports fast and has no torch
dependency, mirroring metrics_common.py.
"""
import re

import numpy as np

_SLICE_RE = re.compile(r'_s\d+$')


def patient_of(name):
    """Slice name/stem -> patient id (drop trailing _sNNN and any .npy)."""
    stem = name[:-4] if name.endswith('.npy') else name
    return _SLICE_RE.sub('', stem)


def _by_patient(patients):
    """patient-id array -> {pid: np.array(row indices)}."""
    groups = {}
    for i, p in enumerate(patients):
        groups.setdefault(p, []).append(i)
    return {p: np.asarray(ix) for p, ix in groups.items()}


def cluster_bootstrap_ci(values, patients, agg=np.nanmean, n_boot=2000, seed=0,
                         alpha=0.05):
    """Patient-clustered bootstrap CI for a single metric.

    values   : (N,) per-slice metric values (may contain NaN, e.g. empty-ROI slices).
    patients : (N,) patient id per slice.
    Returns  : (point, lo, hi) where point = agg(values) and [lo, hi] is the
               (1-alpha) percentile CI over patient resamples.
    """
    values = np.asarray(values, dtype=np.float64)
    patients = np.asarray(patients)
    groups = _by_patient(patients)
    pids = np.array(list(groups.keys()))
    rng = np.random.default_rng(seed)

    point = float(agg(values))
    if len(pids) < 2:
        return point, np.nan, np.nan

    boots = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, len(pids), size=len(pids))
        idx = np.concatenate([groups[pids[j]] for j in pick])
        boots[b] = agg(values[idx])
    lo = float(np.nanpercentile(boots, 100 * alpha / 2))
    hi = float(np.nanpercentile(boots, 100 * (1 - alpha / 2)))
    return point, lo, hi


def naive_ci(values, alpha=0.05):
    """Textbook per-slice CI: mean +/- z * standard-error, treating every slice as an
    independent sample. This is the (over-optimistic) 'unclustered' counterpart to
    cluster_bootstrap_ci -- it ignores that slices from one patient are correlated, so
    its interval is the narrower of the two. Returns (point, lo, hi)."""
    v = np.asarray(values, dtype=np.float64)
    v = v[~np.isnan(v)]
    point = float(np.mean(v)) if v.size else np.nan
    if v.size < 2:
        return point, np.nan, np.nan
    se = float(np.std(v, ddof=1) / np.sqrt(v.size))
    z = 1.959963985  # 95% normal quantile
    return point, point - z * se, point + z * se


def patient_ci(values, patients, alpha=0.05):
    """Simple patient-level CI: average within each patient first (one number per
    patient), then take mean +/- t * standard-error over those per-patient means. This
    is the intuitive 'do the stats on the ~12 patients, not the ~300 slices' clustering,
    and it agrees closely with cluster_bootstrap_ci. Returns (point, lo, hi)."""
    from scipy.stats import t as _t
    v = np.asarray(values, dtype=np.float64)
    patients = np.asarray(patients)
    groups = _by_patient(patients)
    pm = np.array([np.nanmean(v[ix]) for ix in groups.values()])
    pm = pm[~np.isnan(pm)]
    point = float(np.mean(pm)) if pm.size else np.nan
    if pm.size < 2:
        return point, np.nan, np.nan
    se = float(np.std(pm, ddof=1) / np.sqrt(pm.size))
    crit = float(_t.ppf(1 - alpha / 2, pm.size - 1))
    return point, point - crit * se, point + crit * se


def paired_cluster_bootstrap(a, b, patients, n_boot=2000, seed=0, alpha=0.05):
    """Paired patient-clustered bootstrap on the difference (a - b).

    a, b     : (N,) per-slice metric values for two models on the SAME slices.
    patients : (N,) patient id per slice.
    Returns  : dict(delta, lo, hi, p) — mean difference, its (1-alpha) CI, and a
               two-sided bootstrap p-value (H0: mean difference = 0).
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    diff = a - b
    patients = np.asarray(patients)
    valid = ~np.isnan(diff)
    diff, patients = diff[valid], patients[valid]

    groups = _by_patient(patients)
    pids = np.array(list(groups.keys()))
    rng = np.random.default_rng(seed)

    delta = float(np.mean(diff)) if diff.size else np.nan
    if len(pids) < 2 or diff.size == 0:
        return {'delta': delta, 'lo': np.nan, 'hi': np.nan, 'p': np.nan}

    boots = np.empty(n_boot)
    for i in range(n_boot):
        pick = rng.integers(0, len(pids), size=len(pids))
        idx = np.concatenate([groups[pids[j]] for j in pick])
        boots[i] = np.mean(diff[idx])
    lo = float(np.percentile(boots, 100 * alpha / 2))
    hi = float(np.percentile(boots, 100 * (1 - alpha / 2)))
    # two-sided bootstrap p: twice the smaller tail mass past zero (min 1/n_boot).
    frac_le = float(np.mean(boots <= 0))
    frac_ge = float(np.mean(boots >= 0))
    p = min(1.0, 2 * min(frac_le, frac_ge))
    p = max(p, 1.0 / n_boot)
    return {'delta': delta, 'lo': lo, 'hi': hi, 'p': p}


def wilcoxon_signed_rank(a, b):
    """Distribution-free paired test (complements the bootstrap). Returns (stat, p);
    (nan, nan) if too few non-zero paired differences."""
    from scipy.stats import wilcoxon
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    m = ~(np.isnan(a) | np.isnan(b))
    a, b = a[m], b[m]
    if a.size < 1 or np.allclose(a, b):
        return np.nan, np.nan
    try:
        res = wilcoxon(a, b, zero_method='wilcox', alternative='two-sided')
        return float(res.statistic), float(res.pvalue)
    except ValueError:
        return np.nan, np.nan


def holm_correction(pvalues):
    """Holm-Bonferroni step-down adjustment for a family of p-values.
    Returns adjusted p-values in the original order (each clipped to <=1)."""
    p = np.asarray(pvalues, dtype=np.float64)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * p[idx]
        running = max(running, val)  # enforce monotonicity
        adj[idx] = min(1.0, running)
    return adj


def fmt_ci(point, lo, hi, prec=3):
    """'0.463 [0.441, 0.487]' — compact point + CI for tables/console."""
    if np.isnan(lo) or np.isnan(hi):
        return f'{point:.{prec}f}'
    return f'{point:.{prec}f} [{lo:.{prec}f}, {hi:.{prec}f}]'
