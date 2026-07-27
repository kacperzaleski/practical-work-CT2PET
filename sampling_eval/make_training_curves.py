"""Uniform training-dynamics figures for the thesis (thesis/figures/).

Design goal (supervisor + author request): ONE consistent figure showing the
train and validation loss of every one of the four generators side by side,
instead of the previous ad-hoc mix where some models had rich plots and others none.

  training_losses.png  — 2x2, one panel per generator (CPDM, concat-diffusion,
                         concat-flow, PMRF), each with train (solid) + val (dashed)
                         loss on a shared log-y axis. This is the centrepiece.
  training_support.png — 1x3 supporting stages (VQGAN reconstruction, attention-UNet
                         val IoU, PMRF Stage-1 posterior-mean), clearly separated.

Data sources (unified): the three Lightning generators logged their per-epoch
train/val loss to Weights & Biases; those histories are fetched once with `--fetch`
and cached to logs/curves/*.json so the figure is reproducible offline and the thesis
build never touches the network. CPDM logs locally to TensorBoard (per-step train loss
aggregated to per-epoch means + per-epoch val loss).

Usage:
  python sampling_eval/make_training_curves.py --fetch   # refresh wandb cache, then plot
  python sampling_eval/make_training_curves.py           # plot from cache (offline)
"""
import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))
import argparse
import glob
import json
import os
import re
import ast

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUTDIR = 'thesis/figures'
CURVES_DIR = 'logs/curves'
ENTITY = 'teamchaspi'
CPDM_TB = 'results/CT2PET_autoPET_fullbody/CPDM_focal_scratch/log'
CPDM_ITERS_PER_EPOCH = 625  # max_samples.train 5000 / batch 8 (see the CPDM fb64 config)

# model key -> (wandb project, run id). The three Lightning generators + PM anchor.
WANDB_RUNS = {
    'PMRF':        ('CT2PET-PMRF', 'ebto44cb'),
    'Concat-flow': ('CT2PET-PMRF', '7b3cma7k'),
    'Concat-diff': ('CT2PET-ConcatDiff', 'ox8n9bxn'),
    'PM-stage1':   ('CT2PET-PMRF', '655elwv1'),
}
TRAIN_COL, VAL_COL = 'tab:blue', 'tab:red'


# ------------------------------- data loading --------------------------------
def fetch_and_cache():
    """Pull train_loss_epoch + val_loss for each wandb run into logs/curves/<key>.json."""
    import wandb
    os.makedirs(CURVES_DIR, exist_ok=True)
    api = wandb.Api(timeout=30)
    for key, (proj, rid) in WANDB_RUNS.items():
        r = api.run(f'{ENTITY}/{proj}/{rid}')
        tr, va = {}, {}
        # NB: full history (no keys=) — train_loss_epoch and val_loss are logged at
        # different steps, so a keys=[...] filter (which needs all keys per row) is empty.
        for d in r.scan_history():
            e = d.get('epoch')
            if e is None:
                continue
            if d.get('train_loss_epoch') is not None:
                tr[int(e)] = float(d['train_loss_epoch'])
            if d.get('val_loss') is not None:
                va[int(e)] = float(d['val_loss'])
        ep = sorted(set(tr) | set(va))
        out = {'epoch': ep,
               'train': [tr.get(e) for e in ep],
               'val': [va.get(e) for e in ep]}
        json.dump(out, open(os.path.join(CURVES_DIR, f'{key}.json'), 'w'))
        print(f'cached {key}: {len(ep)} epochs  ({proj}/{rid})')


def load_cached(key):
    path = os.path.join(CURVES_DIR, f'{key}.json')
    if not os.path.exists(path):
        return None
    d = json.load(open(path))
    return np.array(d['epoch'], float), \
        np.array([np.nan if v is None else v for v in d['train']], float), \
        np.array([np.nan if v is None else v for v in d['val']], float)


def cpdm_curves():
    """CPDM (local TB): per-epoch train = mean of per-step loss/train; val = val_epoch/loss."""
    from tensorboard.backend.event_processing import event_accumulator
    files = sorted(glob.glob(os.path.join(CPDM_TB, 'events.out.tfevents.*')))
    if not files:
        return None
    tr_step, val = [], {}
    for f in files:
        ea = event_accumulator.EventAccumulator(f, size_guidance={'scalars': 0})
        ea.Reload()
        sc = ea.Tags()['scalars']
        if 'loss/train' in sc:
            tr_step += [(s.step, s.value) for s in ea.Scalars('loss/train')]
        if 'val_epoch/loss' in sc:
            for s in ea.Scalars('val_epoch/loss'):
                val[int(s.step)] = s.value
    if not tr_step:
        return None
    # aggregate per-step train loss into per-epoch means
    per = {}
    for step, v in tr_step:
        per.setdefault(step // CPDM_ITERS_PER_EPOCH, []).append(v)
    tr_ep = sorted(per)
    train = np.array([np.mean(per[e]) for e in tr_ep], float)
    vep = sorted(val)
    return np.array(tr_ep, float), train, np.array(vep, float), np.array([val[e] for e in vep], float)


# --------------------------------- plotting ----------------------------------
def panel(ax, title, e_tr, tr, e_val, val):
    if e_tr is not None and np.isfinite(tr).any():
        ax.plot(e_tr, tr, '-', color=TRAIN_COL, lw=1.6, label='train')
    if e_val is not None and np.isfinite(val).any():
        ax.plot(e_val, val, '--', color=VAL_COL, lw=1.6, label='validation')
    ax.set_yscale('log')
    ax.set_title(title, fontsize=11)
    ax.set_xlabel('epoch'); ax.set_ylabel('loss'); ax.grid(alpha=.3, which='both')
    ax.legend(fontsize=8, frameon=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fetch', action='store_true', help='refresh the wandb cache first')
    args = ap.parse_args()
    if args.fetch:
        fetch_and_cache()
    os.makedirs(OUTDIR, exist_ok=True)

    # ---------- training_losses.png : the four generators, train + val ----------
    fig, ax = plt.subplots(2, 2, figsize=(10, 7.2))
    a = ax.ravel()

    cp = cpdm_curves()
    if cp is not None:
        panel(a[0], 'CPDM (Brownian-bridge latent diffusion)', cp[0], cp[1], cp[2], cp[3])
    else:
        a[0].set_title('CPDM — (training in progress)', fontsize=11); a[0].axis('off')

    for idx, (key, title) in enumerate([
            ('Concat-diff', 'Concatenation-conditioned latent diffusion'),
            ('Concat-flow', 'Concatenation-conditioned rectified flow'),
            ('PMRF', 'PMRF (posterior-mean rectified flow, Stage 2)')], start=1):
        c = load_cached(key)
        if c is None:
            a[idx].set_title(f'{title} — (no cached curve)', fontsize=11); a[idx].axis('off'); continue
        e, tr, val = c
        panel(a[idx], title, e, tr, e, val)

    fig.suptitle('Training and validation loss of the four generators '
                 '(log scale; objectives differ, so compare shapes not levels)', fontsize=12)
    fig.tight_layout()
    fig.savefig(f'{OUTDIR}/training_losses.png', dpi=130, bbox_inches='tight'); plt.close(fig)
    print(f'wrote {OUTDIR}/training_losses.png')

    # ---------- training_support.png : supporting stages ----------
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    # (a) VQGAN reconstruction (text log)
    ev, tot, rec = _parse_vqgan('logs/vqgan_fb64.log')
    if rec:
        ax[0].plot(ev, tot, '-o', ms=3, color='tab:red', label='total val loss')
        ax[0].plot(ev, rec, '-^', ms=3, color='tab:pink', label='reconstruction $L_1$')
        ax[0].legend(fontsize=8, frameon=False)
    ax[0].set_title('VQGAN autoencoder (val loss)', fontsize=10)
    ax[0].set_xlabel('epoch'); ax[0].grid(alpha=.3)
    # (b) attention-UNet IoU
    ea, va = _parse_att_iou('logs/att_retrain.log')
    ax[1].plot(ea, va, '-o', ms=3, color='tab:cyan')
    ax[1].set_title('Attention-UNet best val IoU (focal target)', fontsize=10)
    ax[1].set_xlabel('improvement step'); ax[1].set_ylabel('IoU'); ax[1].grid(alpha=.3)
    # (c) PM stage-1 posterior mean train+val
    c = load_cached('PM-stage1')
    if c is not None:
        e, tr, val = c
        ax[2].plot(e, tr, '-', color=TRAIN_COL, lw=1.6, label='train')
        ax[2].plot(e, val, '--', color=VAL_COL, lw=1.6, label='validation')
        ax[2].set_yscale('log'); ax[2].legend(fontsize=8, frameon=False)
    ax[2].set_title('Posterior-mean U-Net / PMRF Stage 1 (MSE)', fontsize=10)
    ax[2].set_xlabel('epoch'); ax[2].set_ylabel('loss'); ax[2].grid(alpha=.3, which='both')

    fig.suptitle('Supporting stages (shared across the diffusion / flow models)', fontsize=12)
    fig.tight_layout()
    fig.savefig(f'{OUTDIR}/training_support.png', dpi=130, bbox_inches='tight'); plt.close(fig)
    print(f'wrote {OUTDIR}/training_support.png')


def _parse_vqgan(path):
    eps, tot, rec = [], [], []
    if not os.path.exists(path):
        return eps, tot, rec
    pat = re.compile(r'\[val\s*\]\s*e(\d+)\s*(\{.*\})')
    for ln in open(path):
        m = pat.search(ln)
        if m:
            try:
                d = ast.literal_eval(m.group(2))
                eps.append(int(m.group(1))); tot.append(float(d['val_loss'])); rec.append(float(d['val_rec_loss']))
            except Exception:
                pass
    return eps, tot, rec


def _parse_att_iou(path):
    vals = [float(m.group(1)) for ln in open(path)
            for m in [re.search(r'New best score:\s*([\d.]+)', ln)] if m] if os.path.exists(path) else []
    return list(range(len(vals))), vals


if __name__ == '__main__':
    main()
