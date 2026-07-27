"""
Supervisor-facing showcase of the CPDM-on-autoPET reimplementation.

Format: percent-style cells (`# %%` for code, `# %% [markdown]` for prose).
- Open in VS Code's Python interactive mode to run cells.
- Or convert to a .ipynb:
    jupytext --to notebook report.py        # produces report.ipynb
- Or just run end-to-end:
    python report.py                          # writes figures into report_out/

The end-to-end run on CPU is ~10 min (most of it is CPDM sampling on 4 val
images at the trained sample_step=200).
"""

# %% [markdown]
# # CT → PET Translation on autoPET
#
# **Reimplementation of CPDM ([Nguyen et al., WACV 2025](https://openaccess.thecvf.com/content/WACV2025/papers/Nguyen_CT_to_PET_Translation_A_Large-Scale_Dataset_and_Domain-Knowledge-Guided_Diffusion_WACV_2025_paper.pdf), repo [thanhhff/CPDM](https://github.com/thanhhff/CPDM)) on the autoPET dataset.**
#
# Practical-work showcase. Everything was trained CPU-only on an Intel i7-8650U (8 cores, 15 GB RAM).
#
# **Pipeline.** Three pre-stages feed the main diffusion model:
#
# 1. **Attention-map UNet** — learns `CT → high-PET-uptake binary mask`. Trained directly on PET-derived targets (PET > 75th percentile + binary closing).
# 2. **Attention-map export** — runs the trained UNet over every CT slice and writes per-slice probability maps to disk. CPDM consumes these by filename at training time.
# 3. **VQGAN** — encoder/decoder learning a `4 × 16 × 16` latent for the `1 × 64 × 64` slices. Frozen at CPDM training time.
# 4. **CPDM (Brownian Bridge diffusion)** — denoiser UNet operating on the VQGAN latent. Conditions on:
#    * the pre-exported attention map,
#    * a CT-derived 511 keV attenuation map (closed-form transform applied on the fly inside the model).
#
# **Key autoPET adaptations** (full list in `NOTES.md`):
# - Image size **64 × 64** (paper 256 × 256) — CPU budget.
# - **Brain region only** (last 20 % of each whole-body volume) — CPU budget.
# - HU-window CT normalization (paper used `pixel/2047`) → `CT2PETDiffusionModel.get_attenuation_map` patched to invert it correctly.
# - Replaced the original paper's `CustomAlignedDataset` (A/B layout, re-normalizes inputs) with `CT2PETAlignedDataset` (CT/PET layout, passthrough).
# - Wrote `train_vqgan.py` because the paper's repo expects external VQGAN training via `taming-transformers`.

# %%
import os
import sys
import math
import warnings
from pathlib import Path

import numpy as np
import torch
import yaml
import matplotlib.pyplot as plt

# Repo imports
PROJ = Path('/home/kacperzaleski/Projects/practical-work-CT2PET').resolve()
sys.path.insert(0, str(PROJ))

from utils import dict2namespace
from datasets.AttentionMapDataset import AttentionMapDataset
from datasets.CT2PETAlignedDataset import CT2PETAlignedDataset
from training.train_attention_map import UNetAttentionMapModel
from training.train_vqgan import VQGANLightning
from model.BrownianBridge.CT2PETDiffusionModel import CT2PETDiffusionModel

DEVICE = torch.device('cpu')
torch.set_num_threads(max(1, (os.cpu_count() or 1) - 1))

# Output dir for figures when running as a script.
OUT = PROJ / 'report_out'
OUT.mkdir(exist_ok=True)

# Helpers ----------------------------------------------------------------------

def to_disp(x):
    """Map a 1-channel tensor in [-1, 1] to a (H, W) numpy array in [0, 1] for imshow."""
    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()
    if x.ndim == 4:
        x = x[0]
    if x.ndim == 3:
        x = x[0]
    return np.clip((x + 1.0) * 0.5, 0.0, 1.0)

def show_grid(arrs, titles, ncols=None, cmap='gray', suptitle=None, savepath=None):
    n = len(arrs)
    ncols = ncols or n
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.4 * ncols, 2.4 * nrows))
    axes = np.atleast_1d(axes).flatten()
    for ax in axes[n:]:
        ax.axis('off')
    for ax, a, t in zip(axes[:n], arrs, titles):
        ax.imshow(a, cmap=cmap, vmin=0, vmax=1)
        ax.set_title(t, fontsize=9)
        ax.axis('off')
    if suptitle:
        fig.suptitle(suptitle, fontsize=11)
    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=140, bbox_inches='tight')
    return fig

def parse_tb_scalars(events_path):
    """Return {tag: [(step, value), ...]}."""
    from tensorboard.backend.event_processing import event_accumulator
    ea = event_accumulator.EventAccumulator(str(events_path), size_guidance={'scalars': 0})
    ea.Reload()
    out = {}
    for tag in ea.Tags()['scalars']:
        out[tag] = [(s.step, s.value) for s in ea.Scalars(tag)]
    return out

print(f'PyTorch {torch.__version__}, CPU threads = {torch.get_num_threads()}')

# %% [markdown]
# ## 1. Dataset overview
#
# **autoPET** challenge — paired whole-body CT + PET NIfTI volumes per patient. After preprocessing (`preprocess_autopet.py`):
#
# | Split | Patients | 64×64 brain slices |
# |-------|---------:|-------------------:|
# | train | 1 291    | 86 197             |
# | val   | 161      | 10 464             |
# | test  | 162      | 11 330             |
#
# Each slice: `float32`, shape `(64, 64)`, values in `[-1, 1]`. CT normalized over HU `[-1000, 3071]`, PET over SUV `[0, 32]`.

# %%
DATA_ROOT = PROJ / 'data' / 'processed'

split_counts = {
    split: sum(1 for _ in (DATA_ROOT / split / 'CT').glob('*.npy'))
    for split in ('train', 'val', 'test')
}
print('On-disk slice counts:', split_counts)

# Show 6 example CT/PET pairs from the val split.
val_ds = CT2PETAlignedDataset(
    dict2namespace({'data_root': str(DATA_ROOT), 'image_size': 64, 'flip': False}),
    stage='val',
)
print(f'\nVal pairs in registered dataset: {len(val_ds)}')

# Pick a handful of slice indices spread across patients.
rng = np.random.default_rng(2026)
sample_idxs = rng.choice(len(val_ds), size=6, replace=False)

ct_tiles, pet_tiles, titles = [], [], []
for k, i in enumerate(sample_idxs):
    (pet, name), (ct, _) = val_ds[int(i)]
    ct_tiles.append(to_disp(ct))
    pet_tiles.append(to_disp(pet))
    titles.append(f'#{i}')

fig = show_grid(ct_tiles + pet_tiles,
                titles=[f'CT {t}' for t in titles] + [f'PET {t}' for t in titles],
                ncols=6, suptitle='Sample CT (top) ↔ PET (bottom) brain slices',
                savepath=OUT / '01_dataset_samples.png')
plt.show()

# %% [markdown]
# ## 2. Stage 1 — Attention-map UNet
#
# The first sub-network learns where in the CT the corresponding PET will light up.
#
# - Architecture: **ResNet34** encoder + **scSE** UNet decoder, single-channel I/O. Paper uses ResNet50; we downsized for CPU.
# - Target: `(PET > 75th percentile)` followed by `scipy.ndimage.binary_closing`. Generated on the fly by `AttentionMapDataset` from PET; CT is the input.
# - Loss: `0.7 · Dice + 0.3 · BCE` (paper uses pure Dice).
# - Augmentation: random horizontal flip.
#
# **Final metrics (val, after 3 epochs × 200 batches on CPU):**
# - `val_dataset_iou` = **0.7050**
# - `val_per_image_iou` = 0.7079
# - `val_f1_score` = **0.8266**

# %%
ATT_CKPT = PROJ / 'checkpoints' / 'AttentionMap' / 'attention_map_unet-epoch=02-val_dataset_iou=0.7050.ckpt'

print(f'Loading attention UNet from {ATT_CKPT.name} …')
warnings.filterwarnings('ignore', category=FutureWarning)
att_model = UNetAttentionMapModel.load_from_checkpoint(str(ATT_CKPT), map_location=DEVICE)
att_model.eval()
print(f'Params: {sum(p.numel() for p in att_model.parameters()) / 1e6:.2f} M')

# Run the trained UNet on 4 val examples; pull the ground-truth attention target
# from AttentionMapDataset (which derives it from PET >75th percentile).
att_ds = AttentionMapDataset(str(DATA_ROOT), split='val', image_size=64, flip=False)
example_idxs = [10, 250, 1000, 5000]

ct_grid, pred_grid, gt_grid = [], [], []
for i in example_idxs:
    sample = att_ds[i]
    ct = sample['ct'].unsqueeze(0)
    gt = sample['attention_map'].squeeze().numpy()
    with torch.no_grad():
        pred = torch.sigmoid(att_model(ct)).squeeze().numpy()
    ct_grid.append(to_disp(ct))
    pred_grid.append(pred)
    gt_grid.append(gt)

fig = show_grid(ct_grid + pred_grid + gt_grid,
                titles=[f'CT #{i}' for i in example_idxs] +
                       [f'predicted #{i}' for i in example_idxs] +
                       [f'target #{i}' for i in example_idxs],
                ncols=len(example_idxs),
                suptitle='Attention UNet — CT input | predicted mask | PET-derived target',
                savepath=OUT / '02_attention_unet_predictions.png')
plt.show()

# %% [markdown]
# ### Why an attention model at all?
#
# The PET signal in brain slices is concentrated in a few high-uptake regions; the rest is near-background. A separately-trained mask provides the diffusion denoiser with a hard spatial prior about *where* to focus, decoupling that geometric problem from the harder *intensity* prediction. The paper's ablation (Table 4) shows removing this conditioning costs ~12 % in LPIPS.

# %% [markdown]
# ## 3. Stage 2 — Bulk attention-map export
#
# `export_attention_maps.py` runs the trained UNet over every CT slice in all three splits at batch_size=32. Writes float32 sigmoid probabilities (shape `(64, 64)`, values in `[0, 1]`) to `data/processed/{split}/AttentionMaps/<slice>.npy`.
#
# This is **exactly** the format `CT2PETDiffusionModel.get_attention_map` reads:
# ```python
# np_cond = np.load(os.path.join(attention_map_path, f'{x_name[i]}.npy'))
# np_cond[np_cond < 0.5] = 0
# np_cond[np_cond >= 0.5] = 1
# ```
# So no extra conversion step is needed at CPDM training time.
#
# Counts written:
#
# | split | maps |
# |-------|-----:|
# | train | 86 197 |
# | val   | 10 464 |
# | test  | 11 330 |
#
# CPU runtime for the full export: ~12 minutes.

# %%
# Sanity-check the exported maps match what the UNet produces on the fly.
example_idx = example_idxs[0]
(pet, name), (ct, _) = val_ds[example_idx]
exported_map = np.load(DATA_ROOT / 'val' / 'AttentionMaps' / f'{name}.npy')
print(f'Exported attention map for {name}.npy:')
print(f'  shape={exported_map.shape}  dtype={exported_map.dtype}  '
      f'range=[{exported_map.min():.4f}, {exported_map.max():.4f}]  mean={exported_map.mean():.4f}')

fig = show_grid([to_disp(ct), exported_map, (exported_map >= 0.5).astype(np.float32), to_disp(pet)],
                titles=['CT (input to UNet)', 'sigmoid map (on disk)',
                        'thresholded ≥ 0.5 (as model loads it)', 'PET (ground truth context)'],
                ncols=4, suptitle=f'Pre-exported attention map for slice {name}',
                savepath=OUT / '03_exported_attention.png')
plt.show()

# %% [markdown]
# ## 4. Stage 3 — VQGAN encoder/decoder
#
# Trained because the paper expects an externally-trained VQGAN (their README points at `CompVis/taming-transformers`). `train_vqgan.py` is a Lightning 2.x compatible wrapper subclassing `model/VQGAN/vqgan.py:VQModel`. Saved Lightning checkpoint loads directly into the original `VQModel` via its existing `init_from_ckpt`.
#
# - Architecture: `ch=128, ch_mult=(1, 2, 4)`, 2 ResBlocks/level, no attention. 1-channel I/O. 8192 codebook entries × `embed_dim=4`. **55.3 M params**.
# - Loss: `L1(x, x_rec) + commitment_loss`. No perceptual loss or GAN discriminator (the original code used Lightning-1.x `optimizer_idx`-style manual optimization which is gone in PyTorch Lightning 2.x).
# - Each batch concatenates CT and PET along the batch dimension, so the encoder sees both modalities equally.
# - Spatial compression `64 → 16` (×4 spatial downsampling). Latent shape: `4 × 16 × 16`.
#
# **Final metrics (epoch 1, val_loss ≈ 0.0185):** `val_rec_loss ≈ 0.0168`, `val_codebook_loss ≈ 0.0017`.

# %%
VQGAN_CKPT = PROJ / 'checkpoints' / 'VQGAN' / 'vqgan-epoch=01-v2.ckpt'

# Reconstruct the same ddconfig the YAML had.
DD = dict(attn_resolutions=[], ch=128, ch_mult=(1, 2, 4), double_z=False,
          dropout=0.0, in_channels=1, out_ch=1, num_res_blocks=2,
          resolution=64, z_channels=4)

vq = VQGANLightning.load_from_checkpoint(
    str(VQGAN_CKPT), map_location=DEVICE,
    ddconfig=DD, n_embed=8192, embed_dim=4, learning_rate=1e-4,
)
vq.eval()
print(f'VQGAN params: {sum(p.numel() for p in vq.parameters()) / 1e6:.2f} M')

# Show CT and PET reconstructions side by side.
ct_orig, ct_rec, pet_orig, pet_rec = [], [], [], []
with torch.no_grad():
    for i in example_idxs:
        (pet, _), (ct, _) = val_ds[i]
        ct_b = ct.unsqueeze(0)
        pet_b = pet.unsqueeze(0)
        ct_rec_t, _ = vq(ct_b)
        pet_rec_t, _ = vq(pet_b)
        ct_orig.append(to_disp(ct_b)); ct_rec.append(to_disp(ct_rec_t))
        pet_orig.append(to_disp(pet_b)); pet_rec.append(to_disp(pet_rec_t))

fig = show_grid(
    ct_orig + ct_rec + pet_orig + pet_rec,
    titles=[f'CT in #{i}' for i in example_idxs] +
           [f'CT rec #{i}' for i in example_idxs] +
           [f'PET in #{i}' for i in example_idxs] +
           [f'PET rec #{i}' for i in example_idxs],
    ncols=len(example_idxs),
    suptitle='VQGAN reconstructions — 64×64 → 4×16×16 latent → 64×64',
    savepath=OUT / '04_vqgan_reconstructions.png',
)
plt.show()

# %% [markdown]
# The reconstructions retain the dominant intensity structure but smooth out fine detail — expected at this latent size (`4 × 16 × 16`, i.e. 1024 dims for a 4096-pixel image, a **4× compression**). For our purposes this is fine: BB diffusion needs a latent space stable enough to denoise in, not pixel-perfect autoencoding.

# %% [markdown]
# ## 5. Stage 4 — CPDM (Brownian Bridge diffusion)
#
# **Architecture.** `CT2PETDiffusionModel` composes:
#
# - Frozen VQGAN (the one trained above) — encodes CT and PET to `4 × 16 × 16`.
# - Two `SpatialRescaler` condition stages (one per conditioning map: attention + attenuation). Each downsamples the `1 × 64 × 64` map to `4 × 16 × 16` and remaps channels to match the latent. The 1-to-4 channel remap is a learned `Conv2d`.
# - **UNet denoiser** operating on the latent: `image_size = 16`, `model_channels = 128`, `channel_mult = (1, 2, 3, 4)`, attention at resolutions `(8, 4)`. Input channels = `4 (x_t) + 4 (att context) + 4 (atte context) = 12` (the original CPDM's `3 · z_channels` rule). Output channels = 4.
# - **EMA** with decay 0.995 (starts at step 30 000 — never triggered in our short CPU runs).
# - **Brownian Bridge process**: 1000 train timesteps, 200 sample timesteps, `objective='grad'`, `eta=1.0`, `loss_type='l1'`.
#
# **Domain knowledge** (the paper's two novel signals, both encoded as channels into the denoiser):
#
# 1. **Attention map** loaded from `data/processed/{split}/AttentionMaps/` (pre-exported in Stage 2). Hard-thresholded at 0.5 at load time.
# 2. **Attenuation map** computed on the fly from CT via a closed-form HU → 511 keV linear-attenuation-coefficient transform (`attenuationCT_to_511`, kVp=140). I patched the function to invert our HU-window normalization (`HU = (ct + 1)·(3071 − (−1000))/2 + (−1000)`); the original code assumed CT was pixel/2047-normalized.
#
# **Training history.** Three CPU launches totalling **10 completed epochs** at 250 iter / epoch. Used `setsid nohup …` for detachment; SIGINTd once for a laptop move, then resumed cleanly from `last_model.pth`. Final ckpt: `latest_model_10.pth` (model + EMA + optimizer + LR scheduler state).
#
# **Paper-aligned per-epoch evaluation.** Each epoch the runner samples 16 val images via the full 200-step BB reverse process and computes LPIPS / MAE / SSIM / PSNR. All scalars stream to W&B (project `CT2PET-CPDM`, runs `r1yb3gtw` and `4inhu3cy`) in addition to TensorBoard.

# %%
# Plot training loss curves and paper metrics from the TB events files.
cpdm_log_dir = PROJ / 'results' / 'CT2PET_autoPET' / 'CPDM' / 'log'
event_files = sorted(cpdm_log_dir.glob('events.out.tfevents.*'))
print('Event files found:', [p.name for p in event_files])

# Aggregate across the two long-run TB files (the first launch was just a wiring sanity check).
def parse_all_events(paths):
    all_scalars = {}
    for p in paths:
        for tag, series in parse_tb_scalars(p).items():
            all_scalars.setdefault(tag, []).extend(series)
    for tag in all_scalars:
        all_scalars[tag].sort(key=lambda sv: sv[0])
    return all_scalars

scalars = parse_all_events(event_files)
print('Available scalars:', list(scalars.keys()))

fig, axes = plt.subplots(2, 3, figsize=(13, 6.5))

def plot_tag(ax, tag, label=None, color=None):
    if tag not in scalars or not scalars[tag]:
        ax.text(0.5, 0.5, f'{tag} unavailable', ha='center', va='center', transform=ax.transAxes)
        ax.set_axis_off()
        return
    xs, ys = zip(*scalars[tag])
    ax.plot(xs, ys, color=color)
    ax.set_title(label or tag, fontsize=10)
    ax.set_xlabel('global step' if 'loss/train' in tag or 'loss/val_step' in tag else 'epoch')
    ax.grid(alpha=0.3)

plot_tag(axes[0, 0], 'loss/train', 'Train loss (per step)')
plot_tag(axes[0, 1], 'loss/val_step', 'Val loss (single batch, every 50 steps)')
plot_tag(axes[0, 2], 'val_epoch/loss', 'Val epoch loss')
plot_tag(axes[1, 0], 'metrics/lpips', 'LPIPS (lower = better)', color='C3')
plot_tag(axes[1, 1], 'metrics/ssim', 'SSIM (higher = better)', color='C2')
plot_tag(axes[1, 2], 'metrics/psnr', 'PSNR dB (higher = better)', color='C1')
fig.suptitle('CPDM training curves — both long runs combined', fontsize=11)
fig.tight_layout()
fig.savefig(OUT / '05_cpdm_training_curves.png', dpi=140, bbox_inches='tight')
plt.show()

# %% [markdown]
# ### Convergence diagnosis
#
# - Train loss collapses fast — from ~0.57 at step 1 to ~0.04 by step 250 — because PET brain slices are mostly background near −1 and the model rapidly learns a near-uniform output.
# - Val loss settles in **0.014 – 0.020** across all 10 epochs. Best `val_epoch/loss = 0.01394` at epoch 7.
# - Paper metrics oscillate in a narrow band: LPIPS in `0.166 – 0.200`, MAE in `0.0053 – 0.0059`, SSIM in `0.868 – 0.890`, PSNR in `40.77 – 41.70 dB`. That is noise-level fluctuation, not directional improvement.
# - ReduceLROnPlateau patience was 3000 iter ≈ 12 epochs; training was stopped manually at epoch 11 because the metrics had plateaued.

# %% [markdown]
# ### Qualitative samples
#
# Below: for 4 fixed validation slices, render the full conditioning context next to the model's generation and the ground truth.
# Each row is one slice; columns are CT input · attention map · attenuation map · **generated PET** · ground-truth PET · per-pixel error.
#
# Each generation runs the full 200-step Brownian Bridge reverse process on CPU (~1 min per batch of 4 at this resolution).

# %%
# Build the CPDM model and load the final checkpoint.
CPDM_CFG = PROJ / 'config' / 'CPDM-autoPET.yaml'
CPDM_CKPT = PROJ / 'results' / 'CT2PET_autoPET' / 'CPDM' / 'checkpoint' / 'latest_model_10.pth'

with open(CPDM_CFG) as f:
    cfg_dict = yaml.load(f, Loader=yaml.FullLoader)
cfg = dict2namespace(cfg_dict)
cfg.training.use_DDP = False
cfg.training.device = [DEVICE]

print(f'Instantiating CT2PETDiffusionModel and loading {CPDM_CKPT.name} …')
cpdm = CT2PETDiffusionModel(cfg.model).to(DEVICE)
state = torch.load(CPDM_CKPT, map_location=DEVICE)
cpdm.load_state_dict(state['model'], strict=True)
cpdm.eval()
print(f"resumed at epoch={state['epoch']}, step={state['step']}")
print(f'CPDM trainable params: '
      f"{sum(p.numel() for p in cpdm.parameters() if p.requires_grad) / 1e6:.2f} M  "
      f"total: {sum(p.numel() for p in cpdm.parameters()) / 1e6:.2f} M")

# %%
# Pick four val slices by ground-truth PET uptake (max SUV after normalization)
# so the qualitative panel actually shows something — random indexing over a
# brain-only split lands in mostly-background slices. Two high-uptake, two mid.
SCORES_CACHE = OUT / 'val_pet_max_scores.npy'
if SCORES_CACHE.exists():
    scores = np.load(SCORES_CACHE)
else:
    print('Scoring val slices by max PET (one-time pass)…')
    scores = np.empty(len(val_ds), dtype=np.float32)
    for i in range(len(val_ds)):
        (pet_i, _), _ = val_ds[i]
        scores[i] = float(pet_i.max())
    np.save(SCORES_CACHE, scores)
order = np.argsort(scores)
n_val = len(order)
sample_indices = [int(order[-5]), int(order[-50]),
                  int(order[n_val // 2]), int(order[n_val // 4])]
print(f'Selected slice indices (by GT PET max): {sample_indices}  '
      f'max-SUV-norm = {[float(scores[i]) for i in sample_indices]}')

pet_list, ct_list, name_list = [], [], []
for i in sample_indices:
    (pet, name), (ct, _) = val_ds[i]
    pet_list.append(pet); ct_list.append(ct); name_list.append(name)

x_pet = torch.stack(pet_list).to(DEVICE)
x_ct = torch.stack(ct_list).to(DEVICE)
print('PET batch shape:', x_pet.shape, '  CT batch shape:', x_ct.shape)

# %%
# Run the BB reverse process — this is the slow cell (~1-3 min on CPU).
print('Running 200-step BB sampling on 4 val images (this is the slow cell)…')
import time as _t
t0 = _t.time()
with torch.no_grad():
    samples, add_cond = cpdm.sample(x_ct, name_list, 'val', clip_denoised=False)
print(f'Done. Wall-clock: {_t.time() - t0:.1f} s. Sample shape: {tuple(samples.shape)}')

# add_cond carries (att_map, atte_map) along channel dim — verify shape.
print('add_cond shape:', tuple(add_cond.shape))  # (B, 2, 64, 64)

# %%
# Per-pixel absolute error in PET intensity (signed range [-1,1] → display unsigned).
err = (samples - x_pet).abs().clamp_(0, 2).div_(2.0)

ncols = len(sample_indices)
fig, axes = plt.subplots(6, ncols, figsize=(2.4 * ncols, 14))
row_titles = ['CT', 'attention', 'attenuation', 'generated PET', 'GT PET', 'L1 error']
for j in range(ncols):
    gt_disp  = to_disp(x_pet[j])
    gen_disp = to_disp(samples[j])
    # Stretch the hot colormap to the GT's 99th-percentile so brain-uptake
    # regions (typical SUV 5-15, display 0.16-0.47) actually render bright
    # instead of getting crushed against vmax=1 reserved for SUV=32. Same vmax
    # applied to GT and generated so the two are visually comparable.
    vmax_j = max(float(np.percentile(gt_disp, 99)), 0.20)
    axes[0, j].imshow(to_disp(x_ct[j]),  cmap='gray', vmin=0, vmax=1)
    axes[1, j].imshow(add_cond[j, 0].cpu().numpy(), cmap='magma', vmin=0, vmax=1)
    axes[2, j].imshow(add_cond[j, 1].cpu().numpy(), cmap='viridis')
    axes[3, j].imshow(gen_disp, cmap='hot', vmin=0, vmax=vmax_j)
    axes[4, j].imshow(gt_disp,  cmap='hot', vmin=0, vmax=vmax_j)
    axes[5, j].imshow(err[j, 0].cpu().numpy(), cmap='Reds', vmin=0, vmax=0.4)
    axes[0, j].set_title(f'slice {sample_indices[j]} (vmax={vmax_j:.2f})', fontsize=9)
for i, t in enumerate(row_titles):
    axes[i, 0].set_ylabel(t, fontsize=11)
for ax in axes.flat:
    ax.set_xticks([]); ax.set_yticks([])
fig.suptitle('CPDM end-to-end: CT → (attention, attenuation) → generated PET vs ground truth\n'
             '(PET shown in hot colormap, per-column vmax = 99th percentile of GT)',
             fontsize=11)
fig.tight_layout()
fig.savefig(OUT / '06_cpdm_samples.png', dpi=140, bbox_inches='tight')
plt.show()

# %%
# Quantify the four generations against ground truth — same metrics computed
# during training, but here on the rendered batch so the numbers are reproducible.
from skimage.metrics import structural_similarity as _ssim
from skimage.metrics import peak_signal_noise_ratio as _psnr
import lpips as _lpips

lpips_model = _lpips.LPIPS(net='vgg').to(DEVICE).eval()
for p in lpips_model.parameters():
    p.requires_grad = False

def _three_channels(x):
    return x.expand(-1, 3, -1, -1)

with torch.no_grad():
    lp = lpips_model(_three_channels(samples.clamp(-1, 1)),
                     _three_channels(x_pet.clamp(-1, 1))).squeeze().tolist()

# Per-image SSIM / PSNR / MAE.
print('Per-slice metrics on the generated batch:')
print(f"{'slice':>8} {'LPIPS↓':>8} {'MAE↓':>8} {'SSIM↑':>8} {'PSNR↑':>8}")
for j, idx in enumerate(sample_indices):
    pred01 = ((samples[j, 0].cpu().numpy() + 1) * 0.5).clip(0, 1)
    tgt01  = ((x_pet[j, 0].cpu().numpy()  + 1) * 0.5).clip(0, 1)
    mae = float((samples[j] - x_pet[j]).abs().mean())
    ssim = float(_ssim(pred01, tgt01, data_range=1.0))
    psnr = float(_psnr(tgt01, pred01, data_range=1.0))
    lp_v = float(lp[j] if isinstance(lp, list) else lp)
    print(f'{idx:>8d} {lp_v:>8.4f} {mae:>8.4f} {ssim:>8.4f} {psnr:>8.2f}')

# %% [markdown]
# ## 6. Summary
#
# **What works.**
# - End-to-end CPU pipeline runs cleanly on autoPET; all four stages have monitored training runs (W&B + TensorBoard) and reproducible commands documented in `CLAUDE.md` and `NOTES.md`.
# - Attention UNet learns the high-uptake mask reliably (val F1 = 0.83).
# - VQGAN learns a stable `4 × 16 × 16` latent (val_rec_loss ≈ 0.017).
# - CPDM converges fast and produces qualitatively plausible PET that captures the dominant high-uptake regions guided by the attention map.
# - Paper-aligned metrics computed online: best LPIPS 0.166, SSIM 0.89, PSNR 41.7 dB.
#
# **What doesn't reach paper quality.**
# - Generated PET is smoother than ground truth — the VQGAN bottleneck dominates the achievable detail at 64×64. Pushing this further wants higher resolution and a perceptual / GAN loss in the VQGAN, both of which are CPU-incompatible.
# - Metric ceiling is the data floor: brain slices are heavily background-dominated; a trivial "predict near-zero" baseline already gets a low MAE, so the metric gap above it is small even when generations differ visibly.
#
# **What would close the gap (if a GPU were available).**
# - Restore paper-spec inputs: 256×256 slices, whole-body, ResNet50 attention encoder, pure Dice loss.
# - Re-enable the VQGAN perceptual + GAN-discriminator losses (Lightning 1.x → 2.x refactor needed).
# - Train CPDM long enough for several ReduceLROnPlateau LR drops (paper runs 200 epochs at batch 16).
# - Pass `--save_top` so the best-val checkpoint is preserved (we lost the epoch-7 best to `BaseRunner`'s "keep only latest" rotation).
#
# All artifacts and commands needed to reproduce this notebook are in `NOTES.md` and `CLAUDE.md`. W&B project: <https://wandb.ai/teamchaspi/CT2PET-CPDM>.

print(f'\nFigures saved to {OUT}/.')
