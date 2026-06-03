# Project Protocol — CPDM on autoPET

Reimplementation of the WACV 2025 paper *"CT to PET Translation: A Large-scale Dataset and Domain-Knowledge-Guided Diffusion Approach"* (Nguyen et al., [thanhhff/CPDM](https://github.com/thanhhff/CPDM)) on the **autoPET** challenge dataset instead of the paper's curated CT/PET dataset.

## Hardware
- CPU only — Intel i7-8650U, 8 cores @ 1.9 GHz
- 15 GB RAM
- No GPU available; all stages trained on CPU.

## Dataset

Source: **autoPET** challenge — paired whole-body CT + PET NIfTI volumes.

### Preprocessing — `preprocess_autopet.py`
- Loads `imagesTr/*_0000.nii.gz` (CT) and `*_0001.nii.gz` (PET), `labelsTr/*.nii.gz` (tumor mask).
- Extracts **brain region**: last 20 % of slices from each volume (paper uses whole-body — this is an intentional scope reduction for CPU compute budget).
- Resizes each slice to **64 × 64** via `scipy.ndimage.zoom` (paper uses 256 × 256 — intentional reduction).
- CT normalized to [−1, 1] by HU window `[-1000, 3071]` (paper uses pixel/2047; the attenuation pipeline was adapted, see Issue #1 below).
- PET normalized to [−1, 1] by SUV window `[0, 32]`.
- Output: `data/processed/{train,val,test}/{CT,PET,Labels}/<study>_s<idx>.npy` (float32, shape (64, 64)).

### Splits and counts
Patient split (from `splits_80_10_10.json`, autoPET official):
| split | patients | 64 × 64 brain slices |
|-------|---------:|--------------------:|
| train | 1 291    | 86 197              |
| val   | 161      | 10 464              |
| test  | 162      | 11 330              |

## Pipeline

The paper's original sequence is **Segmentation (attention) → VQGAN → CPDM**. We follow the same order.

### Stage 1 — Attention-map UNet
Goal: learn `CT → binary attention mask` (regions where PET will have high SUV uptake). Target derived from PET > 75th-percentile + morphological closing, exactly per paper §4.2.

**Architecture & loss** — `train_attention_map.py`
- Encoder: ResNet34 + scSE decoder (paper uses ResNet50; smaller backbone chosen for CPU).
- Loss: 0.7 · Dice + 0.3 · BCE (paper uses pure Dice).
- Dataset: `datasets/AttentionMapDataset.py` (CT input, attention map generated on the fly from PET).
- Optimizer: Adam, lr 1 × 10⁻³, CosineAnnealingLR.

**Run config** — CPU demo
- batch_size = 8, 3 epochs, 200 train batches/epoch cap (`--limit-train-batches 200`), 50 val batches/epoch.
- Random horizontal flip augmentation.

**Final metrics** (best checkpoint, epoch 2)
- val_dataset_iou = **0.7050**
- val_per_image_iou = 0.7079
- val_f1_score = **0.8266**
- train_dataset_iou (epoch end) = 0.669
- train_f1_score = 0.802
- Loss: 0.246 (train epoch), 0.220 (val)

**Artifacts**
- Best ckpt: `checkpoints/AttentionMap/attention_map_unet-epoch=02-val_dataset_iou=0.7050.ckpt`
- W&B run: [UNet-AttentionMap-CPU-demo](https://wandb.ai/teamchaspi/CT2PET-AttentionMap/runs/xft9m4va)

### Stage 2 — Bulk attention-map export
After UNet training, `export_attention_maps.py` runs inference over **every** CT slice in all three splits and saves a per-slice sigmoid probability map (float32, shape (64, 64), values in [0, 1]) to `data/processed/{split}/AttentionMaps/<slice>.npy`. This is exactly the format `CT2PETDiffusionModel.get_attention_map` reads (threshold 0.5 on load).

| split | maps written |
|-------|-------------:|
| train | 86 197       |
| val   | 10 464       |
| test  | 11 330       |

CPU inference at bs=32 took ~12 min total. Sample stats: range [0, 1], mean ≈ 0.32 across splits.

### Stage 3 — VQGAN encoder/decoder
Goal: learn `image ↔ 4×16×16 latent` so CPDM can run BB diffusion in latent space.

**Implementation note**. The original repo expects VQGAN to be trained externally with `CompVis/taming-transformers`. No `VQGANRunner` exists in this codebase. Instead we wrote `train_vqgan.py` — a CPU-friendly Lightning wrapper around `model/VQGAN/vqgan.py:VQModel`. Skips perceptual + GAN-discriminator losses (which assumed Lightning 1.x manual optimization); trains on L1 reconstruction + codebook quantization loss only. Saved Lightning checkpoint is keyed identically to `VQModel`, so `CT2PETDiffusionModel.__init__` loads it directly via the existing `VQModel.init_from_ckpt`.

**Architecture** — from `config/VQGAN-autoPET.yaml`
- ddconfig: ch=128, ch_mult=(1, 2, 4), num_res_blocks=2, in_channels=1, out_ch=1, z_channels=4, resolution=64, no attention.
- n_embed=8192, embed_dim=4.
- 55.3 M trainable params (Encoder 22.3 M, Decoder 33.0 M, VQ 32.8 K).
- Input 1 × 64 × 64 → latent 4 × 16 × 16 (×4 spatial downsampling).

**Training**
- Optimizer: Adam, lr 1 × 10⁻⁴, betas (0.5, 0.9).
- Loss: `L1(x, x_rec) + commitment_loss(z, z_q)`.
- Each batch concatenates CT and PET across the batch dim, so the encoder sees both modalities equally.
- Dataset: `datasets/CT2PETAlignedDataset.py` (registered as `ct2pet_aligned`; reads `{split}/CT` and `{split}/PET` directly without re-normalizing).

**Run config** — CPU long run
- batch_size = 4 (effective 8 after CT||PET concat).
- max_epochs = 5, limit_train_batches = 500, limit_val_batches = 100.
- Sanity validation 2 batches, validation every epoch.
- Random horizontal flip (`flip: True`).
- num_workers = 2, no pin_memory.

**Final metrics** (stopped manually mid-epoch 2 after laptop-resume gap; epoch 1 checkpoint used)
- val_loss = **0.0185**  (val_rec = 0.0168, val_codebook = 0.0017) — after 500 train batches.
- train rec loss converges 0.88 → 0.12 → 0.03 in the first 50 steps; the plateau at ~0.02 is the data floor (PET brain slices are mostly empty background near −1, so the trivial "predict −1" gets most of the way there).
- Per-step ~2.0 – 2.6 s on CPU at bs=4.
- Wall-clock: ~1 h CPU time for the epoch 1 checkpoint (laptop-sleep accounted for the 21 h calendar elapsed).

**Artifacts**
- Best ckpt used downstream: `checkpoints/VQGAN/vqgan-epoch=01-v2.ckpt` (664 MB).
- W&B run: [VQGAN-CPU-full](https://wandb.ai/teamchaspi/CT2PET-VQGAN) (cloud heartbeats dropped due to network — local training continued; metrics in `logs/VQGAN/run.log`).

### Stage 4 — CPDM (Brownian Bridge diffusion in latent space)

Main model from `model/BrownianBridge/CT2PETDiffusionModel.py`. Composes the frozen VQGAN, two `SpatialRescaler` conditioning paths (one each for attention and attenuation maps), and a UNet denoiser. BB forward / reverse follows paper §4.3.

**Architecture** — from `config/CPDM-autoPET.yaml`
- UNet denoiser: model_channels=128, channel_mult=(1, 2, 3, 4), num_res_blocks=2, attention at resolutions {32, 16, 8}, num_heads=8, head_channels=64, condition via SpatialRescaler.
- in_channels = **11** ( = 4 latent CT + 4 latent PET + 1 attention + 1 attenuation + 1 BB concatenation indicator ); out_channels = 4 (latent).
- BB: num_timesteps = 1000 (train), sample_step = 200, mt_type = linear, eta = 1.0, loss_type = L1, objective = `grad`, skip_sample = True.
- EMA: decay 0.995, update every 8 steps, starts at global_step 30 000 (won't trigger in this demo).
- Optimizer: Adam, lr 1 × 10⁻⁴, no weight decay. ReduceLROnPlateau (cooldown 3000, factor 0.5, min_lr 5 × 10⁻⁷, patience 3000).

**Domain-knowledge conditioning** (key paper novelty)
- **Attention map**: loaded from `data/processed/{train,val}/AttentionMaps/<name>.npy` (pre-exported in Stage 2); thresholded at 0.5 inside the model.
- **Attenuation map**: computed on the fly from CT via closed-form HU → 511 keV LAC transform (`attenuationCT_to_511`, kVp = 140). **Inverts the HU-window normalization** used by `preprocess_autopet.normalize_ct` so the formula sees true HU values (see Issue #1 below).

**Run config** — CPU demo (subject to wall-clock check)
- batch_size = 8, accumulate_grad_batches = 1.
- max_samples cap inside `CT2PETAlignedDataset`: train 2 000 / val 200 / test 200 → 1 epoch = 250 train iterations.
- max_epoch = 3, max_steps = 400 (n_steps caps the loop at the start of each epoch).
- sample_interval = 100 (effectively disables full BB sampling during training — too expensive on CPU; reverse process is 200 timesteps × forward cost).
- validation_interval = 1, save_interval = 1.

**Training history** — three CPU launches:

1. **First demo run** (no W&B, fixed `max_epoch=3 max_steps=400`, ~32 min). Stopped by `n_steps>400` after 2 full epochs. Final `val_epoch/loss = 0.0373`. Used only to validate the wiring; superseded by the long run below.

2. **Long run with W&B + paper metrics + early stopping** (`run r1yb3gtw`, started 2026-06-02 12:32, SIGINTd manually at epoch 8 / iter ~1760 because the laptop needed to be moved). Epochs 0–8 completed.

3. **Resume from `last_model.pth` (epoch 9)** (`run 4inhu3cy`, started 2026-06-03 16:37). Loaded the saved optimizer + LR scheduler state so ReduceLROnPlateau patience and any LR adjustments continued in place. Epochs 9–10 completed; manually stopped mid-epoch-11 because metrics had plateaued.

**Final metric trajectory** (val pass + 16-sample BB-sampled paper-metric eval per epoch; all logged to both TensorBoard and W&B via `WandbTBWriter`):

| epoch | val_epoch/loss | LPIPS↓ | MAE↓   | SSIM↑  | PSNR↑ (dB) |
|-------|----------------|--------|--------|--------|-------------|
| 0     | 0.02184        | 0.186  | 0.0055 | 0.880  | 41.35       |
| 1     | 0.02003        | 0.176  | **0.0053** | **0.890** | **41.70** |
| 2     | 0.02053        | 0.174  | 0.0055 | 0.876  | 41.41       |
| 3     | 0.02046        | 0.171  | 0.0054 | 0.879  | 41.55       |
| 4     | 0.02103        | 0.200  | 0.0059 | 0.880  | 40.94       |
| 5     | 0.02080        | 0.192  | 0.0057 | 0.874  | 41.16       |
| 6     | 0.01977        | **0.166** | 0.0056 | 0.875  | 41.44       |
| 7     | **0.01394**    | 0.176  | 0.0055 | 0.879  | 41.45       |
| 8     | 0.01712        | 0.200  | 0.0059 | 0.868  | 40.77       |
| 9     | —              | 0.174  | 0.0055 | 0.878  | 41.46       |

(Epoch 9 happened in the resumed run; its `val_epoch/loss` scalar landed in TB but I didn't extract it before stopping. Train loss values: epoch 9 → 0.012-0.018 range, epoch 10 → similar, epoch 11 → 0.012-0.014.)

**Convergence diagnosis.** Best `val_epoch/loss` = **0.01394** at epoch 7; subsequent epochs 8-10 stayed in `0.014-0.020`. Across 10 epochs, LPIPS oscillates within ±0.03, MAE ±0.0006, SSIM ±0.02, PSNR ±0.9 dB — noise-level fluctuation, not directional improvement. ReduceLROnPlateau patience (3000 iter ≈ 12 epochs) hadn't triggered yet; even if it had, the typical follow-on pattern of "small one-epoch bump then re-plateau" wouldn't push past the current floor. Stopped manually.

**Why the floor.** Most PET brain-region slices are background near −1 SUV; the model learns trivial near-uniform output in the first epoch (which gets most of the MAE for free) and the residual signal is concentrated in the small high-uptake regions. Pushing beyond this floor likely needs (a) a larger VQGAN, (b) input resolution above 64×64, or (c) a region-weighted loss — all outside the CPU budget.

**Final artifacts**
- Latest ckpt: `results/CT2PET_autoPET/CPDM/checkpoint/latest_model_10.pth` (1.0 GB) — model state + EMA shadow + epoch=10 + global_step=2750.
- Optimizer ckpt: `latest_optim_sche_10.pth` (0.8 GB) — Adam state + ReduceLROnPlateau internal counters.
- `last_model.pth` is identical to `latest_model_10.pth` (rolling alias).
- TensorBoard: `results/CT2PET_autoPET/CPDM/log/events.out.tfevents.*` — three event files (one per launch).
- W&B runs (project [CT2PET-CPDM](https://wandb.ai/teamchaspi/CT2PET-CPDM)):
  - `r1yb3gtw` — long run, epochs 0–8.
  - `4inhu3cy` — resumed run, epochs 9–10 + partial 11.
- BaseRunner's auto-rotation deleted earlier per-epoch checkpoints; only `latest_model_10.pth` survives. (Add `--save_top` next time to also keep the best-val checkpoint.)

**Best single-metric recommendation for downstream eval**: use `latest_model_10.pth` (current best available; `val_epoch/loss=0.01394` checkpoint was rotated out).

## Outstanding / known deviations

1. **CT normalization vs. attenuation formula**. Original code assumed CT in [−1, 1] was produced by `pixel/2047`. Our preprocessor uses an HU-window mapping. Patched `CT2PETDiffusionModel.get_attenuation_map` to invert our normalization (`HU = (ct + 1) · ½ · (3071 − (−1000)) − 1000`) so the closed-form 511 keV LAC sees correct HU values.
2. **Dataset wiring**. Original `CustomAlignedDataset` expected `{split}/A` (cond) and `{split}/B` (target) layouts and would corrupt pre-normalized inputs by re-applying its own normalization. Replaced with `CT2PETAlignedDataset` (registered as `ct2pet_aligned`) reading `{split}/CT` and `{split}/PET` directly with pass-through values, and supporting a `max_samples` cap per split.
3. **No VQGAN runner**. Original repo expected external VQGAN training (`CompVis/taming-transformers`). Replaced with `train_vqgan.py` — a Lightning wrapper saving VQModel-compatible checkpoints.
4. **Image size 64 × 64** vs. paper 256 × 256 — intentional, compute budget.
5. **Brain region only** (last 20 % of each volume) vs. paper whole-body — intentional, compute budget.
6. **Attention UNet encoder** ResNet34 + 0.7·Dice + 0.3·BCE vs. paper ResNet50 + pure Dice — minor implementation choices, output format unchanged.
7. **BaseRunner.train hardcodes `num_workers=24`**. Left as-is; on 8-core CPU it oversubscribes but the per-step cost is dominated by model compute, not I/O.

## Reproducibility — commands

```bash
# 1. Preprocess (one-time, ~ minutes)
python preprocess_autopet.py --raw-dir data/raw/autoPET --output-dir data/processed --target-size 64

# 2. Attention-map UNet (CPU demo: 3 epochs × 200 batches ≈ 6 min)
python train_attention_map.py --batch-size 8 --max-epochs 3 \
  --limit-train-batches 200 --limit-val-batches 50 \
  --wandb-name "UNet-AttentionMap-CPU-demo"

# 3. Export sigmoid attention maps for all splits (~12 min)
python export_attention_maps.py \
  --ckpt checkpoints/AttentionMap/attention_map_unet-epoch=02-val_dataset_iou=0.7050.ckpt \
  --data-root data/processed

# 4. VQGAN (CPU; train to convergence or budget)
python train_vqgan.py --batch-size 4 --max-epochs 5 \
  --limit-train-batches 500 --limit-val-batches 100 \
  --wandb-name "VQGAN-CPU-full"

# 5. CPDM (CPU demo)
python main.py -c config/CPDM-autoPET.yaml -t --gpu_ids -1
```

## Key files
- `preprocess_autopet.py` — NIfTI → 64 × 64 .npy.
- `datasets/CT2PETAlignedDataset.py` — registered `ct2pet_aligned`, used by VQGAN + CPDM.
- `datasets/AttentionMapDataset.py` — used by `train_attention_map.py`; generates target on the fly.
- `train_attention_map.py`, `export_attention_maps.py`, `train_vqgan.py` — standalone Lightning trainers / inference.
- `main.py` + `runners/DiffusionBasedModelRunners/CPDMRunner.py` — registry-driven CPDM training entry point.
- `model/BrownianBridge/CT2PETDiffusionModel.py` — the CPDM diffusion model; modifies attenuation pipeline per Issue #1.
- `config/{CPDM,VQGAN,AttentionMap-UNet}-autoPET.yaml` — autoPET-specific configs.
- `CLAUDE.md` — architecture overview for future AI-assisted sessions (gitignored).
