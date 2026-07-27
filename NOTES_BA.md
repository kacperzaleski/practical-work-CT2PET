# NOTES_BA.md — Bachelor-thesis track: flow-based CT→PET on full-body autoPET

This log records the **bachelor-thesis** extension of the practical work: adding
a **flow-based generative model (PMRF)** as a second architecture to compare
against CPDM on the autoPET dataset, and moving from a brain-only crop to a
**full-body** preprocessed dataset. The original CPDM work and its per-stage
protocol live in `NOTES.md`; this file is additive and does not change the CPDM
pipeline.

---

## 1. Source paper

**"Synthesizing Accurate and Realistic T1-weighted Contrast-Enhanced MR Images
using Posterior-Mean Rectified Flow"** — Brandstötter & Kobler, JKU Linz, 2025
(`Flow Based Generation.pdf`, arXiv:2508.12640). Code: `github.com/bbasti/pmrf-virtual-contrast`.

### What it does
Synthesises contrast-enhanced (CE) T1w brain MRI from non-contrast MRI. The
central idea is the **perception–distortion trade-off** (Blau & Michaeli): an
MSE-optimal model is accurate but blurry; a realism-optimal model is sharp but
drifts from true intensities. One network cannot minimise both. PMRF (Ohayon et
al., ICLR 2025) decouples them into two stages:

1. **Stage 1 — Posterior-Mean Predictor.** A residual U-Net `f_θ(x)` trained
   with plain voxel-wise **MSE** ⇒ converges to the conditional mean ⇒
   high-fidelity but over-smoothed output `ŷ_PM`. This *is* the distortion
   baseline.
2. **Stage 2 — Rectified-Flow Refiner.** Perturb the PM output,
   `z₀ = ŷ_PM + σ_s·ε` (σ_s = 0.1), then learn a **time-conditioned U-Net vector
   field** `v_φ(z,t)` by **flow-matching** along the straight optimal-transport
   path `z_t = (1−t)z₀ + t·y`. Target field is the constant `y − z₀`, so the
   loss is `‖v_φ(z_t,t) − (y − z₀)‖²` — no ODE integration during training.
   At inference, integrate the ODE with `K` explicit Euler steps.

### Baselines compared in the paper (the comparison we replicate)
- **Residual U-Net** = Stage 1 alone → best MSE/PSNR/SSIM, worst FID/KID.
- **RF baseline** = rectified flow conditioned directly on the *noised input*
  (`z₀ = x + σ_s·ε`), same architecture/loss as Stage 2 → best FID/KID, worst
  distortion.
- **PMRF** = the two-stage method → the sweet spot (FID ≈ ⅓ of the U-Net's,
  MSE only ~27 % worse).

### Paper hyper-parameters (Appendix A), for reference
3D 64³ patches; Z-score norm; AdamW lr 5e-4; cosine annealing / 200 epochs;
early stopping patience 20 on val loss; Stage-1 batch 128, Stage-2 batch 64;
K_train = 100; inference K = 200 with patch overlap 32 + Hann blending (best
operating point). Metrics: FID/KID on 2D axial slices; MSE/PSNR/SSIM.

---

## 2. Design decisions for the autoPET port (confirmed with supervisor/user)

| Decision | Choice | Rationale |
|---|---|---|
| Dimensionality | **2D slices** | Matches the existing CPDM/VQGAN/attention pipeline → directly comparable; CPU-feasible. Math is dimension-agnostic. |
| Resolution | **64×64** | Reverted from 128² after a CPU-budget review (§3.1). The body-bbox crop (below) recovers most of the lost effective resolution because anatomy now fills the frame instead of air, so 64² stays adequate for qualitative figures at ~¼ the compute. CPDM/VQGAN return to their original 64² configs. |
| Body region | **Head + torso (vertex→pelvis), legs dropped** | Compromise between brain-only (too narrow; loses organ FDG variety) and full-body (too diverse to converge in the CPU budget). Keeps brain + FDG-rich thorax/abdomen/pelvis; drops the long, low-information leg region. Implemented as the top 60% of each scan's foreground slices. Deliberately *harder* than brain-only — more anatomical variability for the models to learn. |
| Background removal | **Per-volume body-bbox crop, square-padded** | The CT FOV is mostly air; a full-frame resize wastes the 64² grid on background and inflates pixel metrics. Cropping to the body makes anatomy fill the frame and the metrics honest (§3.2). |
| Scope | **All three models**: PM U-Net + RF baseline + PMRF | Full perception–distortion comparison (3 models) + CPDM as a 4th architecture. |
| Conditioning maps | **Dropped** | Attention / 511 keV attenuation maps are CPDM-specific. PMRF feeds conditioning through `z₀`, not extra channels. |
| Normalization | **Kept CT/PET HU/SUV → [−1,1]** | Comparability with CPDM (paper used Z-score for MRI; not applicable here). |
| Latent space | **None — pixel space** | Paper operates in voxel/pixel space; no VQGAN needed. |

---

## 3. Data: head+torso preprocessing (64×64, background-cropped)

`training/preprocess_autopet.py` was extended with three `--body-region` modes (legacy
brain mode preserved):

- `--body-region head_torso` (**default**): keep the top `--keep-top-fraction`
  (default 0.6) of the *foreground* slices — i.e. from the vertex down through
  the pelvis, **dropping the legs**. The head is at the high-index end of axis 2
  (consistent with the legacy brain crop). Foreground = body-voxel fraction
  (raw CT HU > −500) ≥ `--min-foreground` (default 0.02), which removes air-only
  slices.
- `--body-region full`: every foreground slice (whole-body).
- `--body-region brain`: legacy (last `--brain-percent` of the stack), CPDM data.
- CT/PET normalization unchanged: CT clip [−1000, 3071] HU → [−1,1]; PET clip
  [0, 32] SUV → [−1,1]. Output `--target-size` (now **64**) `.npy`.
- **Body-bbox crop on by default** (background removal, §3.2); `--no-crop-body` disables it.
- `--no-labels` to skip the (unused) segmentation slices.

**Dataset folder:** `data/processed_fullbody/{train,val,test}/{CT,PET,Labels}`
(**64×64**, head+torso, background-cropped) — kept separate from the brain-only
`data/processed` so the legacy CPDM is untouched. (Folder name retained for
continuity; contents are cropped head+torso.)

Generated with:
```bash
python training/preprocess_autopet.py --output-dir data/processed_fullbody \
    --body-region head_torso --target-size 64
```
Full run log: `logs/preprocess_fullbody.log`.

### 3.2 Background removal — per-volume body bounding-box crop

**Motivation (the problem).** A whole-body autoPET CT/PET axial slice is mostly
*air*: the patient is a roughly centred blob and the surrounding field-of-view is
near-constant background (CT ≈ −1000 HU, PET ≈ 0 SUV). If we resize the raw slice
straight to 64×64, a large fraction of those 4096 pixels are spent encoding
background that carries no anatomical information. This hurts in two concrete
ways:
1. **Wasted resolution.** At 64² every pixel is precious. Letting air occupy, say,
   half the frame halves the *effective* resolution of the anatomy we actually
   care about — exactly the qualitative detail a thesis figure needs.
2. **Dishonest metrics.** Background is trivially predictable (a constant). A model
   that nails only the air still scores high PSNR/SSIM, because those pixel
   metrics average over the whole frame. The large background region inflates the
   scores and *compresses the differences* between models — undermining the
   CPDM-vs-PMRF perception–distortion comparison this thesis is built around.

**The method (how it works).** For each patient volume we compute one shared
crop window and apply it to every kept slice:
1. **Body mask.** Threshold the raw CT at HU > −500 (the same `is_foreground`
   rule used for slice selection): tissue is above it, air/bed below.
2. **Union over the stack.** OR the per-slice masks of all *kept* slices into a
   single 2-D mask, then take its bounding box `(rmin, rmax, cmin, cmax)`. Using
   the **union over the whole volume** (not a per-slice box) is the key choice:
   it gives **one crop window per patient**, so every slice is zoomed by the same
   factor and anatomical scale stays consistent down the stack. A per-slice box
   would zoom a narrow neck slice and a wide pelvis slice differently, destroying
   inter-slice scale and confusing the model.
3. **Margin.** Expand the box by a relative `--crop-margin` (default 5%) so skin
   and body contour are not clipped.
4. **Square pad, not stretch.** Crop to the (rectangular) box, then pad it to a
   centred **square** with the background fill (CT −1000 HU, PET 0, labels 0)
   *before* resizing. Padding preserves the true aspect ratio — a non-square
   resize would stretch the anatomy and distort organ shapes.
5. **Resize** the square crop to 64×64 and normalize as usual.

The crop is derived from CT and applied **identically to CT, PET and labels**
(they are co-registered), so the modalities stay pixel-aligned.

**Why this is the right trade-off here.** The crop is a *geometric* operation
only — it changes which pixels we keep and at what zoom, never the intensity
normalization — so models trained on the cropped set remain directly comparable
to each other, and the CT/PET value semantics are unchanged. It is computed once
per patient from data we already load, so it adds negligible preprocessing cost.
Net effect: anatomy fills the 64² frame (recovering most of the resolution we
gave up by dropping from 128² to 64²) and the evaluation metrics reflect
performance on *tissue* rather than on trivially-correct empty space.

**Implementation.** `training/preprocess_autopet.py`: `compute_body_bbox()` (union mask →
margined bbox per volume) and `crop_to_square()` (crop + centred square pad).
Controlled by `crop_body` (CLI `--no-crop-body` to disable) and `--crop-margin`.
A `None` bbox (no body found) safely falls back to the old full-frame resize.

**Measured effect (1-patient smoke, head_torso @64²).** Mean tissue fraction
(CT > −0.85 norm ≈ HU > −695) per slice rose from **8.2% without the crop to
18.4% with it — 2.2× more of the 64² frame is anatomy**. CT/PET stay (64,64),
in [−1,1], NaN-free and filename-aligned.

### 3.1 CPU compute budget (why 64² + crop)

Single i7-8650U; target ~1–2 nights per model to early-stop, ~3 days ceiling.
The 128² track did not fit comfortably: the residual U-Net measured **~3.6 s /
train step** (batch 8), VQGAN-128 was ~22–24 s/step and memory-tight enough to
freeze the machine, and CPDM-128 plus three heavier PMRF models added up to a
week-plus of nights. **64² is ~4× cheaper per step** (¼ the pixels) and, crucially,
lets CPDM reuse its original 64² VQGAN/diffusion configs. The body-bbox crop
(§3.2) buys back most of the visual detail lost by halving resolution, so 64²
remains good enough for the qualitative figures.

Exact 64² arch knobs (PMRF `base_ch`/`ch_mult`, `max_samples`) are retuned when
training is set up; the headline is the 4× step-time reduction and that VQGAN/CPDM
no longer need the 128² variants. PMRF Stage 2 still runs the frozen PM forward
each step. Levers for more time / sharper results: raise `max_samples`,
`base_ch`, or `--batch-size`.

The registered dataset `ct2pet_aligned` (`datasets/CT2PETAlignedDataset.py`) is
reused unchanged — just point `data_root` at `data/processed_fullbody`.

---

## 4. New code

```
model/PMRF/
  unet.py   — 2D residual U-Net. use_time=False → posterior-mean predictor;
              use_time=True → time-conditioned vector field (sinusoidal t embed
              + FiLM bias in every ResBlock). GroupNorm+SiLU+3×3, no attention.
  flow.py   — RectifiedFlow: perturb(z0), interpolate, target field (y−z0),
              flow_matching_loss, euler_sample (K explicit Euler steps).
pmrf_common.py        — shared config/dataloader/Trainer helpers (AdamW + cosine
                        + early-stop patience 20; W&B or --no-wandb TensorBoard).
training/train_pmrf_stage1.py  — Stage 1 posterior-mean U-Net (MSE). = distortion baseline.
training/train_flow.py         — rectified flow; --mode rf (z0=CT+noise) or
                        --mode pmrf (z0=PM(CT)+noise, needs --pm-ckpt).
sampling_eval/sample_pmrf.py        — test-set inference + metrics: MSE/PSNR/SSIM/LPIPS and
                        (optional --fid) FID/KID; saves CT|pred|GT panels.
config/PMRF-autoPET.yaml — hyper-parameters / paths / W&B block.
```

Dependency added: **`torch-fidelity`** (for torchmetrics FID/KID — same package
the paper used). Pinned in `pyproject.toml` / `uv.lock`.

Architecture/training defaults (arch, max_samples, batch) come from
`config/PMRF-autoPET.yaml`; `--base-ch`, `--batch-size`, `--max-samples`,
`--max-epochs` override on the CLI. Default arch is `base_ch=32`,
`ch_mult=(1,2,4,8)` (~3.4 M params) — compact enough for CPU at 128².

---

## 5. How to run (the comparison)

Defaults already match the CPU budget (batch 8, max-epochs 100, caps from YAML):

```bash
# 1. Stage-1 posterior mean (also the "Residual U-Net" distortion baseline)
python training/train_pmrf_stage1.py --data-root data/processed_fullbody

# 2a. RF baseline (perception extreme)
python training/train_flow.py --mode rf --data-root data/processed_fullbody

# 2b. PMRF Stage-2 refiner (needs the Stage-1 checkpoint)
python training/train_flow.py --mode pmrf --pm-ckpt checkpoints/PMRF_stage1/last.ckpt \
    --data-root data/processed_fullbody

# 3. Evaluate each model on the test set (K=200 Euler steps for rf/pmrf)
python sampling_eval/sample_pmrf.py --model pm   --pm-ckpt checkpoints/PMRF_stage1/last.ckpt --fid
python sampling_eval/sample_pmrf.py --model rf   --flow-ckpt checkpoints/PMRF_rf/last.ckpt   --num-steps 200 --fid
python sampling_eval/sample_pmrf.py --model pmrf --pm-ckpt checkpoints/PMRF_stage1/last.ckpt \
    --flow-ckpt checkpoints/PMRF_pmrf/last.ckpt --num-steps 200 --fid --save-images 8
```

For longer training (better HW / more time): raise `--max-samples`, `--base-ch`,
or `--batch-size`. Use `--no-wandb` for offline TensorBoard logging.

Checkpoints: `checkpoints/PMRF_stage1/`, `checkpoints/PMRF_rf/`,
`checkpoints/PMRF_pmrf/` (each with `last.ckpt` + top-2 by val loss).

---

## 6. Verification done

- Preprocessing smoke-tested (1 patient/split): full mode 416/282/193 slices @64²;
  head_torso mode 250 train slices @128² (top 60% of foreground), CT/PET in [−1,1].
- ResUNet + RectifiedFlow forward/backward shapes verified.
- All three trainers run end-to-end at the production 128² / `ch_mult=(1,2,4,8)`
  config on the smoke set (no-wandb, CPU); config-driven arch + ckpt reload OK.
- Timing measured: ~3.6 s/train step at base_ch=32, batch 8, 128² (budget basis).
- `sampling_eval/sample_pmrf.py` produces MSE/PSNR/SSIM/LPIPS + FID/KID and CT|pred|GT panels
  for all three models.
- (Metric values from the smoke runs are meaningless — real training pending the
  full dataset.)

## 7. (Dropped) 128² track

The 128² resolution track was **abandoned** for the CPU budget (§3.1) and all of its
artifacts deleted on 2026-06-29: `checkpoints/{VQGAN_128,AttentionMap_128}` (~3 GB), the
empty `logs/`/`results/` 128 dirs, and the three `config/*-autoPET-128.yaml` files. Every
model in the comparison runs at **64²** on the single shared `data/processed_fullbody` set
(see §8). Nothing in the active pipeline depends on the 128² configs.

## 8. Training results log (shared 64² head+torso, background-cropped set)

> **Reset note (resolution change to 64²).** The earlier 128² head+torso dataset
> (252,812 / 30,852 / 32,918 slices), its 128² attention UNet (`checkpoints/
> AttentionMap_128/`, best val IoU 0.653, wandb `7gubzu3e`), exported 128²
> attention maps, and the in-progress VQGAN-128 (wandb `xlieqkbx`) were all part
> of the abandoned 128² track. The 69 GB dataset was deleted to free disk before
> regenerating at 64². Those artifacts are superseded — the perceptual-loss
> finding below still holds for the 64² VQGAN. Everything from here is the 64²
> background-cropped track.

**Dataset (64², cropped) — DONE.** train 252,812 slices / 1291 patients;
val 30,852 / 161; test 32,918 / 162 (316,582 total). Same deterministic slice
selection as the 128² set (top 60% foreground), now body-bbox cropped @64². CT/PET
counts match exactly on disk; Labels also written. Integrity scan (400 random
slices/split): bad_shape=0, out_of_range=0, nan=0, missing_PET=0, CT range
[−1,1], mean tissue fraction **~25%** (vs ~8% uncropped). 240 GB free after the
69 GB 128² delete. Log: `logs/preprocess_fullbody.log`.

**fb64 config set (full CPDM retrain on the cropped set).** New configs writing to
separate `*_fb64` dirs so the brain-64 CPDM (`results/CT2PET_autoPET/CPDM`) is
preserved: `config/AttentionMap-UNet-autoPET-fb64.yaml`,
`config/VQGAN-autoPET-fb64.yaml` (perceptual 0.1), `config/CPDM-autoPET-fb64.yaml`
(dataset_name `CT2PET_autoPET_fullbody`, max_samples 5000/800/1000 matching PMRF).

**Attention-map UNet (64², cropped) — DONE.** ResNet34-UNet (24.5 M), 10 epochs ×
250 train / 60 val batches (batch 32), CPU, ~45 min. wandb run `hy062ezu`.
- Best **val dataset IoU = 0.9001** (epoch 9) — markedly higher than the
  uncropped 128² run's 0.653, because the body-bbox crop removes the trivial
  background imbalance and the high-uptake foreground fills the frame.
- Best ckpt `checkpoints/AttentionMap_fb64/attention_map_unet-epoch=09-val_dataset_iou=0.9001.ckpt`.
- Maps being exported for all splits to `data/processed_fullbody/{split}/AttentionMaps/`.

**PM / RF / PMRF (64²).** _pending_ — record MSE/PSNR/SSIM/LPIPS/FID/KID per model.

**VQGAN (64², cropped, +perceptual) — DONE.** `config/VQGAN-autoPET-fb64.yaml`,
55.3 M params (`ch_mult (1,2,4)` → 4×16×16 latent), batch 16, 100 train/20 val
batches/epoch, perceptual_weight 0.1. wandb run `tbefhcwc`. ~10 s/step (≈½ the
128² cost). Stopped at **epoch 17** (loss had plateaued e12→e16 then nudged down
again at e17). Best/final **val_loss 0.0333** (val_rec 0.0221, val_perceptual
0.0985, val_codebook 0.0013) — down from the e0 baseline 0.94. Checkpoint:
`checkpoints/VQGAN_fb64/last.ckpt` (= `vqgan-epoch=17.ckpt`), consumed by
`config/CPDM-autoPET-fb64.yaml` `model.VQGAN.params.ckpt_path`. (Trained across two
launches: an initial run interrupted by a laptop-suspend freeze at e1, then
resumed via `--resume-ckpt last.ckpt`; epoch numbering in the log restarts at the
resume.)

**CPDM (64²) — PAUSED at epoch 21/22 (resumable).** `config/CPDM-autoPET-fb64.yaml`,
`--gpu_ids -1` (CPU), results under `results/CT2PET_autoPET_fullbody/CPDM/`
(separate from the brain run). Latent 4×16×16, BB-UNet @16 in_channels 12,
n_stages 2, max_samples 5000/800/1000. wandb run `ivqq5kak`
(`CPDM-AutoPET-64-headtorso-cropped`), 625 iters/epoch, ~33 min/epoch (~3.1 s/it).
Per-epoch val paper metrics: **LPIPS steadily falling 0.291 (e0) → 0.167 (e7) →
0.135 (e20)** = the real signal; MAE/SSIM/PSNR ~flat at ~0.017 / 0.81 / 32 (global
metrics pinned near the background floor — see §8.1; honest read comes from
`sampling_eval/eval_masked.py` at the end). LPIPS still improving when paused, so more to gain.
- **Latest pause: epoch 27, step 16875** (resumed once from epoch 21, trained to
  ~28, stopped mid-epoch so no save race this time; `last_model.pth` +
  `last_optim_sche.pth` both load clean). LPIPS plateaued ~0.129 by epoch 26
  (val-loss best 0.01901; early-stop patience was creeping). Resumable any time.
- **Resume point (earlier note): epoch 21** — `last_model.pth` + `last_optim_sche.pth`
  (both load clean, step 13125, restores model+EMA+optim+scheduler). Resume:
  `main.py -c config/CPDM-autoPET-fb64.yaml -t --gpu_ids -1
  --resume_model .../last_model.pth --resume_optim .../last_optim_sche.pth`.
- `latest_model_22.pth` is a valid epoch-22 weights-only ckpt, but
  `latest_optim_sche_22.pth` was **truncated** by the stop landing mid-save (kill
  raced the epoch-22 checkpoint write) — do **not** resume from epoch 22. Lesson:
  stop *after* `last_model.pth` mtime updates, not when `latest_model_N` appears.
- VQGAN keeps the optional **VGG-LPIPS perceptual term** (`training/train_vqgan.py
  --perceptual-weight`, or `training.perceptual_weight` in the YAML) on top of
  L1 + codebook, to sharpen the decode for nicer CPDM thesis figures. The LPIPS
  VGG is wrapped so it stays **out of the saved checkpoint** (verified: 0 VGG
  keys) — CPDM's `VQModel.init_from_ckpt` loads it unchanged. Affects CPDM only
  (PMRF is pixel-space). Set weight 0 to disable.

### 8.1 Honest evaluation — global vs. high-uptake-masked metrics

**Motivation.** Within the body crop the test PET is still **88.9 % near-zero
SUV** (mean SUV ≈ 0.33). Aggregate MAE/PSNR/SSIM are therefore dominated by the
dark/low-uptake background, and a model's true job — the sparse ~11 % of
high-uptake voxels (tumours, brain, bladder, kidneys, heart) — is barely weighted.
Quantified with trivial predictors on the test set (no model):

| predictor | MAE | global-SSIM |
|---|---|---|
| predict −1 (empty) everywhere | 0.020 | 0.630 |
| predict per-image mean | 0.030 | — |
| predict CT directly | 0.116 | 0.319 |

i.e. predicting *nothing* already scores MAE 0.02 / SSIM 0.63. The headline
numbers look great for free; they are not evidence the model is good. (This is
exactly the perception–distortion point: distortion metrics are misleading on
sparse targets.)

**Remedy — masked metrics, one shared ruler.** `metrics_common.py` +
`sampling_eval/eval_masked.py` compute MAE/PSNR/SSIM **globally and inside GT-derived ROI
masks**: *active* (SUV > 0.5) and *lesion/high-uptake* (SUV > 2.5). `sampling_eval/eval_masked.py
--fid` additionally computes **global FID + KID** (torchmetrics InceptionV3,
normalize=True, KID subset auto = min(50, n//2) with error bar) over the matched
set — distributional perception metrics, set-level so not maskable; both models
get them from the same code path. Smoke-tested end-to-end (finite FID/KID with
KID std). Supervisor-requested; primary perception-realism number is KID (unbiased
at our small test-set size), FID secondary. Masks come
from the **ground truth**, never a model's own output, so no model can game them.
SSIM-in-ROI = full SSIM map averaged within the mask. LPIPS/FID stay global
(patch/feature-based, not maskable). Both models feed the **same** evaluator:
CPDM via `sample_to_eval` ([0,1] `.npy`), PMRF via `sampling_eval/sample_pmrf.py --save-npy`.
Applying the same ruler to both is *required* for fairness — a metric is not part
of the task, so this is not an asymmetric advantage (unlike loss-reweighting,
which would be, and which we deliberately did **not** add to keep both models on
their native objectives).

**Validation (120 test slices).** The lesion tier is ~35× more discriminating
than the global tier:

| predictor | global SSIM | active SSIM | lesion SSIM |
|---|---|---|---|
| predict-nothing (zeros) | 0.723 | 0.114 | **0.019** |
| blurred GT (oracle, smeared) | 0.945 | 0.822 | **0.666** |

Global SSIM compresses these two (0.72 vs 0.95); the lesion ROI separates them
cleanly (0.019 vs 0.67). Headline-vs-lesion gap is the finding to report per
model. Trivial-baseline floor row still to be added to the final results table.

**PMRF Stage 1 (posterior-mean U-Net) — DONE.** `training/train_pmrf_stage1.py`, ResUNet
**13.5 M** params (base 32, ch_mult (1,2,4,8); 64²→8² bottleneck — heavier than the
~3.4 M the old note guessed), batch 8, wandb `655elwv1` (`CT2PET-PMRF`).
**Early-stopped at epoch 25, best at epoch 5** (MSE posterior mean converges fast
then plateaus — expected for the distortion anchor). **Best val MSE ≈ 0.00123**
([−1,1] space; low because the dominant background is trivially predicted).
Checkpoints `checkpoints/PMRF_stage1/`: `pm-epoch=05.ckpt` (best, verified loads),
`pm-epoch=04.ckpt`, `last.ckpt` (epoch 25). Feeds Stage 2 via `--pm-ckpt`.
**PMRF Stage 2 (rectified-flow refiner) — DONE.** `training/train_flow.py --mode pmrf`,
flow source `z0 = PM(ct) + σ_s·ε` from the frozen `pm-epoch=05.ckpt`. 27.4 M total
(13.9 M trainable flow + 13.5 M frozen PM). wandb `CT2PET-PMRF`. **Early-stopped at
epoch 55, best at epoch 35, best val flow-loss 0.001299** (low because the
posterior-mean init makes the target velocity `y−z0` small). Checkpoints
`checkpoints/PMRF_pmrf/`: `pmrf-epoch=35.ckpt` (best, verified loads), `last.ckpt`.
Evaluate: `sampling_eval/sample_pmrf.py --model pmrf --pm-ckpt .../pm-epoch=05.ckpt --flow-ckpt
.../pmrf-epoch=35.ckpt --num-steps 200 --save-npy ...` → `sampling_eval/eval_masked.py`.

**concat-flow (rectified flow from noise, CT concatenated) — DONE.** `training/train_flow.py
--mode cond`, wandb `CT2PET-PMRF` run `7b3cma7k` (`PMRF-cond`), 5000/800 train/val, batch 8,
max-epochs 100. **Early-stopped at epoch 42, best at epoch 22** (patience 20). 13.9 M
trainable flow net, `in_channels=2` (z_t ⊕ CT). Checkpoint
`checkpoints/PMRF_cond/cond-epoch=22.ckpt`. Evaluated on the shared 256-slice ruler — see
**§10** for the final cross-model results table.

Models trained: posterior-mean U-Net (distortion anchor) · CPDM (Brownian-bridge latent
diffusion) · concat-diffusion (plain conditional ε-DDPM, same VQGAN latent) · concat-flow
(rectified flow from noise) · PMRF (posterior-mean + rectified flow). The 2×2 = 2 diffusion
× 2 flow, plus the PM anchor. **Final metrics + per-model explanation: §10.**

**concat-flow vs. PMRF — why they are genuinely different models, not a relabel.**
Both share the *identical* rectified-flow machinery: the same `RectifiedFlow` loss
`‖v_φ(·) − (y−z0)‖²`, the same straight-line interpolation `z_t=(1−t)z0+t·y`, the same
`ResUNet` (13.9 M, `use_time=True`), and the same K=200 Euler sampler. They differ in
**two coupled, decisive places** — the flow's *starting point* `z0` and *how the CT
conditioning enters* (`training/train_flow.py:_step`, line 79):

| | **concat-flow** (`--mode cond`) | **PMRF** (`--mode pmrf`) |
|---|---|---|
| Start `z0` | **pure Gaussian noise** `randn_like(pet)` | **posterior-mean estimate** `PM(ct)+σ_s·ε`, σ_s=0.1 |
| CT conditioning | **concatenated** as a 2nd input channel → net is `v_φ(z_t, CT, t)`, `in_channels=2` | **none in the net** — net is single-channel `v_φ(z_t, t)`; CT enters *only* baked into `z0` via the frozen Stage-1 PM net |
| Transport distance | long: noise → PET (must synthesise the whole image while reading CT each step) | short: `PM(ct)` is already an MSE-optimal (blurry) estimate near the target; the flow only adds realistic texture / undoes regression-to-the-mean blur |
| Extra trainable parts | none | frozen 13.5 M PM net (`pm-epoch=05.ckpt`) supplies `z0` |
| Typical flow-loss scale | ~O(1) early (target `y−z0` is large) | small (~1e-3; target `y−z0` is small because `z0≈y`) |

In one line: **concat-flow = "generate PET from noise, told what the CT is"; PMRF =
"start from the best blurry guess, then make it look real."** PMRF is the theoretically
motivated entry (posterior mean minimises distortion → RF moves it onto the data
manifold with provably minimal extra MSE; Ohayon et al.); concat-flow is the plain
conditional-flow contrast and the flow-side mirror of concat-diffusion. They sit at
different points on the perception–distortion plane by construction.

### 8.2 PMRF verification + the two concat-conditioned models

**PMRF verification — faithful.** Checked `model/PMRF/flow.py`, `training/train_pmrf_stage1.py`,
`training/train_flow.py` against the repo paper (the PDF is **Brandstötter & Kobler, "PMRF for
Realistic, Accurate Virtual Contrast MRI"** — the medical-translation application of PMRF,
likely `arXiv:2508.12640`; *not* the original Ohayon PMRF). Eqs (1) Stage-1 MSE, (2)
`z0=ŷ_PM+σ_s·ε`, (3) interpolation, (4) `v*=y−z0`, (6) flow-matching loss, (7) Euler
sampling all match line-for-line; RF vs PMRF differ only in the `_base` (ct vs PM(ct)),
exactly as the paper states. Only deviation: **2D slices vs the paper's 3D 64³ patches** —
intentional (CPU budget, matches the CPDM 2D pipeline). No correctness fixes. Citation
fixed in `thesis-extra.bib` (`brandstotter2025pmrf` added; cited in the thesis PMRF
section). Original PMRF (`ohayon2025pmrf`) kept as the method reference.

**The 2×2 is now 2 diffusion + 2 flow (RF baseline dropped per decision).**
- **Flow 1 — concat-flow (NEW, implemented).** `training/train_flow.py --mode cond`: velocity net
  `v(z_t, CT, t)` takes CT as a 2nd input channel, flows from **pure noise** (`z0=randn`,
  not `base+σ_s`). `RectifiedFlow` now takes an optional `cond=` (concatenated to z_t);
  `sampling_eval/sample_pmrf.py --model cond`. Smoke-tested train→sample→`eval_masked` OK.
- **Diffusion 2 — concat-diffusion (NEW, implemented).** `training/train_concat_diffusion.py` +
  `sampling_eval/sample_concat_diffusion.py` + `config/ConcatDiff-autoPET-fb64.yaml` +
  `model/ConcatDiff/ddpm.py`. Standard ε-prediction DDPM in the **same frozen VQGAN
  latent** as CPDM (encode/decode mirror `CT2PETDiffusionModel`), conditioned by
  concatenating the CT latent (`UNetModel` `condition_key=concat`, `in=8`). LDM-style
  `scale_factor` set on first batch (persisted as buffer). **Lighter denoiser** (~16 M, no
  attention) than CPDM by decision — latent identical so the latent-space axis is
  controlled; param count reported. DDIM (200 steps) sampling. Smoke-tested
  train→sample→`eval_masked` OK.
- Both new models dump `[0,1]` `.npy` → straight into `sampling_eval/eval_masked.py` (+ `--fid` FID/KID).

## 9. Final model comparison — the four models + PM anchor

This section consolidates what each trained model *is* and how they score on the **one
shared ruler** (`sampling_eval/eval_masked.py` over the same 256 test slices, GT-derived ROI masks,
global FID/KID), so the thesis Results chapter can be lifted straight from here. All five
read the identical 64² head+torso background-cropped set; all preds are `[0,1]` `.npy`.

### 9.1 What each model is (mechanics, one paragraph each)

**Posterior-mean U-Net — distortion anchor (not part of the 2×2).** `ResUNet f_θ(CT)→PET`
(`use_time=False`, 13.5 M) trained with plain voxel **MSE**. MSE is minimised by the
*conditional mean* E[PET|CT], so the network converges to an averaged, **deliberately blurry**
estimate: distortion-optimal, perception-poor. It is both the lower bound on MAE/PSNR and the
`z₀` source for PMRF. Ckpt `checkpoints/PMRF_stage1/pm-epoch=05.ckpt`.

**CPDM — Brownian-bridge latent diffusion (Diffusion 1).** The original practical-work model.
Diffusion runs in the frozen VQGAN latent (4×16×16); the forward process is a **Brownian
bridge** `x_t=(1−m_t)·x₀+m_t·y+σ_t·ε` that interpolates *PET-latent → CT-latent* (not
noise → data as in a plain DDPM), objective `grad`, L1 loss. Conditioning is rich: a learned
**attention map** (CT→high-uptake mask) and a CT-derived **511 keV attenuation map**, both
SpatialRescaler'd into the UNet (in_channels=12). 100 M trainable. The most informed model —
a benchmark, not a controlled ablation peer. **Not yet scored on the shared ruler — see §9.3.**

**Concat-diffusion — plain conditional ε-DDPM (Diffusion 2).** Textbook DDPM in the **same
frozen VQGAN latent as CPDM** (`q_sample: x_t=√ᾱ_t·x₀+√(1−ᾱ_t)·ε`, ε-MSE loss), but
conditioned only by **channel-concatenating the CT latent** to the noisy PET latent
(`UNetModel condition_key=concat`, in=8). Deterministic **DDIM** (200 steps). Denoiser is
**lighter than CPDM by design** (~16 M, no attention) — latent identical to CPDM so the
latent-space axis is controlled and only capacity + the BB-vs-plain-DDPM mechanism differ.
Ckpt `checkpoints/ConcatDiff/cd-epoch=25.ckpt`.

**Concat-flow — rectified flow from noise (Flow 1).** Same `RectifiedFlow` machinery as PMRF
(loss `‖v_φ(·)−(y−z₀)‖²`, straight path `z_t=(1−t)z₀+t·y`, K=200 Euler) but **z₀ = pure
Gaussian noise** and CT enters by **concatenation as a 2nd input channel** → `v_φ(z_t,CT,t)`,
`in_channels=2`. It must synthesise the whole PET from noise while reading CT each step (a
*long* transport). 13.9 M. Ckpt `checkpoints/PMRF_cond/cond-epoch=22.ckpt`.

**PMRF — posterior-mean + rectified flow (Flow 2).** Identical RF machinery to concat-flow but
**z₀ = PM(CT)+σ_s·ε** (σ_s=0.1, frozen Stage-1 PM net) and **no CT in the net** — conditioning
lives entirely in z₀. The PM output is already an MSE-optimal (blurry) estimate near the
target, so the flow only has to add realistic texture / undo regression-to-the-mean blur (a
*short* transport). The theoretically motivated entry: posterior mean minimises distortion →
RF moves it onto the data manifold with provably minimal extra MSE (Ohayon et al.). 13.9 M
trainable flow + 13.5 M frozen PM. Ckpts `pm-epoch=05.ckpt` + `pmrf-epoch=35.ckpt`.

> One-line contrast of the two flows: **concat-flow = "generate PET from noise, told what the
> CT is"; PMRF = "start from the best blurry guess, then make it look real."** (Full table: §8.1.)

### 9.2 Results on the shared 256-slice test ruler

Distortion (MAE/PSNR/SSIM) reported **globally and in GT-derived ROI tiers** — active SUV>0.5
(21.7 % of pixels), lesion SUV>2.5 (1.6 %). Perception = LPIPS (VGG, per-image) + global
FID/KID (InceptionV3, KID subset 50). PET is 89 % near-zero, so the **lesion tier and FID/KID
are the honest signal**; the global tier is near the trivial background floor (§8.1).

**Global:**

| Model | MAE↓ | PSNR↑ | SSIM↑ | LPIPS↓ | FID↓ | KID↓ |
|---|---|---|---|---|---|---|
| Posterior-mean (anchor) | **0.0047** | **38.11** | **0.931** | 0.142 | 89.2 | 0.0479 ± 0.0051 |
| CPDM (ep27, unconverged) | 0.0065 | 35.78 | 0.879 | 0.140 | 87.1 | 0.0406 ± 0.0047 |
| Concat-diffusion | 0.0065 | 35.85 | 0.877 | **0.128** | 65.7 | 0.0227 ± 0.0037 |
| Concat-flow | 0.0085 | 35.31 | 0.787 | 0.167 | 66.7 | 0.0264 ± 0.0061 |
| **PMRF** | 0.0059 | 36.10 | 0.892 | 0.129 | **53.3** | **0.0131 ± 0.0035** |

**Active (SUV>0.5) / Lesion (SUV>2.5) tiers — MAE / PSNR / SSIM:**

| Model | active MAE↓ | active PSNR↑ | active SSIM↑ | lesion MAE↓ | lesion PSNR↑ | lesion SSIM↑ |
|---|---|---|---|---|---|---|
| Posterior-mean | **0.0175** | **31.75** | **0.804** | **0.0816** | **22.33** | **0.543** |
| CPDM (ep27, unconverged) | 0.0244 | 29.27 | 0.685 | 0.1055 | 19.93 | 0.385 |
| Concat-diffusion | 0.0241 | 29.35 | 0.683 | 0.1044 | 19.94 | 0.398 |
| Concat-flow | 0.0235 | 29.54 | 0.681 | 0.0930 | 20.94 | 0.455 |
| **PMRF** | 0.0223 | 29.77 | **0.710** | 0.0890 | 21.18 | 0.463 |

(n=256 each, drawn from **only 2 test patients** — see §9.3 caveat; PM/PMRF dumped
earlier, concat-flow/concat-diffusion/CPDM sampled at K/DDIM/BB=200 steps.)

### 9.3 How to read it (the perception–distortion story)

- **The PM anchor behaves exactly as theory predicts:** best distortion on *every* tier
  (global MAE 0.0047, lesion SSIM 0.543) but the **worst perception by a wide margin**
  (FID 89.2, KID 0.048) — the over-smoothed conditional mean. It is the distortion ceiling,
  not a usable PET synthesiser.
- **PMRF is the standout generative model:** **lowest FID (53.3) and lowest KID (0.013)** of
  all — ~40 % better FID than the anchor and clearly ahead of both concat models — *while*
  keeping the best distortion among the generative models (highest active SSIM 0.710, second
  only to PM on global MAE/PSNR/SSIM). This is precisely the PMRF promise: realism of a flow
  with distortion close to the posterior mean. The posterior-mean init pays off.
- **Concat-flow vs PMRF (the controlled flow-vs-flow comparison)** — same RF machinery, so
  the gap isolates the effect of the starting point. Starting from **noise** (concat-flow)
  instead of **PM(CT)** (PMRF) costs heavily: SSIM 0.787 vs 0.892 global, FID 66.7 vs 53.3,
  LPIPS 0.167 vs 0.129. Conditioning baked into a good z₀ beats concatenating CT and flowing
  from noise. This is the cleanest single result in the thesis.
- **Concat-flow vs concat-diffusion (the within-noise-start pair):** both start from noise and
  concat the condition; concat-diffusion (latent DDPM) edges concat-flow on global SSIM
  (0.877 vs 0.787) and LPIPS (0.128 vs 0.167) at similar FID (~66) — the VQGAN latent + DDPM
  is a slightly stronger noise→image route than pixel-space RF-from-noise at this budget.
- **CPDM (the domain-prior diffusion) underperforms — but it is unconverged.** At epoch 27 its
  perception (FID 87.1, KID 0.041) is barely better than the blurry PM anchor and clearly worse
  than the prior-free concat-diffusion (FID 65.7) that shares its exact VQGAN latent and is
  *lighter* (~16 M vs 100 M). Distortion is essentially tied with concat-diffusion (global
  0.0065/0.879). Read honestly: the attention + attenuation priors did **not** buy a better
  trade-off *at this training budget* — but CPDM is the heaviest model (100 M trainable, vs
  ~14–16 M for the others) and was paused well before convergence (LPIPS still falling). Its
  row is a **lower bound**, not CPDM's ceiling; a converged run is the obvious follow-up.
- **Two structural caveats.** (1) The 2×2 is not perfectly orthogonal — both diffusion models
  live in the VQGAN latent, both flows in pixel space, so cross-row "flow vs diffusion" claims
  are confounded; clean comparisons are *within-row* (the two flows, the two diffusions). CPDM
  also carries extra information (attention + attenuation maps) the others don't. (2) **The
  shared test set is only 2 patients** (170 + 86 slices) — the consequence of `max_samples.test
  = 256` slicing the patient-ordered test split. 256 *slices* is a reasonable image-level n for
  SSIM/FID, but patient-level n is tiny, so absolute numbers should be read as *internally
  comparable* (same slices for every model) rather than population estimates. Widening to more
  patients (raise the test cap + re-sample all five) is the cheapest robustness win available.

### 9.4 CPDM evaluation — done, with two gotchas recorded

CPDM **is** now on the shared ruler (row in §9.2; FID 87.1 / KID 0.0406 / global 0.0065/0.879;
LPIPS 0.140). Two issues had to be cleared first, both worth remembering:
1. **Drive must be mounted.** `data/` is a symlink to the external drive
   (`/run/media/.../data`); when unmounted, `sample_to_eval` dies with `FileNotFoundError` on
   the (then-missing) test CT/attention maps.
2. **`get_attention_map` has no test path.** `CT2PETDiffusionModel.get_attention_map` only knows
   `attention_map_train_path` / `attention_map_val_path`, and `sample_to_eval` passes
   `stage='val_step'` — so for the *test* loader it reads the **val** AttentionMaps folder and
   can't find the test-slice names. Workaround used: temporarily point
   `attention_map_val_path` at `…/test/AttentionMaps` in `config/CPDM-autoPET-fb64.yaml` for the
   sampling run (commented in the YAML; **revert before any further training**). A cleaner fix
   would add a `attention_map_test_path` + a stage→path map in `get_attention_map`.

Reproduce (drive mounted; YAML val-path temporarily → test/AttentionMaps):
```bash
python main.py -c config/CPDM-autoPET-fb64.yaml --sample_to_eval --gpu_ids -1 \
    --resume_model results/CT2PET_autoPET_fullbody/CPDM/checkpoint/last_model.pth
python sampling_eval/eval_masked.py \
    --pred-dir results/CT2PET_autoPET_fullbody/CPDM/sample_to_eval/200 \
    --gt-dir   results/CT2PET_autoPET_fullbody/CPDM/sample_to_eval/ground_truth --fid
```

## 10. TODO / open items

- [x] Regenerate head+torso **64², background-cropped** dataset + verify.
- [x] Train 64² attention-map UNet on the cropped set; export train/val maps for CPDM.
- [x] Train PM / PMRF (64²) + concat-flow + concat-diffusion; record test metrics (§9.2).
- [x] Train CPDM (64², VQGAN → diffusion) — trained to epoch 27 (paused, not converged).
- [x] Evaluate PM / concat-diffusion / concat-flow / PMRF on the shared 256-slice ruler (§9.2).
- [ ] **CPDM not yet on the shared ruler** — export `test/AttentionMaps` + run `sample_to_eval`
      once the external drive is remounted (recipe in §9.4); then add its row to §9.2.
- [ ] **Drive was unmounted** (`data/` symlink dead) — remount before any CPDM eval or
      qualitative-figure work (`sampling_eval/make_qualitative.py` reads `data/processed_fullbody/test/CT`).
- [ ] Build the qualitative comparison figure (`sampling_eval/make_qualitative.py` → `thesis/figures/
      qualitative.png`) and wire metrics + figure into `thesis/Thesis.tex` (needs drive mounted).
- [x] CPDM evaluated on the shared ruler (FID 87.1 / KID 0.041 — §9.2). Drive remounted;
      attention-path workaround documented (§9.4).
- [ ] **CPDM blurry-cloud fix in progress (§11):** focal attention maps → retrain att-UNet →
      re-export → fresh CPDM run with EMA engaged. Update §9.2 / thesis once it converges.
- [ ] (Optional) sweep inference `K` to plot the perception–distortion curve
      (paper Fig. 2), and add the models to `sampling_eval/report.py`.

## 11. Why CPDM produced "blurry red clouds", and the fix

The epoch-27 CPDM (and the original practical-work CPDM) output a **diffuse central cloud**
with no focal uptake. Diagnosed it properly instead of just training longer:

**What it is NOT.** The VQGAN is not the bottleneck. Round-tripping test PET through the exact
CPDM encode→decode (`quant_conv`→`quantize`→`decode`) reconstructs **sharp** images with the
focal hot-spots intact (global MAE 0.0041, lesion MAE 0.069; the latent *can* hold sharp PET).
The learning rate is not collapsed either — still 1e-4 at the pause (ReduceLROnPlateau had not
fired). So the latent and the optimiser are fine.

**What it IS — the attention map was a body blob (a real bug).**
`AttentionMapDataset._generate_attention_map` thresholded PET at its **75th percentile computed
over the whole slice**. But the background-cropped frame is only ~25 % body (rest is −1 air),
so the 75th percentile lands at the body/air boundary ⇒ "PET > p75" ≈ **the entire body**
(measured: 25 % of the frame flagged). The CT→attention UNet learned that trivially (IoU 0.86–
0.90) — but a body-shaped mask carries **no localization**. CPDM's only spatial conditioning
(attention map + CT-derived attenuation) therefore told it "uptake is somewhere in the body",
and under the L1/grad bridge objective the safe prediction is the spatial average ⇒ a cloud.
Confirmed visually: the predicted attention map is a clean filled ellipse of the body, nothing
focal. (Secondary issue: EMA never engaged — `start_ema_step=30000` > the 16 875 steps reached
— so sampling used raw, un-averaged weights and got zero EMA benefit.)

**The fix.**
1. **Focal attention target.** `_generate_attention_map` now thresholds at an **absolute SUV**
   (`suv_threshold=2.0`, converted to the [−1,1] scale) instead of a whole-slice percentile.
   Active fraction drops 25 % → ~3.7 %; the target now localizes the genuinely high-uptake
   organs/lesions (heart, kidneys, bladder, brain, liver, focal disease) the diffusion must
   place. These sit at anatomically consistent, CT-visible locations, so the UNet can learn
   real localization (physiological uptake; tumours remain only partly predictable from CT —
   honest, not a regression). Old blob ckpts moved to `checkpoints/AttentionMap_fb64_blob_OLD/`.
2. **Retrain att-UNet** on the focal target (`training/train_attention_map.py … --wandb-name
   AttMap-fb64-focalSUV2.0`), then **re-export** all splits' maps into
   `data/processed_fullbody/{split}/AttentionMaps/` (overwriting the blob maps).
3. **EMA engages early** (focal config only): `start_ema_step 30000 → 5000`, early-stopping
   `patience 15 → 25`. The original `config/CPDM-autoPET-fb64.yaml` is left **pristine** (EMA
   30000, patience 15, blob maps, `model_name: CPDM`) so it remains a clean fallback.
4. **Fine-tune** (not fresh) from the epoch-27 blob weights on the focal conditioning —
   decided 2026-06-29: the denoiser already learned PET-latent generation; only the use of the
   conditioning needs to adapt, so a fine-tune converges in well under a day vs ~1.5–2.5 days
   fresh. `--resume_model` loads weights+EMA (step 16875 restored ⇒ EMA engages immediately at
   start 5000); **no `--resume_optim`** ⇒ fresh Adam at lr 1e-4 + fresh ReduceLROnPlateau so the
   model has room to adapt.

**Fallback is fully preserved (nothing old is overwritten).** Three separations:
- Focal maps → `data/processed_fullbody/{split}/AttentionMaps_focal/` (new `--out-dir-name`
  flag on `sampling_eval/export_attention_maps.py`); blob maps in `.../AttentionMaps/` untouched.
- Focal run → `model_name: CPDM_focal` ⇒ `results/.../CPDM_focal/`; the blob run
  `results/.../CPDM/` (FID 87.1, current §9.2 row) is untouched. Its checkpoint is *also*
  copied to `results/.../CPDM/checkpoint_blob_ep27_BACKUP/` for belt-and-braces.
- Separate config `config/CPDM-autoPET-fb64-focal.yaml` (the blob config is unchanged).
- Old blob att-UNet ckpts in `checkpoints/AttentionMap_fb64_blob_OLD/`.
To resume the **usual blob training** instead: `main.py -c config/CPDM-autoPET-fb64.yaml -t
--gpu_ids -1 --resume_model results/.../CPDM/checkpoint/last_model.pth --resume_optim
results/.../CPDM/checkpoint/last_optim_sche.pth`.

**Overnight orchestration (`run_focal_finetune.sh`, launched 2026-06-29 ~00:07).** One detached
`setsid` chain so it runs unattended: (1) wait for the att-UNet to finish, (2) pick the best
att ckpt by val IoU, (3) export focal maps for all splits, (4) back up the epoch-27 CPDM ckpt,
(5) launch the CPDM fine-tune (focal config, `--resume_model` blob weights, fresh optimiser).
Logs: `logs/att_retrain.log` (att-UNet, wandb `AttMap-fb64-focalSUV2.0`),
`logs/focal_chain.log` (chain), CPDM fine-tune wandb `CPDM-focal-finetune-SUV2.0`. Focal
att-UNet val IoU plateaued ~0.60 (vs the blob's meaningless 0.90). **Morning to-do:** sample
`CPDM_focal` on the 256-slice test set + `eval_masked`, compare to blob CPDM (FID 87.1), and
if better update §9.2 + the thesis; if the fine-tune did *not* help, the blob fallback above is
intact.

Diagnostic scripts: `scratchpad/vqgan_ceiling.py` (latent ceiling), `scratchpad/att_check.py`
(blob vs focal target visualisation).

## 12. Blob-CPDM continuation run — converged at this budget (no gain)

To settle the "CPDM is just under-trained" objection, the **blob** CPDM run was resumed from
its epoch-27 checkpoint and trained ~25 more epochs (to epoch 52). Two changes vs the original
run: resumed **with** optimiser+scheduler state (true continuation, not a fresh-optimiser
fine-tune), and `EMA.start_ema_step` lowered 30000→5000 in `config/CPDM-autoPET-fb64.yaml` so
EMA actually engaged (the blob run only reached step 16875, below the old 30000, so its EMA had
never averaged — it just copied live weights). Launched via `setsid nohup` with `--save_top`;
log `logs/cpdm_blob_continue.log`.

**Result: flat.** Per-epoch val metrics epoch 27→51 (2-batch paper metrics):
- LPIPS 0.133 → 0.135, bouncing 0.125–0.143 (best 0.1248 @ ep45 is a noise spike).
- SSIM ≈ 0.82, PSNR ≈ 32.0, MAE ≈ 0.0177 — all flat within run-to-run noise.
- BB L1 val loss drifted 0.0191 → 0.0184 (~4%), i.e. the training objective inched down but
  none of it reached the reported perceptual/distortion metrics.

**Conclusion:** CPDM has effectively **converged at this CPU budget**; more epochs are the wrong
lever (ceiling is capacity/resolution/data, not training time). The thesis wording was updated
accordingly — the epoch-27 blob row is now framed as *representative / converged*, not an
"unconverged lower bound" (Results intro, Table caption `\dag`, Discussion "domain prior"
paragraph, Hypotheses-revisited, Limitations, Conclusion). Fallbacks preserved:
`checkpoint_blob_ep27_BACKUP/` (epoch-27 blob), plus this run's `last_model.pth` (epoch 52) and
`top_model_epoch_51.pth` (best-val). Optional follow-up (deferred): run the full 256-slice
FID/KID on the epoch-52 weights to confirm it matches the reported FID 87.1.

## 13. Scientific-rigor pass (supervisor feedback) — infra built, CPDM re-training from scratch

Supervisor asked for: (a) CPDM retrained from **scratch** on the focal masks (not fine-tuned),
(b) hypotheses **statistically** tested (CIs + test statistics), (c) test set spanning **~10
patients** (no train/val leakage), (d) state PMRF's **novelty** for CT→PET, (e) drop Future
Work. "15 pages is fine — content/presentation matter."

**Key finding:** the test split already has **162 patients** (val 161, train 1291). The
"2-patient" test set was purely the `max_samples:{test:256}` cap taking the first-256 sorted
slices. Fixed by a shared manifest, not new data.

Built + validated this session:
- `sampling_eval/build_test_manifest.py` → `config/test_manifest_fb64.txt`: **300 slices, 12
  patients (7 FDG / 5 PSMA), 94.7% lesion-bearing**. Verified 0 overlap with train/val.
- `datasets/CT2PETAlignedDataset.py`: honors `dataset_config.test_names_file` (test stage only;
  supersedes the max_samples cap; train/val untouched). All 5 samplers + CPDM `sample_to_eval`
  funnel through this one class, so the manifest is the single shared lever.
- `stats_common.py`: patient-**clustered** bootstrap CIs, paired cluster-bootstrap + Wilcoxon,
  Holm correction. (Cluster bootstrap is wider than naive per-slice — correctly captures
  within-patient correlation.)
- `sampling_eval/eval_masked.py --ci`: patient-clustered 95% CIs per tier/metric.
- `sampling_eval/compare_models.py`: pre-registered H2/H3/H4/H5 paired contrasts, Holm-corrected.
  Distributional axis uses **KID** (unbiased poly-MMD; FID's 2048-d cov is rank-deficient at
  n≈300). KID CI/contrasts use **leave-one-patient-out jackknife** (a cluster bootstrap of a
  U-statistic self-matches duplicated patients and biases KID upward — verified + fixed).
- `--names-file` added to `sample_pmrf.py` and `sample_concat_diffusion.py`.
- `config/CPDM-autoPET-fb64-focal-scratch.yaml` (model_name CPDM_focal_scratch, focal maps,
  manifest). **Launched from scratch** (fresh init, loss 0.58→…, ~35 min/epoch), log
  `logs/cpdm_focal_scratch.log`, results `results/CT2PET_autoPET_fullbody/CPDM_focal_scratch/`.
  Blob (CPDM/) + epoch-27 fine-tune (CPDM_focal/) preserved.

Thesis (number-independent edits done; builds clean, 21 pp): deleted Future Work + fixed the
dangling ref; reframed hypotheses to tested **H2/H3/H4 + H5 ablation**, each naming its test;
added a **Statistical methodology** subsection; added the **PMRF-novelty** contribution;
updated the test-set description/limitation to 12 patients / 300 slices.

### §14. Final manifest recompute + the from-scratch-vs-fine-tune CPDM decision
The from-scratch focal CPDM plateaued at epoch 15 (per-epoch val LPIPS 0.28→0.12). Full re-sample
of all 7 model dirs on the 296–300-slice / 12-patient manifest, then `eval_masked --ci --fid`
per model + `compare_models --kid` (`scripts/recompute_manifest_eval.sh`, log
`logs/eval_manifest.log`; contrasts `logs/contrasts_ft.log`). **Per-model (manifest):**
PMRF FID **42.0** / KID **0.008** (best perception, best-generator global SSIM 0.890);
concat-flow 46.9/0.011; concat-diff 50.5/0.015; **CPDM-focal (fine-tuned) 62.3/0.025**;
CPDM-blob 63.1/0.028; CPDM-scratch 68.2/0.030; PM anchor 77.0/0.036 (worst perception, best
distortion 0.906/0.698/0.533).

**Decision (user gate: swap to scratch only if "not worse" than fine-tune):** scratch wins the
SSIM/MAE tiers (global SSIM 0.880 vs 0.876) but is **worse on both perception metrics** (FID
68 vs 62, KID 0.030 vs 0.025) and visually flatter → **does NOT pass**. Kept the **fine-tuned**
CPDM as the headline (best CPDM perception; cleanly wins H5 vs blob on FID+KID+all SSIM). The
from-scratch run is used only as the **convergence control** (App. cpdm-compare, new figure
`cpdm_scratch_vs_finetune.png`), letting us honestly drop the "not converged / epoch-27 / lower
bound" language while keeping the fine-tuned numbers. `compare_models` was re-run with
`cpdm=CPDM_focal` (fine-tuned) so the hypothesis-test table is consistent with the headline.

Thesis numeric edits: Results table + caption, abstract, discussion (all 4 paragraphs),
hypotheses-revisited table, limitations, conclusion all updated to manifest numbers; added the
hypothesis-test table (`tab:hyptests`) from `logs/contrasts_ft.log`; regenerated
`qualitative.png` (manifest + fine-tuned CPDM) and `training_losses.png` (CPDM panel = 15-epoch
scratch run); fixed 3 table cells mangled to `,` by the earlier em-dash sweep (`---`→`n/a`).

### §15. Stats made accessible + clustered-vs-unclustered (supervisor/author readability pass)
Author found the stats section too advanced. Rewrote `\subsection{Statistical methodology}` in
plain language anchored on the $t$-test / standard-error the reader knows: error bars → the
independence catch (300 slices are really only 12 patients) → the fix (count patients: per-patient
mean + $t$-interval, cross-checked by the patient bootstrap) → paired comparisons + Wilcoxon →
Holm as "guarding against lucky hits" → FID/KID as a distribution distance with leave-one-patient-out.
Dropped the jargon (U-statistic, MMD, rank-deficient covariance, block bootstrap). Added
`stats_common.naive_ci` (per-slice mean±z·SE) and `patient_ci` (per-patient mean±t·SE);
`sampling_eval/clustered_vs_unclustered.py` prints per-slice vs patient CIs. Finding: **clustered
CIs are ~2–3× wider than naive per-slice** and the bootstrap ≈ per-patient-$t$ (agree to rounding),
now shown in new `tab:ci-compare` (global + lesion SSIM, per-slice vs per-patient). Also fixed a
`manifold`→`manifest` typo. Re: the reader's "section 7 says 256 slices" — the source already reads
300/12 everywhere (only `$256^2$` = GPU resolution remains); they were viewing a pre-edit PDF.
Build clean, 22 pp.

**PENDING (needs the converged from-scratch CPDM, ~tomorrow):**
1. Re-sample all 5 models on the manifest (`--names-file config/test_manifest_fb64.txt`;
   CPDM: point attention_map_val_path at test/AttentionMaps_focal during sampling — §9.4 gotcha).
2. `eval_masked --ci` per model + `compare_models --kid` for the contrasts.
3. Fill the Results table (CIs) + new hypothesis-test table; headline CPDM = focal-scratch,
   converged blob = H5 ablation, drop fine-tune row; update the Hypotheses-revisited verdicts;
   reconcile H1 mentions in Discussion. (One `\todo` marks this.)
4. Regenerate `thesis/figures/training_losses.png` so the CPDM panel is complete
   (`python sampling_eval/make_training_curves.py`; add `--fetch` only to refresh wandb cache).

### Training-dynamics figures (restructured this session)
Replaced the ad-hoc `training_cpdm.png` + `training_stages.png` with a uniform pair:
**`training_losses.png`** (2x2, **train+val loss for all four generators**, log-y) and
`training_support.png` (VQGAN / attention-IoU / PMRF Stage-1). The three Lightning generators
logged train loss only to **wandb** (local TB had 1 point; text logs had val only), so
`make_training_curves.py --fetch` pulls full train+val from `teamchaspi/CT2PET-{PMRF,ConcatDiff}`
→ caches `logs/curves/*.json` (offline-reproducible; thesis build never touches the network).
CPDM reads local TB (`loss/train` per step → per-epoch mean + `val_epoch/loss`). Run ids:
PMRF=ebto44cb, concat-flow=7b3cma7k, concat-diff=ox8n9bxn, PM-stage1=655elwv1.

---

## 16. Learning companion — the concepts, explained from scratch

Written *for the author*, to consolidate the take-aways. It focuses on the parts flagged as
unclear (metrics, statistics, the SSIM tiers, the model conditioning) and on machinery that was
introduced without being asked for. Nothing here is new work; it is the "why" behind what is
already in the repo and thesis.

### 16.0 Where you are (honest read from your questions)
You already have the hard parts of the *engineering* mindset: you follow implementation detail,
you catch inconsistencies (256-vs-300 slices, 0.021-vs-0.024 deltas), and you understand the
high-level ML ideas (EMA, diffusion, the trade-off). The genuine gaps — and the real
intellectual content to *own* by the defense — are three:
1. **Inferential statistics.** What a confidence interval actually is, why a plain $t$-test can
   mislead here, and what clustering / bootstrap / correction each buy you.
2. **The evaluation metrics.** What each number measures, and how the three masked SSIM tiers are
   actually computed (it is one SSIM, three averaging regions).
3. **The conditioning mechanics.** Precisely how the CT enters each of the four models. This is
   what makes them *different models* and not relabelings, and it is the spine of the whole 2×2.
Everything else (preprocessing, training loops, checkpointing) is engineering scaffolding.

### 16.1 The spine: the perception–distortion trade-off
One idea underlies the entire thesis. Predicting PET from CT is **ill-posed**: many different
PET images are compatible with the same CT (CT cannot see where a tumour is metabolically hot).
So there is no single right answer, only a *distribution* of plausible PETs.

- If you train a network to minimise pixel error (MSE), the mathematically optimal output is the
  **average** of all plausible PETs. Averages are smooth, so the output is **accurate but
  blurry**. This is the *distortion* corner (our posterior-mean U-Net).
- If you instead force the output to *look like* a real PET (sharp, textured), you gain realism
  but drift from the true intensities. This is the *perception* corner.

Blau & Michaeli proved you **cannot** have both at once: pushing perception up necessarily pushes
distortion up. Every model is a point on this curve. The whole thesis is "where does each
architecture land, and why." PMRF's trick is to start at the distortion corner (the blurry mean)
and take the *shortest* step toward realism, landing at the best knee.

### 16.2 The metrics — what each number actually measures

**Pixel-accuracy metrics (distortion axis).**
- **MAE** = mean of `|pred − gt|` over pixels. Raw intensity error. Lower is better.
- **PSNR** = a log-scaled version of the mean *squared* error, in decibels. Higher is better.
  Same information as MSE, just on a friendlier scale.
- **SSIM** (structural similarity, 0–1) is different in kind. Instead of comparing pixels one by
  one, it slides a small Gaussian window over the image and compares **local brightness, contrast
  and structure**. It answers "do the *patterns* match," not "are the numbers equal." Higher is
  better.

**The three SSIM tiers — the part you asked about.** They are **the same SSIM computation**. The
only difference is *which pixels get averaged into the final score*, chosen by a mask derived
from the **ground-truth** PET (never from a model's output, so no model can cheat). From
`metrics_common.py`:
```python
def _ssim_masked(pred, gt, mask, data_range):
    score, smap = _ssim(gt, pred, data_range=data_range, full=True)  # full SSIM *map*
    if mask is None:
        return float(score)          # GLOBAL: average over all pixels
    return float(smap[mask].mean())  # TIER:   average only inside the mask
```
The SSIM *map* is always computed over the whole image (so each window has proper context); only
the **averaging region** changes:
- **global** (`mask=None`): average over all 4096 pixels. But PET is ~89 % near-zero background,
  which is trivially predictable, so every model scores high (~0.8–0.9) and they all look similar.
  The background *drowns out* the signal.
- **active** (`mask = GT SUV > 0.5`, ~16 % of pixels): average only where there is real uptake.
- **lesion** (`mask = GT SUV > 2.5`, ~2 % of pixels): average only over the hottest voxels
  (tumours, brain, bladder, kidneys) — the diagnostically important part, and the **most
  discriminating** tier (models that all score ~0.87 globally spread out to 0.40–0.53 here).
The mask is a SUV threshold on the GT (`uptake_mask: suv(gt01) > thresh`, where `suv = value*32`).
So "global vs active vs lesion" = "whole image vs signal region vs hot-spots," same ruler,
progressively stricter about *where* it looks. **The global-vs-lesion gap is the honest story**:
a model can look great globally and fail exactly where it matters.

**Perception metrics (perception axis).** These cannot be per-pixel or masked, because "realism"
is about the *look* of the whole set of images, not any single pixel.
- **LPIPS**: feeds `pred` and `gt` through a pretrained VGG and compares deep features. A
  per-image perceptual distance. Lower is better.
- **FID / KID**: compare the **distribution** of all generated images to the distribution of all
  real images, in the feature space of a pretrained InceptionV3. Think "does the *cloud* of
  generated images sit on top of the *cloud* of real ones." FID assumes those feature clouds are
  Gaussian and measures the distance between the two Gaussians; **KID** uses a kernel and makes no
  Gaussian assumption. Lower = more realistic. **We lead with KID** because FID is biased upward
  when the test set is small (it needs a 2048×2048 covariance, wobbly at n≈300); KID stays
  reliable there.

### 16.3 The statistics — from a t-test up

**What a confidence interval is.** Our metric is a mean over slices. If we reran the whole study
on fresh patients we would get a slightly different mean each time. A 95 % CI is the range that
would contain the *true* value 95 % of the time. Its **width is the uncertainty**. The textbook
recipe (and what sits behind a $t$-test) is `mean ± ~2 × standard error`, with
`standard error = std / √n`. The `√n` is why more samples give tighter intervals.

**Why the obvious version lies here.** That recipe assumes `n` **independent** samples. Our 300
slices are **not** independent: they come from only **12 patients** (~25 slices each), and two
slices from the same patient are near-copies (same anatomy, scanner, tracer). So the *effective*
sample size is closer to 12 than 300. Plugging `n = 300` makes `√n` too big, the standard error
too small, the interval too narrow, and a plain $t$-test too eager to call things significant.
This is the single most important statistical idea in the thesis.

**The fix — count patients, not slices.** Two equivalent ways, both reported:
1. **Simple:** average each patient's slices into one number (→ 12 numbers), then do the ordinary
   `mean ± t·SE` on those 12. Honest `n = 12`. (`stats_common.patient_ci`.)
2. **Bootstrap (no bell-curve assumption):** draw 12 patients at random *with replacement* 2000
   times, recompute the metric each time, and take the middle 95 % of the results.
   (`stats_common.cluster_bootstrap_ci`.)
They agree to rounding, and both come out **2–3× wider** than the naive per-slice interval
(`stats_common.naive_ci`) — that widening *is* the correction. `tab:ci-compare` in the thesis
shows this side by side.

**Comparing two models (paired, and the 0.021-vs-0.024 point).** Every model is scored on the
*same* slices, so we compare **per-slice differences** rather than two separate averages. Pairing
cancels "this slice is just hard for everyone" and is far more powerful. It is also why the
contrast Δ is *the mean of the per-slice differences on the shared slices*, which need not equal
`(model-A mean) − (model-B mean)` from the rounded results table — the two models' per-model means
are taken over slightly different slice sets and then rounded, whereas the paired Δ uses only the
shared slices at full precision.

**Guarding against lucky hits (Holm correction).** We run ~11 comparisons. Each single test has a
5 % false-positive chance, so across 11 the chance that *at least one* fires by luck is ~43 %, not
5 %. **Holm–Bonferroni** sorts the p-values and makes the bar progressively stricter for the most
extreme ones, so the *whole family* keeps a 5 % error rate. Only contrasts that clear the raised
bar get a `*`. (`stats_common.holm_correction`.)

**Wilcoxon signed-rank** is a paired test like the $t$-test, but it ranks the differences instead
of using their raw values, so it does **not** assume they are bell-shaped (our metric differences
are skewed and bounded in [0,1]). We report it as a cross-check; it agrees with the bootstrap.

**Jackknife for KID.** KID is one number for a whole *set* of images, so there is no per-slice
value to bootstrap. Instead we recompute KID **12 times, each time leaving one patient out**; the
spread of those 12 gives the error bar. (A bootstrap would draw the same patient twice, and KID's
formula misbehaves when an image is compared with a duplicate of itself — so jackknife, not
bootstrap, for KID.)

### 16.4 The four models — how the CT conditions each (the real differences)

**The unifying frame.** Every generative model here answers the same template:
> *start from some `z0`, then transform it into a PET, using the CT somehow.*

There are only **two knobs**: (1) **what `z0` is** (the starting point), and (2) **how the CT
enters** the transform. "Diffusion vs flow" is only *how the transform is run* (many small
denoising steps that undo added noise, vs following a learned straight-line velocity field). The
conditioning story is the same two knobs for all of them:

| model | family | `z0` (start) | how the CT enters | transform (inference) |
|---|---|---|---|---|
| **PM U-Net** | — (anchor) | — | CT is the *only* input | one forward pass, MSE-trained ⇒ blurry mean |
| **CPDM** | diffusion (latent) | PET-latent, bridged toward the **CT latent** | **auxiliary domain maps**: CT latent as bridge endpoint + attention map + 511 keV attenuation map, all concatenated | Brownian-bridge reverse, 200 steps |
| **concat-diffusion** | diffusion (latent) | pure noise | **CT latent concatenated** as extra channels every step | DDPM/DDIM reverse, 200 steps |
| **concat-flow** | flow (pixel) | pure noise | **CT concatenated** as a 2nd input channel every step | Euler-integrate velocity, 200 steps |
| **PMRF** | flow (pixel) | **`PM(CT) + noise`** | **only through `z0`** — the net never sees CT | Euler-integrate velocity, 200 steps |

**Three conditioning styles** (this is the conceptual pay-off):
1. **Concatenation** (concat-flow, concat-diffusion): glue the CT next to the thing being denoised
   as extra channels; the network reads it at every step. Generic, "prior-free," the obvious
   baseline.
2. **Baked into the start** (PMRF): never hand the network the CT at all. Instead start the whole
   process from a good CT-derived guess `PM(CT)` and let the network only *add realism*. Elegant,
   and the reason PMRF's transport is *short* (it begins near the answer).
3. **Auxiliary domain priors** (CPDM): concatenate not the raw CT but hand-designed, domain-specific
   maps — *where* uptake is likely (attention) and *how* photons attenuate (attenuation) — plus a
   special forward process (Brownian bridge) that starts from the CT latent instead of noise.

**The cleanest illustration — the two flows differ in two lines.** `concat-flow` and `PMRF` share
the *identical* rectified-flow loss and network; only the two knobs change (`training/train_flow.py`):
```python
def _step(self, batch):
    ct, pet = self._unpack(batch)
    if self.mode == 'cond':                       # concat-flow
        z0 = torch.randn_like(pet)                # knob 1: start = pure noise
        return self.flow.flow_matching_loss(self.net, z0, pet, cond=ct)  # knob 2: CT concatenated
    z0 = self.flow.perturb(self._base(ct))        # PMRF: knob 1: start = PM(ct) + noise
    return self.flow.flow_matching_loss(self.net, z0, pet)               # knob 2: no cond — CT only via z0
```
and the concatenation itself is literally one line (`model/PMRF/flow.py`):
```python
def _net_input(z_t, cond):
    return z_t if cond is None else torch.cat([z_t, cond], dim=1)  # glue CT on as extra channels
```
So "concat-flow vs PMRF" is not a relabeling: same machinery, but *start from noise and read the
CT each step* versus *start from the blurry CT-derived guess and add realism*. That single change
is the cleanest controlled result in the thesis (it isolates the effect of the starting point).

**Diffusion side, same story.** `concat-diffusion` adds noise to the PET latent and trains the net
to predict that noise, with the CT latent handed in as `context` (concatenated inside the UNet):
```python
def p_losses(self, model, x0, cond, t, noise):
    x_t = self.q_sample(x0, t, noise)             # forward: x_t = √ᾱ·x0 + √(1−ᾱ)·noise
    eps_pred = model(x_t, timesteps=t, context=cond)   # cond = CT latent, concatenated
    return torch.mean((noise - eps_pred) ** 2)    # predict the noise you added
```
CPDM is the same latent, but swaps plain noise for the **Brownian bridge** (the forward process
walks from the PET latent toward the CT latent rather than toward noise) and swaps the concatenated
raw CT for the attention + attenuation maps. So `CPDM vs concat-diffusion` isolates "domain prior +
bridge" against "plain concatenation" *in the same latent space* — which is exactly why H2 for the
diffusion family is a fair test, and why it is striking that the plain concat model wins on
perception.

**One-line memory hooks.**
- PM U-Net = "the blurry average" (distortion corner).
- concat-diffusion / concat-flow = "generate from noise, told what the CT is."
- PMRF = "start from the best blurry guess, then make it look real."
- CPDM = "generate from the CT latent, with hand-made hints about where uptake and attenuation are."

## 17. Thesis presentation pass (structure, front matter, captions, inline figures)

A non-scientific restructuring pass on `thesis/Thesis.tex` (no numbers changed):
- **Front matter**: title / abstract / ToC each on their own page (`\clearpage`); added `\listoftables`,
  `\listoffigures`, a manual **List of Abbreviations** table, and an **Acknowledgements** section
  (thanks Ass.-Prof. Kobler), all before the Introduction and in the ToC via `\addcontentsline`.
- **Sections**: Related Work merged into a single **"Background and Related Work"** section (medical
  translation prior work folded into a `\subsection`; the duplicate perception--distortion frontier
  paragraph deduped into `sec:pd-plane`). Introduction now ends with a thesis-roadmap paragraph.
  New numbering: 1 Intro, 2 Background & Related Work, 3 Dataset, 4 Methods, 5 Eval, 6 Results,
  7 Discussion, 8 Limitations, 9 Conclusion.
- **Declaration of AI Usage** added just before the appendix (ideas-into-text, knowledge help,
  formatting/LaTeX help; design/analysis/conclusions are the author's own).
- **Abbreviations** expanded at first use throughout (CT, PET, FDG, SUV, HU, MAE/PSNR/SSIM/LPIPS/
  FID/KID, MSE, ODE, ROI, MRI, PSMA, CPDM, PMRF, VQGAN, VQ-VAE, DDPM, DDIM, BBDM); abstract expands
  the ones it introduces since it stands alone.
- **Captions**: all 7 table captions moved **above** the tabular (figure captions stay below); every
  caption trimmed to a short descriptor (with `\caption[short]{...}` for clean list entries) and the
  explanatory content relocated into body prose next to each `\ref`; every float now referenced in text.
- **Inline figure rows**: `make_appendix_figs.py` gained single-row variants (`appendix_pmrf_row.png`,
  `appendix_cpdm_row.png`) via a second `panel()` call on `rows[:1]`; the PM-vs-PMRF and blob-vs-focal
  rows are shown inline in Results with supporting paragraphs, full grids kept in the appendix.
- **Appendix figures** height-capped (`0.72`/`0.58\textheight`, `keepaspectratio`, `[!ht]` + `\clearpage`
  before each appendix subsection) so each grid fits on one page under its subsection title without
  crowding the page number. Builds clean (no undefined refs, no overfull > 3pt).

Follow-up polish:
- Table captions: added `\usepackage{caption}` + `\captionsetup[table]{skip=9pt}` so above-table
  captions have breathing room before the table.
- Inline rows placed out of flow: `make_qualitative.py` now also emits `qualitative_short.png`
  (1 high + 1 med row); the main-text qualitative figure uses the short version, full 6-row grid
  moved to appendix A.1 (`app:qual`/`fig:app-qual`). All three inline Results figures
  (qualitative-short, pmrf-row, cpdm-row) switched to `[H]` (needs `float` package) so they sit
  immediately after their introducing paragraph instead of floating into the Discussion. 30 pp.

Further follow-up:
- ToC now fits one page via `etoolbox \patchcmd{\l@section}` (tocloft.sty is not installed on this
  TeXLive). Acknowledgements moved to before the ToC, on its own page. Supporting-stages figure
  (`training_support.png`) removed from the thesis (kept in the repo). Standalone Limitations
  section dissolved into a `Limitations.` paragraph at the end of the Discussion (kept: 2D slices,
  masked-metric thresholds, test-set size, latent-vs-pixel confound, one resolution/not-SOTA
  sentence; dropped: CPDM convergence, VQGAN simplification, resolution+compute, scanner-domain).
- **LPIPS + PSNR gap** (author noticed the table only had MAE/SSIM/FID/KID). New
  `sampling_eval/extra_metrics.py` computes global LPIPS (VGG) + PSNR for all models on the shared
  296-slice/12-patient manifest with patient-clustered CIs. LPIPS (lower=better): pm 0.155 (worst),
  cpdm_focal 0.119 (best pt), concatdiff 0.123, cpdm_blob 0.124, pmrf 0.125, cond 0.150. PSNR (dB):
  pm 37.1 (best = MSE-optimal), then 36.5/36.2/36.0/35.7/35.6, all CIs overlapping (34--39).
  Decision (author): **add LPIPS as a table column; explain PSNR in text** (monotone in MSE, dup of
  MAE). Honest reading in thesis: LPIPS = per-image full-reference perception (distinct from
  distributional FID/KID); it ranks pm (blur) and concat-flow (noise) worst and leaves
  CPDM/concatdiff/PMRF tied, so it complements but does not overturn FID/KID. Teaching point: CPDM's
  weak perception is specifically distributional (FID/KID), not per-image (LPIPS on par with best).

## 18. Reviewer fixes + adversarial (GAN) VQGAN

Reviewer-flagged corrections (`thesis/Thesis.tex` / `practical_work/References.bib`):
- Abstract no longer contradicts H3: the negative result is bounded to CPDM's hand-designed domain
  prior (diffusion family), and it states PMRF's posterior-mean init IS the deciding mechanism.
  Same conflation fixed in the results summary and the H2 discussion ("domain-native paradigm" ->
  "paradigm-native member").
- Notation: `y` now consistently means source/condition (BBDM, CPDM, PMRF); the rectified-flow
  target is `x_0` (was `y`) in the RF eq, the Flow-2 methods line, and training-dynamics.
- Dataset split: both sections say "by patient" (was "by study" in one).
- Removed the leftover "TODO verify" from the Jeblick bib entry + the stale bib header comment.

**GAN-VQGAN** (author decision to implement CPDM's VQGAN faithfully, with the PatchGAN the original
uses). New, under NEW names so current results are untouched:
- `model/VQGAN/discriminator.py`: reused the existing taming `NLayerDiscriminator`/`ActNorm`; added
  `weights_init`, `hinge_d_loss`, `vanilla_d_loss`, `adopt_weight`, `calculate_adaptive_weight`.
- `training/train_vqgan_gan.py`: vanilla-PyTorch two-optimizer loop. Loss = L1 + VGG-LPIPS +
  codebook + adaptive-weighted PatchGAN, disc warmed up after `--disc-start`. wandb `VQGAN-GAN-fb64`.
  Ckpts `checkpoints/VQGAN_gan_fb64/`: `best.ckpt` slim (state_dict only ~220MB, CPDM-ready),
  `last.ckpt` full (resumable). Verified on synthetic data: GAN+adaptive-weight work, warmup gates
  the disc, `best.ckpt` loads into a fresh `VQModel` with 0 missing / 0 unexpected keys.
- `scripts/run_vqgan_gan.sh`: detached launcher (setsid nohup), resume-friendly. NOT run yet (data
  drive unmounted this session). VGG-LPIPS on CPU is the bottleneck; defaults kept modest + resumable.
- **Downstream cascade before any results change** (new VQGAN = new latent space): retrain
  CPDM-focal AND concat-diffusion in the new latent (old denoisers invalid), re-sample, re-eval.
  Flows and attention/attenuation maps unaffected. Do it under new config/ckpt names; update the
  thesis only once val+test numbers are in.

Cascade configs drafted (point at `checkpoints/VQGAN_gan_fb64/best.ckpt`, write to NEW result dirs):
- `config/CPDM-autoPET-fb64-ganvq-focal.yaml` (from scratch; model_name `CPDM_ganvq_focal`).
- `config/ConcatDiff-autoPET-fb64-ganvq.yaml` (train with `--ckpt-dir checkpoints/ConcatDiff_ganvq`;
  added a `--ckpt-dir` arg to `training/train_concat_diffusion.py`).
Cascade order once the VQGAN is done: (1) `main.py -c config/CPDM-autoPET-fb64-ganvq-focal.yaml -t
--gpu_ids -1 --save_top`; (2) `training/train_concat_diffusion.py --config
config/ConcatDiff-autoPET-fb64-ganvq.yaml --ckpt-dir checkpoints/ConcatDiff_ganvq`; (3) sample both
+ `sampling_eval/eval_masked.py` / `compare_models.py` into `results/eval/*_ganvq`; (4) only then
update the thesis if the numbers hold up.

GAN-VQGAN training LAUNCHED (this session): `bash scripts/run_vqgan_gan.sh` (batch 8, 5000-slice cap,
disc-start 600, ~14s/step CPU => ~2h/epoch). wandb run `CT2PET-VQGAN/VQGAN-GAN-fb64`. The launcher
auto-resumes from `checkpoints/VQGAN_gan_fb64/last.ckpt` (written every 200 steps + each epoch), so
re-running the same script continues where it stopped.

## 19. Cover page (JKU style) + metric-keyed CPDM checkpointing

- **JKU-style cover page** (`thesis/Thesis.tex`): replaced the plain `\maketitle` with a self-contained
  `titlepage` (no `jkureport.sty`, no new packages) — JKU header + rule, title/subtitle, "Bachelor's
  Thesis / to obtain ... Bachelor of Science in the Computer Science programme", and a Submitted-by /
  Supervisor / Institute block over a footer rule. Fields are `\newcommand`s at the top of the file;
  **verify `\thesisprogram` and `\thesisinstitute`** (defaults: Computer Science / Institute of
  Computational Perception). Drop the official logo via the commented `\includegraphics` line if wanted.
  A full port to the JKU report template was rejected (report/chapter + ACM/bibtex + split files =
  churns the whole doc). Builds clean, 31 pp.
- **Metric-keyed checkpoints** (`runners/DiffusionBasedModelRunners/CPDMRunner.py`): the CPDM VQGAN-blob
  run showed distortion metrics (SSIM/PSNR/MAE) peaking ~epoch 7–10 then drifting while `val_epoch/loss`
  (latent) kept falling to ~epoch 50; the reported blob number used `last_model.pth` (epoch 52), so it
  sat ~0.02 SSIM / ~0.6 dB below its own SSIM/PSNR peak. (LPIPS, though, keeps improving past epoch 10 —
  it is a distortion↔perception trade, not a strict "peak", so re-selecting epoch 10 is not unambiguously
  fairer.) Fix for the *retrain*: `validation_epoch` now saves `best_ssim_model.pth` and
  `best_lpips_model.pth` from the metrics already computed each val epoch (one extra `torch.save` per
  improvement, no extra sampling; model weights only, overwritten in place). At eval, point
  `--resume_model` at whichever selection you report; still honest to keep `last_model.pth` alongside.

## 20. VQGAN-GAN converged, checkpoint frozen, CPDM retrain launched

- **VQGAN-GAN converged** at 14 epochs (CPU, ~1 week wall incl. overnight sleeps). Val curves:
  rec 0.067->0.033 (plateau/oscillation ~0.033-0.047 from epoch 2), perc 0.343->0.143 (plateau
  ~0.143-0.170 from epoch ~10). Stopped at epoch 14 because it entered **discriminator dominance**
  (step-level d->~0.01, g rising to ~2.5): training further risks degrading the decoder.
- **Checkpoint choice = `best.ckpt` (epoch 11)**, decided by a decode check
  (`sampling_eval/eval_vqgan_recon.py`, 6 fixed val slices, figures in `results/vqgan_recon/`):
  best vs last -> CT L1 0.045/0.059, CT LPIPS 0.188/0.218, PET L1 0.018/0.016, PET LPIPS 0.120/0.100.
  last.ckpt (epoch 14) washed out CT internal structure (late disc-dominance skew toward PET) for
  only a modest PET gain; best.ckpt is the balanced autoencoder for both CPDM's PET decode and CT
  conditioning. Config `CPDM-autoPET-fb64-ganvq-focal.yaml` now points at best.ckpt.
  The recon grids double as the supervisor-requested **VQ-VAE reconstruction figure**.
- **CPDM retrain LAUNCHED from scratch** on the new latent: `main.py -c
  config/CPDM-autoPET-fb64-ganvq-focal.yaml -t --gpu_ids -1 --save_top`, detached, `WANDB_MODE=offline`
  (wandb DNS was down; offline run under the result dir, `wandb sync` later). ~4 s/it, 625 it/epoch =>
  ~42 min/epoch (100.4M trainable). Metric-keyed checkpoints (`best_ssim_model.pth`/`best_lpips_model.pth`)
  now active. Log `logs/cpdm_ganvq_focal.log`.
- **Supervisor feedback on the practical-work part (apply to thesis, do not repeat):** (1) hyper-params
  hardly explored -> add a hyper-parameter table + justification (CPU budget) in the thesis; (2) no
  qualitative VQ-VAE reconstruction -> add the `results/vqgan_recon/` figure; (3) validation loss not
  defined -> define it explicitly per stage; (4) "stage-4 val loss down but PSNR down is strange" ->
  this is the latent-loss vs image-metric divergence we already diagnosed (§18/§19); explain it in the
  thesis and report CPDM from a metric-selected checkpoint, not the latent-loss one.

- **Old vs new VQGAN reconstruction (direct measurement, `sampling_eval/eval_vqgan_recon.py` +
  scratch masked script).** The adversarial VQGAN is NOT uniformly better — it is a global-vs-lesion
  trade. Global recon (8 val slices): OLD beats NEW on every metric (CT L1 0.026/0.045, CT LPIPS
  0.118/0.191, PET L1 0.011/0.019, PET LPIPS 0.083/0.122). Masked PET recon (60 val slices, MAE in
  SUV): OLD global 0.180 / active 0.589 / lesion 1.612; NEW 0.253 / 0.664 / **1.351** — NEW is ~16%
  better on the SUV>2.5 lesions (the diagnostically relevant, thesis-privileged region) while worse on
  the near-zero background that dominates global metrics. So the retrain is DEFENSIBLE on the lesion
  axis, not a clean win. Honest caveat: both autoencoders reconstruct lesion peaks poorly (MAE ~1.3-1.6
  SUV, i.e. ~50%+ of the SUV>2.5 threshold) — the VQGAN is a real bottleneck capping BOTH CPDMs.
  Nothing guarantees the end-to-end CPDM improves; the deciding test is the shared-ruler eval_masked
  (global+active+lesion, patient-clustered CIs) of NEW-CPDM vs OLD-CPDM after the retrain. Report the
  trade honestly whichever way it lands.

- **NEW-CPDM vs OLD-CPDM head-to-head (matched eval_masked, same 296 slices / 12 patients, --fid --ci).**
  NEW = CPDM_ganvq_focal (adversarial VQGAN latent, best_ssim epoch-30 ckpt); OLD = CPDM_focal
  (disc-free VQGAN, latent-loss ckpt). Clean perception/lesion vs global-distortion trade, as the
  VQGAN recon test predicted:
  - OLD better on global/active distortion: global SSIM 0.876 vs 0.866, global PSNR 36.5 vs 34.8,
    active SSIM 0.658 vs 0.611.
  - NEW better on all lesion tiers + distributional perception: lesion MAE 0.1044->0.0915, lesion PSNR
    19.99->21.11, lesion SSIM 0.4295->0.4457, FID 62.3->58.6, KID 0.0256->0.0214.
  Direction consistent across 5 metrics but every per-patient CI overlaps => directional, NOT
  significant at n=12. NEW reported from its SSIM-best (distortion-favoring) ckpt, so its FID/lesion
  edge is if anything understated (LPIPS-best epoch-12 ckpt would push perception further).
  Caveat: NEW benefits from both the new latent AND metric-keyed ckpt selection; OLD used its
  latent-loss ckpt, so this is new-pipeline vs old-pipeline, not a clean VQGAN-only ablation.
  DECISION PENDING: adopting the new VQGAN for the thesis requires retraining concat-diffusion on the
  new latent too (shared latent = fairness for H2/H5), i.e. finishing the ConcatDiff-ganvq cascade.
  Alternative: keep OLD VQGAN as the main results and report the adversarial-VQGAN run as a documented
  P-D-trade exploration (no extra retrain).
