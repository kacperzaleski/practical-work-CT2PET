#!/usr/bin/env bash
# Train the adversarial (GAN) VQGAN, a faithful reimplementation of the taming/CPDM VQGAN
# loss (L1 + VGG-LPIPS + codebook + PatchGAN, adaptive-weighted, warmed up after disc-start).
# Kept under a NEW checkpoint name (checkpoints/VQGAN_gan_fb64) and a NEW wandb run so nothing
# in the current results moves until the retrained diffusion models are evaluated.
#
# Detached so a harness teardown / laptop suspend does not kill it (CLAUDE.md convention).
# Requires the external data drive mounted at data/processed_fullbody.
#   bash scripts/run_vqgan_gan.sh          # launch detached, tail logs/vqgan_gan_fb64.log
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
LOG=logs/vqgan_gan_fb64.log
mkdir -p logs checkpoints/VQGAN_gan_fb64

if [ ! -d data/processed_fullbody/train/CT ]; then
  echo "ERROR: data/processed_fullbody/train/CT not found (is the external drive mounted?)" >&2
  exit 1
fi

# Auto-resume: if a checkpoint from an earlier session exists, continue from it.
RESUME=""
if [ -f checkpoints/VQGAN_gan_fb64/last.ckpt ]; then
  RESUME="--resume checkpoints/VQGAN_gan_fb64/last.ckpt"
  echo "found checkpoints/VQGAN_gan_fb64/last.ckpt -> resuming"
fi

# The VGG-LPIPS perceptual term (faithful to the taming/CPDM loss) is the CPU bottleneck: ~12s/step
# at batch 8, so an epoch's wall-clock scales with --max-samples (5000 slices x2 CT+PET is ~2h/epoch).
# The subset is kept to 5000 (matching the diffusion models' train cap, so the VQGAN is trained on a
# comparable slice budget), a checkpoint is written every epoch, early stopping (patience 12) cuts it
# once val reconstruction plateaus, and disc-start 600 (~1 epoch) reaches the adversarial phase the
# same day. Resume across sessions with --resume checkpoints/VQGAN_gan_fb64/last.ckpt.
setsid nohup "$PY" training/train_vqgan_gan.py \
  --config config/VQGAN-autoPET-fb64.yaml \
  --data-root data/processed_fullbody \
  --batch-size 8 --num-workers 4 \
  --max-samples 5000 --val-samples 800 \
  --max-epochs 40 --disc-start 600 --patience 12 \
  --log-every 25 --ckpt-every-steps 200 \
  --ckpt-dir checkpoints/VQGAN_gan_fb64 \
  --wandb-project CT2PET-VQGAN --wandb-name VQGAN-GAN-fb64 \
  $RESUME ${EXTRA_ARGS:-} < /dev/null >> "$LOG" 2>&1 &

echo "launched GAN-VQGAN training (PID $!). Tail: tail -f $LOG"
echo "resume after interruption: add --resume checkpoints/VQGAN_gan_fb64/last.ckpt"
