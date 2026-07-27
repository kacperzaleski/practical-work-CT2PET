#!/usr/bin/env bash
# Overnight sequential pipeline template (single CPU -> run one at a time, full cores each):
#   1. wait for the already-running attention-map export to finish
#   2. retrain VQGAN on the 64x64 head+torso set
#   3. PMRF stage 1 (posterior-mean U-Net), capped at 30 epochs
# Each stage logs to its own file; this driver logs stage boundaries + exit codes.
set -u
cd /home/kacperzaleski/Projects/practical-work-CT2PET
PY=.venv/bin/python
EXPORT_PID="${1:-}"

log() { echo "[$(date '+%F %T')] $*"; }

# --- 1. wait for export ---------------------------------------------------
if [[ -n "$EXPORT_PID" ]]; then
  log "waiting for attention export (pid $EXPORT_PID) to finish..."
  while kill -0 "$EXPORT_PID" 2>/dev/null; do sleep 60; done
  log "export process gone."
fi

# --- 2. VQGAN (64x64) -----------------------------------------------------
log "STARTING VQGAN retrain"
$PY training/train_vqgan.py \
  --config config/VQGAN-autoPET-fb64.yaml \
  --data-root data/processed_fullbody \
  --batch-size 16 --num-workers 4 \
  --max-epochs 5 --limit-train-batches 500 --limit-val-batches 100 \
  > logs/vqgan_fb64.log 2>&1
log "VQGAN finished (exit $?)"

# --- 3. PMRF stage 1 ------------------------------------------------------
log "STARTING PMRF stage 1 (max 30 epochs)"
$PY training/train_pmrf_stage1.py \
  --data-root data/processed_fullbody \
  --batch-size 32 --num-workers 4 \
  --max-epochs 30 \
  > logs/pmrf_stage1.log 2>&1
log "PMRF stage 1 finished (exit $?)"

log "ALL STAGES DONE"
