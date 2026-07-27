#!/usr/bin/env bash
# Overnight chain: wait for the focal attention-UNet to finish -> export focal maps
# (to AttentionMaps_focal, preserving the blob maps) -> back up the epoch-27 CPDM
# checkpoint -> fine-tune CPDM from those blob weights on the focal conditioning.
# Everything that preserves a fallback is explicit. Logs to logs/focal_chain.log.
set -u
cd /home/kacperzaleski/Projects/practical-work-CT2PET
PY=.venv/bin/python
ROOT=data/processed_fullbody
ATT_DIR=checkpoints/AttentionMap_fb64
OLD_CPDM_CKPT=results/CT2PET_autoPET_fullbody/CPDM/checkpoint/last_model.pth
log(){ echo "=== $(date '+%F %T') $* ==="; }

# 1) Wait for the attention-UNet training to finish.
log "STEP 1: waiting for attention-UNet (train_attention_map) to finish..."
while pgrep -f train_attention_map.py >/dev/null; do sleep 60; done
log "attention-UNet finished."

# 2) Pick the best attention checkpoint (highest val_dataset_iou in the filename).
BEST=$(ls ${ATT_DIR}/attention_map_unet-epoch=*.ckpt 2>/dev/null | sort -t= -k3 -n | tail -1)
[ -z "$BEST" ] && BEST="${ATT_DIR}/last.ckpt"
log "STEP 2: best attention ckpt = $BEST"

# 3) Export focal maps to AttentionMaps_focal (blob maps in AttentionMaps/ untouched).
log "STEP 3: exporting focal attention maps -> {split}/AttentionMaps_focal"
$PY sampling_eval/export_attention_maps.py --ckpt "$BEST" --data-root "$ROOT" \
    --splits train val test --out-dir-name AttentionMaps_focal --batch-size 32 \
    || { log "EXPORT FAILED — aborting"; exit 1; }
for s in train val test; do
  printf 'focal maps %-5s: %s\n' "$s" "$(ls ${ROOT}/${s}/AttentionMaps_focal 2>/dev/null | wc -l)"
done

# 4) Back up the epoch-27 CPDM checkpoint dir (extra safety; the focal run also writes
#    to a SEPARATE model dir CPDM_focal, so the original is preserved regardless).
log "STEP 4: backing up the epoch-27 CPDM checkpoint"
BK=results/CT2PET_autoPET_fullbody/CPDM/checkpoint_blob_ep27_BACKUP
mkdir -p "$BK"
cp -n results/CT2PET_autoPET_fullbody/CPDM/checkpoint/last_model.pth "$BK/" 2>/dev/null
cp -n results/CT2PET_autoPET_fullbody/CPDM/checkpoint/last_optim_sche.pth "$BK/" 2>/dev/null
ls -la "$BK/"

# 5) Fine-tune CPDM: load epoch-27 weights (+EMA), FRESH optimizer (no --resume_optim),
#    focal conditioning, EMA engaged, writes to results/.../CPDM_focal/.
[ ! -f "$OLD_CPDM_CKPT" ] && { log "OLD CPDM ckpt missing ($OLD_CPDM_CKPT) — aborting"; exit 1; }
log "STEP 5: launching CPDM fine-tune (focal) from $OLD_CPDM_CKPT"
$PY main.py -c config/CPDM-autoPET-fb64-focal.yaml -t --gpu_ids -1 \
    --resume_model "$OLD_CPDM_CKPT"
log "CPDM fine-tune process exited."
log "ALL DONE"
