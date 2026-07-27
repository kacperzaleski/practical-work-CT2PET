#!/usr/bin/env bash
# Re-sample every model on the shared 12-patient / 300-slice manifest and dump [0,1]
# pred/gt .npy for the unified evaluator. Order: the two CPDM variants first (scratch,
# finetune) so the scratch-vs-finetune decision can be made early, then blob (H5 arm)
# and the four flow/concat models. Single CPU -> strictly sequential.
set -u
cd /home/kacperzaleski/Projects/practical-work-CT2PET
PY=.venv/bin/python
MAN=config/test_manifest_fb64.txt
log(){ echo "[$(date '+%F %T')] $*"; }

# ---- clear stale dumps (old 256-slice / 2-patient sets) so eval sees ONLY the manifest ----
log "clearing stale sample dirs"
rm -rf results/CT2PET_autoPET_fullbody/CPDM_focal_scratch/sample_to_eval \
       results/CT2PET_autoPET_fullbody/CPDM_focal/sample_to_eval \
       results/CT2PET_autoPET_fullbody/CPDM/sample_to_eval \
       results/eval/pm results/eval/pmrf results/eval/cond results/eval/concatdiff

# ---- CPDM variants (sample_to_eval writes .../<model>/sample_to_eval/{200,ground_truth}) ----
cpdm(){  # $1=config  $2=model_dir  $3=tag
  log "CPDM $3: sampling on manifest"
  $PY main.py -c "$1" --sample_to_eval --gpu_ids -1 \
    --resume_model "results/CT2PET_autoPET_fullbody/$2/checkpoint/last_model.pth" \
    > "logs/sample_${3}.log" 2>&1
  log "CPDM $3 done (exit $?)"
}
cpdm config/_sample_cpdm_scratch.yaml   CPDM_focal_scratch  scratch
cpdm config/_sample_cpdm_finetune.yaml  CPDM_focal          finetune
cpdm config/_sample_cpdm_blob.yaml      CPDM                blob

# ---- flow / concat models (dump to results/eval/<name>/{pred,gt}) ----
log "PM: sampling"
$PY sampling_eval/sample_pmrf.py --model pm --pm-ckpt checkpoints/PMRF_stage1/last.ckpt \
  --names-file "$MAN" --num-steps 200 --save-npy results/eval/pm > logs/sample_pm.log 2>&1
log "PMRF: sampling"
$PY sampling_eval/sample_pmrf.py --model pmrf --pm-ckpt checkpoints/PMRF_stage1/last.ckpt \
  --flow-ckpt checkpoints/PMRF_pmrf/last.ckpt --names-file "$MAN" --num-steps 200 \
  --save-npy results/eval/pmrf > logs/sample_pmrf.log 2>&1
log "concat-flow: sampling"
$PY sampling_eval/sample_pmrf.py --model cond --flow-ckpt checkpoints/PMRF_cond/last.ckpt \
  --names-file "$MAN" --num-steps 200 --save-npy results/eval/cond > logs/sample_cond.log 2>&1
log "concat-diff: sampling"
$PY sampling_eval/sample_concat_diffusion.py --ckpt checkpoints/ConcatDiff/last.ckpt \
  --names-file "$MAN" --num-steps 200 --save-npy results/eval/concatdiff > logs/sample_concatdiff.log 2>&1

log "ALL SAMPLING DONE"

# ============================ unified evaluation ============================
EVLOG=logs/eval_manifest.log
: > "$EVLOG"
SE=sample_to_eval
declare -A PRED=(
  [scratch]="results/CT2PET_autoPET_fullbody/CPDM_focal_scratch/$SE/200"
  [finetune]="results/CT2PET_autoPET_fullbody/CPDM_focal/$SE/200"
  [blob]="results/CT2PET_autoPET_fullbody/CPDM/$SE/200"
  [pm]="results/eval/pm/pred" [pmrf]="results/eval/pmrf/pred"
  [cond]="results/eval/cond/pred" [concatdiff]="results/eval/concatdiff/pred")
declare -A GT=(
  [scratch]="results/CT2PET_autoPET_fullbody/CPDM_focal_scratch/$SE/ground_truth"
  [finetune]="results/CT2PET_autoPET_fullbody/CPDM_focal/$SE/ground_truth"
  [blob]="results/CT2PET_autoPET_fullbody/CPDM/$SE/ground_truth"
  [pm]="results/eval/pm/gt" [pmrf]="results/eval/pmrf/gt"
  [cond]="results/eval/cond/gt" [concatdiff]="results/eval/concatdiff/gt")

for m in scratch finetune blob pm pmrf cond concatdiff; do
  log "EVAL $m (masked + CI + FID/KID)"
  { echo "############## MODEL: $m ##############"
    $PY sampling_eval/eval_masked.py --pred-dir "${PRED[$m]}" --gt-dir "${GT[$m]}" --ci --fid
  } >> "$EVLOG" 2>&1
done

log "COMPARE: pre-registered contrasts (cpdm = from-scratch focal)"
{ echo "############## PRE-REGISTERED CONTRASTS ##############"
  $PY sampling_eval/compare_models.py \
    --model pm:"${PRED[pm]}":"${GT[pm]}" \
    --model pmrf:"${PRED[pmrf]}":"${GT[pmrf]}" \
    --model cond:"${PRED[cond]}":"${GT[cond]}" \
    --model concatdiff:"${PRED[concatdiff]}":"${GT[concatdiff]}" \
    --model cpdm:"${PRED[scratch]}":"${GT[scratch]}" \
    --model cpdm_blob:"${PRED[blob]}":"${GT[blob]}" --kid
} >> "$EVLOG" 2>&1

log "ALL EVAL DONE"
