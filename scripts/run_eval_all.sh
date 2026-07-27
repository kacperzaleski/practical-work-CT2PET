#!/usr/bin/env bash
# Sample the 3 not-yet-dumped models on the shared 256 test slices, then run the
# unified masked + FID/KID eval on all five. PM and PMRF were already dumped to
# results/eval/{pm,pmrf}. Logs to logs/eval_all.log.
set -u
PY=.venv/bin/python
ROOT=data/processed_fullbody
N=256
log(){ echo "=== $(date +%H:%M:%S) $* ==="; }

# 1) concat-flow (cond): z0=noise, CT concatenated, K=200 Euler
log "SAMPLE concat-flow (cond)"
$PY sampling_eval/sample_pmrf.py --model cond \
  --flow-ckpt checkpoints/PMRF_cond/cond-epoch=22.ckpt \
  --num-steps 200 --max-samples $N --batch-size 8 \
  --save-npy results/eval/cond || echo "cond sampling FAILED"

# 2) concat-diffusion: DDIM 200 steps in the VQGAN latent
log "SAMPLE concat-diffusion"
$PY sampling_eval/sample_concat_diffusion.py \
  --ckpt "checkpoints/ConcatDiff/cd-epoch=25.ckpt" \
  --num-steps 200 --max-samples $N --batch-size 8 \
  --save-npy results/eval/concatdiff || echo "concatdiff sampling FAILED"

# 3) CPDM: Brownian-bridge 200-step reverse, dumps into its sample_to_eval tree
log "SAMPLE CPDM (sample_to_eval)"
$PY main.py -c config/CPDM-autoPET-fb64.yaml --sample_to_eval --gpu_ids -1 \
  --resume_model results/CT2PET_autoPET_fullbody/CPDM/checkpoint/last_model.pth \
  || echo "CPDM sampling FAILED"

# 4) Eval everyone on the same ruler (global + active + lesion + FID/KID)
CPDM_PRED=results/CT2PET_autoPET_fullbody/CPDM/sample_to_eval/200
CPDM_GT=results/CT2PET_autoPET_fullbody/CPDM/sample_to_eval/ground_truth
for spec in \
  "Posterior-mean|results/eval/pm/pred|results/eval/pm/gt" \
  "CPDM|$CPDM_PRED|$CPDM_GT" \
  "ConcatDiffusion|results/eval/concatdiff/pred|results/eval/concatdiff/gt" \
  "ConcatFlow|results/eval/cond/pred|results/eval/cond/gt" \
  "PMRF|results/eval/pmrf/pred|results/eval/pmrf/gt" ; do
  name=${spec%%|*}; rest=${spec#*|}; pred=${rest%%|*}; gt=${rest#*|}
  log "EVAL $name"
  $PY sampling_eval/eval_masked.py --pred-dir "$pred" --gt-dir "$gt" --fid \
    || echo "$name eval FAILED"
done
log "ALL DONE"
