#!/usr/bin/env bash
# Wait for the CPDM_focal sample_to_eval to finish, then run masked + FID/KID + LPIPS.
set -u
cd /home/kacperzaleski/Projects/practical-work-CT2PET
PY=.venv/bin/python
PRED=results/CT2PET_autoPET_fullbody/CPDM_focal/sample_to_eval/200
GT=results/CT2PET_autoPET_fullbody/CPDM_focal/sample_to_eval/ground_truth
log(){ echo "=== $(date '+%F %T') $* ==="; }

log "waiting for CPDM_focal sampling to finish..."
while pgrep -f "CPDM-autoPET-fb64-focal.yaml --sample_to_eval" >/dev/null; do sleep 30; done
log "sampling done. preds=$(ls $PRED 2>/dev/null | wc -l) gt=$(ls $GT 2>/dev/null | wc -l)"

log "MASKED + FID/KID"
$PY sampling_eval/eval_masked.py --pred-dir "$PRED" --gt-dir "$GT" --fid 2>&1 | grep -viE "warning|warn\("

log "LPIPS (vgg)"
$PY - <<'PYEOF' 2>&1 | grep -viE "warning|warn\("
import glob, os, numpy as np, torch, lpips
net=lpips.LPIPS(net='vgg',verbose=False).eval()
def load(p):
    a=np.clip(np.load(p).astype(np.float32).squeeze(),0,1)
    return (torch.from_numpy(a)[None,None]*2-1).repeat(1,3,1,1)
d='results/CT2PET_autoPET_fullbody/CPDM_focal/sample_to_eval'
names=sorted(os.path.basename(f) for f in glob.glob(d+'/200/*.npy'))
with torch.no_grad():
    v=[net(load(f'{d}/200/{n}'),load(f'{d}/ground_truth/{n}')).item() for n in names]
print(f'CPDM_focal LPIPS={np.mean(v):.4f} (n={len(v)})')
PYEOF
log "ALL DONE"
