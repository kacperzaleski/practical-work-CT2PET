"""Build a shared, patient-diverse test-slice manifest for the CT->PET comparison.

The on-disk test split is patient-disjoint (official autoPET 80/10/10), but the
`max_samples: {test: N}` cap in the configs takes the first N *alphabetically
sorted* slices -- which all come from the first one or two patients. That makes the
reported test set effectively single-patient. This script instead picks a fixed set
of patients spread across the test split and a body-spread selection of slices from
each, writing the chosen slice stems to a manifest file. Every model's sampler then
restricts its test loader to exactly these slices (via `dataset_config.test_names_file`),
so the comparison is apples-to-apples AND patient-diverse, with zero train/val leakage
(only the `test/` split is ever read).

Determinism: fully seeded, so re-running yields the identical manifest.

Normalization (from preprocess_autopet): CT HU[-1000,3071]->[-1,1], PET SUV[0,32]->[-1,1].
  body voxel  : CT   > -0.7544  (HU > -500)
  lesion voxel: PET  > -0.8438  (SUV > 2.5)   -- ensures the lesion metric tier is populated.

Usage:
  python sampling_eval/build_test_manifest.py                 # 12 patients x 25 slices
  python sampling_eval/build_test_manifest.py --n-patients 12 --n-slices 25 --seed 0
"""
import sys as _sys, pathlib as _pathlib  # repo-root bootstrap (script lives in a subfolder)
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))
import argparse
import random
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

CT_BODY_NORM = -0.7544    # HU > -500
PET_LESION_NORM = -0.8438  # SUV > 2.5
SLICE_RE = re.compile(r'_s(\d+)$')


def patient_of(stem):
    """Slice stem -> patient/study id (drop the trailing _sNNN)."""
    return SLICE_RE.sub('', stem)


def tracer_of(pid):
    return 'psma' if pid.startswith('psma_') else 'fdg'


def slice_index(stem):
    m = SLICE_RE.search(stem)
    return int(m.group(1)) if m else 0


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data-root', default='data/processed_fullbody')
    p.add_argument('--split', default='test')
    p.add_argument('--n-patients', type=int, default=12)
    p.add_argument('--n-slices', type=int, default=25, help='target slices per patient')
    p.add_argument('--min-body', type=float, default=0.05,
                   help='drop slices whose body-voxel fraction is below this')
    p.add_argument('--min-lesion-slices', type=int, default=3,
                   help='guarantee at least this many lesion-bearing slices per patient')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out', default='config/test_manifest_fb64.txt')
    args = p.parse_args()

    rng = random.Random(args.seed)
    ct_dir = Path(args.data_root) / args.split / 'CT'
    pet_dir = Path(args.data_root) / args.split / 'PET'
    if not ct_dir.is_dir() or not pet_dir.is_dir():
        raise SystemExit(f'Missing CT/PET dirs under {args.data_root}/{args.split}')

    stems = sorted(f.stem for f in ct_dir.glob('*.npy'))
    by_patient = defaultdict(list)
    for s in stems:
        by_patient[patient_of(s)].append(s)

    # --- stratified patient selection: keep the pool's FDG/PSMA proportion ---
    patients = sorted(by_patient)
    fdg = sorted(pid for pid in patients if tracer_of(pid) == 'fdg')
    psma = sorted(pid for pid in patients if tracer_of(pid) == 'psma')
    n_psma = round(args.n_patients * len(psma) / len(patients))
    n_psma = max(1, min(args.n_patients - 1, n_psma))
    n_fdg = args.n_patients - n_psma
    chosen = sorted(rng.sample(fdg, n_fdg) + rng.sample(psma, n_psma))
    print(f'pool: {len(patients)} test patients ({len(fdg)} FDG, {len(psma)} PSMA)')
    print(f'chosen: {len(chosen)} patients ({n_fdg} FDG, {n_psma} PSMA), '
          f'target {args.n_slices} slices each\n')

    manifest = []
    total_lesion = 0
    for pid in chosen:
        # per-slice body fraction + lesion flag
        cand = []
        for s in sorted(by_patient[pid], key=slice_index):
            ct = np.load(ct_dir / f'{s}.npy')
            pet = np.load(pet_dir / f'{s}.npy')
            body = float((ct > CT_BODY_NORM).mean())
            if body < args.min_body:
                continue
            has_les = bool((pet > PET_LESION_NORM).any())
            cand.append((s, body, has_les))
        if not cand:
            print(f'  [warn] {pid}: no body slices above --min-body; skipped')
            continue

        # evenly-spaced pick across the retained (body-sorted-by-slice-index) list
        k = min(args.n_slices, len(cand))
        idx = np.linspace(0, len(cand) - 1, k).round().astype(int)
        idx = sorted(set(idx.tolist()))
        picked = {cand[i][0] for i in idx}

        # guarantee lesion coverage: top-body lesion slices swapped in for lowest-body picks
        les_slices = [c for c in cand if c[2]]
        have = sum(1 for i in idx if cand[i][2])
        if have < args.min_lesion_slices and les_slices:
            need = min(args.min_lesion_slices, len(les_slices)) - have
            extra = [c[0] for c in sorted(les_slices, key=lambda c: -c[1])
                     if c[0] not in picked][:max(0, need)]
            # drop the lowest-body non-lesion picks to make room
            droppable = sorted([cand[i] for i in idx if not cand[i][2]], key=lambda c: c[1])
            for e, d in zip(extra, droppable):
                picked.discard(d[0]); picked.add(e)

        les_here = sum(1 for s in picked if dict((c[0], c[2]) for c in cand)[s])
        total_lesion += les_here
        manifest.extend(sorted(picked, key=slice_index))
        print(f'  {pid[:60]:60s} {len(picked):3d} slices ({les_here} lesion-bearing)')

    manifest = sorted(set(manifest))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text('\n'.join(manifest) + '\n')

    man_patients = sorted({patient_of(s) for s in manifest})
    print(f'\nwrote {out}  ({len(manifest)} slices, {len(man_patients)} patients, '
          f'{sum(1 for p in man_patients if tracer_of(p)=="fdg")} FDG / '
          f'{sum(1 for p in man_patients if tracer_of(p)=="psma")} PSMA)')
    print(f'lesion-bearing slices: {total_lesion}/{len(manifest)} '
          f'({100*total_lesion/max(1,len(manifest)):.1f}%)')


if __name__ == '__main__':
    main()
