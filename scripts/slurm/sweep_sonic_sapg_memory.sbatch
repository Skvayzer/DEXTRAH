#!/usr/bin/env bash
#SBATCH --job-name=wb_mem_sweep
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx_6000_ada:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=02:30:00
#SBATCH --output=slurm-wb-mem-sweep-%j.out
# Memory sweep for the frozen-SONIC trainer: optional PRE_CHECK command, then for each
# N in SWEEP_ENVS a short run (MAX_EPOCHS) of scripts/slurm/train_g1_sonic_sapg.sbatch
# with the caller's settings, NUM_ENVS=N and MINIBATCH_SIZE=N*16/10 (10 minibatches per
# epoch); reports device memory per size (SWEEP_RESULT). Stops at the first failure.
set -uo pipefail
project="${WHOLEBODY_REPO:?Set WHOLEBODY_REPO}"
cd "$project"
if [[ -n "${PRE_CHECK:-}" ]]; then
    echo "=== pre-check: $PRE_CHECK"
    bash -c "$PRE_CHECK" || echo "PRE_CHECK_FAILED"
fi
for n in ${SWEEP_ENVS:?}; do
    echo "=== sweep: NUM_ENVS=$n"
    prefix="${SWEEP_PREFIX:-0_wb_mem_sweep}_n${n}"
    NUM_ENVS=$n MINIBATCH_SIZE=$((n*16/10)) MAX_EPOCHS="${MAX_EPOCHS:-6}" RUN_PREFIX="$prefix" \
        bash scripts/slurm/train_g1_sonic_sapg.sbatch > "slurm-wb-mem-sweep-${SLURM_JOB_ID}-n${n}.out" 2>&1
    status=$?
    out="outputs/${prefix}_${SLURM_JOB_ID}"
    python3 - "$out" "$n" "$status" <<'PY'
import json, sys
from pathlib import Path
out, n, status = Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
r = dict(envs=n, exit=status)
p = out/'progress.json'
if p.exists():
    d = json.loads(p.read_text())
    r.update({k: round(v, 2) for k, v in d.items() if 'gib' in k}, epoch=d.get('epoch'))
print('SWEEP_RESULT '+json.dumps(r), flush=True)
PY
    grep -E "fps step  |fps total|out of memory|OutOfMemory|Traceback" "slurm-wb-mem-sweep-${SLURM_JOB_ID}-n${n}.out" | tail -4
    if [[ $status != 0 ]]; then echo "SWEEP_STOP at NUM_ENVS=$n (exit $status)"; break; fi
done
