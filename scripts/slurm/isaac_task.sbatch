#!/usr/bin/env bash
#SBATCH --job-name=g1_isaac
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx_6000_ada:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-isaac-%j.out
# Isaac backend: runs each "script args" line of $ISAAC_COMMANDS in order (one GPU job).
set -euo pipefail
project=/data1/users/konstantin.smirnov/DEXTRAH-G1-WholeBody
export FULLBODY_SOURCE_COMMIT="$(git -C "$project" rev-parse HEAD)"
src="$(mktemp -d "$project/outputs/isaac_source.XXXXXX")"
git -C "$project" archive "$FULLBODY_SOURCE_COMMIT" | tar -x -C "$src"
export PATH="/data1/users/konstantin.smirnov/venvs/g1_sonic/bin:/data1/users/konstantin.smirnov/miniconda3/envs/adept_dextrah/bin:$PATH"
export PYTHONPATH="$src" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1 PXR_WORK_THREAD_LIMIT=8
export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y WANDB_MODE=disabled
unset DISPLAY
cd "$src"
nvidia-smi --query-gpu=name,driver_version,memory.used,memory.total --format=csv
while IFS= read -r line; do
  [[ -z "$line" ]] && continue
  echo "=== RUN $line"
  eval "python $line" || echo "=== FAILED $line"
done <<< "${ISAAC_COMMANDS:?Set ISAAC_COMMANDS}"
