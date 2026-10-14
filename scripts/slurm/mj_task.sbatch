#!/usr/bin/env bash
#SBATCH --job-name=g1_mjlab
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx_6000_ada:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-mjlab-%j.out
# mjlab backend: runs each "script args" line of $MJ_COMMANDS in order (one GPU job).
set -euo pipefail
project=/data1/users/konstantin.smirnov/DEXTRAH-G1-WholeBody
export FULLBODY_SOURCE_COMMIT="$(git -C "$project" rev-parse HEAD)"
src="$(mktemp -d "$project/outputs/mj_source.XXXXXX")"
git -C "$project" archive "$FULLBODY_SOURCE_COMMIT" | tar -x -C "$src"
export PATH="/data1/users/konstantin.smirnov/venvs/mjlab/bin:$PATH"
export PYTHONPATH="$src" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1 WANDB_MODE=disabled
if [[ "${MJ_CONE_HESSIAN_FIX:-0}" == 1 ]]; then export PYTHONPATH="$src/third_party/mujoco_warp_cone_fix:$PYTHONPATH"; fi
export MUJOCO_GL=egl
cd "$src"
nvidia-smi --query-gpu=name,driver_version,memory.used,memory.total --format=csv
while IFS= read -r line; do
  [[ -z "$line" ]] && continue
  echo "=== RUN $line"
  eval "python $line" || echo "=== FAILED $line"
done <<< "${MJ_COMMANDS:?Set MJ_COMMANDS}"
