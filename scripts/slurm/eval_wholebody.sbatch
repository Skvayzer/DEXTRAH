#!/usr/bin/env bash
#SBATCH --job-name=wb_eval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx_6000_ada:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=slurm-wb-eval-%j.out
# Evaluate a whole-body (frozen SONIC) mjlab checkpoint (scripts/eval_wholebody_checkpoint.py):
# same simulator settings as training (vendored cone-Hessian fix), validated fresh starts.
set -euo pipefail
project="${WHOLEBODY_REPO:-/data1/users/konstantin.smirnov/DEXTRAH-G1-WholeBody}"
commit="$(git -C "$project" rev-parse HEAD)"
source_snapshot="$(mktemp -d "$project/outputs/eval_source.XXXXXX")"
git -C "$project" archive "$commit" | tar -x -C "$source_snapshot"
export PATH="/data1/users/konstantin.smirnov/venvs/mjlab/bin:$PATH"
export PYTHONPATH="$source_snapshot/third_party/mujoco_warp_cone_fix:$source_snapshot:$source_snapshot/scripts"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$source_snapshot"
echo "Commit: $commit"
python scripts/eval_wholebody_checkpoint.py --checkpoint "${CHECKPOINT:?}" --output "${OUTPUT:?}" ${EXTRA_ARGS:-}
