#!/usr/bin/env bash
#SBATCH --job-name=mj_walk
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx_6000_ada:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=01:30:00
#SBATCH --output=slurm-mj-walk-%j.out
# Inference-only walking / table-to-table carry check in the mjlab whole-body env
# (scripts/mj_walk_check.py): frozen SONIC + released NVIDIA planner, vendored cone-Hessian fix.
set -euo pipefail
project="${WHOLEBODY_REPO:-/data1/users/konstantin.smirnov/DEXTRAH-G1-WholeBody}"
commit="$(git -C "$project" rev-parse HEAD)"
source_snapshot="$(mktemp -d "$project/outputs/walk_source.XXXXXX")"
git -C "$project" archive "$commit" | tar -x -C "$source_snapshot"
export PATH="/data1/users/konstantin.smirnov/venvs/mjlab/bin:$PATH"
export PYTHONPATH="$source_snapshot/third_party/mujoco_warp_cone_fix:$source_snapshot:$source_snapshot/scripts:/data1/users/konstantin.smirnov/G1-SONIC-planner/python-deps"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
cd "$source_snapshot"
echo "Commit: $commit"
python scripts/mj_walk_check.py --checkpoint "${CHECKPOINT:?}" --output "${OUTPUT:?}" ${EXTRA_ARGS:-}
