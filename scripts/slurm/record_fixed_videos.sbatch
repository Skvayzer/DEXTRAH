#!/usr/bin/env bash
#SBATCH --job-name=fixed_videos
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx_6000_ada:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=slurm-fixed-videos-%j.out
# Videos of a fixed-body mjlab checkpoint (scripts/record_fixed_policy_videos.py):
# same simulator settings as training (cone-Hessian fix, arm armature), EGL rendering.
set -euo pipefail
project="${WHOLEBODY_REPO:-/data1/users/konstantin.smirnov/DEXTRAH-G1-WholeBody}"
commit="$(git -C "$project" rev-parse HEAD)"
source_snapshot="$(mktemp -d "$project/outputs/videos_source.XXXXXX")"
git -C "$project" archive "$commit" | tar -x -C "$source_snapshot"
export PATH="/data1/users/konstantin.smirnov/venvs/mjlab/bin:$PATH"
export PYTHONPATH="$source_snapshot/third_party/mujoco_warp_cone_fix:$source_snapshot"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
cd "$source_snapshot"
echo "Commit: $commit"
python scripts/record_fixed_policy_videos.py --checkpoint "${CHECKPOINT:?}" --output "${OUTPUT:?}" ${EXTRA_ARGS:-}
