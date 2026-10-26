#!/usr/bin/env bash
#SBATCH --job-name=mj_t2t
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx_6000_ada:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=slurm-mj-t2t-%j.out
# Phase 2 inference-only evaluation (scripts/mj_table_to_table.py): table-to-table transfer with a hard
# switch M -> C -> M, released planner route, table B placement with support-gated release.
set -euo pipefail
project="${WHOLEBODY_REPO:-/data1/users/konstantin.smirnov/DEXTRAH-G1-WholeBody}"
commit="$(git -C "$project" rev-parse HEAD)"
source_snapshot="$(mktemp -d "$project/outputs/t2t_source.XXXXXX")"
git -C "$project" archive "$commit" | tar -x -C "$source_snapshot"
export PATH="/data1/users/konstantin.smirnov/venvs/mjlab/bin:$PATH"
export PYTHONPATH="$source_snapshot/third_party/mujoco_warp_cone_fix:$source_snapshot:$source_snapshot/scripts:/data1/users/konstantin.smirnov/G1-SONIC-planner/python-deps"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
cd "$source_snapshot"
echo "Commit: $commit"
python scripts/mj_table_to_table.py --m-checkpoint "${M_CHECKPOINT:?}" --c-checkpoint "${C_CHECKPOINT:?}" --output "${OUTPUT:?}" ${EXTRA_ARGS:-}
