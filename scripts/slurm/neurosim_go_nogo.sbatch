#!/usr/bin/env bash
#SBATCH --job-name=neurosim
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx_6000_ada:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=110G
#SBATCH --time=03:00:00
#SBATCH --output=slurm-neurosim-%j.out
# Neural simulator go / no-go (feature/g1-neurosim): log carry transitions with a carry v2 expert, then
# train the hand-object world model and report held-out-object metrics.
#   CHECKPOINT=<carry v2 .pth> OUT=<new dir> [COLLECT_ARGS=...] [TRAIN_ARGS=...] [DATA=<existing collection>] sbatch ...
set -uo pipefail
project="${NEUROSIM_REPO:-/data1/users/konstantin.smirnov/DEXTRAH-G1-WholeBody-neurosim}"
commit="$(git -C "$project" rev-parse HEAD)"
src="$(mktemp -d /data1/users/konstantin.smirnov/DEXTRAH-G1-WholeBody/outputs/neurosim_source.XXXXXX)"
git -C "$project" archive "$commit" | tar -x -C "$src"
export PATH="/data1/users/konstantin.smirnov/venvs/mjlab/bin:$PATH"
export PYTHONPATH="$src/third_party/mujoco_warp_cone_fix:$src:$src/scripts:/data1/users/konstantin.smirnov/G1-SONIC-planner/python-deps"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$src"
echo "Commit: $commit (feature/g1-neurosim)"
OUT="${OUT:?}"
DATA="${DATA:-$OUT/data}"
if [[ ! -d "$DATA" ]]; then
  python scripts/neurosim/collect_carry_transitions.py --checkpoint "${CHECKPOINT:?}" --output "$DATA" ${COLLECT_ARGS:-} || { echo NEUROSIM_COLLECT_FAILED; exit 1; }
fi
python scripts/neurosim/train_world_model.py --data "$DATA" --output "$OUT/model" ${TRAIN_ARGS:-} || echo NEUROSIM_TRAIN_FAILED
