#!/usr/bin/env bash
#SBATCH --job-name=neurosim_v2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx_6000_ada:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=110G
#SBATCH --time=03:00:00
#SBATCH --output=slurm-neurosim-v2-%j.out
# Neural simulator stage 1, v2 (review fixes): coverage collection (30% perturbed worlds: executed-command noise,
# loosened grips, extra object pushes), then training with pre-step inputs, drop-only labels, low-pass twist,
# importance-weighted pre-drop oversampling, k 16, chordal orientation loss; final report incl. smoothed
# criterion 1 and the controlled drop report.
#   CHECKPOINT=<carry v2 .pth> OUT=<new dir> sbatch neurosim_go_nogo_v2.sbatch
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
if [[ ! -d "$OUT/data" ]]; then
  python scripts/neurosim/collect_carry_transitions.py --checkpoint "${CHECKPOINT:?}" --output "$OUT/data" --num-envs 2048 \
    --seconds 90 --action-noise 0.1 --perturb-fraction 0.3 ${COLLECT_ARGS:-} || { echo NEUROSIM_COLLECT_FAILED; exit 1; }
fi
python scripts/neurosim/train_world_model.py --data "$OUT/data" --output "$OUT/model" ${TRAIN_ARGS:-} || echo NEUROSIM_TRAIN_FAILED
