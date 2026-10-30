#!/usr/bin/env bash
#SBATCH --job-name=m_cylinders
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx_6000_ada:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=slurm-m-cylinders-%j.out
# Frozen M on flat-ended cylinders (scripts/eval_cylinders_frozen_m.py): upright cylinders with training
# randomization, the same without randomization, and the tools through the same path (control).
set -uo pipefail
project="${WHOLEBODY_REPO:-/data1/users/konstantin.smirnov/DEXTRAH-G1-WholeBody}"
commit="$(git -C "$project" rev-parse HEAD)"
source_snapshot="$(mktemp -d "$project/outputs/eval_source.XXXXXX")"
git -C "$project" archive "$commit" | tar -x -C "$source_snapshot"
export PATH="/data1/users/konstantin.smirnov/venvs/mjlab/bin:$PATH"
export PYTHONPATH="$source_snapshot/third_party/mujoco_warp_cone_fix:$source_snapshot:$source_snapshot/scripts"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$source_snapshot"
echo "Commit: $commit"
CK="${CHECKPOINT:?}"; OUT="${OUTPUT:?}"
COMMON="--checkpoint $CK --bps-cache $project/outputs/bps_cache_cylinders ${EXTRA_ARGS:-}"
python scripts/eval_cylinders_frozen_m.py $COMMON --output "$OUT/cylinders_upright" || echo RUN_FAILED cylinders_upright
python scripts/eval_cylinders_frozen_m.py $COMMON --output "$OUT/cylinders_upright_nodr" --no-randomization || echo RUN_FAILED cylinders_upright_nodr
python scripts/eval_cylinders_frozen_m.py $COMMON --output "$OUT/tools_control" --object-set tools || echo RUN_FAILED tools_control
echo CYLINDER_EVAL_DONE
