#!/usr/bin/env bash
#SBATCH --job-name=wb_profile
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx_6000_ada:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=01:30:00
#SBATCH --output=slurm-wb-profile-%j.out
# Profile variants of the whole-body env step (scripts/profile_wholebody_step.py, or SCRIPT=...), one process each.
set -euo pipefail
project="${WHOLEBODY_REPO:-/data1/users/konstantin.smirnov/DEXTRAH-G1-WholeBody}"
commit="$(git -C "$project" rev-parse HEAD)"
source_snapshot="$(mktemp -d "$project/outputs/profile_source.XXXXXX")"
git -C "$project" archive "$commit" | tar -x -C "$source_snapshot"
export PATH="/data1/users/konstantin.smirnov/venvs/mjlab/bin:$PATH"
export PYTHONPATH="$source_snapshot/third_party/mujoco_warp_cone_fix:$source_snapshot:$source_snapshot/scripts"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$source_snapshot"
echo "Commit: $commit"
C="${CHECKPOINT:?}"
IFS=';' read -ra VARIANTS <<< "${VARIANTS:?}"
for v in "${VARIANTS[@]}"; do
    echo "=== variant: $v"
    script="${SCRIPT:-profile_wholebody_step.py}"
    if [[ "$v" == script=* ]]; then script="${v%% *}"; script="${script#script=}"; v="${v#* }"; fi
    python "scripts/$script" --checkpoint "$C" $v || echo "VARIANT_FAILED $v"
done
