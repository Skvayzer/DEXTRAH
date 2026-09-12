#!/usr/bin/env bash
#SBATCH --job-name=g1_ground_touch
#SBATCH --partition=batch
#SBATCH --gres=gpu:rtx_6000_ada:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=00:10:00
#SBATCH --output=slurm-ground-touch-%j.out
set -euo pipefail
project="${WHOLEBODY_REPO:?Set WHOLEBODY_REPO}"
source_snapshot="$(mktemp -d "$project/outputs/ground_touch_source.XXXXXX")"
git -C "$project" archive HEAD | tar -x -C "$source_snapshot"
export PATH="/data1/users/konstantin.smirnov/venvs/g1_sonic/bin:/data1/users/konstantin.smirnov/miniconda3/envs/adept_dextrah/bin:$PATH"
export PYTHONPATH="$source_snapshot${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONUNBUFFERED=1
export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y
unset DISPLAY
cd "$source_snapshot"
python scripts/validate_ground_touch_filter.py --headless --device cuda:0 \
    --output "$project/outputs/ground_touch_${SLURM_JOB_ID}"
