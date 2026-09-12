#!/usr/bin/env bash
#SBATCH --job-name=g1_failure_video
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=slurm-g1-failure-render-%j.out
set -euo pipefail
project="${WHOLEBODY_REPO:?Set WHOLEBODY_REPO}"
recording="${FAILURE_RECORDING:?Set FAILURE_RECORDING}"
source_snapshot="$(mktemp -d "$project/outputs/failure_render_source.XXXXXX")"
git -C "$project" archive HEAD | tar -x -C "$source_snapshot"
export PATH="/data1/users/konstantin.smirnov/miniconda3/envs/adept_dextrah/bin:$PATH"
export PYTHONPATH="$source_snapshot${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
unset DISPLAY
cd "$source_snapshot"
python scripts/render_g1_failure.py "$recording"
