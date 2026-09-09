#!/usr/bin/env bash
#SBATCH --job-name=g1_bps_reposing_render
#SBATCH --partition=batch
#SBATCH --gres=gpu:rtx_6000_ada:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-bps-render-%j.out
set -euo pipefail
project="${RECORDING_REPO:?Set RECORDING_REPO to the staged source copy}"
recordings="${RECORDING_OUTPUT:-$project/outputs/reposing}"
python=/data1/users/konstantin.smirnov/miniconda3/envs/adept_dextrah/bin/python
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
unset DISPLAY
cd "$project"
if [[ "${RENDER_SMOKE:-0}" == 1 ]]; then
    "$python" scripts/render_g1_bps_reposing.py "$recordings/clips/hammer" \
        --video-name "${SMOKE_VIDEO_NAME:-smoke.mp4}" --limit-frames 90
else
    for family in hammer spatula brush eraser; do
        "$python" scripts/render_g1_bps_reposing.py "$recordings/clips/$family" \
            --video-name "sapg-bps128-g1-$family-60s.mp4"
    done
fi
