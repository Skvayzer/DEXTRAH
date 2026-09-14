#!/usr/bin/env bash
#SBATCH --job-name=g1_sonic_video
#SBATCH --partition=batch
#SBATCH --gres=gpu:rtx_6000_ada:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=slurm-sonic-video-%j.out
set -euo pipefail
project="${WHOLEBODY_REPO:?Set WHOLEBODY_REPO}"
export FULLBODY_SOURCE_COMMIT="$(git -C "$project" rev-parse HEAD)"
source_snapshot="$(mktemp -d "$project/outputs/sonic_video_source.XXXXXX")"
git -C "$project" archive "$FULLBODY_SOURCE_COMMIT" | tar -x -C "$source_snapshot"
export PATH="/data1/users/konstantin.smirnov/venvs/g1_sonic/bin:/data1/users/konstantin.smirnov/miniconda3/envs/adept_dextrah/bin:$PATH"
export PYTHONPATH="$source_snapshot${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1
export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y WANDB_MODE=disabled
unset DISPLAY
cd "$source_snapshot"
recording="$project/outputs/sonic_checkpoint_video_${SLURM_JOB_ID}"
python scripts/record_g1_sonic_checkpoint.py --headless --device cuda:0 \
    --checkpoint "${RECORD_CHECKPOINT:?Set RECORD_CHECKPOINT}" --output "$recording" \
    --num-envs "${NUM_ENVS:-1200}" --seconds "${RECORD_SECONDS:-60}"
export PATH="/data1/users/konstantin.smirnov/miniconda3/envs/adept_dextrah/bin:$PATH"
for family in hammer brush spatula; do
    python scripts/render_g1_sonic_checkpoint.py "$recording/$family"
done
