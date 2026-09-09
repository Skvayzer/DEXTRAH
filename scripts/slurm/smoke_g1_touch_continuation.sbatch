#!/usr/bin/env bash
#SBATCH --job-name=g1_touch_smoke
#SBATCH --partition=batch
#SBATCH --gres=gpu:rtx_6000_ada:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-g1-touch-smoke-%j.out
set -euo pipefail
project=/data1/users/konstantin.smirnov/DEXTRAH-BPS128-TOUCH
p2p=/data1/users/konstantin.smirnov/play2perfect
python=/data1/users/konstantin.smirnov/miniconda3/envs/adept_dextrah/bin/python
source_run=/data1/users/konstantin.smirnov/DEXTRAH-BPS128/dextrah_lab/rl_games/logs/rl_games/g1_sapg_bps128_warmstart/0_g1_bps128_warmstart_seed_42_355
export PYTHONPATH="$project:$p2p:$p2p/rl_games${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OMNI_KIT_ACCEPT_EULA=YES
unset DISPLAY
cd "$project"
extra=()
if [[ -n "${CALIBRATION:-}" ]]; then extra+=(--calibration "$CALIBRATION"); fi
if [[ -n "${RESUME:-}" ]]; then extra+=(--resume "$RESUME"); fi
"$python" scripts/train_g1_touch_continuation.py --headless --device cuda:0 \
    --source-run "$source_run" \
    --source-checkpoint "$source_run/nn/last_g1_sapg_bps128_warmstart_frame_8000372736_rew__7400.633_.pth" \
    --output "${RUN_OUTPUT:-$project/outputs/0_touch_smoke_$SLURM_JOB_ID}" \
    --num-envs "${NUM_ENVS:-1200}" --max-epochs "${MAX_EPOCHS:-30}" --max-frames -1 "${extra[@]}"
