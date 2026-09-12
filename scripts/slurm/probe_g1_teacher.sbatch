#!/usr/bin/env bash
#SBATCH --job-name=g1_teacher_smoke
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:08:00
#SBATCH --output=slurm-teacher-smoke-%j.out
set -euo pipefail
project="${WHOLEBODY_REPO:?Set WHOLEBODY_REPO}"
p2p=/data1/users/konstantin.smirnov/play2perfect
python=/data1/users/konstantin.smirnov/miniconda3/envs/adept_dextrah/bin/python
run=/data1/users/konstantin.smirnov/DEXTRAH-BPS128/dextrah_lab/rl_games/logs/rl_games/g1_sapg_bps128_warmstart/0_g1_bps128_warmstart_seed_42_355
export PYTHONPATH="$project:$p2p:$p2p/rl_games${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OMNI_KIT_ACCEPT_EULA=YES
unset DISPLAY
cd "$project"
# NOT a representative policy evaluation: only test startup and finite steps.
timeout --signal=TERM --kill-after=10 360 "$python" scripts/record_g1_bps_reposing.py \
    --headless --device cuda:0 --run "$run" --play2perfect-root "$p2p" \
    --checkpoint "$run/nn/last_g1_sapg_bps128_warmstart_frame_8000372736_rew__7400.633_.pth" \
    --output "outputs/teacher_smoke_${SLURM_JOB_ID}" \
    --seconds 2 --video-seconds 2 --num-envs 12 --seed 42 --metrics-only
