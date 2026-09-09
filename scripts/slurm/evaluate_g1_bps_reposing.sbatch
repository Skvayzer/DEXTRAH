#!/usr/bin/env bash
#SBATCH --job-name=g1_bps_reposing_eval
#SBATCH --partition=batch
#SBATCH --gres=gpu:rtx_6000_ada:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-bps-reposing-%j.out
set -euo pipefail
project="${RECORDING_REPO:?Set RECORDING_REPO to the staged source copy}"
p2p=/data1/users/konstantin.smirnov/play2perfect
python=/data1/users/konstantin.smirnov/miniconda3/envs/adept_dextrah/bin/python
run=/data1/users/konstantin.smirnov/DEXTRAH-BPS128/dextrah_lab/rl_games/logs/rl_games/g1_sapg_bps128_warmstart/0_g1_bps128_warmstart_seed_42_355
export PYTHONPATH="$project:$p2p:$p2p/rl_games${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OMNI_KIT_ACCEPT_EULA=YES
unset DISPLAY
cd "$project"
"$python" -m pytest -q tests/test_bps_evaluation.py tests/test_bps_bank.py tests/test_object_bps.py
"$python" scripts/record_g1_bps_reposing.py --headless --device cuda:0 \
    --run "$run" --play2perfect-root "$p2p" \
    --checkpoint "$run/nn/last_g1_sapg_bps128_warmstart_frame_8000372736_rew__7400.633_.pth" \
    --output "${RECORDING_OUTPUT:-$project/outputs/reposing}" \
    --seconds 120 --video-seconds 60 --fps 30 --num-envs 1200 --seed 42
