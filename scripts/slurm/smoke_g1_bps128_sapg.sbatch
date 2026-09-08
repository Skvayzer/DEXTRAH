#!/usr/bin/env bash
#SBATCH --job-name=g1_bps128_smoke
#SBATCH --partition=batch
#SBATCH --gres=gpu:rtx_6000_ada:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --output=slurm-g1-bps128-smoke-%j.out
set -euo pipefail
export REPO_ROOT="${REPO_ROOT:-/data1/users/konstantin.smirnov/DEXTRAH-BPS128}"
export PLAY2PERFECT_ROOT="${PLAY2PERFECT_ROOT:-/data1/users/konstantin.smirnov/play2perfect}"
export PYTHONPATH="${REPO_ROOT}:${PLAY2PERFECT_ROOT}:${PLAY2PERFECT_ROOT}/rl_games${PYTHONPATH:+:${PYTHONPATH}}"
export NUM_ENVS=1200
export MINIBATCH_SIZE=19200
export MAX_ITERATIONS=50
export WANDB_ENABLED=False
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
TRAIN_PYTHON="${TRAIN_PYTHON:-/data1/users/konstantin.smirnov/miniconda3/envs/adept_dextrah/bin/python}"
cd "${REPO_ROOT}"
"${TRAIN_PYTHON}" -m pytest -q tests/test_object_bps.py tests/test_bps_bank.py tests/test_bps_warmstart.py tests/test_g1_direct_task_contract.py
"${TRAIN_PYTHON}" scripts/validate_g1_bps128.py --headless --num_envs=1200
bash scripts/slurm/train_g1_bps128_sapg.sbatch
