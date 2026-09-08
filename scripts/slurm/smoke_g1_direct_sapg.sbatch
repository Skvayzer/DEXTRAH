#!/usr/bin/env bash
#SBATCH --job-name=g1_direct_smoke
#SBATCH --partition=batch
#SBATCH --gres=gpu:rtx_6000_ada:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=slurm-g1-direct-smoke-%j.out

set -euo pipefail
REPO_ROOT="${REPO_ROOT:-/data1/users/konstantin.smirnov/DEXTRAH-ADEPT}"
TRAIN_PYTHON="${TRAIN_PYTHON:-/data1/users/konstantin.smirnov/miniconda3/envs/adept_dextrah/bin/python}"
cd "${REPO_ROOT}"
"${TRAIN_PYTHON}" -m pytest -q tests/test_g1_direct_task_contract.py
"${TRAIN_PYTHON}" scripts/validate_g1_direct_control.py --headless --num_envs=96
cd "${REPO_ROOT}/dextrah_lab/rl_games"
"${TRAIN_PYTHON}" train.py --headless \
    --task=G1-Revo2-SimToolReal-Repose-Direct --seed=42 --num_envs=96 \
    --max_iterations=2 --run_name="0_g1_direct_smoke_${SLURM_JOB_ID}" \
    env.assets.num_assets_per_type=1 agent.wandb_activate=False \
    agent.params.config.minibatch_size=1536 \
    agent.params.config.central_value_config.minibatch_size=1536 \
    agent.params.config.expl_coef_block_size=16
