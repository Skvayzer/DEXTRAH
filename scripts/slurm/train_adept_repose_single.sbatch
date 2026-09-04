#!/usr/bin/env bash
#SBATCH --job-name=adept_repose
#SBATCH --partition=batch
#SBATCH --gres=gpu:rtx_6000_ada:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=7-00:00:00
#SBATCH --output=slurm-adept-repose-train-%j.out

set -euo pipefail

ADEPT_ENV_PATH="${ADEPT_ENV_PATH:-${HOME}/data1/miniconda3/envs/adept_dextrah}"
REPO_ROOT="${REPO_ROOT:-${HOME}/data1/DEXTRAH-ADEPT}"
NUM_ENVS="${NUM_ENVS:-4096}"
MAX_ITERATIONS="${MAX_ITERATIONS:-750000}"
SEED="${SEED:-42}"
USE_CUDA_GRAPH="${USE_CUDA_GRAPH:-false}"
WANDB_ACTIVATE="${WANDB_ACTIVATE:-true}"
RUN_NAME="${RUN_NAME:-0_adept_repose_seed_${SEED}}"

cd "${REPO_ROOT}/dextrah_lab/rl_games"
"${ADEPT_ENV_PATH}/bin/python" train.py \
    --headless \
    --task=Adept-Kuka-Allegro-Repose \
    --seed="${SEED}" \
    --num_envs="${NUM_ENVS}" \
    --max_iterations="${MAX_ITERATIONS}" \
    --run_name="${RUN_NAME}" \
    agent.wandb_activate="${WANDB_ACTIVATE}" \
    env.use_cuda_graph="${USE_CUDA_GRAPH}"
