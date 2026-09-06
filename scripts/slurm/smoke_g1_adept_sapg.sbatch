#!/usr/bin/env bash
#SBATCH --job-name=g1_adept_smoke
#SBATCH --partition=batch
#SBATCH --gres=gpu:rtx_6000_ada:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=slurm-g1-adept-smoke-%j.out

set -euo pipefail

ADEPT_ENV_PATH="${ADEPT_ENV_PATH:-${HOME}/data1/miniconda3/envs/adept_dextrah}"
REPO_ROOT="${REPO_ROOT:-${HOME}/data1/DEXTRAH-ADEPT}"
NUM_ENVS="${NUM_ENVS:-64}"
NUM_ASSETS_PER_TYPE="${NUM_ASSETS_PER_TYPE:-1}"
MAX_ITERATIONS="${MAX_ITERATIONS:-40}"
BLOCK_SIZE="${BLOCK_SIZE:-32}"
MINIBATCH_SIZE="${MINIBATCH_SIZE:-1024}"
RUN_NAME="${RUN_NAME:-0_g1_adept_smoke_${SLURM_JOB_ID}}"

cd "${REPO_ROOT}/dextrah_lab/rl_games"
"${ADEPT_ENV_PATH}/bin/python" train.py \
    --headless \
    --task=Adept-G1-Revo2-SimToolReal-Repose \
    --num_envs="${NUM_ENVS}" \
    --max_iterations="${MAX_ITERATIONS}" \
    --run_name="${RUN_NAME}" \
    env.assets.num_assets_per_type="${NUM_ASSETS_PER_TYPE}" \
    agent.wandb_activate=False \
    agent.params.config.minibatch_size="${MINIBATCH_SIZE}" \
    agent.params.config.central_value_config.minibatch_size="${MINIBATCH_SIZE}" \
    agent.params.config.expl_coef_block_size="${BLOCK_SIZE}" \
    agent.params.config.save_frequency=1000
