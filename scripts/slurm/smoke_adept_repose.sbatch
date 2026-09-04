#!/usr/bin/env bash
#SBATCH --job-name=adept_repose_smoke
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=slurm-adept-repose-%j.out

set -euo pipefail

ADEPT_ENV_PATH="${ADEPT_ENV_PATH:-${HOME}/data1/miniconda3/envs/adept_dextrah}"
REPO_ROOT="${REPO_ROOT:-${HOME}/data1/DEXTRAH-ADEPT}"
NUM_ENVS="${NUM_ENVS:-16}"
MAX_ITERATIONS="${MAX_ITERATIONS:-1}"
USE_CUDA_GRAPH="${USE_CUDA_GRAPH:-false}"
MINIBATCH_SIZE="${MINIBATCH_SIZE:-64}"
RUN_NAME="${RUN_NAME:-0_adept_smoke_${SLURM_JOB_ID}}"
CHECKPOINT="${CHECKPOINT:-}"

RESUME_ARGS=()
if [[ -n "${CHECKPOINT}" ]]; then
    if [[ ! -f "${CHECKPOINT}" ]]; then
        echo "Checkpoint does not exist: ${CHECKPOINT}" >&2
        exit 2
    fi
    RESUME_ARGS+=(--checkpoint="${CHECKPOINT}")
fi

cd "${REPO_ROOT}/dextrah_lab/rl_games"
"${ADEPT_ENV_PATH}/bin/python" train.py \
    --headless \
    --task=Adept-Kuka-Allegro-Repose \
    --num_envs="${NUM_ENVS}" \
    --max_iterations="${MAX_ITERATIONS}" \
    --run_name="${RUN_NAME}" \
    "${RESUME_ARGS[@]}" \
    agent.params.config.minibatch_size="${MINIBATCH_SIZE}" \
    agent.params.config.central_value_config.minibatch_size="${MINIBATCH_SIZE}" \
    agent.wandb_activate=False \
    env.use_cuda_graph="${USE_CUDA_GRAPH}"
