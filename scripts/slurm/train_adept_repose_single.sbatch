#!/usr/bin/env bash
#SBATCH --job-name=adept_repose
#SBATCH --partition=batch
#SBATCH --gres=gpu:rtx_6000_ada:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=7-00:00:00
#SBATCH --requeue
#SBATCH --output=slurm-adept-repose-train-%j.out

set -euo pipefail

ADEPT_ENV_PATH="${ADEPT_ENV_PATH:-${HOME}/data1/miniconda3/envs/adept_dextrah}"
REPO_ROOT="${REPO_ROOT:-${HOME}/data1/DEXTRAH-ADEPT}"
NUM_ENVS="${NUM_ENVS:-4096}"
MAX_ITERATIONS="${MAX_ITERATIONS:-750000}"
MAX_FRAMES="${MAX_FRAMES:-8000000000}"
SEED="${SEED:-42}"
USE_CUDA_GRAPH="${USE_CUDA_GRAPH:-true}"
WANDB_ACTIVATE="${WANDB_ACTIVATE:-true}"
RUN_NAME="${RUN_NAME:-0_adept_repose_seed_${SEED}}"
AUTO_RESUME="${AUTO_RESUME:-true}"
CHECKPOINT="${CHECKPOINT:-}"

RUN_ROOT="${REPO_ROOT}/dextrah_lab/rl_games/logs/rl_games/adept_kuka_allegro_repose/${RUN_NAME}"
if [[ -z "${CHECKPOINT}" && "${AUTO_RESUME}" == "true" ]]; then
    LAST_CHECKPOINT="${RUN_ROOT}/last/model.pth"
    if [[ -f "${LAST_CHECKPOINT}" ]]; then
        CHECKPOINT="${LAST_CHECKPOINT}"
    fi
fi

RESUME_ARGS=()
if [[ -n "${CHECKPOINT}" ]]; then
    if [[ ! -f "${CHECKPOINT}" ]]; then
        echo "Checkpoint does not exist: ${CHECKPOINT}" >&2
        exit 2
    fi
    RESUME_ARGS+=(--checkpoint="${CHECKPOINT}")
    echo "Resuming ${RUN_NAME} from ${CHECKPOINT}"
else
    echo "Starting fresh run ${RUN_NAME}"
fi

cd "${REPO_ROOT}/dextrah_lab/rl_games"
"${ADEPT_ENV_PATH}/bin/python" train.py \
    --headless \
    --task=Adept-Kuka-Allegro-Repose \
    --seed="${SEED}" \
    --num_envs="${NUM_ENVS}" \
    --max_iterations="${MAX_ITERATIONS}" \
    --run_name="${RUN_NAME}" \
    "${RESUME_ARGS[@]}" \
    agent.wandb_activate="${WANDB_ACTIVATE}" \
    agent.params.config.max_frames="${MAX_FRAMES}" \
    env.use_cuda_graph="${USE_CUDA_GRAPH}"
