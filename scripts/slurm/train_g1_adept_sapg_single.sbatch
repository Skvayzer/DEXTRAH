#!/usr/bin/env bash
#SBATCH --job-name=g1_adept_sapg
#SBATCH --partition=batch
#SBATCH --gres=gpu:rtx_6000_ada:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=192G
#SBATCH --time=7-00:00:00
#SBATCH --requeue
#SBATCH --output=slurm-g1-adept-sapg-%j.out

set -euo pipefail

ADEPT_ENV_PATH="${ADEPT_ENV_PATH:-${HOME}/data1/miniconda3/envs/adept_dextrah}"
REPO_ROOT="${REPO_ROOT:-${HOME}/data1/DEXTRAH-ADEPT}"
NUM_ENVS="${NUM_ENVS:-24576}"
MAX_ITERATIONS="${MAX_ITERATIONS:-1000000}"
MAX_FRAMES="${MAX_FRAMES:-8000000000}"
SAVE_FREQUENCY="${SAVE_FREQUENCY:-250}"
SEED="${SEED:-42}"
WANDB_ACTIVATE="${WANDB_ACTIVATE:-true}"
RUN_NAME="${RUN_NAME:-0_g1_adept_sapg_simtoolreal_absolute_v2_seed_${SEED}}"
AUTO_RESUME="${AUTO_RESUME:-true}"
CHECKPOINT="${CHECKPOINT:-}"

RUN_ROOT="${REPO_ROOT}/dextrah_lab/rl_games/logs/rl_games/g1_adept_sapg_simtoolreal/${RUN_NAME}"
if [[ -z "${CHECKPOINT}" && "${AUTO_RESUME}" == "true" && -d "${RUN_ROOT}/nn" ]]; then
    CHECKPOINT="$({
        find "${RUN_ROOT}/nn" -maxdepth 1 -type f -name 'last_*.pth' \
            -printf '%T@ %p\n' | sort -nr | head -n 1 | cut -d' ' -f2-
    } || true)"
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
    --task=Adept-G1-Revo2-SimToolReal-Repose \
    --seed="${SEED}" \
    --num_envs="${NUM_ENVS}" \
    --max_iterations="${MAX_ITERATIONS}" \
    --run_name="${RUN_NAME}" \
    "${RESUME_ARGS[@]}" \
    agent.wandb_activate="${WANDB_ACTIVATE}" \
    agent.params.config.max_frames="${MAX_FRAMES}" \
    agent.params.config.save_frequency="${SAVE_FREQUENCY}"
