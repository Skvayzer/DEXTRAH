#!/usr/bin/env bash
#SBATCH --job-name=g1_bps128_sapg
#SBATCH --partition=batch
#SBATCH --gres=gpu:rtx_6000_ada:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=192G
#SBATCH --time=7-00:00:00
#SBATCH --output=slurm-g1-bps128-%j.out

set -euo pipefail
REPO_ROOT="${REPO_ROOT:-/data1/users/konstantin.smirnov/DEXTRAH-BPS128}"
PLAY2PERFECT_ROOT="${PLAY2PERFECT_ROOT:-/data1/users/konstantin.smirnov/play2perfect}"
TRAIN_PYTHON="${TRAIN_PYTHON:-/data1/users/konstantin.smirnov/miniconda3/envs/adept_dextrah/bin/python}"
SOURCE_RUN="${PLAY2PERFECT_ROOT}/outputs/2026-08-20/10-49-23"
SOURCE_NAME="0_g1_brainco_right_mesh_collision_factorized_multiusd_objects_play_sapg_24576_scratch_20260820_retry1"
SOURCE_CHECKPOINT="${SOURCE_RUN}/${SOURCE_NAME}/nn/${SOURCE_NAME}.pth"
SOURCE_SHA256="dc4454d1db23c90cab0b7bfa44e9c95f5ed2061530f480d3bfd1d7a4948ba5b2"
NUM_ENVS="${NUM_ENVS:-24576}"
MAX_ITERATIONS="${MAX_ITERATIONS:-1000000}"
MAX_FRAMES="${MAX_FRAMES:-8000000000}"
MINIBATCH_SIZE="${MINIBATCH_SIZE:-98304}"
SEED="${SEED:-42}"
WANDB_ENABLED="${WANDB_ENABLED:-True}"
RUN_NAME="${RUN_NAME:-0_g1_bps128_warmstart_seed_${SEED}_${SLURM_JOB_ID}}"
if (( NUM_ENVS % 6 != 0 )); then
    echo 'Six SAPG groups require num_envs divisible by six' >&2
    exit 1
fi
export PYTHONPATH="${REPO_ROOT}:${PLAY2PERFECT_ROOT}:${PLAY2PERFECT_ROOT}/rl_games${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export WANDB_MODE="${WANDB_MODE:-online}"
export HYDRA_FULL_ERROR=1
test -f "${SOURCE_CHECKPOINT}"
git -C "${REPO_ROOT}" log -1 --format='Training commit: %H'
git -C "${PLAY2PERFECT_ROOT}" log -1 --format='Play2Perfect dependency: %H'
echo "Slurm job ${SLURM_JOB_ID}: ${NUM_ENVS} envs; CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=index,uuid,name,memory.total --format=csv
cd "${REPO_ROOT}/dextrah_lab/rl_games"
"${TRAIN_PYTHON}" train.py --headless \
    --task=G1-Revo2-SimToolReal-Repose-BPS128 --seed="${SEED}" \
    --num_envs="${NUM_ENVS}" --run_name="${RUN_NAME}" --max_iterations="${MAX_ITERATIONS}" \
    --bps_warmstart_checkpoint="${SOURCE_CHECKPOINT}" \
    --bps_source_config="${SOURCE_RUN}/.hydra/config.yaml" --bps_source_sha256="${SOURCE_SHA256}" \
    env.bps_cache_dir="${REPO_ROOT}/outputs/bps128_cache" \
    agent.wandb_activate="${WANDB_ENABLED}" \
    agent.params.config.max_frames="${MAX_FRAMES}" \
    agent.params.config.minibatch_size="${MINIBATCH_SIZE}" \
    agent.params.config.central_value_config.minibatch_size="${MINIBATCH_SIZE}" \
    agent.params.config.expl_coef_block_size="$((NUM_ENVS / 6))"
