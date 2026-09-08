#!/usr/bin/env bash
#SBATCH --job-name=g1_direct_sapg
#SBATCH --partition=batch
#SBATCH --gres=gpu:rtx_6000_ada:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=192G
#SBATCH --time=7-00:00:00
#SBATCH --output=slurm-g1-direct-sapg-%j.out

set -euo pipefail
REPO_ROOT="${REPO_ROOT:-/data1/users/konstantin.smirnov/DEXTRAH-ADEPT}"
TRAIN_PYTHON="${TRAIN_PYTHON:-/data1/users/konstantin.smirnov/miniconda3/envs/adept_dextrah/bin/python}"
NUM_ENVS="${NUM_ENVS:-24576}"
MAX_FRAMES="${MAX_FRAMES:-8000000000}"
SEED="${SEED:-42}"
RUN_NAME="${RUN_NAME:-0_g1_direct_nofabric_nopca_keypoints_seed_${SEED}_${SLURM_JOB_ID}}"

# Scratch control experiment: never auto-resume the failed fabric checkpoint.
# The separate shape-input experiment will have its own explicit task/config.
cd "${REPO_ROOT}/dextrah_lab/rl_games"
"${TRAIN_PYTHON}" train.py --headless \
    --task=G1-Revo2-SimToolReal-Repose-Direct --seed="${SEED}" \
    --num_envs="${NUM_ENVS}" --run_name="${RUN_NAME}" \
    agent.wandb_activate=True agent.params.config.max_frames="${MAX_FRAMES}" \
    agent.params.config.save_frequency=250
