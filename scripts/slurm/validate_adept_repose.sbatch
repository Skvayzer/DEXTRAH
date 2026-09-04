#!/usr/bin/env bash
#SBATCH --job-name=adept_validate
#SBATCH --partition=batch
#SBATCH --gres=gpu:rtx_6000_ada:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=slurm-adept-validation-%j.out

set -euo pipefail

ADEPT_ENV_PATH="${ADEPT_ENV_PATH:-${HOME}/data1/miniconda3/envs/adept_dextrah}"
REPO_ROOT="${REPO_ROOT:-${HOME}/data1/DEXTRAH-ADEPT}"
NUM_ENVS="${NUM_ENVS:-16}"
ROLLOUT_STEPS="${ROLLOUT_STEPS:-64}"
CONTACT_STEPS="${CONTACT_STEPS:-32}"
CONTACT_GAP="${CONTACT_GAP:-0.02}"
CONTACT_VELOCITY="${CONTACT_VELOCITY:--1.0}"

cd "${REPO_ROOT}/dextrah_lab/rl_games"
"${ADEPT_ENV_PATH}/bin/python" -u validate_adept_repose.py \
    --headless \
    --task=Adept-Kuka-Allegro-Repose \
    --num_envs="${NUM_ENVS}" \
    --rollout_steps="${ROLLOUT_STEPS}" \
    --contact_steps="${CONTACT_STEPS}" \
    --contact_gap="${CONTACT_GAP}" \
    --contact_velocity="${CONTACT_VELOCITY}"
