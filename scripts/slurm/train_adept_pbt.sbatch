#!/usr/bin/env bash
#SBATCH --job-name=adept_pbt
#SBATCH --partition=batch
#SBATCH --array=0-15
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=7-00:00:00
#SBATCH --output=slurm-adept-pbt-%A_%a.out

set -euo pipefail

ADEPT_ENV_PATH="${ADEPT_ENV_PATH:-${HOME}/data1/miniconda3/envs/adept_dextrah}"
REPO_ROOT="${REPO_ROOT:-${HOME}/data1/DEXTRAH-ADEPT}"
PBT_SHARED_DIR="${PBT_SHARED_DIR:-${HOME}/data1/adept_pbt}"

cd "${REPO_ROOT}/dextrah_lab/rl_games"
"${ADEPT_ENV_PATH}/bin/python" train.py \
    --headless \
    --task=Adept-Kuka-Allegro-Repose \
    --seed="$((4200 + SLURM_ARRAY_TASK_ID))" \
    --num_envs=4096 \
    --pbt_worker_id="${SLURM_ARRAY_TASK_ID}" \
    --pbt_dir="${PBT_SHARED_DIR}" \
    agent.params.config.full_experiment_name="${SLURM_ARRAY_TASK_ID}_pbt" \
    agent.wandb_activate=False \
    env.use_cuda_graph=True
