#!/usr/bin/env bash
#SBATCH --job-name=adept_fmb_post
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=7-00:00:00
#SBATCH --output=slurm-adept-fmb-post-%j.out

set -euo pipefail

: "${BC_CHECKPOINT:?Set BC_CHECKPOINT to the completed actor-BC checkpoint}"
ADEPT_ENV_PATH="${ADEPT_ENV_PATH:-${HOME}/data1/miniconda3/envs/adept_dextrah}"
REPO_ROOT="${REPO_ROOT:-${HOME}/data1/DEXTRAH-ADEPT}"
TASK="${TASK:-Adept-Kuka-Allegro-FMB-Star}"

cd "${REPO_ROOT}/dextrah_lab/rl_games"
"${ADEPT_ENV_PATH}/bin/python" train.py \
    --headless \
    --task="${TASK}" \
    --adept_posttrain \
    --adept_actor_checkpoint="${BC_CHECKPOINT}" \
    --num_envs="${NUM_ENVS:-4096}" \
    agent.wandb_activate=False
