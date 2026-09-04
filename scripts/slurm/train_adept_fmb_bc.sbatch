#!/usr/bin/env bash
#SBATCH --job-name=adept_fmb_bc
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=3-00:00:00
#SBATCH --output=slurm-adept-fmb-bc-%j.out

set -euo pipefail

: "${TEACHER_CHECKPOINT:?Set TEACHER_CHECKPOINT to the Stage-1 checkpoint}"
ADEPT_ENV_PATH="${ADEPT_ENV_PATH:-${HOME}/data1/miniconda3/envs/adept_dextrah}"
REPO_ROOT="${REPO_ROOT:-${HOME}/data1/DEXTRAH-ADEPT}"
OUTPUT_CHECKPOINT="${OUTPUT_CHECKPOINT:-${REPO_ROOT}/logs/adept_fmb_bc.pth}"
TASK="${TASK:-Adept-Kuka-Allegro-FMB-Star}"

cd "${REPO_ROOT}/dextrah_lab/rl_games"
"${ADEPT_ENV_PATH}/bin/python" adept_actor_bc.py \
    --headless \
    --task="${TASK}" \
    --teacher_checkpoint="${TEACHER_CHECKPOINT}" \
    --output="${OUTPUT_CHECKPOINT}" \
    --num_envs="${NUM_ENVS:-4096}" \
    --iterations=40000
