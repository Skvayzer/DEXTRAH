#!/usr/bin/env bash
#SBATCH --job-name=adept_rgb_dagger
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=7-00:00:00
#SBATCH --output=slurm-adept-rgb-%j.out

set -euo pipefail

: "${TEACHER_CHECKPOINT:?Set TEACHER_CHECKPOINT to the post-trained FMB teacher}"
ADEPT_ENV_PATH="${ADEPT_ENV_PATH:-${HOME}/data1/miniconda3/envs/adept_dextrah}"
REPO_ROOT="${REPO_ROOT:-${HOME}/data1/DEXTRAH-ADEPT}"
OUTPUT_CHECKPOINT="${OUTPUT_CHECKPOINT:-${REPO_ROOT}/logs/adept_rgb_student.pth}"
TASK="${TASK:-Adept-Kuka-Allegro-FMB-Star-Vision}"
STAGE1_ARGS=()
if [[ -n "${STAGE1_STUDENT:-}" ]]; then
    STAGE1_ARGS+=(--stage1_student="${STAGE1_STUDENT}")
fi

cd "${REPO_ROOT}/dextrah_lab/rl_games"
"${ADEPT_ENV_PATH}/bin/python" adept_vision_dagger.py \
    --headless \
    --task="${TASK}" \
    --teacher_checkpoint="${TEACHER_CHECKPOINT}" \
    --teacher_stage="${TEACHER_STAGE:-downstream}" \
    --output="${OUTPUT_CHECKPOINT}" \
    --num_envs="${NUM_ENVS:-256}" \
    "${STAGE1_ARGS[@]}"
