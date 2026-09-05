#!/usr/bin/env bash
#SBATCH --job-name=adept_viser
#SBATCH --partition=batch
#SBATCH --gres=gpu:rtx_6000_ada:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-adept-viser-%j.out

set -euo pipefail

ADEPT_ENV_PATH="${ADEPT_ENV_PATH:-${HOME}/data1/miniconda3/envs/adept_dextrah}"
REPO_ROOT="${REPO_ROOT:-${HOME}/data1/DEXTRAH-ADEPT}"
VISER_PORT="${VISER_PORT:-8088}"
NUM_ENVS="${NUM_ENVS:-16}"
ADR_LEVEL="${ADR_LEVEL:-50}"

echo "ADEPT Viser will listen on compute-node loopback port ${VISER_PORT}."
echo "From your laptop run:"
echo "  ssh -N -L ${VISER_PORT}:127.0.0.1:${VISER_PORT} konstantin.smirnov@tl-server-0"
echo "Then open: http://localhost:${VISER_PORT}"

cd "${REPO_ROOT}/dextrah_lab/rl_games"
"${ADEPT_ENV_PATH}/bin/python" -u visualize_adept_repose.py \
    --headless \
    --task=Adept-Kuka-Allegro-Repose \
    --num_envs="${NUM_ENVS}" \
    --adr_level="${ADR_LEVEL}" \
    --viser_host=127.0.0.1 \
    --viser_port="${VISER_PORT}"
