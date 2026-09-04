#!/usr/bin/env bash
#SBATCH --job-name=adept_repose_smoke
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=slurm-adept-repose-%j.out

set -euo pipefail

ADEPT_ENV_PATH="${ADEPT_ENV_PATH:-${HOME}/data1/miniconda3/envs/adept_dextrah}"
REPO_ROOT="${REPO_ROOT:-${HOME}/data1/DEXTRAH-ADEPT}"

cd "${REPO_ROOT}/dextrah_lab/rl_games"
"${ADEPT_ENV_PATH}/bin/python" train.py \
    --headless \
    --task=Adept-Kuka-Allegro-Repose \
    --num_envs=16 \
    agent.params.config.max_epochs=1 \
    agent.params.config.minibatch_size=64 \
    agent.params.config.central_value_config.minibatch_size=64 \
    agent.wandb_activate=False \
    env.objects_dir=visdex_objects \
    env.use_cuda_graph=False
