#!/usr/bin/env bash
#SBATCH --job-name=g1_transfer_teacher
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=slurm-transfer-teacher-%j.out
set -euo pipefail
project="${WHOLEBODY_REPO:?Set WHOLEBODY_REPO}"
export PATH=/data1/users/konstantin.smirnov/miniconda3/envs/adept_dextrah/bin:$PATH
export PYTHONPATH="$project:/data1/users/konstantin.smirnov/play2perfect${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONUNBUFFERED=1
export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y
cd "$project"
python scripts/capture_g1_transfer_teacher.py --candidate "${TEACHER_CANDIDATE:-bps_baseline}" \
    --seconds "${CAPTURE_SECONDS:-30}" --output "outputs/transfer_teacher_${SLURM_JOB_ID}"
