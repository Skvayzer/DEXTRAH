#!/usr/bin/env bash
#SBATCH --job-name=g1_grasp_bank
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:rtx_6000_ada:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=slurm-grasp-bank-%j.out
# Inference only: reposing expert captures secure grasps for carry training.
set -euo pipefail
project="${WHOLEBODY_REPO:?Set WHOLEBODY_REPO}"
export FULLBODY_SOURCE_COMMIT="$(git -C "$project" rev-parse HEAD)"
source_snapshot="$(mktemp -d "$project/outputs/grasp_bank_source.XXXXXX")"
git -C "$project" archive "$FULLBODY_SOURCE_COMMIT" | tar -x -C "$source_snapshot"
export PATH="/data1/users/konstantin.smirnov/venvs/g1_sonic/bin:/data1/users/konstantin.smirnov/miniconda3/envs/adept_dextrah/bin:$PATH"
export PYTHONPATH="$source_snapshot${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1
export PXR_WORK_THREAD_LIMIT=8 OPENBLAS_NUM_THREADS=4
export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y WANDB_MODE=disabled
unset DISPLAY
cd "$source_snapshot"
python scripts/build_grasp_bank.py --headless --device cuda:0 \
    --kit_args="--/plugins/carb.tasking.plugin/threadCount=8 --/plugins/omni.tbb.globalcontrol/maxThreadCount=8" \
    --checkpoint "${BANK_CHECKPOINT:?Set BANK_CHECKPOINT}" --output "$project/outputs/grasp_bank_${SLURM_JOB_ID}" \
    --num-envs "${NUM_ENVS:-9216}" --seconds "${BANK_SECONDS:-60}"
