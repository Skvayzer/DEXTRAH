#!/usr/bin/env bash
#SBATCH --job-name=g1_sonic_source
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=slurm-sonic-source-%j.out
set -euo pipefail
project="${WHOLEBODY_REPO:?Set WHOLEBODY_REPO}"
export FULLBODY_SOURCE_COMMIT="$(git -C "$project" rev-parse HEAD)"
source_snapshot="$(mktemp -d "$project/outputs/sonic_source_task.XXXXXX")"
git -C "$project" archive "$FULLBODY_SOURCE_COMMIT" | tar -x -C "$source_snapshot"
export PATH="/data1/users/konstantin.smirnov/venvs/g1_sonic/bin:/data1/users/konstantin.smirnov/miniconda3/envs/adept_dextrah/bin:$PATH"
export PYTHONPATH="$source_snapshot${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONUNBUFFERED=1
export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y
cd "$source_snapshot"
nvidia-smi --query-gpu=index,uuid,name,memory.used,memory.total --format=csv
python -m pytest -q tests/test_wholebody_timed_history.py
extra_args=()
if [[ "${EXERCISE_HAND:-0}" == 1 ]]; then
    extra_args+=(--exercise-hand)
fi
python scripts/validate_g1_sonic_source_task.py --headless --device cuda:0 \
    --output "$project/outputs/sonic_source_task_${SLURM_JOB_ID}" \
    --num-envs "${NUM_ENVS:-12}" --seconds "${PROBE_SECONDS:-30}" "${extra_args[@]}"
