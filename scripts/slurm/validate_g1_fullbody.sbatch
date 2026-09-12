#!/usr/bin/env bash
#SBATCH --job-name=g1_fullbody_probe
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=slurm-fullbody-probe-%j.out
set -euo pipefail
project="${WHOLEBODY_REPO:?Set WHOLEBODY_REPO}"
environment=/data1/users/konstantin.smirnov/venvs/g1_sonic
export PATH="$environment/bin:/data1/users/konstantin.smirnov/miniconda3/envs/adept_dextrah/bin:$PATH"
export PYTHONPATH="$project${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONUNBUFFERED=1
export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y
cd "$project"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv
extra_args=()
if [[ "${EXERCISE_HANDS:-0}" == 1 ]]; then
    extra_args+=(--exercise-hands)
fi
python scripts/validate_g1_fullbody.py --headless --device cuda:0 \
    --output "outputs/fullbody_probe_${SLURM_JOB_ID}" \
    --num-envs "${NUM_ENVS:-4}" --seconds "${PROBE_SECONDS:-10}" "${extra_args[@]}"
