#!/usr/bin/env bash
#SBATCH --job-name=g1_sonic_distill
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-sonic-distill-%j.out
set -euo pipefail
project="${WHOLEBODY_REPO:?Set WHOLEBODY_REPO}"
export FULLBODY_SOURCE_COMMIT="$(git -C "$project" rev-parse HEAD)"
mkdir -p "$project/outputs"
source_snapshot="$(mktemp -d "$project/outputs/distill_source.XXXXXX")"
git -C "$project" archive "$FULLBODY_SOURCE_COMMIT" | tar -x -C "$source_snapshot"
export PATH="/data1/users/konstantin.smirnov/venvs/g1_sonic/bin:/data1/users/konstantin.smirnov/miniconda3/envs/adept_dextrah/bin:$PATH"
export PYTHONPATH="$source_snapshot${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONUNBUFFERED=1
cd "$source_snapshot"
nvidia-smi --query-gpu=index,uuid,name,memory.used,memory.total --format=csv
python -m pytest -q tests/test_sonic_student.py tests/test_sonic_bootstrap.py tests/test_teacher_data.py
extra_args=()
if [[ "${REUSE_SAPG_FEATURES:-0}" == 1 ]]; then
    extra_args+=(--reuse-sapg-features)
fi
python scripts/train_g1_sonic_distillation.py \
    --capture "$project/outputs/transfer_teacher_459" \
    --standing "$project/outputs/fullbody_probe_470" \
    --contract "$project/outputs/student_contract_475/validation.json" \
    --output "$project/outputs/sonic_distillation_${SLURM_JOB_ID}" \
    --updates "${DISTILL_UPDATES:-1200}" --batch-size "${DISTILL_BATCH:-8}" "${extra_args[@]}"
