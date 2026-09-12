#!/usr/bin/env bash
#SBATCH --job-name=g1_student_contract
#SBATCH --partition=batch
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=slurm-student-contract-%j.out
set -euo pipefail
project="${WHOLEBODY_REPO:?Set WHOLEBODY_REPO}"
export FULLBODY_SOURCE_COMMIT="$(git -C "$project" rev-parse HEAD)"
mkdir -p "$project/outputs"
source_snapshot="$(mktemp -d "$project/outputs/student_source.XXXXXX")"
git -C "$project" archive "$FULLBODY_SOURCE_COMMIT" | tar -x -C "$source_snapshot"
export PATH="/data1/users/konstantin.smirnov/venvs/g1_sonic/bin:/data1/users/konstantin.smirnov/miniconda3/envs/adept_dextrah/bin:$PATH"
export PYTHONPATH="$source_snapshot${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONUNBUFFERED=1
cd "$source_snapshot"
python -m pytest -q tests/test_sonic_student.py tests/test_sonic_history.py tests/test_teacher_data.py tests/test_transfer_labels.py
python scripts/validate_g1_sonic_student.py --output "$project/outputs/student_contract_${SLURM_JOB_ID}"
