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
export FULLBODY_SOURCE_COMMIT="$(git -C "$project" rev-parse HEAD)"
source_snapshot="$(mktemp -d "$project/outputs/fullbody_source.XXXXXX")"
git -C "$project" archive "$FULLBODY_SOURCE_COMMIT" | tar -x -C "$source_snapshot"
environment=/data1/users/konstantin.smirnov/venvs/g1_sonic
export PATH="$environment/bin:/data1/users/konstantin.smirnov/miniconda3/envs/adept_dextrah/bin:$PATH"
export PYTHONPATH="$source_snapshot${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONUNBUFFERED=1
export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y
cd "$source_snapshot"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv
extra_args=()
if [[ "${EXERCISE_HANDS:-0}" == 1 ]]; then
    extra_args+=(--exercise-hands)
fi
if [[ "${FULL_HAND_RANGE:-0}" == 1 ]]; then
    extra_args+=(--full-hand-range)
fi
if [[ "${NO_PHYSICS_REPLICATION:-0}" == 1 ]]; then
    extra_args+=(--no-physics-replication)
fi
case "${HAND_PHYSICS_PRESET:-original}" in
    original) ;;
    thumb_pair_diagnostic) extra_args+=(--mimic-frequency 0 --filter-thumb-housing) ;;
    palm_decomposition) extra_args+=(--mimic-frequency 0 --decompose-palms --palm-shrink-wrap --contact-offset 0.001) ;;
    palm_decomposition16) extra_args+=(--mimic-frequency 0 --decompose-palms --palm-shrink-wrap --contact-offset 0.001 --position-iterations 16) ;;
    palm_decomposition32) extra_args+=(--mimic-frequency 0 --decompose-palms --palm-shrink-wrap --contact-offset 0.001 --position-iterations 32) ;;
    *) printf 'Unknown HAND_PHYSICS_PRESET\n' >&2; exit 2 ;;
esac
if [[ -n "${TEACHER_REFERENCE:-}" ]]; then
    extra_args+=(--reference "$TEACHER_REFERENCE")
fi
if [[ -n "${STUDENT_CHECKPOINT:-}" ]]; then
    extra_args+=(--student-checkpoint "$STUDENT_CHECKPOINT")
fi
python scripts/validate_g1_fullbody.py --headless --device cuda:0 \
    --output "$project/outputs/fullbody_probe_${SLURM_JOB_ID}" \
    --num-envs "${NUM_ENVS:-4}" --seconds "${PROBE_SECONDS:-10}" "${extra_args[@]}"
