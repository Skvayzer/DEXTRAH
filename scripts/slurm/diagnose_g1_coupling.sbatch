#!/usr/bin/env bash
#SBATCH --job-name=g1_coupling_diagnostic
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=slurm-coupling-%j.out
set -euo pipefail
project="${WHOLEBODY_REPO:?Set WHOLEBODY_REPO}"
environment=/data1/users/konstantin.smirnov/venvs/g1_sonic
export PATH="$environment/bin:/data1/users/konstantin.smirnov/miniconda3/envs/adept_dextrah/bin:$PATH"
export PYTHONPATH="$project${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONUNBUFFERED=1
export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y
cd "$project"
# One GPU, sequential fresh processes. Expected diagnostic failures are retained;
# each child writes validation.json/failure.json, not hidden by a successful sweep.
for variant in rigid rigid32 compliant32 no_self low_gain; do
    extra=()
    case "$variant" in
        rigid) extra+=(--mimic-frequency 0) ;;
        rigid32) extra+=(--mimic-frequency 0 --position-iterations 32) ;;
        compliant32) extra+=(--position-iterations 32) ;;
        no_self) extra+=(--disable-self-collisions) ;;
        low_gain) extra+=(--hand-stiffness 20 --hand-damping 0.2) ;;
    esac
    if python scripts/validate_g1_fullbody.py --headless --device cuda:0 \
        --output "outputs/coupling_${SLURM_JOB_ID}_${variant}" \
        --num-envs 4 --seconds 6 --exercise-hands "${extra[@]}"; then
        printf 'DIAGNOSTIC_PASS %s\n' "$variant"
    else
        printf 'DIAGNOSTIC_FAIL %s\n' "$variant"
    fi
done
