#!/usr/bin/env bash
#SBATCH --job-name=g1_sonic_sapg
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --signal=B:USR1@180
#SBATCH --output=slurm-sonic-sapg-%j.out
set -euo pipefail
project="${WHOLEBODY_REPO:?Set WHOLEBODY_REPO}"
export FULLBODY_SOURCE_COMMIT="$(git -C "$project" rev-parse HEAD)"
source_snapshot="$(mktemp -d "$project/outputs/sonic_sapg_source.XXXXXX")"
git -C "$project" archive "$FULLBODY_SOURCE_COMMIT" | tar -x -C "$source_snapshot"
export PATH="/data1/users/konstantin.smirnov/venvs/g1_sonic/bin:/data1/users/konstantin.smirnov/miniconda3/envs/adept_dextrah/bin:$PATH"
export PYTHONPATH="$source_snapshot${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1
export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y WANDB_MODE=online
unset DISPLAY
cd "$source_snapshot"
nvidia-smi --query-gpu=index,uuid,name,memory.used,memory.total --format=csv
extra_args=()
if [[ "${CONTINUOUS:-0}" == 1 ]]; then extra_args+=(--continuous); fi
if [[ -n "${RESUME_CHECKPOINT:-}" ]]; then extra_args+=(--resume "$RESUME_CHECKPOINT"); fi
# Forward the advance time-limit notification to Python for an atomic save at
# an optimizer boundary. The actual Slurm limit remains authoritative.
training_pid=''
trap 'if [[ -n "$training_pid" ]]; then kill -USR1 "$training_pid"; fi' USR1 TERM
python scripts/train_g1_sonic_sapg.py --headless --device cuda:0 \
    --output "$project/outputs/0_sonic_sapg_touch_${SLURM_JOB_ID}" \
    --bootstrap "${STUDENT_CHECKPOINT:?Set STUDENT_CHECKPOINT to the validated touch-derived bootstrap}" \
    --num-envs "${NUM_ENVS:-768}" --max-epochs "${MAX_EPOCHS:-120}" "${extra_args[@]}" &
training_pid=$!
wait "$training_pid" || { result=$?; if kill -0 "$training_pid" 2>/dev/null; then wait "$training_pid"; else exit "$result"; fi; }
