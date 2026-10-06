#!/usr/bin/env bash
#SBATCH --job-name=g1_walk_clocks
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:rtx_6000_ada:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-walk-clocks-%j.out
# Phase 0: frozen SONIC + released planner on empty floor, no SAPG residual.
# Separates SONIC-native 200/50 Hz timing from the 120/60 Hz manipulation
# training clock, and the harness floor/solver from the touch-383 settings.
set -euo pipefail
project="${WHOLEBODY_REPO:?Set WHOLEBODY_REPO}"
export FULLBODY_SOURCE_COMMIT="$(git -C "$project" rev-parse HEAD)"
source_snapshot="$(mktemp -d "$project/outputs/walk_clocks_source.XXXXXX")"
git -C "$project" archive "$FULLBODY_SOURCE_COMMIT" | tar -x -C "$source_snapshot"
export PATH="/data1/users/konstantin.smirnov/venvs/g1_sonic/bin:/data1/users/konstantin.smirnov/miniconda3/envs/adept_dextrah/bin:$PATH"
export PYTHONPATH="$source_snapshot${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1
export PXR_WORK_THREAD_LIMIT=8 OPENBLAS_NUM_THREADS=4
export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y WANDB_MODE=disabled
unset DISPLAY
cd "$source_snapshot"
root="$project/outputs/walk_clocks_${SLURM_JOB_ID}"
mkdir -p "$root"
cases=(idle forward_040 backward_040 left_040 right_040 right_040_yaw90 turn_left turn_right)
for condition in ${CONDITIONS:-native:harness task:harness native:training task:training}; do
    clock="${condition%%:*}" physics="${condition##*:}"
    out="$root/${clock}_${physics}"
    echo "WALK_CONDITION_START clock=$clock physics=$physics output=$out"
    # Kit has previously hung in teardown AFTER saving results; bound each run
    # and judge completion from metadata.json, not the process exit status.
    status=0
    timeout --signal=KILL 30m python scripts/diagnose_sonic_empty_floor.py --headless --device cuda:0 \
        --kit_args="--/plugins/carb.tasking.plugin/threadCount=8 --/plugins/omni.tbb.globalcontrol/maxThreadCount=8" \
        --output "$out" --robots upstream revo2 --clock "$clock" --physics "$physics" \
        --case "${cases[@]}" || status=$?
    python -c "import json,sys; m=json.load(open(sys.argv[1])); print('WALK_CONDITION_DONE', sys.argv[2], 'completed=', m.get('completed'), 'exit=', sys.argv[3])" \
        "$out/metadata.json" "$condition" "$status" || echo "WALK_CONDITION_NO_METADATA $condition exit=$status"
done
