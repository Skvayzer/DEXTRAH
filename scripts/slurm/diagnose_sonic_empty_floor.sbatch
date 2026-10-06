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
# ONNX Runtime for the released planner is isolated outside the training venv;
# append it so it can never shadow the snapshot or venv packages.
export PYTHONPATH="$source_snapshot${PYTHONPATH:+:$PYTHONPATH}:/data1/users/konstantin.smirnov/G1-SONIC-planner/python-deps"
python -c "import onnxruntime; print('ONNXRUNTIME', onnxruntime.__version__)"
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
    # Kit hangs in app.close() after results (or an error) are saved. Judge the
    # run from metadata.json and kill the process after a short grace period.
    python scripts/diagnose_sonic_empty_floor.py --headless --device cuda:0 \
        --kit_args="--/plugins/carb.tasking.plugin/threadCount=8 --/plugins/omni.tbb.globalcontrol/maxThreadCount=8" \
        --output "$out" --robots upstream revo2 --clock "$clock" --physics "$physics" \
        --case "${cases[@]}" &
    pid=$! status=running deadline=$((SECONDS+1800))
    while kill -0 "$pid" 2>/dev/null; do
        if python -c "import json,sys; m=json.load(open(sys.argv[1])); sys.exit(0 if m.get('completed') or 'error' in m else 1)" \
                "$out/metadata.json" 2>/dev/null; then
            sleep 60; kill -9 "$pid" 2>/dev/null || true; status=killed_after_saving; break
        fi
        if (( SECONDS > deadline )); then kill -9 "$pid" 2>/dev/null || true; status=timeout; break; fi
        sleep 10
    done
    wait "$pid" 2>/dev/null || true
    python -c "import json,sys; m=json.load(open(sys.argv[1])); print('WALK_CONDITION_DONE', sys.argv[2], 'completed=', m.get('completed'), 'exit=', sys.argv[3])" \
        "$out/metadata.json" "$condition" "$status" || echo "WALK_CONDITION_NO_METADATA $condition exit=$status"
done
