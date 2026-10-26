#!/usr/bin/env bash
#SBATCH --job-name=g1_sonic_sapg
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx_6000_ada:1
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
launch_args=(--headless)
if [[ "${BACKEND:-isaac}" == mjlab ]]; then
    # MuJoCo Warp port (dextrah_lab/mj): separate venv, no Kit, no AppLauncher.
    export PATH="/data1/users/konstantin.smirnov/venvs/mjlab/bin:$PATH"
    launch_args=(--backend mjlab)
    if [[ -n "${MJ_NCONMAX:-}" ]]; then extra_args+=(--mj-nconmax "$MJ_NCONMAX"); fi
    if [[ -n "${MJ_NJMAX:-}" ]]; then extra_args+=(--mj-njmax "$MJ_NJMAX"); fi
    if [[ "${MJ_NAN_CAPTURE:-0}" == 1 ]]; then extra_args+=(--mj-nan-capture); fi
    if [[ "${MJ_FAST_STEP:-0}" == 1 ]]; then extra_args+=(--mj-fast-step); fi
    if [[ "${MJ_CONE_HESSIAN_FIX:-0}" == 1 ]]; then
        # Vendored mujoco_warp 3.11.0 with the elliptic cone Hessian fix (third_party/.../README.md).
        export PYTHONPATH="$source_snapshot/third_party/mujoco_warp_cone_fix:$PYTHONPATH"
        extra_args+=(--mj-cone-hessian-fix)
    fi
else
    python -m pytest -q tests/test_resume_optimizer_devices.py tests/test_training_stack_watchdog.py tests/test_sapg_boundary_gc.py
fi
if [[ "${FROZEN_PRETRAINED_SONIC:-0}" == 1 ]]; then
    if [[ "${BACKEND:-isaac}" != mjlab ]]; then
        python -m pytest -q tests/test_frozen_sonic_sapg.py tests/test_sonic_source_actions.py
    fi
    extra_args+=(--frozen-pretrained-sonic --adapter-learning-rate "${ADAPTER_LEARNING_RATE:-0.0003}")
else
    extra_args+=(--bootstrap "${STUDENT_CHECKPOINT:?Set STUDENT_CHECKPOINT to the validated touch-derived bootstrap}")
fi
if [[ -n "${SIM_WORKER_THREADS:-}" ]]; then
    if [[ ! "$SIM_WORKER_THREADS" =~ ^[1-9][0-9]*$ ]]; then
        echo "SIM_WORKER_THREADS must be a positive integer" >&2
        exit 2
    fi
    extra_args+=("--kit_args=--/plugins/carb.tasking.plugin/threadCount=$SIM_WORKER_THREADS --/plugins/omni.tbb.globalcontrol/maxThreadCount=$SIM_WORKER_THREADS")
fi
if [[ "${MEMORY_TRACE:-0}" == 1 ]]; then extra_args+=(--memory-trace); fi
if [[ "${LEAN_SAPG_BATCHES:-0}" == 1 ]]; then extra_args+=(--lean-sapg-batches); fi
if [[ -n "${DUAL_CLIP:-}" ]]; then extra_args+=(--dual-clip "$DUAL_CLIP"); fi
if [[ -n "${GRASP_REWARD_SCALE:-}" ]]; then extra_args+=(--grasp-reward-scale "$GRASP_REWARD_SCALE"); fi
if [[ "${HAND_SURFACE_ARM_LINKS:-0}" == 1 ]]; then extra_args+=(--hand-surface-arm-links); fi
if [[ "${HAND_SURFACE_GOAL:-0}" == 1 ]]; then extra_args+=(--hand-surface-goal); fi
if [[ -n "${OBJECT_ENERGY_SCALE:-}" ]]; then extra_args+=(--object-energy-scale "$OBJECT_ENERGY_SCALE"); fi
if [[ -n "${SONIC_ACTION_PENALTY_SCALE:-}" ]]; then extra_args+=(--sonic-action-penalty-scale "$SONIC_ACTION_PENALTY_SCALE"); fi
if [[ -n "${SONIC_ACTION_BOUND:-}" ]]; then extra_args+=(--sonic-action-bound "$SONIC_ACTION_BOUND"); fi
if [[ -n "${FALL_PENALTY:-}" ]]; then extra_args+=(--fall-penalty "$FALL_PENALTY"); fi
if [[ "${BOUNDARY_GC_FREEZE:-0}" == 1 ]]; then extra_args+=(--boundary-gc-freeze); fi
if [[ -n "${BOUNDARY_GC_INTERVAL:-}" ]]; then
    extra_args+=(--boundary-gc-interval "$BOUNDARY_GC_INTERVAL")
fi
if [[ -n "${MEMORY_GC_PROBE_EPOCH:-}" ]]; then
    extra_args+=(--memory-gc-probe-epoch "$MEMORY_GC_PROBE_EPOCH")
fi
if [[ -n "${NUMERICAL_FAILURE_MODE:-}" ]]; then
    extra_args+=(--numerical-failure-mode "$NUMERICAL_FAILURE_MODE")
fi
if [[ "${CONTINUOUS:-0}" == 1 ]]; then extra_args+=(--continuous); fi
if [[ "${CARRY:-0}" == 1 ]]; then
    python -m pytest -q tests/test_carry.py
    extra_args+=(--carry --carry-clips "${CARRY_CLIPS:?Set CARRY_CLIPS}" --grasp-bank "${GRASP_BANK:?Set GRASP_BANK}")
    extra_args+=(--critic-warmup-epochs "${CRITIC_WARMUP_EPOCHS:-0}")
fi
if [[ "${MJ_CARRY:-0}" == 1 ]]; then
    # mjlab carry expert (dextrah_lab/mj/carry_env.py, after DexCompose)
    python -m pytest -q tests/test_carry_compose.py
    extra_args+=(--mj-carry --carry-clips "${CARRY_CLIPS:?Set CARRY_CLIPS}" --carry-holding-state "${CARRY_HOLDING_STATE:?Set CARRY_HOLDING_STATE}")
    extra_args+=(--carry-stage "${CARRY_STAGE:-1}" --carry-episode-seconds "${CARRY_EPISODE_SECONDS:-20}")
    extra_args+=(--critic-warmup-epochs "${CRITIC_WARMUP_EPOCHS:-0}" --carry-settings "${CARRY_SETTINGS:-{\}}")
    if [[ -n "${CARRY_LEVEL:-}" ]]; then extra_args+=(--carry-level "$CARRY_LEVEL"); fi
fi
if [[ "${NO_TABLE_BODY_SUPPORT:-0}" == 1 ]]; then extra_args+=(--no-table-body-support); fi
if [[ "${BODY_TABLE_CONTACT_TERMINATION:-0}" == 1 ]]; then extra_args+=(--body-table-contact-termination); fi
if [[ "${PER_ENV_OBJECT_PHYSICS:-0}" == 1 ]]; then extra_args+=(--per-env-object-physics); fi
if [[ "${SPAWN_ON_TABLE:-0}" == 1 ]]; then extra_args+=(--spawn-on-table); fi
if [[ "${BODY_TABLE_CONTACT_MONITOR_ONLY:-0}" == 1 ]]; then extra_args+=(--body-table-contact-monitor-only); fi
if [[ -n "${INIT_WEIGHTS:-}" ]]; then extra_args+=(--init-weights "$INIT_WEIGHTS"); fi
if [[ -n "${RESUME_CHECKPOINT:-}" ]]; then extra_args+=(--resume "$RESUME_CHECKPOINT"); fi
if [[ "${RESET_BEST:-0}" == 1 ]]; then extra_args+=(--reset-best); fi
if [[ -n "${BEST_DELAY_EPOCHS:-}" ]]; then extra_args+=(--best-delay-epochs "$BEST_DELAY_EPOCHS"); fi
if [[ -n "${TEACHER_CHECKPOINT:-}" ]]; then extra_args+=(--teacher-checkpoint "$TEACHER_CHECKPOINT"); fi
if [[ -n "${TASK_RUN:-}" ]]; then extra_args+=(--task-run "$TASK_RUN"); fi
if [[ -n "${MINIBATCH_SIZE:-}" ]]; then extra_args+=(--minibatch-size "$MINIBATCH_SIZE"); fi
if [[ "${SGS:-0}" == 1 ]]; then
    # Success Guided Sampling (dextrah_lab/sgs, docs/sgs_success_guided_sampling.md).
    extra_args+=(--sgs --sgs-families "${SGS_FAMILIES:-reaching,stable_grasp,near_goal}")
    if [[ -n "${SGS_NUM_CONFIGS:-}" ]]; then extra_args+=(--sgs-num-configs "$SGS_NUM_CONFIGS"); fi
    if [[ -n "${SGS_EVAL_EVERY:-}" ]]; then extra_args+=(--sgs-eval-every "$SGS_EVAL_EVERY"); fi
    if [[ -n "${SGS_HARVEST_MAX_STEPS:-}" ]]; then extra_args+=(--sgs-harvest-max-steps "$SGS_HARVEST_MAX_STEPS"); fi
    if [[ "${SGS_VALIDATE_SPAWNS:-0}" == 1 ]]; then extra_args+=(--sgs-validate-spawns); fi
    if [[ -n "${SGS_REHARVEST_EVERY:-}" ]]; then extra_args+=(--sgs-reharvest-every "$SGS_REHARVEST_EVERY"); fi
    if [[ -n "${SGS_REHARVEST_STEPS:-}" ]]; then extra_args+=(--sgs-reharvest-steps "$SGS_REHARVEST_STEPS"); fi
    if [[ -n "${SGS_REHARVEST_FRACTION:-}" ]]; then extra_args+=(--sgs-reharvest-fraction "$SGS_REHARVEST_FRACTION"); fi
    if [[ -n "${SGS_EDIT_FRACTION:-}" ]]; then extra_args+=(--sgs-edit-fraction "$SGS_EDIT_FRACTION"); fi
    if [[ -n "${SGS_HARD_OBJECT_SHARE:-}" ]]; then extra_args+=(--sgs-hard-object-share "$SGS_HARD_OBJECT_SHARE"); fi
    if [[ -n "${SGS_HELD_MAX_BODY_ACTION:-}" ]]; then extra_args+=(--sgs-held-max-body-action "$SGS_HELD_MAX_BODY_ACTION"); fi
    if [[ -n "${SGS_EVAL_POOL:-}" ]]; then extra_args+=(--sgs-eval-pool "$SGS_EVAL_POOL"); fi
fi
if [[ -n "${BEST_METRIC:-}" ]]; then extra_args+=(--best-metric "$BEST_METRIC"); fi
# Hand-to-surface observation: default ON for mjlab frozen-SONIC runs (user, 2026-10-21); HAND_SURFACE_OBS=0 opts out.
if [[ "${BACKEND:-isaac}" == mjlab && "${FROZEN_PRETRAINED_SONIC:-0}" == 1 && "${HAND_SURFACE_OBS:-1}" == 1 ]]; then
    extra_args+=(--hand-surface-obs)
fi
if [[ -n "${SGS_STATE:-}" ]]; then extra_args+=(--sgs-state "$SGS_STATE"); fi
if [[ -n "${DIAGNOSTIC_CAPTURE_SECONDS:-}" ]]; then
    extra_args+=(--diagnostic-capture-seconds "$DIAGNOSTIC_CAPTURE_SECONDS")
fi
# Forward the advance time-limit notification to Python for an atomic save at
# an optimizer boundary. The actual Slurm limit remains authoritative.
training_pid=''
trap 'if [[ -n "$training_pid" ]]; then kill -USR1 "$training_pid"; fi' USR1 TERM
python scripts/train_g1_sonic_sapg.py "${launch_args[@]}" --device cuda:0 \
    --output "$project/outputs/${RUN_PREFIX:-0_sonic_sapg_touch}_${SLURM_JOB_ID}" \
    --num-envs "${NUM_ENVS:-768}" --max-epochs "${MAX_EPOCHS:-120}" "${extra_args[@]}" &
training_pid=$!
wait "$training_pid" || { result=$?; if kill -0 "$training_pid" 2>/dev/null; then wait "$training_pid"; else exit "$result"; fi; }
