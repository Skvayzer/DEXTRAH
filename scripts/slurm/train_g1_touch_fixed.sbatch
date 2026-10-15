#!/usr/bin/env bash
#SBATCH --job-name=g1_touch_fixed
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx_6000_ada:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=2-00:00:00
#SBATCH --signal=B:USR1@180
#SBATCH --output=slurm-g1-touch-fixed-%j.out
# Fixed-body G1 + Revo2 BPS-128 + 70 Hz touch SAPG teacher (the run-383 task: robot
# base fixed, arm + right hand; no SONIC), continued from a tactile checkpoint.
# PER_ENV_OBJECT_PHYSICS=1: every env simulates its own object (fixes the
# replicated-template bug), verified against URDF masses at startup.
set -euo pipefail
project="${WHOLEBODY_REPO:-/data1/users/konstantin.smirnov/DEXTRAH-G1-WholeBody}"
commit="$(git -C "$project" rev-parse HEAD)"
source_snapshot="$(mktemp -d "$project/outputs/touch_fixed_source.XXXXXX")"
git -C "$project" archive "$commit" | tar -x -C "$source_snapshot"
export PATH="/data1/users/konstantin.smirnov/venvs/g1_sonic/bin:/data1/users/konstantin.smirnov/miniconda3/envs/adept_dextrah/bin:$PATH"
export PYTHONPATH="$source_snapshot${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1
export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y WANDB_MODE=online
unset DISPLAY
cd "$source_snapshot"
echo "Training commit: $commit"
nvidia-smi --query-gpu=index,uuid,name,memory.used,memory.total --format=csv
source_run=/data1/users/konstantin.smirnov/DEXTRAH-BPS128/dextrah_lab/rl_games/logs/rl_games/g1_sapg_bps128_warmstart/0_g1_bps128_warmstart_seed_42_355
resume="${RESUME_CHECKPOINT:?Set RESUME_CHECKPOINT (e.g. run 383 complete_17364025344.pth)}"
test -f "$resume"
extra_args=()
launch_args=(--headless)
if [[ "${BACKEND:-isaac}" == mjlab ]]; then
    # MuJoCo Warp port (dextrah_lab/mj, fixed_body=True): separate venv, no Kit.
    export PATH="/data1/users/konstantin.smirnov/venvs/mjlab/bin:$PATH"
    launch_args=(--backend mjlab --headless)
    if [[ -n "${MJ_NCONMAX:-}" ]]; then extra_args+=(--mj-nconmax "$MJ_NCONMAX"); fi
    if [[ -n "${MJ_NJMAX:-}" ]]; then extra_args+=(--mj-njmax "$MJ_NJMAX"); fi
    if [[ "${MJ_NAN_CAPTURE:-0}" == 1 ]]; then extra_args+=(--mj-nan-capture); fi
    if [[ -n "${MJ_NAN_CAPTURE_MAX_QVEL:-}" ]]; then extra_args+=(--mj-nan-capture-max-qvel "$MJ_NAN_CAPTURE_MAX_QVEL"); fi
    if [[ "${MJ_ARM_ARMATURE:-0}" == 1 ]]; then extra_args+=(--mj-arm-armature); fi
    if [[ "${MJ_CONE_HESSIAN_FIX:-0}" == 1 ]]; then
        export PYTHONPATH="$source_snapshot/third_party/mujoco_warp_cone_fix:$PYTHONPATH"
        extra_args+=(--mj-cone-hessian-fix)
    fi
fi
if [[ "${PER_ENV_OBJECT_PHYSICS:-0}" == 1 ]]; then extra_args+=(--per-env-object-physics); fi
if [[ -n "${MINIBATCH_SIZE:-}" ]]; then extra_args+=(--minibatch-size "$MINIBATCH_SIZE"); fi
if [[ -n "${BEST_DELAY_EPOCHS:-}" ]]; then extra_args+=(--best-delay-epochs "$BEST_DELAY_EPOCHS"); fi
if [[ "${WANDB:-1}" == 1 ]]; then extra_args+=(--wandb); fi
training_pid=''
trap 'if [[ -n "$training_pid" ]]; then kill -USR1 "$training_pid"; fi' USR1 TERM
python scripts/train_g1_touch_continuation.py "${launch_args[@]}" --device cuda:0 \
    --source-run "$source_run" \
    --source-checkpoint "$source_run/nn/last_g1_sapg_bps128_warmstart_frame_8000372736_rew__7400.633_.pth" \
    --resume "$resume" --continuous --touch-sensor-hz 70 --touch-publish-hz 70 \
    --num-envs "${NUM_ENVS:?Set NUM_ENVS}" --snapshot-epochs "${SNAPSHOT_EPOCHS:-0}" \
    --output "$project/outputs/${RUN_PREFIX:-0_bps128_touch70_fixed}_${SLURM_JOB_ID}" "${extra_args[@]}" &
training_pid=$!
wait "$training_pid" || { result=$?; if kill -0 "$training_pid" 2>/dev/null; then wait "$training_pid"; else exit "$result"; fi; }
