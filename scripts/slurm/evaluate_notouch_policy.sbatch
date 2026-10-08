#!/usr/bin/env bash
#SBATCH --job-name=g1_notouch_eval
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:rtx_6000_ada:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=slurm-notouch-eval-%j.out
# Inference only, one GPU: (1) seed-42 hammer/brush/spatula reposing videos with real
# table collisions + body-table contact termination, (2) standing check (table removed
# after a held grasp) for the same checkpoint and run 560's best.
set -euo pipefail
project="${WHOLEBODY_REPO:?Set WHOLEBODY_REPO}"
export FULLBODY_SOURCE_COMMIT="$(git -C "$project" rev-parse HEAD)"
source_snapshot="$(mktemp -d "$project/outputs/notouch_eval_source.XXXXXX")"
git -C "$project" archive "$FULLBODY_SOURCE_COMMIT" | tar -x -C "$source_snapshot"
export PATH="/data1/users/konstantin.smirnov/venvs/g1_sonic/bin:/data1/users/konstantin.smirnov/miniconda3/envs/adept_dextrah/bin:$PATH"
export PYTHONPATH="$source_snapshot${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1
export PXR_WORK_THREAD_LIMIT=8 OPENBLAS_NUM_THREADS=4
export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y WANDB_MODE=disabled
unset DISPLAY
cd "$source_snapshot"
kit="--/plugins/carb.tasking.plugin/threadCount=8 --/plugins/omni.tbb.globalcontrol/maxThreadCount=8"
out="$project/outputs/notouch_eval_${SLURM_JOB_ID}"
python scripts/record_g1_sonic_checkpoint.py --headless --device cuda:0 --kit_args="$kit" \
    --checkpoint "${EVAL_CHECKPOINT:?Set EVAL_CHECKPOINT}" --output "$out/videos" \
    --num-envs 120 --seconds 60 --body-table-contact-termination || echo "RECORDING_FAILED"
export PATH="/data1/users/konstantin.smirnov/miniconda3/envs/adept_dextrah/bin:$PATH"
for family in hammer brush spatula; do
    python scripts/render_g1_sonic_checkpoint.py "$out/videos/$family" || echo "RENDER_FAILED $family"
done
export PATH="/data1/users/konstantin.smirnov/venvs/g1_sonic/bin:$PATH"
python scripts/eval_standing_without_table.py --headless --device cuda:0 --kit_args="$kit" \
    --physics contact --num-envs 120 --seconds 30 --torch-memory-limit-gib 30 \
    --checkpoints "$EVAL_CHECKPOINT" "${BASELINE_CHECKPOINT:?Set BASELINE_CHECKPOINT}" \
    --labels notouch_best run560_best --output "$out/standing"
