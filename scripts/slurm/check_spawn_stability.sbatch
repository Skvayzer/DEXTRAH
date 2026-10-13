#!/usr/bin/env bash
#SBATCH --job-name=g1_spawn_check
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx_6000_ada:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=slurm-spawn-check-%j.out
set -euo pipefail
project=/data1/users/konstantin.smirnov/DEXTRAH-G1-WholeBody
backend="${BACKEND:-isaac}"
export FULLBODY_SOURCE_COMMIT="$(git -C "$project" rev-parse HEAD)"
src="$(mktemp -d "$project/outputs/spawn_check_source.XXXXXX")"
git -C "$project" archive "$FULLBODY_SOURCE_COMMIT" | tar -x -C "$src"
if [[ "$backend" == isaac ]]; then
  export PATH="/data1/users/konstantin.smirnov/venvs/g1_sonic/bin:/data1/users/konstantin.smirnov/miniconda3/envs/adept_dextrah/bin:$PATH"
  extra=(--headless --kit_args="--/plugins/carb.tasking.plugin/threadCount=8 --/plugins/omni.tbb.globalcontrol/maxThreadCount=8")
else
  export PATH="/data1/users/konstantin.smirnov/venvs/mjlab/bin:$PATH"
  extra=()
fi
export PYTHONPATH="$src" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1 PXR_WORK_THREAD_LIMIT=8
export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y WANDB_MODE=disabled
unset DISPLAY
cd "$src"
out="$project/outputs/spawn_check_${backend}${SPAWN_ON_TABLE:+_fixed}_${SLURM_JOB_ID}"
python scripts/check_spawn_stability.py --backend "$backend" --output "$out" \
    --num-envs "${NUM_ENVS:-1200}" --seconds "${SECONDS_SIM:-40}" ${SPAWN_ON_TABLE:+--spawn-on-table} "${extra[@]}"
