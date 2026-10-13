#!/usr/bin/env bash
#SBATCH --job-name=g1_isaac_ref
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx_6000_ada:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=slurm-isaac-ref-%j.out
# Isaac reference for the mjlab port: model dump + PD step responses, then the
# spawn stability check. One GPU job, both parts sequential.
set -euo pipefail
project=/data1/users/konstantin.smirnov/DEXTRAH-G1-WholeBody
export FULLBODY_SOURCE_COMMIT="$(git -C "$project" rev-parse HEAD)"
src="$(mktemp -d "$project/outputs/isaac_ref_source.XXXXXX")"
git -C "$project" archive "$FULLBODY_SOURCE_COMMIT" | tar -x -C "$src"
export PATH="/data1/users/konstantin.smirnov/venvs/g1_sonic/bin:/data1/users/konstantin.smirnov/miniconda3/envs/adept_dextrah/bin:$PATH"
export PYTHONPATH="$src" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1 PXR_WORK_THREAD_LIMIT=8
export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y WANDB_MODE=disabled
unset DISPLAY
cd "$src"
kit=(--headless --kit_args="--/plugins/carb.tasking.plugin/threadCount=8 --/plugins/omni.tbb.globalcontrol/maxThreadCount=8")
python scripts/dump_isaac_reference.py --output "$project/outputs/isaac_reference_${SLURM_JOB_ID}" \
    --num-envs 1200 --step-response "${kit[@]}" || echo "REFERENCE_DUMP_FAILED"
python scripts/check_spawn_stability.py --backend isaac --output "$project/outputs/spawn_check_isaac_${SLURM_JOB_ID}" \
    --num-envs 1200 --seconds 40 "${kit[@]}"
