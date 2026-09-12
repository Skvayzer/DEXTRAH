#!/usr/bin/env bash
#SBATCH --job-name=g1_runtime_probe
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=00:08:00
#SBATCH --output=slurm-runtime-probe-%j.out
set -uo pipefail
project="${WHOLEBODY_REPO:?Set WHOLEBODY_REPO}"
export PYTHONPATH="$project${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONUNBUFFERED=1 OMNI_KIT_ACCEPT_EULA=YES
unset DISPLAY
cd "$project"
python=/data1/users/konstantin.smirnov/miniconda3/envs/adept_dextrah/bin/python
# Sequentially compare CUDA and CPU physics in ONE allocated GPU job.
# Timeouts preserve logs and prevent a native initialization hang occupying it indefinitely.
timeout --signal=TERM --kill-after=10 150 "$python" scripts/probe_isaac_runtime.py \
    --headless --device cuda:0 --output "outputs/runtime_${SLURM_JOB_ID}_cuda"
gpu_status=$?
timeout --signal=TERM --kill-after=10 150 "$python" scripts/probe_isaac_runtime.py \
    --headless --device cpu --output "outputs/runtime_${SLURM_JOB_ID}_cpu"
cpu_status=$?
printf 'Runtime probe exit codes: CUDA=%s CPU=%s\n' "$gpu_status" "$cpu_status"
[[ "$gpu_status" == 0 && "$cpu_status" == 0 ]]
