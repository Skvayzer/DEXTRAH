#!/usr/bin/env bash
#SBATCH --job-name=g1_teacher_suite
#SBATCH --partition=batch
#SBATCH --gres=gpu:rtx_6000_ada:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=slurm-teacher-suite-%j.out
set -euo pipefail
project="${TEACHER_REPO:?Set TEACHER_REPO to the immutable staged source}"
output="${TEACHER_OUTPUT:?Set TEACHER_OUTPUT to the suite output directory}"
p2p=/data1/users/konstantin.smirnov/play2perfect
python=/data1/users/konstantin.smirnov/miniconda3/envs/adept_dextrah/bin/python
export PYTHONPATH="$project:$p2p:$p2p/rl_games${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OMNI_KIT_ACCEPT_EULA=YES
unset DISPLAY
cd "$project"
"$python" -m pytest -q tests/test_teacher_suite.py tests/test_bps_evaluation.py tests/test_teacher_runner.py
"$python" scripts/run_g1_teacher_suite.py --output "$output" --execute --render-winner
