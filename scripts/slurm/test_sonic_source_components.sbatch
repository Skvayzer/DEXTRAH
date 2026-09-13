#!/usr/bin/env bash
#SBATCH --job-name=g1_source_tests
#SBATCH --partition=batch
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=slurm-sonic-components-%j.out
set -euo pipefail
project="${WHOLEBODY_REPO:?Set WHOLEBODY_REPO}"
source_snapshot="$(mktemp -d "$project/outputs/sonic_component_tests.XXXXXX")"
git -C "$project" archive HEAD | tar -x -C "$source_snapshot"
export PATH="/data1/users/konstantin.smirnov/venvs/g1_sonic/bin:/data1/users/konstantin.smirnov/miniconda3/envs/adept_dextrah/bin:$PATH"
export PYTHONPATH="$source_snapshot${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1
cd "$source_snapshot"
python -m pytest -q tests/test_wholebody_timed_history.py tests/test_sonic_source_contract.py \
    tests/test_sonic_sapg_network.py tests/test_sonic_source_actions.py tests/test_sonic_bootstrap.py tests/test_sapg_features.py \
    tests/test_body_termination.py tests/test_failure_recording.py tests/test_sapg_memory_trace.py
