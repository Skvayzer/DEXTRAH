#!/usr/bin/env bash
#SBATCH --job-name=nsim_drops
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-nsim-drops-%j.out
# Drop-prediction report of a trained hand-object world model on CPU (feature/g1-neurosim):
#   MODEL=<world_model.pt> DATA=<collection dir> OUTPUT=<report.json> sbatch neurosim_drop_report.sbatch
set -euo pipefail
project="${NEUROSIM_REPO:-/data1/users/konstantin.smirnov/DEXTRAH-G1-WholeBody-neurosim}"
commit="$(git -C "$project" rev-parse HEAD)"
src="$(mktemp -d /data1/users/konstantin.smirnov/DEXTRAH-G1-WholeBody/outputs/neurosim_source.XXXXXX)"
git -C "$project" archive "$commit" | tar -x -C "$src"
export PATH="/data1/users/konstantin.smirnov/venvs/mjlab/bin:$PATH" PYTHONPATH="$src" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES= PYTHONUNBUFFERED=1
cd "$src"
echo "Commit: $commit (feature/g1-neurosim)"
python scripts/neurosim/evaluate_world_model.py --model "${MODEL:?}" --data "${DATA:?}" --output "${OUTPUT:?}" ${EXTRA_ARGS:-}
