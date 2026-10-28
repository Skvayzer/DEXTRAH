#!/usr/bin/env bash
#SBATCH --job-name=nsim_pairdiag
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=19G
#SBATCH --time=01:00:00
#SBATCH --output=/data1/users/konstantin.smirnov/DEXTRAH-G1-WholeBody/outputs/neurosim/slurm-pairdiag-%j.out
# Per-component agreement on the held-out paired counterfactuals: base vs pair-fine-tuned world model (CPU).
set -uo pipefail
project=/data1/users/konstantin.smirnov/DEXTRAH-G1-WholeBody-neurosim
repo=/data1/users/konstantin.smirnov/DEXTRAH-G1-WholeBody
commit="$(git -C "$project" rev-parse HEAD)"
src="$(mktemp -d $repo/outputs/neurosim_source.XXXXXX)"
git -C "$project" archive "$commit" | tar -x -C "$src"
export PATH="/data1/users/konstantin.smirnov/venvs/mjlab/bin:$PATH"
export PYTHONPATH="$src:$src/scripts" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=
cd "$src"
echo "Commit: $commit (feature/g1-neurosim)"
NS=$repo/outputs/neurosim
CF=$NS/counterfactual
python scripts/neurosim/pair_diagnostic.py --pairs $CF/pairs_v1 --models $NS/stage15/combined_single/world_model.pt \
  $CF/finetune_v1/world_model.pt --finetune-report $CF/finetune_v1/report.json --output $CF/pair_diagnostic.json
