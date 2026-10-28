#!/usr/bin/env bash
#SBATCH --job-name=nsim_actsens
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=19G
#SBATCH --time=01:00:00
#SBATCH --output=/data1/users/konstantin.smirnov/DEXTRAH-G1-WholeBody/outputs/neurosim/slurm-actsens-%j.out
# One-step action sensitivity of the stage-1.5 world model vs ridge regression on its data (CPU).
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
python scripts/neurosim/action_sensitivity.py --model $NS/stage15/combined_single/world_model.pt --data $NS/go_nogo_v2/data \
  --output $NS/stage15/action_sensitivity.json --max-chunks ${CHUNKS:-2}
