#!/usr/bin/env bash
#SBATCH --job-name=neurosim_n1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx_6000_ada:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=110G
#SBATCH --time=03:00:00
#SBATCH --output=/data1/users/konstantin.smirnov/DEXTRAH-G1-WholeBody/outputs/neurosim/slurm-neurosim-night1-%j.out
# Neural simulator stage 1.5 (feature/g1-neurosim): (1) gradient check of the v2 world model against MuJoCo Warp
# finite differences; (2) single model on both collections (v1 + v2, 22M transitions); (3) 5-member
# probabilistic ensemble (3 x 512) on both collections.
set -uo pipefail
project=/data1/users/konstantin.smirnov/DEXTRAH-G1-WholeBody-neurosim
repo=/data1/users/konstantin.smirnov/DEXTRAH-G1-WholeBody
commit="$(git -C "$project" rev-parse HEAD)"
src="$(mktemp -d $repo/outputs/neurosim_source.XXXXXX)"
git -C "$project" archive "$commit" | tar -x -C "$src"
export PATH="/data1/users/konstantin.smirnov/venvs/mjlab/bin:$PATH"
export PYTHONPATH="$src/third_party/mujoco_warp_cone_fix:$src:$src/scripts:/data1/users/konstantin.smirnov/G1-SONIC-planner/python-deps"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$src"
echo "Commit: $commit (feature/g1-neurosim)"
NS=$repo/outputs/neurosim
C2=$repo/outputs/0_mjlab_wb_carry_v2_1587/nn/complete_351989760.pth
echo "=== 1. gradient check"
python scripts/neurosim/gradient_check.py --checkpoint $C2 --world-model $NS/go_nogo_v2/model/world_model.pt \
  --output $NS/stage15/gradient_check_v2.json --states 256 || echo GRADIENT_CHECK_FAILED
echo "=== 2. single model, v1 + v2 data"
python scripts/neurosim/train_world_model.py --data $NS/go_nogo_v1/data $NS/go_nogo_v2/data --output $NS/stage15/combined_single \
  || echo COMBINED_SINGLE_FAILED
echo "=== 3. ensemble 5 x (3 x 512), v1 + v2 data"
python scripts/neurosim/train_world_model.py --data $NS/go_nogo_v1/data $NS/go_nogo_v2/data --output $NS/stage15/combined_ensemble \
  --members 5 --hidden 512,512,512 || echo COMBINED_ENSEMBLE_FAILED
echo NIGHT1_DONE
