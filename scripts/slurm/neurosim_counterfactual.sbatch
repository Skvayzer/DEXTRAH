#!/usr/bin/env bash
#SBATCH --job-name=nsim_cf
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx_6000_ada:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=110G
#SBATCH --time=02:00:00
#SBATCH --output=/data1/users/konstantin.smirnov/DEXTRAH-G1-WholeBody/outputs/neurosim/slurm-counterfactual-%j.out
# Can paired counterfactual data cure the world model's action-blindness? (2026-10-28)
#  1. collect: 2048 holding states x 8 bit-identical copies (copy 0 = carry expert C v2 1587, copies 1-7 = finger
#     command perturbations: constant / pulse / noise), 30 rounds, 8 control steps; held-out objects excluded
#  2. fine-tune the stage-1.5 model (combined_single) on its data + the difference loss on the pairs (W&B)
#  3. gradient check v5 at 0.1 s on the model's held-out objects only: base model, fine-tuned model
set -uo pipefail
project=/data1/users/konstantin.smirnov/DEXTRAH-G1-WholeBody-neurosim
repo=/data1/users/konstantin.smirnov/DEXTRAH-G1-WholeBody
commit="$(git -C "$project" rev-parse HEAD)"
src="$(mktemp -d $repo/outputs/neurosim_source.XXXXXX)"
git -C "$project" archive "$commit" | tar -x -C "$src"
export PATH="/data1/users/konstantin.smirnov/venvs/mjlab/bin:$PATH"
export PYTHONPATH="$src/third_party/mujoco_warp_cone_fix:$src:$src/scripts:/data1/users/konstantin.smirnov/G1-SONIC-planner/python-deps"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
cd "$src"
echo "Commit: $commit (feature/g1-neurosim)"
NS=$repo/outputs/neurosim
C2=$repo/outputs/0_mjlab_wb_carry_v2_1587/nn/complete_351989760.pth
WM=$NS/stage15/combined_single/world_model.pt
CF=$NS/counterfactual
mkdir -p $CF
echo "=== 1. collect paired counterfactuals"
python scripts/neurosim/collect_counterfactual_pairs.py --checkpoint $C2 --world-model $WM --output $CF/pairs_v1 || { echo COLLECT_FAILED; exit 1; }
echo "=== 2. fine-tune with the difference loss"
python scripts/neurosim/train_world_model.py --data $NS/go_nogo_v1/data $NS/go_nogo_v2/data --init $WM --pairs $CF/pairs_v1 \
  --iters 6000 --lr 1e-4 --eval-every 1000 --output $CF/finetune_v1 || echo FINETUNE_FAILED
echo "=== 3a. gradient check, held-out objects, base model"
python scripts/neurosim/gradient_check.py --checkpoint $C2 --world-model $WM --output $CF/gradcheck_base_h0p1.json \
  --states 256 --deltas 0.02 0.05 0.1 --replicas 2 --t0-s 0.034 --duration-s 0.05 --horizon-s 0.1 --held-out-only || echo GRADCHECK_BASE_FAILED
echo "=== 3b. gradient check, held-out objects, fine-tuned model"
python scripts/neurosim/gradient_check.py --checkpoint $C2 --world-model $CF/finetune_v1/world_model.pt --output $CF/gradcheck_finetune_h0p1.json \
  --states 256 --deltas 0.02 0.05 0.1 --replicas 2 --t0-s 0.034 --duration-s 0.05 --horizon-s 0.1 --held-out-only || echo GRADCHECK_FT_FAILED
echo COUNTERFACTUAL_JOB_DONE
