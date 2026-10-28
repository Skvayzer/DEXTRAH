#!/usr/bin/env bash
#SBATCH --job-name=nsim_grad_v5
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx_6000_ada:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=110G
#SBATCH --time=01:30:00
#SBATCH --output=/data1/users/konstantin.smirnov/DEXTRAH-G1-WholeBody/outputs/neurosim/slurm-gradcheck-v5-%j.out
# Gradient check v5: perturbation 2 control steps after the restore, horizons 0.1 / 0.2 / 0.5 s.
# Replica divergence (2026-10-28, domain randomisation off): restored copies are bit-identical, then differ at the
# float level after one step (MuJoCo Warp is not bitwise reproducible across worlds) and the hand-object system
# amplifies that ~x200 per 0.1 s (object 0.002 mm at 0.1 s, 0.5-0.7 mm at 0.2 s, 3-5 mm / 11 deg by 0.6 s, also
# with shared expert actions). The earlier checks perturbed at 0.6 s, when copies already differ by millimetres.
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
C2=$repo/outputs/0_mjlab_wb_carry_v2_1587/nn/complete_351989760.pth
WM=$repo/outputs/neurosim/stage15/combined_single/world_model.pt
out=$repo/outputs/neurosim/stage15
for spec in "0.1 0.05 h0p1" "0.2 0.1 h0p2" "0.5 0.25 h0p5"; do
  set -- $spec
  echo "=== gradient check v5: horizon $1 s, perturbation $2 s, from 2 steps after the restore"
  python scripts/neurosim/gradient_check.py --checkpoint $C2 --world-model $WM --output $out/gradient_check_v5_$3.json \
    --states 256 --deltas 0.05 0.1 --replicas 2 --t0-s 0.034 --duration-s $2 --horizon-s $1 || echo "GRADIENT_V5_$3_FAILED"
done
echo GRADCHECK_V5_DONE
