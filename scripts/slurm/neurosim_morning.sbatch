#!/usr/bin/env bash
#SBATCH --job-name=morning_checks
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx_6000_ada:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=110G
#SBATCH --time=02:00:00
#SBATCH --output=/data1/users/konstantin.smirnov/DEXTRAH-G1-WholeBody/outputs/neurosim/slurm-morning-%j.out
# Morning slot: (1) table-to-table route of the overnight carry expert (no carry move, standing settle reference);
# (2) gradient check with a simulator noise floor and small perturbations; (3) bundled (expected-effect) gradient check.
#   C=<carry checkpoint at the pause> sbatch neurosim_morning.sbatch
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
echo "Commit: $commit (feature/g1-neurosim)  C: ${C:?}"
NS=$repo/outputs/neurosim
M=$repo/outputs/0_mjlab_wb_imgen_dualclip_1454/best/best_epoch12662.pth
P=$repo/outputs/0_mjlab_wb_place_1576/nn/complete_5789521920.pth
LIB=$repo/outputs/rest_poses/rest_poses_v1.npz
C2=$repo/outputs/0_mjlab_wb_carry_v2_1587/nn/complete_351989760.pth
WM=$NS/stage15/combined_single/world_model.pt
echo "=== 1. route, overnight C (no carry move, standing settle reference)"
python scripts/mj_table_to_table.py --m-checkpoint $M --c-checkpoint "$C" --place-checkpoint $P --place-goal nearest \
  --place-library $LIB --output $repo/outputs/handoff_harvest/route_c2nomove_night --contact-condim 4 --njmax 400 --no-render || echo ROUTE_FAILED
echo "=== 2. gradient check: noise floor, small deltas"
python scripts/neurosim/gradient_check.py --checkpoint $C2 --world-model $WM --output $NS/stage15/gradient_check_noise.json \
  --states 256 --deltas 0.01 0.02 0.05 --replicas 2 || echo GRADIENT_NOISE_FAILED
echo "=== 3. bundled gradient check (expected effects)"
python scripts/neurosim/gradient_check.py --checkpoint $C2 --world-model $WM --output $NS/stage15/gradient_check_bundled.json \
  --states 128 --deltas 0.05 --replicas 1 --bundle 8 --jitter 0.01 || echo GRADIENT_BUNDLED_FAILED
echo MORNING_DONE
