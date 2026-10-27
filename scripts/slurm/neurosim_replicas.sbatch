#!/usr/bin/env bash
#SBATCH --job-name=nsim_replicas
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx_6000_ada:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=110G
#SBATCH --time=01:00:00
#SBATCH --output=/data1/users/konstantin.smirnov/DEXTRAH-G1-WholeBody/outputs/neurosim/slurm-replicas-%j.out
# Do identical carry worlds stay identical? (scripts/neurosim/replica_divergence.py): the gradient check's
# overrides, + no kept tables, + shared expert actions. Carry expert C v2 1587 (the world model's data source).
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
out=$repo/outputs/neurosim/stage15/replicas
for v in as_check no_table shared; do
  echo "=== variant $v"
  python scripts/neurosim/replica_divergence.py --checkpoint $C2 --variant $v --output $out/$v.json || echo "VARIANT_${v}_FAILED"
done
echo REPLICAS_DONE
