#!/usr/bin/env bash
#SBATCH --job-name=g1_bps128_touch70_long
#SBATCH --partition=batch
#SBATCH --gres=gpu:rtx_6000_ada:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=192G
#SBATCH --time=2-00:00:00
#SBATCH --signal=B:USR1@180
#SBATCH --output=slurm-g1-touch-long-%j.out
set -euo pipefail
project=/data1/users/konstantin.smirnov/DEXTRAH-BPS128-TOUCH
p2p=/data1/users/konstantin.smirnov/play2perfect
python=/data1/users/konstantin.smirnov/miniconda3/envs/adept_dextrah/bin/python
source_run=/data1/users/konstantin.smirnov/DEXTRAH-BPS128/dextrah_lab/rl_games/logs/rl_games/g1_sapg_bps128_warmstart/0_g1_bps128_warmstart_seed_42_355
resume=${RESUME_CHECKPOINT:-$project/outputs/0_bps128_touch_pilot_381/nn/complete_500170752.pth}
export PYTHONPATH="$project:$p2p:$p2p/rl_games${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OMNI_KIT_ACCEPT_EULA=YES WANDB_MODE=online
unset DISPLAY
cd "$project"
test -f "$resume"
git log -1 --format='Training commit: %H'
git -C "$p2p" log -1 --format='Play2Perfect dependency: %H'
echo "Slurm ${SLURM_JOB_ID}: CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
# exec lets Slurm's advance USR1 reach the Python process for an update-boundary save.
# No evaluation process, checkpoint polling loop, or automatic restart is involved.
exec "$python" scripts/train_g1_touch_continuation.py --headless --device cuda:0 \
    --source-run "$source_run" \
    --source-checkpoint "$source_run/nn/last_g1_sapg_bps128_warmstart_frame_8000372736_rew__7400.633_.pth" \
    --resume "$resume" --continuous --max-frames -1 --max-epochs -1 \
    --touch-sensor-hz 70 --touch-publish-hz 70 --num-envs 24576 --wandb \
    --output "$project/outputs/0_bps128_touch70_continuous_${SLURM_JOB_ID}"
