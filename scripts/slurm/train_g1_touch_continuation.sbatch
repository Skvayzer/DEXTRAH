#!/usr/bin/env bash
#SBATCH --job-name=g1_bps128_touch
#SBATCH --partition=batch
#SBATCH --gres=gpu:rtx_6000_ada:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=192G
#SBATCH --time=2-00:00:00
#SBATCH --output=slurm-g1-touch-%j.out
set -euo pipefail
project=/data1/users/konstantin.smirnov/DEXTRAH-BPS128-TOUCH
p2p=/data1/users/konstantin.smirnov/play2perfect
python=/data1/users/konstantin.smirnov/miniconda3/envs/adept_dextrah/bin/python
source_run=/data1/users/konstantin.smirnov/DEXTRAH-BPS128/dextrah_lab/rl_games/logs/rl_games/g1_sapg_bps128_warmstart/0_g1_bps128_warmstart_seed_42_355
export PYTHONPATH="$project:$p2p:$p2p/rl_games${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OMNI_KIT_ACCEPT_EULA=YES WANDB_MODE=online
unset DISPLAY
cd "$project"
test -f "${CALIBRATION:?Set CALIBRATION to the validated frozen-policy sensor statistics}"
git log -1 --format='Training commit: %H'
git -C "$p2p" log -1 --format='Play2Perfect dependency: %H'
echo "Slurm ${SLURM_JOB_ID}: CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
options=()
kind=touch
if [[ "${CONTROL:-0}" == 1 ]]; then options+=(--control); kind=control; fi
"$python" scripts/run_g1_touch_experiment.py \
    --source-run "$source_run" \
    --source-checkpoint "$source_run/nn/last_g1_sapg_bps128_warmstart_frame_8000372736_rew__7400.633_.pth" \
    --play2perfect-root "$p2p" --num-envs "${NUM_ENVS:-24576}" \
    --calibration "$CALIBRATION" \
    --output "$project/outputs/0_bps128_${kind}_pilot_${SLURM_JOB_ID}" "${options[@]}"
