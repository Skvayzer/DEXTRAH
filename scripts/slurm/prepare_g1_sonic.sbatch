#!/usr/bin/env bash
#SBATCH --job-name=g1_sonic_prepare
#SBATCH --partition=batch
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=slurm-sonic-prepare-%j.out
set -euo pipefail
project="${WHOLEBODY_REPO:?Set WHOLEBODY_REPO}"
base_python=/data1/users/konstantin.smirnov/miniconda3/envs/adept_dextrah/bin/python
environment=/data1/users/konstantin.smirnov/venvs/g1_sonic
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONUNBUFFERED=1
if [[ ! -f "$environment/pyvenv.cfg" ]]; then
    "$base_python" -m venv --system-site-packages "$environment"
fi
# Dependencies are installed only in the new overlay; do not upgrade base Torch.
"$environment/bin/python" -m pip install --no-deps \
    'vector-quantize-pytorch==1.28.2' 'loguru==0.7.3' 'einx==0.3.0' 'trl==0.27.2'
"$environment/bin/python" -m pip install \
    'accelerate==1.12.0' 'frozendict==2.4.7' 'transformers==4.56.2' 'datasets==3.6.0'
cd "$project"
"$environment/bin/python" scripts/prepare_g1_sonic.py \
    --grail /data1/users/konstantin.smirnov/GRAIL \
    --output /data1/users/konstantin.smirnov/G1-SONIC-models
