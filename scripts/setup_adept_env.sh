#!/usr/bin/env bash
set -euo pipefail

# Keep ADEPT's legacy FABRICS requirements isolated from other Isaac Lab work.
CONDA_ROOT="${CONDA_ROOT:-${HOME}/data1/miniconda3}"
BASE_ENV_PATH="${BASE_ENV_PATH:-${CONDA_ROOT}/envs/unitree_sim_env}"
ADEPT_ENV_PATH="${ADEPT_ENV_PATH:-${CONDA_ROOT}/envs/adept_dextrah}"
FABRICS_DIR="${FABRICS_DIR:-${HOME}/data1/FABRICS-ADEPT}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ ! -x "${BASE_ENV_PATH}/bin/python" ]]; then
    echo "Base Isaac environment not found: ${BASE_ENV_PATH}" >&2
    exit 1
fi

if [[ ! -x "${ADEPT_ENV_PATH}/bin/python" ]]; then
    "${CONDA_ROOT}/bin/conda" create -y -p "${ADEPT_ENV_PATH}" --clone "${BASE_ENV_PATH}"
fi

"${CONDA_ROOT}/bin/conda" install -y -p "${ADEPT_ENV_PATH}" git-lfs
export PATH="${ADEPT_ENV_PATH}/bin:${PATH}"

if [[ ! -d "${FABRICS_DIR}/.git" ]]; then
    git clone https://github.com/NVlabs/FABRICS.git "${FABRICS_DIR}"
fi

git -C "${FABRICS_DIR}" lfs install --local
git -C "${FABRICS_DIR}" lfs pull
git -C "${REPO_ROOT}" lfs install --local
git -C "${REPO_ROOT}" lfs pull

PYTHON="${ADEPT_ENV_PATH}/bin/python"

# Isaac Sim 5 requires NumPy 1.26 and NetworkX 3.3. urdfpy's package metadata
# pins obsolete versions, but its runtime works with these versions plus the
# compatibility shim in adept_cspace_fabric.py.
"${PYTHON}" -m pip install \
    numpy==1.26.0 \
    networkx==3.3 \
    pycollada==0.9.3 \
    warp-lang==1.8.1
"${PYTHON}" -m pip install --no-deps \
    urdfpy==0.0.22 \
    pyrender==0.1.45 \
    freetype-py==2.5.1 \
    PyOpenGL==3.1.0
"${PYTHON}" -m pip install --no-deps -e "${FABRICS_DIR}" -e "${REPO_ROOT}"

"${PYTHON}" "${REPO_ROOT}/scripts/generate_adept_primitives.py"
"${PYTHON}" "${REPO_ROOT}/scripts/generate_adept_fmb_assets.py"

"${PYTHON}" -m pytest -q "${REPO_ROOT}/tests/test_adept_action_mapping.py"

echo "ADEPT environment ready: ${ADEPT_ENV_PATH}"
echo "Submit the smoke test with: sbatch ${REPO_ROOT}/scripts/slurm/smoke_adept_repose.sbatch"
