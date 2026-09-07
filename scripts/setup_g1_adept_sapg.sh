#!/usr/bin/env bash
set -euo pipefail

CONDA_ROOT="${CONDA_ROOT:-${HOME}/data1/miniconda3}"
ADEPT_ENV_PATH="${ADEPT_ENV_PATH:-${CONDA_ROOT}/envs/adept_dextrah}"
PLAY2PERFECT_ROOT="${PLAY2PERFECT_ROOT:-${HOME}/data1/play2perfect}"
SIMTOOLREAL_ROOT="${SIMTOOLREAL_ROOT:-${HOME}/data1/simtoolreal}"
PCA_ARTIFACT="${PCA_ARTIFACT:-/data2/users/konstantin.smirnov/dex-ycb/revo2-g1-full-v1/revo2_human_motion_pca.npz}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

EXPECTED_PLAY2PERFECT_COMMIT="70e79b5e53f912ef04af294ff8f61ac1c7f42160"
EXPECTED_SIMTOOLREAL_COMMIT="313d5aea1f507c6cfe097b672b62945d7b0bbff5"
EXPECTED_PCA_SHA256="8cea2fe7602958bbefec826fd331a100c15145c7dd837c68406a07895a2237b3"
G1_URDF="${PLAY2PERFECT_ROOT}/unitree_ros/robots/g1_with_brainco_hand/g1_29dof_mode_15_brainco_hand.urdf"

if [[ ! -x "${ADEPT_ENV_PATH}/bin/python" ]]; then
    echo "ADEPT Isaac environment not found: ${ADEPT_ENV_PATH}" >&2
    exit 1
fi
if [[ ! -d "${PLAY2PERFECT_ROOT}/.git" ]]; then
    echo "Play2Perfect checkout not found: ${PLAY2PERFECT_ROOT}" >&2
    exit 1
fi
if [[ "$(git -C "${PLAY2PERFECT_ROOT}" rev-parse HEAD)" != "${EXPECTED_PLAY2PERFECT_COMMIT}" ]]; then
    echo "Play2Perfect must be pinned to ${EXPECTED_PLAY2PERFECT_COMMIT}" >&2
    exit 1
fi
if [[ ! -f "${G1_URDF}" ]]; then
    echo "G1+BrainCo URDF/mesh bundle is missing: ${G1_URDF}" >&2
    exit 1
fi
if [[ ! -d "${SIMTOOLREAL_ROOT}/.git" ]]; then
    git clone https://github.com/tylerlum/simtoolreal.git "${SIMTOOLREAL_ROOT}"
fi
if [[ "$(git -C "${SIMTOOLREAL_ROOT}" rev-parse HEAD)" != "${EXPECTED_SIMTOOLREAL_COMMIT}" ]]; then
    echo "SimToolReal reference checkout must be pinned to ${EXPECTED_SIMTOOLREAL_COMMIT}" >&2
    exit 1
fi
if [[ ! -f "${PCA_ARTIFACT}" ]]; then
    echo "Frozen Revo2 PCA artifact is missing: ${PCA_ARTIFACT}" >&2
    exit 1
fi
if [[ "$(sha256sum "${PCA_ARTIFACT}" | cut -d' ' -f1)" != "${EXPECTED_PCA_SHA256}" ]]; then
    echo "Frozen Revo2 PCA artifact checksum mismatch" >&2
    exit 1
fi

PYTHON="${ADEPT_ENV_PATH}/bin/python"
"${PYTHON}" -m pip install --no-deps -e "${PLAY2PERFECT_ROOT}/rl_games"
"${PYTHON}" -m pip install --no-deps -e "${PLAY2PERFECT_ROOT}" -e "${REPO_ROOT}"
"${PYTHON}" -m pytest -q \
    "${REPO_ROOT}/tests/test_g1_reduced_fabric.py" \
    "${REPO_ROOT}/tests/test_g1_collision_geometry.py" \
    "${REPO_ROOT}/tests/test_g1_isaac_adapter.py" \
    "${REPO_ROOT}/tests/test_revo2_pca_control.py"
"${PYTHON}" "${REPO_ROOT}/scripts/visualize_g1_adept_fabric.py" \
    --validate-only \
    --urdf "${G1_URDF}"

echo "G1 ADEPT/SAPG environment ready: ${ADEPT_ENV_PATH}"
echo "Smoke: sbatch ${REPO_ROOT}/scripts/slurm/smoke_g1_adept_sapg.sbatch"
echo "Train: sbatch ${REPO_ROOT}/scripts/slurm/train_g1_adept_sapg_single.sbatch"
