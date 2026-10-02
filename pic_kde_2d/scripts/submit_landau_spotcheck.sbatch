#!/bin/bash
#SBATCH --job-name=pic-kde-landau-spot
#SBATCH --account=p201103
#SBATCH --partition=cpu
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --output=/mnt/tier2/users/u103576/logs/pic_kde_2d_landau_spot_%j.out
#SBATCH --error=/mnt/tier2/users/u103576/logs/pic_kde_2d_landau_spot_%j.err

set -euo pipefail

PROJECT_ROOT="/home/users/u103576/hexapic-main/pic_kde_2d"
PYTHON_ROOT="/apps/USE/easybuild/release/2024.1/software/Python/3.11.10-GCCcore-13.3.0"
PYTHON_BIN="${PYTHON_ROOT}/bin/python3"
export LD_LIBRARY_PATH="${PYTHON_ROOT}/lib:${PYTHON_ROOT}/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

OUTPUT_DIR="${OUTPUT_DIR:-/mnt/tier2/users/u103576/logs/pic_kde_2d_landau_spot_${SLURM_JOB_ID}}"
mkdir -p "${OUTPUT_DIR}"
cd "${PROJECT_ROOT}"

"${PYTHON_BIN}" -m pip install --user -e . --no-deps

"${PYTHON_BIN}" scripts/run_landau_spotcheck.py \
  --output "${OUTPUT_DIR}" \
  --Nx "${NX:-64}" \
  --Ny "${NY:-8}" \
  --ppc "${PPC:-64}" \
  --dt "${DT:-0.05}" \
  --n-steps "${NSTEPS:-1200}" \
  --diag-stride "${DIAG_STRIDE:-2}" \
  --kx-mode "${KX_MODE:-1}" \
  --k-lambdaD "${K_LAMBDAD:-0.4}" \
  --sigma-dx "${SIGMA_DX:-0.25}" \
  --seed "${SEED:-0}"
