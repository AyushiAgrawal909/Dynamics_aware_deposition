#!/bin/bash
#SBATCH --job-name=pic-kde-2d-p2
#SBATCH --account=p201103
#SBATCH --partition=cpu
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:45:00
#SBATCH --output=/mnt/tier2/users/u103576/logs/pic_kde_2d_phase2_%j.out
#SBATCH --error=/mnt/tier2/users/u103576/logs/pic_kde_2d_phase2_%j.err

set -euo pipefail

PROJECT_ROOT="/home/users/u103576/hexapic-main/pic_kde_2d"
PYTHON_ROOT="/apps/USE/easybuild/release/2024.1/software/Python/3.11.10-GCCcore-13.3.0"
PYTHON_BIN="${PYTHON_ROOT}/bin/python3"
export LD_LIBRARY_PATH="${PYTHON_ROOT}/lib:${PYTHON_ROOT}/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
OUTDIR="/mnt/tier2/users/u103576/logs/pic_kde_2d_phase2_${SLURM_JOB_ID}"

mkdir -p "${OUTDIR}"
cd "${PROJECT_ROOT}"

"${PYTHON_BIN}" -m pip install --user -e . --no-deps
"${PYTHON_BIN}" "${PROJECT_ROOT}/pic_kde_2d/validation/run_phase2_validation.py" \
  --outdir "${OUTDIR}"
