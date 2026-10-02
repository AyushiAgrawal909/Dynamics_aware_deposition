#!/bin/bash
#SBATCH --job-name=pic-kde-E2specMISE
#SBATCH --account=p201103
#SBATCH --partition=gpu
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/tier2/users/u103576/logs/pic_kde_2d_E2_spec_mise_gpu_%j.out
#SBATCH --error=/mnt/tier2/users/u103576/logs/pic_kde_2d_E2_spec_mise_gpu_%j.err

set -euo pipefail

PROJECT_ROOT="/home/users/u103576/hexapic-main/pic_kde_2d"
PYTHON_ROOT="/apps/USE/easybuild/release/2024.1/software/Python/3.11.10-GCCcore-13.3.0"
PYTHON_BIN="${PYTHON_ROOT}/bin/python3"
export LD_LIBRARY_PATH="${PYTHON_ROOT}/lib:${PYTHON_ROOT}/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

TAG="${TAG:-E2_spectral_mise_decomp}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/tier2/users/u103576/logs/pic_kde_2d_${TAG}_${SLURM_JOB_ID}}"

mkdir -p "${OUTPUT_ROOT}"
cd "${PROJECT_ROOT}"

"${PYTHON_BIN}" scripts/run_E2_spectral_mise_decomposition.py \
  --output "${OUTPUT_ROOT}" \
  --seeds 10 \
  --sigma-dx 0.0,0.25,0.30,0.35,0.45,0.68,1.0,3.0,12.0

