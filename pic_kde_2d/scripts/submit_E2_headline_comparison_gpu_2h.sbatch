#!/bin/bash
#SBATCH --job-name=pic-kde-E2headline
#SBATCH --account=p201103
#SBATCH --partition=gpu
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/tier2/users/u103576/logs/pic_kde_2d_E2headline_gpu_%j.out
#SBATCH --error=/mnt/tier2/users/u103576/logs/pic_kde_2d_E2headline_gpu_%j.err

set -euo pipefail

PROJECT_ROOT="/home/users/u103576/hexapic-main/pic_kde_2d"
PYTHON_ROOT="/apps/USE/easybuild/release/2024.1/software/Python/3.11.10-GCCcore-13.3.0"
PYTHON_BIN="${PYTHON_ROOT}/bin/python3"
export LD_LIBRARY_PATH="${PYTHON_ROOT}/lib:${PYTHON_ROOT}/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/mpl"

TAG="${TAG:-e2headline30_gpu2h}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/tier2/users/u103576/logs/pic_kde_2d_${TAG}_${SLURM_JOB_ID}}"

mkdir -p "${OUTPUT_ROOT}"
cd "${PROJECT_ROOT}"

"${PYTHON_BIN}" scripts/run_E2_headline_comparison.py \
  --output "${OUTPUT_ROOT}" \
  --seeds 30 \
  --n-steps 300 \
  --nboot 1000
