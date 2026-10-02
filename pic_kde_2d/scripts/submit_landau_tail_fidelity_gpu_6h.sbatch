#!/bin/bash
#SBATCH --job-name=pic-kde-landau-tail
#SBATCH --account=p201103
#SBATCH --partition=gpu
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --time=06:00:00
#SBATCH --output=/mnt/tier2/users/u103576/logs/pic_kde_2d_landau_tail_gpu_%j.out
#SBATCH --error=/mnt/tier2/users/u103576/logs/pic_kde_2d_landau_tail_gpu_%j.err

set -euo pipefail

PROJECT_ROOT="/home/users/u103576/hexapic-main/pic_kde_2d"
PYTHON_ROOT="/apps/USE/easybuild/release/2024.1/software/Python/3.11.10-GCCcore-13.3.0"
PYTHON_BIN="${PYTHON_ROOT}/bin/python3"
export LD_LIBRARY_PATH="${PYTHON_ROOT}/lib:${PYTHON_ROOT}/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

TAG="${TAG:-landau_tail_fidelity_60}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/tier2/users/u103576/logs/pic_kde_2d_${TAG}_${SLURM_JOB_ID}}"

mkdir -p "${OUTPUT_ROOT}"
cd "${PROJECT_ROOT}"

"${PYTHON_BIN}" scripts/run_landau_tail_fidelity.py \
  --output "${OUTPUT_ROOT}" \
  --sigma-dx 0.25,3,4,6,8,12 \
  --seeds 60 \
  --ppc 64 \
  --nboot 1000

