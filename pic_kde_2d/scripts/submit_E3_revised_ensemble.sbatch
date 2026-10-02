#!/bin/bash
#SBATCH --job-name=pic-kde-E3rev-ens
#SBATCH --account=p201103
#SBATCH --partition=gpu
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --time=04:00:00
#SBATCH --output=/mnt/tier2/users/u103576/logs/pic_kde_2d_E3revised_ensemble_%j.out
#SBATCH --error=/mnt/tier2/users/u103576/logs/pic_kde_2d_E3revised_ensemble_%j.err

set -euo pipefail

PROJECT_ROOT="/home/users/u103576/hexapic-main/pic_kde_2d"
PYTHON_ROOT="/apps/USE/easybuild/release/2024.1/software/Python/3.11.10-GCCcore-13.3.0"
PYTHON_BIN="${PYTHON_ROOT}/bin/python3"
export LD_LIBRARY_PATH="${PYTHON_ROOT}/lib:${PYTHON_ROOT}/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

TAG="${TAG:-e3revised_ensemble20}"
SEED_START="${SEED_START:-0}"
SEEDS="${SEEDS:-20}"
PPC_LOW="${PPC_LOW:-16}"
PPC_REF="${PPC_REF:-256}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/tier2/users/u103576/logs/pic_kde_2d_E3revised_ensemble_${SLURM_JOB_ID}}"

mkdir -p "${OUTPUT_ROOT}"
cd "${PROJECT_ROOT}"

"${PYTHON_BIN}" scripts/run_E3_revised.py \
  --output "${OUTPUT_ROOT}" \
  --tag "${TAG}" \
  --seed-start "${SEED_START}" \
  --seeds "${SEEDS}" \
  --ppc-low "${PPC_LOW}" \
  --ppc-ref "${PPC_REF}"
