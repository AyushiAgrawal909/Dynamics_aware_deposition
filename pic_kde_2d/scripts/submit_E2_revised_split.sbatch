#!/bin/bash
#SBATCH --job-name=pic-kde-E2rev
#SBATCH --account=p201103
#SBATCH --partition=gpu
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --time=02:00:00
#SBATCH --array=0-6
#SBATCH --output=/mnt/tier2/users/u103576/logs/pic_kde_2d_E2revised_%A_%a.out
#SBATCH --error=/mnt/tier2/users/u103576/logs/pic_kde_2d_E2revised_%A_%a.err

set -euo pipefail

PROJECT_ROOT="/home/users/u103576/hexapic-main/pic_kde_2d"
PYTHON_ROOT="/apps/USE/easybuild/release/2024.1/software/Python/3.11.10-GCCcore-13.3.0"
PYTHON_BIN="${PYTHON_ROOT}/bin/python3"
export LD_LIBRARY_PATH="${PYTHON_ROOT}/lib:${PYTHON_ROOT}/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

SIGMAS=(0.25 0.5 0.68 0.75 1.0 1.25 1.5)

TAG="${TAG:-e2revised30}"
SEED_START="${SEED_START:-0}"
SEEDS="${SEEDS:-30}"
PPC="${PPC:-32}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/tier2/users/u103576/logs/pic_kde_2d_E2revised_${SLURM_ARRAY_JOB_ID}}"

SIGMA_DX="${SIGMAS[$SLURM_ARRAY_TASK_ID]}"
SHARD_DIR="${OUTPUT_ROOT}/shards/sigma${SIGMA_DX}"

mkdir -p "${SHARD_DIR}"
cd "${PROJECT_ROOT}"

echo "E2 revised shard: sigma_dx=${SIGMA_DX} seeds=${SEED_START}..$((SEED_START + SEEDS - 1))"

"${PYTHON_BIN}" scripts/run_E2_revised.py \
  --output "${SHARD_DIR}" \
  --tag "${TAG}_s${SIGMA_DX}" \
  --sigma-dx "${SIGMA_DX}" \
  --seed-start "${SEED_START}" \
  --seeds "${SEEDS}" \
  --ppc "${PPC}"

echo "E2 revised shard complete: ${SHARD_DIR}"
