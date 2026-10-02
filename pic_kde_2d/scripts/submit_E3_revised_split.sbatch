#!/bin/bash
#SBATCH --job-name=pic-kde-E3rev
#SBATCH --account=p201103
#SBATCH --partition=gpu
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --time=04:00:00
#SBATCH --array=0-19
#SBATCH --output=/mnt/tier2/users/u103576/logs/pic_kde_2d_E3revised_%A_%a.out
#SBATCH --error=/mnt/tier2/users/u103576/logs/pic_kde_2d_E3revised_%A_%a.err

set -euo pipefail

PROJECT_ROOT="/home/users/u103576/hexapic-main/pic_kde_2d"
PYTHON_ROOT="/apps/USE/easybuild/release/2024.1/software/Python/3.11.10-GCCcore-13.3.0"
PYTHON_BIN="${PYTHON_ROOT}/bin/python3"
export LD_LIBRARY_PATH="${PYTHON_ROOT}/lib:${PYTHON_ROOT}/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

TAG="${TAG:-e3revised20}"
PPC_LOW="${PPC_LOW:-16}"
PPC_REF="${PPC_REF:-256}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/tier2/users/u103576/logs/pic_kde_2d_E3revised_${SLURM_ARRAY_JOB_ID}}"

SEED="${SLURM_ARRAY_TASK_ID}"
SHARD_DIR="${OUTPUT_ROOT}/shards/seed${SEED}"

mkdir -p "${SHARD_DIR}"
cd "${PROJECT_ROOT}"

echo "E3 revised shard: seed=${SEED} ppc_low=${PPC_LOW} ppc_ref=${PPC_REF}"

"${PYTHON_BIN}" scripts/run_E3_revised.py \
  --output "${SHARD_DIR}" \
  --tag "${TAG}_seed${SEED}" \
  --seed-start "${SEED}" \
  --seeds 1 \
  --ppc-low "${PPC_LOW}" \
  --ppc-ref "${PPC_REF}"

echo "E3 revised shard complete: ${SHARD_DIR}"
