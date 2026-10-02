#!/bin/bash
#SBATCH --job-name=pic-kde-E1split
#SBATCH --account=p201103
#SBATCH --partition=cpu
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --array=0-31
#SBATCH --output=/mnt/tier2/users/u103576/logs/pic_kde_2d_E1split_%A_%a.out
#SBATCH --error=/mnt/tier2/users/u103576/logs/pic_kde_2d_E1split_%A_%a.err

set -euo pipefail

PROJECT_ROOT="/home/users/u103576/hexapic-main/pic_kde_2d"
PYTHON_ROOT="/apps/USE/easybuild/release/2024.1/software/Python/3.11.10-GCCcore-13.3.0"
PYTHON_BIN="${PYTHON_ROOT}/bin/python3"
export LD_LIBRARY_PATH="${PYTHON_ROOT}/lib:${PYTHON_ROOT}/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

K_MODES_CSV="${K_MODES_CSV:-1,2,4,8}"
SIGMA_DX_CSV="${SIGMA_DX_CSV:-0.25,0.5,0.75,1.0,1.5,2.0,3.0,4.0}"
TAG="${TAG:-split}"
SEED_START="${SEED_START:-0}"
SEED_COUNT="${SEED_COUNT:-10}"
PPC="${PPC:-64}"
NY="${NY:-32}"
NSTEPS="${NSTEPS:-600}"
FIT_T_MIN="${FIT_T_MIN:-5.0}"
FIT_T_MAX="${FIT_T_MAX:-30.0}"
NBOOT="${NBOOT:-1000}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/tier2/users/u103576/logs/pic_kde_2d_E1split_${SLURM_ARRAY_JOB_ID}}"

IFS=',' read -r -a K_MODES <<< "${K_MODES_CSV}"
IFS=',' read -r -a SIGMAS <<< "${SIGMA_DX_CSV}"

N_SIGMAS="${#SIGMAS[@]}"
K_INDEX=$((SLURM_ARRAY_TASK_ID / N_SIGMAS))
S_INDEX=$((SLURM_ARRAY_TASK_ID % N_SIGMAS))

if (( K_INDEX >= ${#K_MODES[@]} )); then
  echo "Task ${SLURM_ARRAY_TASK_ID} outside configured matrix; exiting."
  exit 0
fi

K_MODE="${K_MODES[$K_INDEX]}"
SIGMA_DX="${SIGMAS[$S_INDEX]}"
SHARD_DIR="${OUTPUT_ROOT}/shards/k${K_MODE}_sigma${SIGMA_DX}"

mkdir -p "${SHARD_DIR}"
cd "${PROJECT_ROOT}"

echo "E1 split shard: k=${K_MODE} sigma_dx=${SIGMA_DX} seeds=${SEED_START}..$((SEED_START + SEED_COUNT - 1))"

"${PYTHON_BIN}" scripts/run_E1_sigma_kmax.py \
  --output "${SHARD_DIR}" \
  --tag "${TAG}_k${K_MODE}_s${SIGMA_DX}" \
  --k-modes "${K_MODE}" \
  --sigma-dx "${SIGMA_DX}" \
  --seed-start "${SEED_START}" \
  --seed-count "${SEED_COUNT}" \
  --ppc "${PPC}" \
  --Ny "${NY}" \
  --n-steps "${NSTEPS}" \
  --fit-t-min "${FIT_T_MIN}" \
  --fit-t-max "${FIT_T_MAX}" \
  --nboot "${NBOOT}"

echo "Shard complete: ${SHARD_DIR}"
