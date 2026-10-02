#!/bin/bash
#SBATCH --job-name=pic-kde-E1rev
#SBATCH --account=p201103
#SBATCH --partition=gpu
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --time=04:00:00
#SBATCH --array=0-5
#SBATCH --output=/mnt/tier2/users/u103576/logs/pic_kde_2d_E1revised_%A_%a.out
#SBATCH --error=/mnt/tier2/users/u103576/logs/pic_kde_2d_E1revised_%A_%a.err

set -euo pipefail

PROJECT_ROOT="/home/users/u103576/hexapic-main/pic_kde_2d"
PYTHON_ROOT="/apps/USE/easybuild/release/2024.1/software/Python/3.11.10-GCCcore-13.3.0"
PYTHON_BIN="${PYTHON_ROOT}/bin/python3"
export LD_LIBRARY_PATH="${PYTHON_ROOT}/lib:${PYTHON_ROOT}/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

K_MODES=(1 2 4 8)
SIGMAS=(0.25 0.5 0.75 1.0 1.5 2.0 2.5 3.0 3.5 4.0 4.5)
CASES_PER_JOB=8

TAG="${TAG:-e1revised30}"
SEED_START="${SEED_START:-0}"
SEEDS="${SEEDS:-30}"
PPC="${PPC:-32}"
NY="${NY:-32}"
NSTEPS="${NSTEPS:-600}"
NBOOT="${NBOOT:-1000}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/tier2/users/u103576/logs/pic_kde_2d_E1revised_${SLURM_ARRAY_JOB_ID}}"

cd "${PROJECT_ROOT}"

# Each case is an independent NumPy process. Avoid oversubscribing BLAS
# threads while filling the eight requested CPU cores of one GPU allocation.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

N_SIGMAS="${#SIGMAS[@]}"
N_CASES=$((${#K_MODES[@]} * N_SIGMAS))
FIRST_CASE=$((SLURM_ARRAY_TASK_ID * CASES_PER_JOB))
LAST_CASE=$((FIRST_CASE + CASES_PER_JOB - 1))
if (( LAST_CASE >= N_CASES )); then
  LAST_CASE=$((N_CASES - 1))
fi

echo "E1 revised packed job: cases=${FIRST_CASE}..${LAST_CASE} seeds=${SEED_START}..$((SEED_START + SEEDS - 1))"

pids=()
for ((CASE_ID=FIRST_CASE; CASE_ID<=LAST_CASE; CASE_ID++)); do
  K_INDEX=$((CASE_ID / N_SIGMAS))
  S_INDEX=$((CASE_ID % N_SIGMAS))
  K_MODE="${K_MODES[$K_INDEX]}"
  SIGMA_DX="${SIGMAS[$S_INDEX]}"
  SHARD_DIR="${OUTPUT_ROOT}/shards/k${K_MODE}_sigma${SIGMA_DX}"
  mkdir -p "${SHARD_DIR}"

  (
    echo "E1 revised shard: case=${CASE_ID} k=${K_MODE} sigma_dx=${SIGMA_DX}"
    "${PYTHON_BIN}" scripts/run_E1_revised.py \
      --output "${SHARD_DIR}" \
      --tag "${TAG}_k${K_MODE}_s${SIGMA_DX}" \
      --k-modes "${K_MODE}" \
      --sigma-dx "${SIGMA_DX}" \
      --seed-start "${SEED_START}" \
      --seeds "${SEEDS}" \
      --ppc "${PPC}" \
      --Ny "${NY}" \
      --n-steps "${NSTEPS}" \
      --fit-mode fixed \
      --nboot "${NBOOT}" \
      >"${SHARD_DIR}/run.out" 2>"${SHARD_DIR}/run.err"
  ) &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done

if (( status != 0 )); then
  echo "One or more E1 revised shards failed." >&2
  exit "${status}"
fi

echo "E1 revised packed job complete: cases=${FIRST_CASE}..${LAST_CASE}"
