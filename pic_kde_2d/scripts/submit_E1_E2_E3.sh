#!/bin/bash
#SBATCH --job-name=pic-kde-E123
#SBATCH --account=p201103
#SBATCH --partition=cpu
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/tier2/users/u103576/logs/pic_kde_2d_phase4_%j.out
#SBATCH --error=/mnt/tier2/users/u103576/logs/pic_kde_2d_phase4_%j.err

set -euo pipefail

PROJECT_ROOT="/home/users/u103576/hexapic-main/pic_kde_2d"
PYTHON_ROOT="/apps/USE/easybuild/release/2024.1/software/Python/3.11.10-GCCcore-13.3.0"
PYTHON_BIN="${PYTHON_ROOT}/bin/python3"
export LD_LIBRARY_PATH="${PYTHON_ROOT}/lib:${PYTHON_ROOT}/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

RUN_E1="${RUN_E1:-1}"
RUN_E2="${RUN_E2:-1}"
RUN_E3="${RUN_E3:-1}"
RUN_PLOTS="${RUN_PLOTS:-1}"
TAG="${TAG:-pilot}"

E1_SEED_START="${E1_SEED_START:-0}"
E1_SEED_COUNT="${E1_SEED_COUNT:-5}"
E1_PPC="${E1_PPC:-32}"
E1_NY="${E1_NY:-8}"
E1_SIGMA_DX="${E1_SIGMA_DX:-0.25,0.5,0.75,1.0,1.5,2.0,3.0,4.0}"
E1_NSTEPS="${E1_NSTEPS:-600}"
E1_FIT_T_MIN="${E1_FIT_T_MIN:-5.0}"
E1_FIT_T_MAX="${E1_FIT_T_MAX:-30.0}"

E2_SEED_START="${E2_SEED_START:-0}"
E2_SEED_COUNT="${E2_SEED_COUNT:-5}"
E2_PPC="${E2_PPC:-16}"

E3_SEED_START="${E3_SEED_START:-0}"
E3_SEED_COUNT="${E3_SEED_COUNT:-5}"
E3_PPC_LOW="${E3_PPC_LOW:-8}"
E3_PPC_REF="${E3_PPC_REF:-64}"

OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/tier2/users/u103576/logs/pic_kde_2d_phase4_${SLURM_JOB_ID}}"
PLOT_DIR="${PLOT_DIR:-${OUTPUT_ROOT}/plots}"

mkdir -p "${OUTPUT_ROOT}/E1" "${OUTPUT_ROOT}/E2" "${OUTPUT_ROOT}/E3"
cd "${PROJECT_ROOT}"

"${PYTHON_BIN}" -m pip install --user -e . --no-deps

if [[ "${RUN_E1}" == "1" ]]; then
  echo "=== Running E1 ==="
  "${PYTHON_BIN}" scripts/run_E1_sigma_kmax.py \
    --output "${OUTPUT_ROOT}/E1" \
    --tag "${TAG}" \
    --seed-start "${E1_SEED_START}" \
    --seed-count "${E1_SEED_COUNT}" \
    --ppc "${E1_PPC}" \
    --Ny "${E1_NY}" \
    --sigma-dx "${E1_SIGMA_DX}" \
    --n-steps "${E1_NSTEPS}" \
    --fit-t-min "${E1_FIT_T_MIN}" \
    --fit-t-max "${E1_FIT_T_MAX}"
fi

if [[ "${RUN_E2}" == "1" ]]; then
  echo "=== Running E2 ==="
  "${PYTHON_BIN}" scripts/run_E2_two_stream.py \
    --output "${OUTPUT_ROOT}/E2" \
    --tag "${TAG}" \
    --seed-start "${E2_SEED_START}" \
    --seed-count "${E2_SEED_COUNT}" \
    --ppc "${E2_PPC}"
fi

if [[ "${RUN_E3}" == "1" ]]; then
  echo "=== Running E3 ==="
  "${PYTHON_BIN}" scripts/run_E3_anisotropic.py \
    --output "${OUTPUT_ROOT}/E3" \
    --tag "${TAG}" \
    --seed-start "${E3_SEED_START}" \
    --seed-count "${E3_SEED_COUNT}" \
    --ppc-low "${E3_PPC_LOW}" \
    --ppc-ref "${E3_PPC_REF}"
fi

if [[ "${RUN_PLOTS}" == "1" ]]; then
  echo "=== Plotting Phase 4 results ==="
  mkdir -p "${PLOT_DIR}"
  "${PYTHON_BIN}" scripts/plot_phase4_results.py \
    --results-root "${OUTPUT_ROOT}" \
    --plot-dir "${PLOT_DIR}" \
    --tag "${TAG}"
fi

echo "Phase 4 batch run complete."
