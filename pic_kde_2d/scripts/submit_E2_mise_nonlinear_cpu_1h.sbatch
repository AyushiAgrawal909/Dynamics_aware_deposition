#!/bin/bash
#SBATCH --job-name=pic-kde-E2miseNL
#SBATCH --account=p201103
#SBATCH --partition=cpu
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/tier2/users/u103576/logs/pic_kde_2d_E2miseNL_cpu_%j.out
#SBATCH --error=/mnt/tier2/users/u103576/logs/pic_kde_2d_E2miseNL_cpu_%j.err

set -euo pipefail

PROJECT_ROOT="/home/users/u103576/hexapic-main/pic_kde_2d"
PYTHON_ROOT="/apps/USE/easybuild/release/2024.1/software/Python/3.11.10-GCCcore-13.3.0"
PYTHON_BIN="${PYTHON_ROOT}/bin/python3"
export LD_LIBRARY_PATH="${PYTHON_ROOT}/lib:${PYTHON_ROOT}/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

TAG="${TAG:-e2mise_nonlinear5_ref512_cpu1h}"
SEEDS="${SEEDS:-5}"
PPC="${PPC:-32}"
REFERENCE_PPC="${REFERENCE_PPC:-512}"
N_STEPS="${N_STEPS:-600}"
EVAL_T_MIN="${EVAL_T_MIN:-20.0}"
EVAL_T_MAX="${EVAL_T_MAX:-30.0}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/tier2/users/u103576/logs/pic_kde_2d_${TAG}_${SLURM_JOB_ID}}"

mkdir -p "${OUTPUT_ROOT}"
cd "${PROJECT_ROOT}"

"${PYTHON_BIN}" scripts/run_E2_mise_gate.py \
  --output "${OUTPUT_ROOT}" \
  --seeds "${SEEDS}" \
  --ppc "${PPC}" \
  --reference-ppc "${REFERENCE_PPC}" \
  --n-steps "${N_STEPS}" \
  --eval-t-min "${EVAL_T_MIN}" \
  --eval-t-max "${EVAL_T_MAX}"
