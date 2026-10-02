#!/bin/bash -l
# Frozen 40-seed PPC-1024 weak-Landau campaign. Each array task runs all three
# arms sequentially from byte-identical initial particle arrays.
#SBATCH --job-name=p1_weak_confirm_low
#SBATCH --account=p201103
#SBATCH --partition=cpu
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --array=0-39%8
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ayushiphysics909@gmail.com
#SBATCH --output=/mnt/tier2/users/u103576/logs/p1_weak_confirm_low_%A_%a.out
#SBATCH --error=/mnt/tier2/users/u103576/logs/p1_weak_confirm_low_%A_%a.err

set -euo pipefail

PROJECT=/home/users/u103576/hexapic-main/pic_kde_2d
CONFIRMATORY=/home/users/u103576/hexapic-main/paper1_source_package/reproducibility
CONFIG=${CONFIRMATORY}/weak_landau_confirmatory_v1.json
PYROOT=/apps/USE/easybuild/release/2024.1/software/Python/3.11.10-GCCcore-13.3.0
PYTHON=${PYROOT}/bin/python3
ROOT=/mnt/tier2/users/u103576/paper1_validation/weak_landau_confirmatory_v1
SEED=$((202609100 + SLURM_ARRAY_TASK_ID))
CACHE=${ROOT}/numba/low_seed${SEED}

export LD_LIBRARY_PATH="${PYROOT}/lib:${PYROOT}/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${PROJECT}:${PROJECT}/scripts:${PYTHONPATH:-}"
export NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mkdir -p "${CACHE}"
cd "${PROJECT}"

env NUMBA_CACHE_DIR="${CACHE}" \
  "${PYTHON}" scripts/run_weak_landau_confirmatory.py --config "${CONFIG}" \
    --preflight "${ROOT}/preflight/preflight.json" \
    simulate-low-seed --output "${ROOT}/low" --seed "${SEED}"
