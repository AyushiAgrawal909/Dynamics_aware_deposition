#!/bin/bash -l
#SBATCH --job-name=p1_SL25_cv
#SBATCH --account=p201103
#SBATCH --partition=cpu
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --time=01:00:00
#SBATCH --array=0-9%5
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=ayushiphysics909@gmail.com
#SBATCH --output=/mnt/tier2/users/u103576/logs/p1_SL25_cv_%A_%a.out
#SBATCH --error=/mnt/tier2/users/u103576/logs/p1_SL25_cv_%A_%a.err

set -euo pipefail
: "${OUTPUT_ROOT:?set OUTPUT_ROOT}"
PROJECT=/home/users/u103576/hexapic-main/pic_kde_2d
PYROOT=/apps/USE/easybuild/release/2024.1/software/Python/3.11.10-GCCcore-13.3.0
PYTHON=${PYROOT}/bin/python3
OUT=${OUTPUT_ROOT}/seed${SLURM_ARRAY_TASK_ID}
mkdir -p "${OUT}/numba"
export LD_LIBRARY_PATH="${PYROOT}/lib:${PYROOT}/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${PROJECT}:${PYTHONPATH:-}"
export NUMBA_CACHE_DIR="${OUT}/numba"
export NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd "${PROJECT}"
"${PYTHON}" scripts/run_landau_lscv_aligned.py scan \
  --output "${OUT}" --seed-start "${SLURM_ARRAY_TASK_ID}" --seeds 1 \
  --ppc 1024 --alpha 0.25 --live-sigma-dx 0.25 \
  --n-steps 240 --diag-stride 2 --eval-t-min 3 --eval-t-max 12 \
  --sigma-dx "0,0.25,0.5,0.75,1,1.1,1.2,1.25,1.3,1.35,1.3958,1.45,1.5,1.6,1.8,2,2.25,2.5,3,4"
