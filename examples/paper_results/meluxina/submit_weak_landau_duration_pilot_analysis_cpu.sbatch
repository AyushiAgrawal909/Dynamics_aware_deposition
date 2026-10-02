#!/bin/bash -l
#SBATCH --job-name=p1_weak_reduce
#SBATCH --account=p201103
#SBATCH --partition=cpu
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ayushiphysics909@gmail.com
#SBATCH --output=/mnt/tier2/users/u103576/logs/p1_weak_reduce_%j.out
#SBATCH --error=/mnt/tier2/users/u103576/logs/p1_weak_reduce_%j.err

set -euo pipefail
: "${RATE_JOB_ID:?set RATE_JOB_ID to the completed pilot array job ID}"

PROJECT=/home/users/u103576/hexapic-main/pic_kde_2d
PYROOT=/apps/USE/easybuild/release/2024.1/software/Python/3.11.10-GCCcore-13.3.0
PYTHON=${PYROOT}/bin/python3
ROOT=/mnt/tier2/users/u103576/paper1_validation/weak_landau_duration_pilot_${RATE_JOB_ID}

export LD_LIBRARY_PATH="${PYROOT}/lib:${PYROOT}/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${PROJECT}:${PROJECT}/scripts:${PYTHONPATH:-}"
export NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd "${PROJECT}"

"${PYTHON}" scripts/run_landau_sigma_refinement.py analyze \
  --input "${ROOT}/gaussian" --output "${ROOT}/analysis_gaussian" \
  --ppcs 1024 --sigma-grid 0,1 --seeds 3 --nboot 2000 \
  --fit-t-start 5 --fit-t-stop-max 30 --noise-floor-start 70 --noise-floor-stop 90 \
  --min-duration 5 --min-model-efoldings 2 --min-r-squared 0.98

"${PYTHON}" scripts/analyze_landau_binomial_passes.py \
  --input "${ROOT}/compensated" --output "${ROOT}/analysis_compensated" \
  --alpha 0.02 --seeds 3 --nboot 2000 \
  --filter-family compensated_binomial --compensation-order 1 --passes 0 16

