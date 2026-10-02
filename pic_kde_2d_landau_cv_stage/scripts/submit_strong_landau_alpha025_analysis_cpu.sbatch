#!/bin/bash -l
#SBATCH --job-name=p1_SL25_an
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
#SBATCH --output=/mnt/tier2/users/u103576/logs/p1_SL25_an_%j.out
#SBATCH --error=/mnt/tier2/users/u103576/logs/p1_SL25_an_%j.err

set -euo pipefail
: "${CV_ROOT:?set CV_ROOT}"
: "${RATE_ROOT:?set RATE_ROOT}"
: "${OUTPUT_ROOT:?set OUTPUT_ROOT}"
PROJECT=/home/users/u103576/hexapic-main/pic_kde_2d
PYROOT=/apps/USE/easybuild/release/2024.1/software/Python/3.11.10-GCCcore-13.3.0
PYTHON=${PYROOT}/bin/python3
export LD_LIBRARY_PATH="${PYROOT}/lib:${PYROOT}/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${PROJECT}:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd "${PROJECT}"
mkdir -p "${OUTPUT_ROOT}/cv" "${OUTPUT_ROOT}/fidelity" "${OUTPUT_ROOT}/merged"
"${PYTHON}" scripts/run_landau_lscv_aligned.py reduce \
  --input-root "${CV_ROOT}" --output "${OUTPUT_ROOT}/cv" \
  --expected-seeds 10 --ppc 1024 --eval-t-min 3 --eval-t-max 12
"${PYTHON}" scripts/analyze_strong_landau_fidelity.py \
  --input "${RATE_ROOT}" --output "${OUTPUT_ROOT}/fidelity" \
  --alpha 0.25 --cv-width 1.3958 \
  --sigmas "0,1,1.25,1.3958,1.5,2,2.453" \
  --fit-start 3 --fit-stop-max 30 --bounce-search-start 8 \
  --bounce-search-stop 35 --min-duration 5 --min-r-squared 0.98 \
  --min-model-efoldings 3
"${PYTHON}" scripts/summarize_strong_landau_alpha025.py \
  --cv "${OUTPUT_ROOT}/cv/landau_lscv_result.json" \
  --fidelity "${OUTPUT_ROOT}/fidelity/strong_landau_fidelity.json" \
  --output "${OUTPUT_ROOT}/merged"
