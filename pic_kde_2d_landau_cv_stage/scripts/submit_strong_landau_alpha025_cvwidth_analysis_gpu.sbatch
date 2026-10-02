#!/bin/bash -l
#SBATCH --job-name=p1_SL25_cvwidth_an
#SBATCH --account=p201103
#SBATCH --partition=gpu
#SBATCH --qos=default
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-node=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ayushiphysics909@gmail.com
#SBATCH --output=/mnt/tier2/users/u103576/logs/p1_SL25_cvwidth_an_%j.out
#SBATCH --error=/mnt/tier2/users/u103576/logs/p1_SL25_cvwidth_an_%j.err

set -euo pipefail
: "${RATE_ROOT:?set RATE_ROOT}"
: "${OUTPUT_ROOT:?set OUTPUT_ROOT}"

PROJECT=/home/users/u103576/hexapic-main/pic_kde_2d
PYROOT=/apps/USE/easybuild/release/2024.1/software/Python/3.11.10-GCCcore-13.3.0
PYTHON=${PYROOT}/bin/python3
SIGMA_DX=${SIGMA_DX:-2.8208226}

export LD_LIBRARY_PATH="${PYROOT}/lib:${PYROOT}/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${PROJECT}:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd "${PROJECT}"
mkdir -p "${OUTPUT_ROOT}"

"${PYTHON}" scripts/analyze_strong_landau_fidelity.py \
  --input "${RATE_ROOT}" --output "${OUTPUT_ROOT}" \
  --alpha 0.25 --cv-width "${SIGMA_DX}" --sigmas "0,${SIGMA_DX}" \
  --fit-start 3 --fit-stop-max 30 --bounce-search-start 8 \
  --bounce-search-stop 35 --min-duration 5 --min-r-squared 0.98 \
  --min-model-efoldings 3
