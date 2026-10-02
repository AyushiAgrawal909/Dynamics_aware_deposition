#!/bin/bash -l
#SBATCH --job-name=p1_cb_an
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
#SBATCH --output=/mnt/tier2/users/u103576/logs/p1_cb_an_%j.out
#SBATCH --error=/mnt/tier2/users/u103576/logs/p1_cb_an_%j.err

set -euo pipefail
: "${RATE_ROOT:?set RATE_ROOT}"
: "${NOISE_ROOT:?set NOISE_ROOT}"
: "${OUTPUT_ROOT:?set OUTPUT_ROOT}"
PROJECT=/home/users/u103576/hexapic-main/pic_kde_2d
PYROOT=/apps/USE/easybuild/release/2024.1/software/Python/3.11.10-GCCcore-13.3.0
PYTHON=${PYROOT}/bin/python3
export LD_LIBRARY_PATH="${PYROOT}/lib:${PYROOT}/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${PROJECT}:${PROJECT}/scripts:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd "${PROJECT}"

mkdir -p "${OUTPUT_ROOT}/landau_rate" "${OUTPUT_ROOT}/e2_rate" \
  "${OUTPUT_ROOT}/landau_noise" "${OUTPUT_ROOT}/e2_noise"
"${PYTHON}" scripts/analyze_landau_binomial_passes.py \
  --input "${RATE_ROOT}/landau/shards" --output "${OUTPUT_ROOT}/landau_rate" \
  --alpha 0.01 --filter-family compensated_binomial \
  --passes 0 1 2 4 8 16
"${PYTHON}" scripts/analyze_E2_binomial_passes.py \
  --input "${RATE_ROOT}/e2/shards" --output "${OUTPUT_ROOT}/e2_rate" \
  --filter-family compensated_binomial --passes 0 1 2 4 8 16
"${PYTHON}" scripts/reduce_binomial_noise.py \
  --input "${NOISE_ROOT}/landau" --output "${OUTPUT_ROOT}/landau_noise"
"${PYTHON}" scripts/reduce_binomial_noise.py \
  --input "${NOISE_ROOT}/e2" --output "${OUTPUT_ROOT}/e2_noise"
"${PYTHON}" scripts/merge_compensated_binomial_campaign.py \
  --landau-rate "${OUTPUT_ROOT}/landau_rate/landau_binomial_passes.json" \
  --landau-noise "${OUTPUT_ROOT}/landau_noise/binomial_noise_summary.json" \
  --e2-rate "${OUTPUT_ROOT}/e2_rate/E2_binomial_passes.json" \
  --e2-noise "${OUTPUT_ROOT}/e2_noise/binomial_noise_summary.json" \
  --output "${OUTPUT_ROOT}/merged"
