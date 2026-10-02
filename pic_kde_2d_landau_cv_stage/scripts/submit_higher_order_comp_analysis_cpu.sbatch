#!/bin/bash -l
#SBATCH --job-name=p1_hoc_an
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
#SBATCH --output=/mnt/tier2/users/u103576/logs/p1_hoc_an_%j.out
#SBATCH --error=/mnt/tier2/users/u103576/logs/p1_hoc_an_%j.err

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

for ORDER in 2 3; do
  BASE=${OUTPUT_ROOT}/p${ORDER}
  mkdir -p "${BASE}/landau_rate" "${BASE}/e2_rate" \
    "${BASE}/landau_noise" "${BASE}/e2_noise"
  "${PYTHON}" scripts/analyze_landau_binomial_passes.py \
    --input "${RATE_ROOT}/p${ORDER}/landau/shards" \
    --output "${BASE}/landau_rate" --alpha 0.01 \
    --filter-family compensated_binomial --compensation-order "${ORDER}" \
    --passes 0 1 2 4
  "${PYTHON}" scripts/analyze_E2_binomial_passes.py \
    --input "${RATE_ROOT}/p${ORDER}/e2/shards" \
    --output "${BASE}/e2_rate" \
    --filter-family compensated_binomial --compensation-order "${ORDER}" \
    --passes 0 1 2 4
  "${PYTHON}" scripts/reduce_binomial_noise.py \
    --input "${NOISE_ROOT}/landau" --output "${BASE}/landau_noise" \
    --compensation-order "${ORDER}"
  "${PYTHON}" scripts/reduce_binomial_noise.py \
    --input "${NOISE_ROOT}/e2" --output "${BASE}/e2_noise" \
    --compensation-order "${ORDER}"
done

"${PYTHON}" scripts/merge_higher_order_compensation.py \
  --input "${OUTPUT_ROOT}" --output "${OUTPUT_ROOT}/merged"
