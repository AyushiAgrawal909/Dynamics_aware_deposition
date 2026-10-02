#!/bin/bash -l
#SBATCH --job-name=p1_cb_noise
#SBATCH --account=p201103
#SBATCH --partition=cpu
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=20G
#SBATCH --time=04:00:00
#SBATCH --array=0-19%8
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=ayushiphysics909@gmail.com
#SBATCH --output=/mnt/tier2/users/u103576/logs/p1_cb_noise_%A_%a.out
#SBATCH --error=/mnt/tier2/users/u103576/logs/p1_cb_noise_%A_%a.err

set -euo pipefail
: "${OUTPUT_ROOT:?set OUTPUT_ROOT for the compensated-binomial noise shards}"
PROJECT=/home/users/u103576/hexapic-main/pic_kde_2d
PYROOT=/apps/USE/easybuild/release/2024.1/software/Python/3.11.10-GCCcore-13.3.0
PYTHON=${PYROOT}/bin/python3
BENCH_INDEX=$((SLURM_ARRAY_TASK_ID / 10))
SEED=$((SLURM_ARRAY_TASK_ID % 10))

export LD_LIBRARY_PATH="${PYROOT}/lib:${PYROOT}/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${PROJECT}:${PYTHONPATH:-}"
export NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd "${PROJECT}"

if [[ "${BENCH_INDEX}" -eq 0 ]]; then
  BENCH=landau
  PARAMS=(--benchmark landau_weak --ppc 1024 --reference-ppc 16384 --alpha 0.01 --n-steps 600 --eval-t-min 5 --eval-t-max 30)
else
  BENCH=e2
  PARAMS=(--benchmark e2 --ppc 32 --reference-ppc 1024 --alpha 0.001 --n-steps 80 --eval-t-min 0.5 --eval-t-max 4)
fi
OUT=${OUTPUT_ROOT}/${BENCH}/seed${SEED}
mkdir -p "${OUT}/numba"
export NUMBA_CACHE_DIR="${OUT}/numba"
"${PYTHON}" scripts/run_binomial_noise_seed.py \
  --output "${OUT}" --seed "${SEED}" \
  --filter-family compensated_binomial \
  --passes 0 1 2 4 8 16 "${PARAMS[@]}"
