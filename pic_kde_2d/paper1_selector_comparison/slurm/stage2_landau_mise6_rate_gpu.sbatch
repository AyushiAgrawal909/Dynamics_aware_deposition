#!/bin/bash -l
#SBATCH --job-name=p1_L_mise6
#SBATCH --account=p201103
#SBATCH --partition=gpu
#SBATCH --qos=default
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --gpus-per-node=1
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ayushiphysics909@gmail.com
#SBATCH --output=/mnt/tier2/users/u103576/logs/p1_L_mise6_%j.out
#SBATCH --error=/mnt/tier2/users/u103576/logs/p1_L_mise6_%j.err

set -euo pipefail

PROJECT=/home/users/u103576/hexapic-main/pic_kde_2d
PYROOT=/apps/USE/easybuild/release/2024.1/software/Python/3.11.10-GCCcore-13.3.0
PYTHON=${PYROOT}/bin/python3
RUNNER=${PROJECT}/scripts/run_landau_sigma_refinement.py
ROOT=${OUTPUT_ROOT:-${PROJECT}/results/landau_mise6_rate_${SLURM_JOB_ID}}
BASELINE=${PROJECT}/results/landau_sigma_refinement_4864819/shards/ppc1024/sigma0

export LD_LIBRARY_PATH="${PYROOT}/lib:${PYROOT}/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${PROJECT}:${PYTHONPATH:-}"
export NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

mkdir -p "${ROOT}/shards/ppc1024/sigma6"
cd "${PROJECT}"

pids=()
for seed in $(seq 0 9); do
  OUT=${ROOT}/shards/ppc1024/sigma6/seed${seed}
  mkdir -p "${OUT}/numba"
  env NUMBA_CACHE_DIR="${OUT}/numba" \
    "${PYTHON}" "${RUNNER}" simulate \
    --output "${OUT}" --ppc 1024 --sigma-dx 6 --seed "${seed}" \
    --alpha 0.01 --n-steps 1800 >"${OUT}/run.log" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  wait "${pid}" || status=1
done
if [ "${status}" -ne 0 ]; then
  exit "${status}"
fi

ln -s "${BASELINE}" "${ROOT}/shards/ppc1024/sigma0"
"${PYTHON}" "${RUNNER}" analyze \
  --input "${ROOT}/shards" \
  --output "${ROOT}/analysis" \
  --ppcs 1024 \
  --sigma-grid 0,6 \
  --seed-start 0 \
  --seeds 10 \
  --nboot 2000

echo "complete output=${ROOT}"
