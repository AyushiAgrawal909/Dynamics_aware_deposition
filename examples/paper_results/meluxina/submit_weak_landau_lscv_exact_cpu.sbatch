#!/bin/bash -l
# Paired dynamics run at the exact quadratic LOO-CV selected width.
#SBATCH --job-name=p1_lscv_exact
#SBATCH --account=p201103
#SBATCH --partition=cpu
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=72G
#SBATCH --time=04:00:00
#SBATCH --array=0-1%2
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ayushiphysics909@gmail.com
#SBATCH --output=/mnt/tier2/users/u103576/logs/p1_lscv_exact_%A_%a.out
#SBATCH --error=/mnt/tier2/users/u103576/logs/p1_lscv_exact_%A_%a.err

set -euo pipefail

PROJECT=/home/users/u103576/hexapic-main/pic_kde_2d
PYROOT=/apps/USE/easybuild/release/2024.1/software/Python/3.11.10-GCCcore-13.3.0
PYTHON=${PYROOT}/bin/python3
ROOT=/mnt/tier2/users/u103576/paper1_validation/weak_landau_lscv_exact_${SLURM_ARRAY_JOB_ID}
SIGMAS=(0 7.244293147028428)
SIGMA=${SIGMAS[${SLURM_ARRAY_TASK_ID}]}
TAG=$(printf "%.8g" "${SIGMA}")
TAG=${TAG/./p}

export LD_LIBRARY_PATH="${PYROOT}/lib:${PYROOT}/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${PROJECT}:${PROJECT}/scripts:${PYTHONPATH:-}"
export NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd "${PROJECT}"

pids=()
for seed in $(seq 0 9); do
  OUT=${ROOT}/shards/ppc1024/sigma${TAG}/seed${seed}
  mkdir -p "${OUT}/numba"
  env NUMBA_CACHE_DIR="${OUT}/numba" \
    "${PYTHON}" scripts/run_landau_sigma_refinement.py simulate \
      --output "${OUT}" --ppc 1024 --sigma-dx "${SIGMA}" --seed "${seed}" \
      --alpha 0.01 --dt 0.05 --n-steps 1800 --diag-stride 2 \
      >"${OUT}/run.log" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do wait "${pid}" || status=1; done
exit "${status}"
