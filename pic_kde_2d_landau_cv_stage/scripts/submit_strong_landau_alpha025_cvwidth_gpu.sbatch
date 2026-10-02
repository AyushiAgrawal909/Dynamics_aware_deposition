#!/bin/bash -l
#SBATCH --job-name=p1_SL25_cvwidth
#SBATCH --account=p201103
#SBATCH --partition=gpu
#SBATCH --qos=default
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ayushiphysics909@gmail.com
#SBATCH --output=/mnt/tier2/users/u103576/logs/p1_SL25_cvwidth_%j.out
#SBATCH --error=/mnt/tier2/users/u103576/logs/p1_SL25_cvwidth_%j.err

set -euo pipefail
: "${OUTPUT_ROOT:?set OUTPUT_ROOT}"

PROJECT=/home/users/u103576/hexapic-main/pic_kde_2d
PYROOT=/apps/USE/easybuild/release/2024.1/software/Python/3.11.10-GCCcore-13.3.0
PYTHON=${PYROOT}/bin/python3
RUNNER=${PROJECT}/scripts/run_landau_sigma_refinement.py
SIGMA_DX=${SIGMA_DX:-2.8208226}
SIGMA_TAG=${SIGMA_DX/./p}

export LD_LIBRARY_PATH="${PYROOT}/lib:${PYROOT}/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${PROJECT}:${PYTHONPATH:-}"
export NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd "${PROJECT}"

pids=()
for seed in $(seq 0 9); do
  OUT=${OUTPUT_ROOT}/ppc1024/sigma${SIGMA_TAG}/seed${seed}
  mkdir -p "${OUT}/numba"
  (
    export NUMBA_CACHE_DIR="${OUT}/numba"
    "${PYTHON}" "${RUNNER}" simulate \
      --output "${OUT}" --ppc 1024 --sigma-dx "${SIGMA_DX}" --seed "${seed}" \
      --alpha 0.25 --Nx 64 --Ny 16 --Ly 8 --kx-mode 1 --k-lambdaD 0.4 \
      --dt 0.05 --n-steps 1200 --diag-stride 2
  ) >"${OUT}/run.log" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  wait "${pid}" || status=1
done
exit "${status}"
