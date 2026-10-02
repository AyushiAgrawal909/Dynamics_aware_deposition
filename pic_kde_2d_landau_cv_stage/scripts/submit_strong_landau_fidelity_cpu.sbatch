#!/bin/bash -l
#SBATCH --job-name=p1_SL_rate
#SBATCH --account=p201103
#SBATCH --partition=cpu
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --array=0-5
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ayushiphysics909@gmail.com
#SBATCH --output=/mnt/tier2/users/u103576/logs/p1_SL_rate_%A_%a.out
#SBATCH --error=/mnt/tier2/users/u103576/logs/p1_SL_rate_%A_%a.err

set -euo pipefail

PROJECT=/home/users/u103576/hexapic-main/pic_kde_2d
PYROOT=/apps/USE/easybuild/release/2024.1/software/Python/3.11.10-GCCcore-13.3.0
PYTHON=${PYROOT}/bin/python3
RUNNER=${PROJECT}/scripts/run_landau_sigma_refinement.py
ROOT=${OUTPUT_ROOT:-${PROJECT}/results/strong_landau_fidelity_${SLURM_ARRAY_JOB_ID}}
SIGMAS=(0 1 1.10785 1.5 2 2.453)
SIGMA=${SIGMAS[${SLURM_ARRAY_TASK_ID}]}
SIGMA_TAG=${SIGMA/./p}

export LD_LIBRARY_PATH="${PYROOT}/lib:${PYROOT}/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${PROJECT}:${PYTHONPATH:-}"
export NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

cd "${PROJECT}"
mkdir -p "${ROOT}/shards/ppc1024/sigma${SIGMA_TAG}"
pids=()
for seed in $(seq 0 9); do
  OUT=${ROOT}/shards/ppc1024/sigma${SIGMA_TAG}/seed${seed}
  mkdir -p "${OUT}/numba"
  (
    export NUMBA_CACHE_DIR="${OUT}/numba"
    "${PYTHON}" "${RUNNER}" simulate \
      --output "${OUT}" --ppc 1024 --sigma-dx "${SIGMA}" --seed "${seed}" \
      --alpha 0.5 --Nx 64 --Ny 16 --Ly 8 --kx-mode 1 --k-lambdaD 0.4 \
      --dt 0.05 --n-steps 900 --diag-stride 2
  ) >"${OUT}/run.log" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  wait "${pid}" || status=1
done
exit "${status}"
