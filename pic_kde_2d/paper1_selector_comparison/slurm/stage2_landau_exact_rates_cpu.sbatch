#!/bin/bash -l
#SBATCH --job-name=p1_L_rates
#SBATCH --account=p201103
#SBATCH --partition=gpu
#SBATCH --qos=default
#SBATCH --array=0-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --gpus-per-node=1
#SBATCH --mem-per-cpu=2G
#SBATCH --time=04:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ayushiphysics909@gmail.com
#SBATCH --output=/mnt/tier2/users/u103576/logs/p1_L_rates_%A_%a.out
#SBATCH --error=/mnt/tier2/users/u103576/logs/p1_L_rates_%A_%a.err

set -euo pipefail
PROJECT=/home/users/u103576/hexapic-main/pic_kde_2d
PYROOT=/apps/USE/easybuild/release/2024.1/software/Python/3.11.10-GCCcore-13.3.0
PYTHON=${PYROOT}/bin/python3
SIGMAS=(1.8329728493314705 60)
SIGMA=${SIGMAS[$SLURM_ARRAY_TASK_ID]}
TAG=${SIGMA/./p}
ROOT=${OUTPUT_ROOT:-/mnt/tier2/users/u103576/logs/p1_landau_rates_${SLURM_ARRAY_JOB_ID}}
export LD_LIBRARY_PATH="${PYROOT}/lib:${PYROOT}/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${PROJECT}:${PYTHONPATH:-}"
export NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mkdir -p "${ROOT}/ppc1024/sigma${TAG}"
cd "${PROJECT}"
pids=()
for seed in $(seq 0 9); do
  OUT=${ROOT}/ppc1024/sigma${TAG}/seed${seed}
  mkdir -p "${OUT}/numba"
  srun --exclusive -N1 -n1 -c1 env NUMBA_CACHE_DIR="${OUT}/numba" \
    "${PYTHON}" scripts/run_landau_sigma_refinement.py simulate \
    --output "${OUT}" --ppc 1024 --sigma-dx "${SIGMA}" --seed "${seed}" \
    --alpha 0.01 --n-steps 1800 >"${OUT}/run.log" 2>&1 &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "${pid}" || status=1; done
exit "${status}"
