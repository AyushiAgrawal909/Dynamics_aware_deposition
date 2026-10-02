#!/bin/bash -l
#SBATCH --job-name=weak-landau-cv60
#SBATCH --account=p201103
#SBATCH --partition=cpu
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --ntasks=3
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ayushiphysics909@gmail.com
#SBATCH --output=/mnt/tier2/users/u103576/logs/weak_landau_cv60_%j.out
#SBATCH --error=/mnt/tier2/users/u103576/logs/weak_landau_cv60_%j.err

set -euo pipefail
PROJECT=/home/users/u103576/hexapic-main/pic_kde_2d
PYTHON_ROOT=/apps/USE/easybuild/release/2024.1/software/Python/3.11.10-GCCcore-13.3.0
PYTHON=${PYTHON_ROOT}/bin/python3
SCRIPT=${PROJECT}/scripts/run_weak_landau_smoothing.py
OUTPUT=${PROJECT}/results/weak_landau_cv_extended_${SLURM_JOB_ID}
SIGMAS="0,0.25,0.5,1,2,3,4,5,6,8,10,12,16,20,24,30,32,36,40,48,60"

export LD_LIBRARY_PATH="${PYTHON_ROOT}/lib:${PYTHON_ROOT}/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${PROJECT}:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mkdir -p "${OUTPUT}"

pids=()
for ppc in 8 16 32; do
  ppc_dir=${OUTPUT}/ppc${ppc}
  mkdir -p "${ppc_dir}"
  (
    export NUMBA_CACHE_DIR="${OUTPUT}/numba_selector_ppc${ppc}"
    mkdir -p "${NUMBA_CACHE_DIR}"
    srun --exclusive -N 1 -n 1 -c 1 "${PYTHON}" "${SCRIPT}" selector \
      --output "${ppc_dir}" --sigma-dx "${SIGMAS}" \
      --alpha 0.01 --ppc "${ppc}" --reference-ppc 512 --seeds 10
  ) > "${ppc_dir}/selector.log" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then status=1; fi
done
exit "${status}"
