#!/bin/bash -l
#SBATCH --job-name=p1_ppc_live
#SBATCH --account=p201103
#SBATCH --partition=cpu
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --array=0-5%3
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ayushiphysics909@gmail.com
#SBATCH --output=/mnt/tier2/users/u103576/logs/p1_ppc_live_%A_%a.out
#SBATCH --error=/mnt/tier2/users/u103576/logs/p1_ppc_live_%A_%a.err

set -euo pipefail
PROJECT=/home/users/u103576/hexapic-main/pic_kde_2d
PYROOT=/apps/USE/easybuild/release/2024.1/software/Python/3.11.10-GCCcore-13.3.0
PYTHON=${PYROOT}/bin/python3
PPCS=(8 32 128 512 1024 4096)
PPC=${PPCS[${SLURM_ARRAY_TASK_ID}]}
ROOT=${OUTPUT_ROOT:-${PROJECT}/results/landau_ppc_dynamic_selectors_${SLURM_ARRAY_JOB_ID}}
OUT=${ROOT}/ppc${PPC}
export LD_LIBRARY_PATH="${PYROOT}/lib:${PYROOT}/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${PROJECT}:${PYTHONPATH:-}"
export NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd "${PROJECT}"
pids=()
for seed in $(seq 0 9); do
  SEED_OUT=${OUT}/seed${seed}
  mkdir -p "${SEED_OUT}/numba"
  (
    export NUMBA_CACHE_DIR="${SEED_OUT}/numba"
    "${PYTHON}" scripts/run_landau_ppc_selector_snapshots.py \
      --output "${SEED_OUT}" --ppc "${PPC}" --seed "${seed}"
  ) >"${SEED_OUT}/run.log" 2>&1 &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "${pid}" || status=1; done
if (( status != 0 )); then exit "${status}"; fi
"${PYTHON}" scripts/reduce_landau_ppc_dynamic_selectors.py \
  --input "${OUT}" --output "${OUT}" --ppc "${PPC}"
