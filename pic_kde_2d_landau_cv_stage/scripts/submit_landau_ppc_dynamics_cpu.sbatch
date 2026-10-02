#!/bin/bash -l
#SBATCH --job-name=p1_ppc_dyn
#SBATCH --account=p201103
#SBATCH --partition=cpu
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --array=0-23%4
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ayushiphysics909@gmail.com
#SBATCH --output=/mnt/tier2/users/u103576/logs/p1_ppc_dyn_%A_%a.out
#SBATCH --error=/mnt/tier2/users/u103576/logs/p1_ppc_dyn_%A_%a.err

set -euo pipefail
: "${SELECTOR_ROOT:?submit with SELECTOR_ROOT pointing to completed direct selectors}"
PROJECT=/home/users/u103576/hexapic-main/pic_kde_2d
PYROOT=/apps/USE/easybuild/release/2024.1/software/Python/3.11.10-GCCcore-13.3.0
PYTHON=${PYROOT}/bin/python3
PPCS=(8 32 128 512 1024 4096)
CHOICES=(cic c4 mise lscv)
PPC_INDEX=$((SLURM_ARRAY_TASK_ID / 4))
CHOICE_INDEX=$((SLURM_ARRAY_TASK_ID % 4))
PPC=${PPCS[${PPC_INDEX}]}
CHOICE=${CHOICES[${CHOICE_INDEX}]}
ROOT=${OUTPUT_ROOT:-${PROJECT}/results/landau_ppc_dynamics_${SLURM_ARRAY_JOB_ID}}
export LD_LIBRARY_PATH="${PYROOT}/lib:${PYROOT}/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${PROJECT}:${PYTHONPATH:-}"
export NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd "${PROJECT}"
pids=()
for seed in $(seq 0 9); do
  OUT=${ROOT}/ppc${PPC}/${CHOICE}/seed${seed}
  mkdir -p "${OUT}/numba"
  (
    export NUMBA_CACHE_DIR="${OUT}/numba"
    "${PYTHON}" scripts/run_landau_ppc_dynamics.py \
      --output "${OUT}" --selector-root "${SELECTOR_ROOT}" \
      --ppc "${PPC}" --choice "${CHOICE}" --seed "${seed}"
  ) >"${OUT}/run.log" 2>&1 &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "${pid}" || status=1; done
exit "${status}"
