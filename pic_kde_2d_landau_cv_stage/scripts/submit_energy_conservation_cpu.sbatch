#!/bin/bash -l
#SBATCH --job-name=p1_energy
#SBATCH --account=p201103
#SBATCH --partition=cpu
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --array=0-8%4
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ayushiphysics909@gmail.com
#SBATCH --output=/mnt/tier2/users/u103576/logs/p1_energy_%A_%a.out
#SBATCH --error=/mnt/tier2/users/u103576/logs/p1_energy_%A_%a.err

set -euo pipefail
PROJECT=/home/users/u103576/hexapic-main/pic_kde_2d
PYROOT=/apps/USE/easybuild/release/2024.1/software/Python/3.11.10-GCCcore-13.3.0
PYTHON=${PYROOT}/bin/python3
ROOT=${OUTPUT_ROOT:-${PROJECT}/results/energy_conservation_${SLURM_ARRAY_JOB_ID}}

INDEX=${SLURM_ARRAY_TASK_ID}
if (( INDEX < 3 )); then
  BENCH=landau_weak; ALPHA=0.01; PPC=1024; NSTEPS=1800
  CASE_INDEX=${INDEX}
elif (( INDEX < 6 )); then
  BENCH=landau_strong; ALPHA=0.5; PPC=1024; NSTEPS=900
  CASE_INDEX=$((INDEX - 3))
else
  BENCH=e2; ALPHA=0.001; PPC=32; NSTEPS=1200
  CASE_INDEX=$((INDEX - 6))
fi
CASES=(cic c4 large)
CASE=${CASES[${CASE_INDEX}]}
if [[ "${CASE}" == "cic" ]]; then
  SX=0; SY=0
elif [[ "${CASE}" == "c4" ]]; then
  if [[ "${BENCH}" == "e2" ]]; then SX=0.1844549; SY=0.1844549; else SX=1; SY=1; fi
else
  if [[ "${BENCH}" == "e2" ]]; then SX=12; SY=12; else SX=6; SY=6; fi
fi

export LD_LIBRARY_PATH="${PYROOT}/lib:${PYROOT}/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${PROJECT}:${PYTHONPATH:-}"
export NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd "${PROJECT}"
pids=()
for seed in $(seq 0 9); do
  OUT=${ROOT}/${BENCH}/${CASE}/seed${seed}
  mkdir -p "${OUT}/numba"
  (
    export NUMBA_CACHE_DIR="${OUT}/numba"
    "${PYTHON}" scripts/run_energy_conservation_case.py \
      --output "${OUT}" --benchmark "${BENCH}" --case "${CASE}" \
      --sigma-x-dx "${SX}" --sigma-y-dy "${SY}" --alpha "${ALPHA}" \
      --ppc "${PPC}" --seed "${seed}" --n-steps "${NSTEPS}"
  ) >"${OUT}/run.log" 2>&1 &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "${pid}" || status=1; done
exit "${status}"
