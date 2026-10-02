#!/bin/bash -l
#SBATCH --job-name=df_linear_ref
#SBATCH --account=p201103
#SBATCH --partition=gpu
#SBATCH --qos=default
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=30
#SBATCH --gpus-per-node=1
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ayushiphysics909@gmail.com
#SBATCH --output=/mnt/tier2/users/u103576/logs/df_linear_ref_%j.out
#SBATCH --error=/mnt/tier2/users/u103576/logs/df_linear_ref_%j.err

set -euo pipefail

PROJECT=/home/users/u103576/hexapic-main/pic_kde_2d
PYROOT=/apps/USE/easybuild/release/2024.1/software/Python/3.11.10-GCCcore-13.3.0
PYTHON=${PYROOT}/bin/python3
ROOT=${OUTPUT_ROOT:-/mnt/tier2/users/u103576/logs/df_linear_ref_${SLURM_JOB_ID}}

export LD_LIBRARY_PATH="${PYROOT}/lib:${PYROOT}/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${PROJECT}:${PYTHONPATH:-}"
export NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mkdir -p "${ROOT}"
cd "${PROJECT}"

pids=()
for seed in $(seq 0 9); do
  OUT=${ROOT}/landau/seed${seed}
  mkdir -p "${OUT}/numba"
  env NUMBA_CACHE_DIR="${OUT}/numba" \
    "${PYTHON}" scripts/run_linearized_delta_f_mode.py \
    --output "${OUT}" --benchmark landau --mode 1 \
    --seed "${seed}" --ppc 512 --alpha 0.001 --Nx 64 --Ny 16 --Ly 8 \
    --dt 0.05 --t-end 90 --diag-stride 2 >"${OUT}/run.log" 2>&1 &
  pids+=("$!")

  for mode in 8 12; do
    OUT=${ROOT}/two_stream/m${mode}/seed${seed}
    mkdir -p "${OUT}/numba"
    T_END=30
    if [ "${mode}" -eq 12 ]; then T_END=60; fi
    env NUMBA_CACHE_DIR="${OUT}/numba" \
      "${PYTHON}" scripts/run_linearized_delta_f_mode.py \
      --output "${OUT}" --benchmark two_stream --mode "${mode}" \
      --seed "${seed}" --ppc 512 --alpha 1e-8 --Nx 128 --Ny 16 --Lx 64 --Ly 16 \
      --v-beam 0.8 --v-thermal 0.05 --dt 0.02 --t-end "${T_END}" --diag-stride 2 \
      >"${OUT}/run.log" 2>&1 &
    pids+=("$!")
  done
done

status=0
for pid in "${pids[@]}"; do
  wait "${pid}" || status=1
done
if [ "${status}" -ne 0 ]; then
  exit "${status}"
fi

"${PYTHON}" scripts/analyze_linearized_delta_f_reference.py \
  --input "${ROOT}" --output "${ROOT}/analysis" --seeds 10 --nboot 2000

echo "complete output=${ROOT}"
