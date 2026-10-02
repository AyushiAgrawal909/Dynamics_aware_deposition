#!/bin/bash -l
#SBATCH --job-name=df_linear_pilot
#SBATCH --account=p201103
#SBATCH --partition=gpu
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --gpus-per-node=1
#SBATCH --mem=32G
#SBATCH --time=00:50:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ayushiphysics909@gmail.com
#SBATCH --output=/mnt/tier2/users/u103576/logs/df_linear_pilot_%j.out
#SBATCH --error=/mnt/tier2/users/u103576/logs/df_linear_pilot_%j.err

set -euo pipefail

PROJECT=/home/users/u103576/hexapic-main/pic_kde_2d
PYROOT=/apps/USE/easybuild/release/2024.1/software/Python/3.11.10-GCCcore-13.3.0
PYTHON=${PYROOT}/bin/python3
ROOT=${OUTPUT_ROOT:-/mnt/tier2/users/u103576/logs/df_linear_pilot_${SLURM_JOB_ID}}

export LD_LIBRARY_PATH="${PYROOT}/lib:${PYROOT}/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${PROJECT}:${PYTHONPATH:-}"
export NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mkdir -p "${ROOT}"
cd "${PROJECT}"

pids=()
for seed in 0 1; do
  "${PYTHON}" scripts/run_linearized_delta_f_mode.py \
    --output "${ROOT}/landau/seed${seed}" --benchmark landau --mode 1 \
    --seed "${seed}" --ppc 32 --alpha 0.001 --Nx 64 --Ny 16 --Ly 8 \
    --dt 0.05 --t-end 30 --diag-stride 2 >"${ROOT}/landau_${seed}.log" 2>&1 &
  pids+=("$!")
  for mode in 8 12; do
    "${PYTHON}" scripts/run_linearized_delta_f_mode.py \
      --output "${ROOT}/two_stream/m${mode}/seed${seed}" \
      --benchmark two_stream --mode "${mode}" --seed "${seed}" \
      --ppc 32 --alpha 1e-8 --Nx 128 --Ny 16 --Lx 64 --Ly 16 \
      --v-beam 0.8 --v-thermal 0.05 --dt 0.02 --t-end 30 --diag-stride 2 \
      >"${ROOT}/two_stream_m${mode}_${seed}.log" 2>&1 &
    pids+=("$!")
  done
done

status=0
for pid in "${pids[@]}"; do
  wait "${pid}" || status=1
done
exit "${status}"
