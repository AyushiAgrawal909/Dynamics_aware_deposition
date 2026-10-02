#!/bin/bash -l
# Complete only the pending protected-band edge modes on GPU-node CPUs.
#SBATCH --job-name=df_band_edge
#SBATCH --account=p201103
#SBATCH --partition=gpu
#SBATCH --qos=default
#SBATCH --array=13-14
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --gpus-per-node=1
#SBATCH --mem=120G
#SBATCH --time=01:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ayushiphysics909@gmail.com
#SBATCH --output=/mnt/tier2/users/u103576/logs/df_band_edge_%A_%a.out
#SBATCH --error=/mnt/tier2/users/u103576/logs/df_band_edge_%A_%a.err

set -euo pipefail

PROJECT=/home/users/u103576/hexapic-main/pic_kde_2d
PYROOT=/apps/USE/easybuild/release/2024.1/software/Python/3.11.10-GCCcore-13.3.0
PYTHON=${PYROOT}/bin/python3
ROOT=${OUTPUT_ROOT:-/mnt/tier2/users/u103576/logs/df_band_1_14_4873221}
MODE=${SLURM_ARRAY_TASK_ID}

export LD_LIBRARY_PATH="${PYROOT}/lib:${PYROOT}/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${PROJECT}:${PROJECT}/scripts:${PYTHONPATH:-}"
export NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd "${PROJECT}"

pids=()
for seed in $(seq 0 9); do
  OUT=${ROOT}/m$(printf '%02d' "${MODE}")/seed${seed}
  mkdir -p "${OUT}/numba"
  env NUMBA_CACHE_DIR="${OUT}/numba" \
    "${PYTHON}" scripts/run_linearized_delta_f_mode.py \
    --output "${OUT}" --benchmark two_stream --mode "${MODE}" \
    --seed "${seed}" --ppc 512 --alpha 1e-8 --sigma-dx 0 \
    --Nx 128 --Ny 16 --Lx 64 --Ly 16 \
    --v-beam 0.8 --v-thermal 0.05 --dt 0.02 --t-end 60 --diag-stride 2 \
    >"${OUT}/run.log" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  wait "${pid}" || status=1
done
exit "${status}"
