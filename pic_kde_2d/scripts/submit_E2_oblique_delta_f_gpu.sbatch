#!/bin/bash -l
# Coherently seeded oblique E2 modes. The GPU allocation provides one node;
# independent CPU shards are packed across its cores because NumPy/Numba is CPU.
#SBATCH --job-name=e2_oblique_df
#SBATCH --account=p201103
#SBATCH --partition=gpu
#SBATCH --qos=long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --gpus-per-node=1
#SBATCH --mem=200G
#SBATCH --time=02:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ayushiphysics909@gmail.com
#SBATCH --output=/mnt/tier2/users/u103576/logs/e2_oblique_df_%j.out
#SBATCH --error=/mnt/tier2/users/u103576/logs/e2_oblique_df_%j.err

set -euo pipefail

PROJECT=/home/users/u103576/hexapic-main/pic_kde_2d
PYROOT=/apps/USE/easybuild/release/2024.1/software/Python/3.11.10-GCCcore-13.3.0
PYTHON=${PYROOT}/bin/python3
ROOT=${OUTPUT_ROOT:-/mnt/tier2/users/u103576/logs/e2_oblique_df_${SLURM_JOB_ID}}

export LD_LIBRARY_PATH="${PYROOT}/lib:${PYROOT}/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${PROJECT}:${PROJECT}/scripts:${PYTHONPATH:-}"
export NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mkdir -p "${ROOT}/runs"
cd "${PROJECT}"

pids=()
for mode_y in 0 1 2 3; do
  for seed in $(seq 0 9); do
    OUT=${ROOT}/runs/my$(printf '%02d' "${mode_y}")/seed${seed}
    mkdir -p "${OUT}/numba"
    env NUMBA_CACHE_DIR="${OUT}/numba" \
      "${PYTHON}" scripts/run_linearized_delta_f_mode.py \
      --output "${OUT}" --benchmark two_stream --mode 12 \
      --ky-mode "${mode_y}" --seed "${seed}" --ppc 512 --alpha 1e-8 \
      --sigma-x-dx 0 --sigma-y-dy 0 \
      --Nx 128 --Ny 16 --Lx 64 --Ly 16 \
      --v-beam 0.8 --v-thermal 0.05 --dt 0.02 --t-end 60 \
      --diag-stride 2 --project-target-mode \
      >"${OUT}/run.log" 2>&1 &
    pids+=("$!")
  done
done

status=0
for pid in "${pids[@]}"; do
  wait "${pid}" || status=1
done
if [ "${status}" -ne 0 ]; then
  echo "one or more oblique delta-f shards failed" >&2
  exit "${status}"
fi

"${PYTHON}" scripts/analyze_E2_oblique_delta_f.py \
  --input "${ROOT}/runs" \
  --output "${ROOT}/analysis" \
  --mode-x 12 --mode-y 0,1,2,3 --seeds 10 --nboot 5000 \
  --fit-t-start 24.32 --fit-t-stop 59.68

echo "complete output=${ROOT}"
