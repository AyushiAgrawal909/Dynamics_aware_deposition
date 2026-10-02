#!/bin/bash -l
#SBATCH --job-name=df_m12_sigma
#SBATCH --account=p201103
#SBATCH --partition=cpu
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=120G
#SBATCH --time=04:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ayushiphysics909@gmail.com
#SBATCH --output=/mnt/tier2/users/u103576/logs/df_m12_sigma_%j.out
#SBATCH --error=/mnt/tier2/users/u103576/logs/df_m12_sigma_%j.err

set -euo pipefail

PROJECT=/home/users/u103576/hexapic-main/pic_kde_2d
PYROOT=/apps/USE/easybuild/release/2024.1/software/Python/3.11.10-GCCcore-13.3.0
PYTHON=${PYROOT}/bin/python3
ROOT=${OUTPUT_ROOT:-/mnt/tier2/users/u103576/logs/df_m12_sigma_${SLURM_JOB_ID}}
BASELINE_ROOT=${BASELINE_ROOT:-/mnt/tier2/users/u103576/logs/df_linear_ref_4872599/two_stream/m12}

export LD_LIBRARY_PATH="${PYROOT}/lib:${PYROOT}/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${PROJECT}:${PROJECT}/scripts:${PYTHONPATH:-}"
export NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mkdir -p "${ROOT}"
cd "${PROJECT}"

if [ ! -d "${BASELINE_ROOT}" ]; then
  echo "missing sigma=0 baseline: ${BASELINE_ROOT}" >&2
  exit 2
fi
ln -s "${BASELINE_ROOT}" "${ROOT}/sigma_0"

pids=()
for sigma in 0.175 0.18445488487247555 0.19872036040909125 0.22; do
  token=${sigma//./p}
  for seed in $(seq 0 9); do
    OUT=${ROOT}/sigma_${token}/seed${seed}
    mkdir -p "${OUT}/numba"
    env NUMBA_CACHE_DIR="${OUT}/numba" \
      "${PYTHON}" scripts/run_linearized_delta_f_mode.py \
      --output "${OUT}" --benchmark two_stream --mode 12 \
      --seed "${seed}" --ppc 512 --alpha 1e-8 --sigma-dx "${sigma}" \
      --Nx 128 --Ny 16 --Lx 64 --Ly 16 \
      --v-beam 0.8 --v-thermal 0.05 --dt 0.02 --t-end 60 --diag-stride 2 \
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

"${PYTHON}" scripts/analyze_linearized_delta_f_m12_sigma.py \
  --input "${ROOT}" --output "${ROOT}/analysis" --seeds 10 --nboot 2000

echo "complete output=${ROOT}"
