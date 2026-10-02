#!/bin/bash -l
#SBATCH --job-name=df_breakdown
#SBATCH --account=p201103
#SBATCH --partition=cpu
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=15
#SBATCH --mem=80G
#SBATCH --time=02:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ayushiphysics909@gmail.com
#SBATCH --output=/mnt/tier2/users/u103576/logs/df_breakdown_%j.out
#SBATCH --error=/mnt/tier2/users/u103576/logs/df_breakdown_%j.err

set -euo pipefail

PROJECT=/home/users/u103576/hexapic-main/pic_kde_2d
PYROOT=/apps/USE/easybuild/release/2024.1/software/Python/3.11.10-GCCcore-13.3.0
PYTHON=${PYROOT}/bin/python3
ROOT=${OUTPUT_ROOT:-/mnt/tier2/users/u103576/logs/df_breakdown_${SLURM_JOB_ID}}

export LD_LIBRARY_PATH="${PYROOT}/lib:${PYROOT}/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${PROJECT}:${PYTHONPATH:-}"
export NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mkdir -p "${ROOT}"
cd "${PROJECT}"

pids=()
for alpha in 0.001 0.01 0.1; do
  token=${alpha//./p}
  for seed in $(seq 0 4); do
    OUT=${ROOT}/alpha_${token}/seed${seed}
    mkdir -p "${OUT}/numba"
    env NUMBA_CACHE_DIR="${OUT}/numba" \
      "${PYTHON}" scripts/run_linearized_delta_f_mode.py \
      --output "${OUT}" --benchmark landau --mode 1 \
      --seed "${seed}" --ppc 128 --alpha "${alpha}" --sigma-dx 0 \
      --Nx 64 --Ny 16 --Ly 8 --dt 0.05 --t-end 60 --diag-stride 2 \
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

"${PYTHON}" scripts/analyze_delta_f_breakdown.py \
  --input "${ROOT}" --output "${ROOT}/analysis" --threshold 0.5 --seeds 5

echo "complete output=${ROOT}"
