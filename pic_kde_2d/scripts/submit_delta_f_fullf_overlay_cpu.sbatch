#!/bin/bash -l
#SBATCH --job-name=df_fullf_overlay
#SBATCH --account=p201103
#SBATCH --partition=cpu
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=60G
#SBATCH --time=01:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ayushiphysics909@gmail.com
#SBATCH --output=/mnt/tier2/users/u103576/logs/df_fullf_overlay_%j.out
#SBATCH --error=/mnt/tier2/users/u103576/logs/df_fullf_overlay_%j.err

set -euo pipefail

PROJECT=/home/users/u103576/hexapic-main/pic_kde_2d
PYROOT=/apps/USE/easybuild/release/2024.1/software/Python/3.11.10-GCCcore-13.3.0
PYTHON=${PYROOT}/bin/python3
ROOT=${OUTPUT_ROOT:-/mnt/tier2/users/u103576/logs/df_fullf_overlay_${SLURM_JOB_ID}}
DELTA_ROOT=${DELTA_ROOT:-/mnt/tier2/users/u103576/logs/df_breakdown_4873222/alpha_0p1}

export LD_LIBRARY_PATH="${PYROOT}/lib:${PYROOT}/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${PROJECT}:${PYTHONPATH:-}"
export NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mkdir -p "${ROOT}/fullf"
cd "${PROJECT}"

pids=()
for seed in $(seq 0 4); do
  OUT=${ROOT}/fullf/seed${seed}
  mkdir -p "${OUT}/numba"
  env NUMBA_CACHE_DIR="${OUT}/numba" \
    "${PYTHON}" scripts/run_landau_sigma_refinement.py simulate \
    --output "${OUT}" --ppc 128 --sigma-dx 0 --seed "${seed}" \
    --alpha 0.1 --Nx 64 --Ny 16 --Ly 8 \
    --dt 0.05 --n-steps 1200 --diag-stride 2 \
    >"${OUT}/run.log" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  wait "${pid}" || status=1
done
if [ "${status}" -ne 0 ]; then
  exit "${status}"
fi

"${PYTHON}" scripts/analyze_delta_f_fullf_overlay.py \
  --fullf "${ROOT}/fullf" --deltaf "${DELTA_ROOT}" \
  --output "${ROOT}/analysis" --seeds 5

echo "complete output=${ROOT}"
