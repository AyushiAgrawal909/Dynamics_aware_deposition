#!/bin/bash -l
#SBATCH --job-name=p1_L_lscv
#SBATCH --account=p201103
#SBATCH --partition=gpu
#SBATCH --qos=long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --gpus-per-node=1
#SBATCH --mem=200G
#SBATCH --time=08:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ayushiphysics909@gmail.com
#SBATCH --output=/mnt/tier2/users/u103576/logs/p1_L_lscv_%j.out
#SBATCH --error=/mnt/tier2/users/u103576/logs/p1_L_lscv_%j.err

set -euo pipefail

PROJECT=/home/users/u103576/hexapic-main/pic_kde_2d
PYROOT=/apps/USE/easybuild/release/2024.1/software/Python/3.11.10-GCCcore-13.3.0
PYTHON=${PYROOT}/bin/python3
ROOT=${OUTPUT_ROOT:-${PROJECT}/results/landau_lscv_aligned_${SLURM_JOB_ID}}

export LD_LIBRARY_PATH="${PYROOT}/lib:${PYROOT}/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${PROJECT}:${PYTHONPATH:-}"
export NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mkdir -p "${ROOT}"
cd "${PROJECT}"

pids=()
for seed in $(seq 0 9); do
  OUT=${ROOT}/seed${seed}
  mkdir -p "${OUT}" "${OUT}/numba"
  (
    export NUMBA_CACHE_DIR="${OUT}/numba"
    "${PYTHON}" scripts/run_landau_lscv_aligned.py scan \
      --output "${OUT}" --seed-start "${seed}" --seeds 1 \
      --ppc 1024 --Nx 64 --Ny 16 --Ly 8 --kx-mode 1 --k-lambdaD 0.4 \
      --alpha 0.01 --live-sigma-dx 0.25 --dt 0.05 --n-steps 600 \
      --diag-stride 10 --eval-t-min 5 --eval-t-max 30
  ) >"${OUT}/scan.log" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  wait "${pid}" || status=1
done
if [[ "${status}" -ne 0 ]]; then
  exit "${status}"
fi

"${PYTHON}" scripts/run_landau_lscv_aligned.py reduce \
  --input-root "${ROOT}" --output "${ROOT}/analysis" \
  --expected-seeds 10 --ppc 1024 --eval-t-min 5 --eval-t-max 30

"${PYTHON}" scripts/plot_landau_lscv_vs_mise.py \
  --lscv-summary "${ROOT}/analysis/landau_lscv_summary.csv" \
  --lscv-result "${ROOT}/analysis/landau_lscv_result.json" \
  --mise-summary /mnt/tier2/users/u103576/logs/p1_landau_noise_4868286/reduced/landau_noise_summary.csv \
  --output "${ROOT}/analysis/landau_lscv_vs_mise"
