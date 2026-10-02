#!/bin/bash -l
#SBATCH --job-name=p1_e2_c4_noise
#SBATCH --account=p201103
#SBATCH --partition=cpu
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=120G
#SBATCH --time=04:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ayushiphysics909@gmail.com
#SBATCH --output=/mnt/tier2/users/u103576/logs/p1_e2_c4_noise_%j.out
#SBATCH --error=/mnt/tier2/users/u103576/logs/p1_e2_c4_noise_%j.err

set -euo pipefail

PROJECT=/home/users/u103576/hexapic-main/pic_kde_2d
PYROOT=/apps/USE/easybuild/release/2024.1/software/Python/3.11.10-GCCcore-13.3.0
PYTHON=${PYROOT}/bin/python3
SWEEP_RESULT=${SWEEP_RESULT:-/mnt/tier2/users/u103576/logs/df_m12_sigma_4872745/analysis/m12_sigma_sweep.json}
ROOT=${OUTPUT_ROOT:-/mnt/tier2/users/u103576/logs/p1_e2_c4_noise_${SLURM_JOB_ID}}

export LD_LIBRARY_PATH="${PYROOT}/lib:${PYROOT}/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${PROJECT}:${PYTHONPATH:-}"
export NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mkdir -p "${ROOT}"
cd "${PROJECT}"

if [ ! -f "${SWEEP_RESULT}" ]; then
  echo "missing completed m=12 sweep result: ${SWEEP_RESULT}" >&2
  exit 2
fi
SIGMA=$("${PYTHON}" -c "import json; print(json.load(open('${SWEEP_RESULT}'))['recommended_conservative_sigma_dx'])")
if [ "${SIGMA}" = "0.0" ] || [ "${SIGMA}" = "0" ]; then
  SIGMAS=0
else
  SIGMAS="0,${SIGMA}"
fi
printf 'selected_sigma_dx=%s\n' "${SIGMA}" | tee "${ROOT}/selected_sigma.txt"

pids=()
for seed in $(seq 0 9); do
  OUT=${ROOT}/seed${seed}
  mkdir -p "${OUT}/mise" "${OUT}/numba"
  env NUMBA_CACHE_DIR="${OUT}/numba" \
    "${PYTHON}" scripts/run_E2_mise_gate.py \
    --output "${OUT}/mise" --sigma-dx "${SIGMAS}" \
    --seed-start "${seed}" --seeds 1 --ppc 32 --reference-ppc 1024 \
    --Nx 128 --Ny 16 --Lx 64 --Ly 16 --kx-mode 12 \
    --v-beam 0.8 --v-thermal 0.05 --alpha 0.001 \
    --live-sigma-dx 0.25 --dt 0.05 --n-steps 80 \
    --diag-stride 2 --eval-t-min 0.5 --eval-t-max 4.0 --nboot 200 \
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

"${PYTHON}" paper1_selector_comparison/scripts/reduce_noise_shards.py \
  --input "${ROOT}" --output "${ROOT}/analysis" \
  --benchmark e2 --seeds 10 --nboot 10000

echo "complete output=${ROOT}"
