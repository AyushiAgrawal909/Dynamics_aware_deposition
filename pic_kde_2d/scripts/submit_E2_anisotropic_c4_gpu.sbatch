#!/bin/bash -l
# Packed anisotropic C4 validation: the GPU allocation supplies one node whose
# CPU cores run independent paired seeds concurrently. The NumPy/Numba solver
# itself does not use CUDA.
#SBATCH --job-name=e2_c4_aniso
#SBATCH --account=p201103
#SBATCH --partition=gpu
#SBATCH --qos=long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --gpus-per-node=1
#SBATCH --mem=200G
#SBATCH --time=08:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ayushiphysics909@gmail.com
#SBATCH --output=/mnt/tier2/users/u103576/logs/e2_c4_aniso_%j.out
#SBATCH --error=/mnt/tier2/users/u103576/logs/e2_c4_aniso_%j.err

set -euo pipefail

PROJECT=/home/users/u103576/hexapic-main/pic_kde_2d
PYROOT=/apps/USE/easybuild/release/2024.1/software/Python/3.11.10-GCCcore-13.3.0
PYTHON=${PYROOT}/bin/python3
ROOT=${OUTPUT_ROOT:-/mnt/tier2/users/u103576/logs/e2_c4_aniso_${SLURM_JOB_ID}}
BASELINE_ROOT=${BASELINE_ROOT:-/mnt/tier2/users/u103576/logs/df_linear_ref_4872599/two_stream/m12}
SIGMA_X=0.18445488487247555
SIGMA_Y=(0.18445488487247555 1 2 4)

export LD_LIBRARY_PATH="${PYROOT}/lib:${PYROOT}/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${PROJECT}:${PROJECT}/scripts:${PYTHONPATH:-}"
export NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mkdir -p "${ROOT}/rate" "${ROOT}/noise"
cd "${PROJECT}"

if [ ! -d "${BASELINE_ROOT}" ]; then
  echo "missing paired sigma=0 m=12 baseline: ${BASELINE_ROOT}" >&2
  exit 2
fi

rate_pids=()
for width_index in "${!SIGMA_Y[@]}"; do
  sigma_y=${SIGMA_Y[${width_index}]}
  for seed in $(seq 0 9); do
    OUT=${ROOT}/rate/sy_${width_index}/seed${seed}
    mkdir -p "${OUT}/numba"
    env NUMBA_CACHE_DIR="${OUT}/numba" \
      "${PYTHON}" scripts/run_linearized_delta_f_mode.py \
      --output "${OUT}" --benchmark two_stream --mode 12 \
      --seed "${seed}" --ppc 512 --alpha 1e-8 \
      --sigma-x-dx "${SIGMA_X}" --sigma-y-dy "${sigma_y}" \
      --Nx 128 --Ny 16 --Lx 64 --Ly 16 \
      --v-beam 0.8 --v-thermal 0.05 --dt 0.02 --t-end 60 --diag-stride 2 \
      >"${OUT}/run.log" 2>&1 &
    rate_pids+=("$!")
  done
done

status=0
for pid in "${rate_pids[@]}"; do
  wait "${pid}" || status=1
done
if [ "${status}" -ne 0 ]; then
  echo "one or more projected delta-f rate shards failed" >&2
  exit "${status}"
fi

noise_pids=()
for seed in $(seq 0 9); do
  OUT=${ROOT}/noise/seed${seed}
  mkdir -p "${OUT}/numba"
  env NUMBA_CACHE_DIR="${OUT}/numba" \
    "${PYTHON}" scripts/run_E2_anisotropic_c4_seed.py \
    --output "${OUT}" --seed "${seed}" \
    --sigma-x-dx "${SIGMA_X}" \
    --sigma-y-dy "0.18445488487247555,1,2,4" \
    --ppc 32 --reference-ppc 1024 \
    --Nx 128 --Ny 16 --Lx 64 --Ly 16 --kx-mode 12 \
    --v-beam 0.8 --v-thermal 0.05 --alpha 0.001 \
    --live-sigma-dx 0.25 --dt 0.05 --n-steps 80 \
    --diag-stride 2 --eval-t-min 0.5 --eval-t-max 4.0 \
    >"${OUT}/run.log" 2>&1 &
  noise_pids+=("$!")
done

status=0
for pid in "${noise_pids[@]}"; do
  wait "${pid}" || status=1
done
if [ "${status}" -ne 0 ]; then
  echo "one or more anisotropic noise/audit shards failed" >&2
  exit "${status}"
fi

"${PYTHON}" scripts/analyze_E2_anisotropic_c4.py \
  --rate-root "${ROOT}/rate" \
  --rate-baseline "${BASELINE_ROOT}" \
  --noise-root "${ROOT}/noise" \
  --output "${ROOT}/analysis" \
  --sigma-y-dy "0.18445488487247555,1,2,4" \
  --seeds 10 --nboot 5000

echo "complete output=${ROOT}"
