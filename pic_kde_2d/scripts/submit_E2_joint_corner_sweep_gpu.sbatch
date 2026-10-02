#!/bin/bash -l
# Joint (sigma_x, sigma_y) sweep at the coherently seeded E2 corner (12,1).
#SBATCH --job-name=e2_joint_corner
#SBATCH --account=p201103
#SBATCH --partition=gpu
#SBATCH --qos=long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --gpus-per-node=1
#SBATCH --mem=220G
#SBATCH --time=03:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ayushiphysics909@gmail.com
#SBATCH --output=/mnt/tier2/users/u103576/logs/e2_joint_corner_%j.out
#SBATCH --error=/mnt/tier2/users/u103576/logs/e2_joint_corner_%j.err

set -euo pipefail

PROJECT=/home/users/u103576/hexapic-main/pic_kde_2d
PYROOT=/apps/USE/easybuild/release/2024.1/software/Python/3.11.10-GCCcore-13.3.0
PYTHON=${PYROOT}/bin/python3
ROOT=${OUTPUT_ROOT:-/mnt/tier2/users/u103576/logs/e2_joint_corner_${SLURM_JOB_ID}}
CONFIGS="base:0:0,y012:0:0.12,y018:0:0.18,y02428:0:0.2428,y030:0:0.30,joint08:0.08:0.21110810,joint12:0.12:0.16299273"

export LD_LIBRARY_PATH="${PYROOT}/lib:${PYROOT}/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${PROJECT}:${PROJECT}/scripts:${PYTHONPATH:-}"
export NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mkdir -p "${ROOT}/rate" "${ROOT}/noise"
cd "${PROJECT}"

run_rate_batch() {
  local -a pids=()
  local status=0
  for entry in "$@"; do
    IFS=: read -r label sigma_x sigma_y seed <<<"${entry}"
    OUT=${ROOT}/rate/${label}/seed${seed}
    mkdir -p "${OUT}/numba"
    env NUMBA_CACHE_DIR="${OUT}/numba" \
      "${PYTHON}" scripts/run_linearized_delta_f_mode.py \
      --output "${OUT}" --benchmark two_stream --mode 12 --ky-mode 1 \
      --seed "${seed}" --ppc 512 --alpha 1e-8 \
      --sigma-x-dx "${sigma_x}" --sigma-y-dy "${sigma_y}" \
      --Nx 128 --Ny 16 --Lx 64 --Ly 16 \
      --v-beam 0.8 --v-thermal 0.05 --dt 0.02 --t-end 60 \
      --diag-stride 2 --project-target-mode \
      >"${OUT}/run.log" 2>&1 &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do
    wait "${pid}" || status=1
  done
  return "${status}"
}

rate_entries=()
IFS=, read -ra config_array <<<"${CONFIGS}"
for config in "${config_array[@]}"; do
  IFS=: read -r label sigma_x sigma_y <<<"${config}"
  for seed in $(seq 0 9); do
    rate_entries+=("${label}:${sigma_x}:${sigma_y}:${seed}")
  done
done

run_rate_batch "${rate_entries[@]:0:40}"
run_rate_batch "${rate_entries[@]:40}"

noise_pids=()
for seed in $(seq 0 9); do
  OUT=${ROOT}/noise/seed${seed}
  mkdir -p "${OUT}/numba"
  env NUMBA_CACHE_DIR="${OUT}/numba" \
    "${PYTHON}" scripts/run_E2_joint_corner_noise_seed.py \
    --output "${OUT}" --seed "${seed}" --configs "${CONFIGS}" \
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
  echo "one or more joint-corner noise shards failed" >&2
  exit "${status}"
fi

"${PYTHON}" scripts/analyze_E2_joint_corner_sweep.py \
  --rate-root "${ROOT}/rate" \
  --noise-root "${ROOT}/noise" \
  --output "${ROOT}/analysis" \
  --seeds 10 --nboot 5000 \
  --fit-t-start 24.32 --fit-t-stop 59.68

echo "complete output=${ROOT}"
