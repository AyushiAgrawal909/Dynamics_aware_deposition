#!/bin/bash -l
# Three fresh diagnostic seeds for the corrected global velocity assignment.
# The NumPy/Numba solver is CPU-backed; the GPU request changes queue placement.
#SBATCH --job-name=p1_weak_global_pilot
#SBATCH --account=p201103
#SBATCH --partition=gpu
#SBATCH --qos=default
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --mem=40G
#SBATCH --time=02:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ayushiphysics909@gmail.com
#SBATCH --output=/mnt/tier2/users/u103576/logs/p1_weak_global_pilot_%j.out
#SBATCH --error=/mnt/tier2/users/u103576/logs/p1_weak_global_pilot_%j.err

set -euo pipefail

PROJECT=/home/users/u103576/hexapic-main/pic_kde_2d
CONFIRMATORY=/home/users/u103576/hexapic-main/paper1_source_package/reproducibility
CONFIG=${CONFIRMATORY}/weak_landau_confirmatory_v1.json
PYROOT=/apps/USE/easybuild/release/2024.1/software/Python/3.11.10-GCCcore-13.3.0
PYTHON=${PYROOT}/bin/python3
ROOT=/mnt/tier2/users/u103576/paper1_validation/weak_landau_global_loading_pilot_v1
BASE_PREFLIGHT=/mnt/tier2/users/u103576/paper1_validation/weak_landau_confirmatory_v1/preflight/preflight.json
PILOT_PREFLIGHT=${ROOT}/preflight/preflight.json

export LD_LIBRARY_PATH="${PYROOT}/lib:${PYROOT}/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${PROJECT}:${PROJECT}/scripts:${PYTHONPATH:-}"
export NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

cd "${PROJECT}"
pids=()
for seed in 202609180 202609181 202609182; do
  target=${ROOT}/runs/seed${seed}
  partial=${target}.partial
  if [[ -e "${target}" || -e "${partial}" ]]; then
    echo "Refusing duplicate pilot seed ${seed}" >&2
    exit 2
  fi
  cache=${ROOT}/numba/seed${seed}
  mkdir -p "${cache}"
  (
    env NUMBA_CACHE_DIR="${cache}" \
      "${PYTHON}" scripts/run_weak_landau_global_loading_pilot.py \
        --config "${CONFIG}" --base-preflight "${BASE_PREFLIGHT}" \
        simulate-seed --pilot-preflight "${PILOT_PREFLIGHT}" \
        --output "${ROOT}/runs" --seed "${seed}"
  ) >"/mnt/tier2/users/u103576/logs/p1_weak_global_pilot_seed${seed}_${SLURM_JOB_ID}.out" \
    2>"/mnt/tier2/users/u103576/logs/p1_weak_global_pilot_seed${seed}_${SLURM_JOB_ID}.err" &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  wait "${pid}" || status=1
done
exit "${status}"
