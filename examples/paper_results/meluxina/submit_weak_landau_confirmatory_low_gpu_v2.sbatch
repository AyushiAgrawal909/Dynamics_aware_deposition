#!/bin/bash -l
# Frozen v2 weak-Landau PPC-1024 campaign with global inverse-CDF velocities.
# Each array element packs ten independent seeds on one GPU-partition node;
# every seed runs the three paired arms sequentially from one particle array.
# The solver is CPU-backed: the GPU request is for queue placement only.
#SBATCH --job-name=p1_weak_v2_low
#SBATCH --account=p201103
#SBATCH --partition=gpu
#SBATCH --qos=default
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --array=0-3%2
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ayushiphysics909@gmail.com
#SBATCH --output=/mnt/tier2/users/u103576/logs/p1_weak_v2_low_%A_%a.out
#SBATCH --error=/mnt/tier2/users/u103576/logs/p1_weak_v2_low_%A_%a.err

set -euo pipefail

PROJECT=/home/users/u103576/hexapic-main/pic_kde_2d
CONFIRMATORY=/home/users/u103576/hexapic-main/paper1_source_package/reproducibility
CONFIG=${CONFIRMATORY}/weak_landau_confirmatory_v2_global_velocity.json
PYROOT=/apps/USE/easybuild/release/2024.1/software/Python/3.11.10-GCCcore-13.3.0
PYTHON=${PYROOT}/bin/python3
ROOT=/mnt/tier2/users/u103576/paper1_validation/weak_landau_confirmatory_v2_global_velocity
PREFLIGHT=${ROOT}/preflight/preflight.json
GROUP_START=$((202609300 + 10 * SLURM_ARRAY_TASK_ID))
GROUP_STOP=$((GROUP_START + 9))

export LD_LIBRARY_PATH="${PYROOT}/lib:${PYROOT}/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${PROJECT}:${PROJECT}/scripts:${PYTHONPATH:-}"
export NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

# Refuse the whole packed task before starting any process if one target or
# partial directory already exists.  This preserves fixed-sample traceability.
for seed in $(seq "${GROUP_START}" "${GROUP_STOP}"); do
  target=${ROOT}/low/seed${seed}
  partial=${target}.partial
  if [[ -e "${target}" || -e "${partial}" ]]; then
    echo "Refusing duplicate v2 low seed ${seed}: ${target} or ${partial} exists" >&2
    exit 2
  fi
done

cd "${PROJECT}"
pids=()
for seed in $(seq "${GROUP_START}" "${GROUP_STOP}"); do
  cache=${ROOT}/numba/low_seed${seed}
  mkdir -p "${cache}"
  (
    env NUMBA_CACHE_DIR="${cache}" \
      "${PYTHON}" scripts/run_weak_landau_confirmatory_v2.py --config "${CONFIG}" \
        --preflight "${PREFLIGHT}" simulate-low-seed \
        --output "${ROOT}/low" --seed "${seed}"
  ) >"/mnt/tier2/users/u103576/logs/p1_weak_v2_low_seed${seed}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out" \
    2>"/mnt/tier2/users/u103576/logs/p1_weak_v2_low_seed${seed}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.err" &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  wait "${pid}" || status=1
done
exit "${status}"
