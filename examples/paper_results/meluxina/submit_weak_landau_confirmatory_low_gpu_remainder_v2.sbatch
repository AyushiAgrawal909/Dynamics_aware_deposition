#!/bin/bash -l
# Non-overlapping low-PPC remainder from canceled array element 5200458_0.
# Seeds 300..303 moved to CPU; this packed GPU task covers only 304..309.
#SBATCH --job-name=p1_weak_v2_low_grem
#SBATCH --account=p201103
#SBATCH --partition=gpu
#SBATCH --qos=default
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ayushiphysics909@gmail.com
#SBATCH --output=/mnt/tier2/users/u103576/logs/p1_weak_v2_low_grem_%j.out
#SBATCH --error=/mnt/tier2/users/u103576/logs/p1_weak_v2_low_grem_%j.err

set -euo pipefail

PROJECT=/home/users/u103576/hexapic-main/pic_kde_2d
CONFIRMATORY=/home/users/u103576/hexapic-main/paper1_source_package/reproducibility
CONFIG=${CONFIRMATORY}/weak_landau_confirmatory_v2_global_velocity.json
PYROOT=/apps/USE/easybuild/release/2024.1/software/Python/3.11.10-GCCcore-13.3.0
PYTHON=${PYROOT}/bin/python3
ROOT=/mnt/tier2/users/u103576/paper1_validation/weak_landau_confirmatory_v2_global_velocity
PREFLIGHT=${ROOT}/preflight/preflight.json

export LD_LIBRARY_PATH="${PYROOT}/lib:${PYROOT}/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${PROJECT}:${PROJECT}/scripts:${PYTHONPATH:-}"
export NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

for seed in 202609304 202609305 202609306 202609307 202609308 202609309; do
  target=${ROOT}/low/seed${seed}
  partial=${target}.partial
  if [[ -e "${target}" || -e "${partial}" ]]; then
    echo "Refusing duplicate v2 low seed ${seed}: ${target} or ${partial} exists" >&2
    exit 2
  fi
done

cd "${PROJECT}"
pids=()
for seed in 202609304 202609305 202609306 202609307 202609308 202609309; do
  cache=${ROOT}/numba/low_seed${seed}_gpu_remainder
  mkdir -p "${cache}"
  (
    env NUMBA_CACHE_DIR="${cache}" \
      "${PYTHON}" scripts/run_weak_landau_confirmatory_v2.py --config "${CONFIG}" \
        --preflight "${PREFLIGHT}" simulate-low-seed \
        --output "${ROOT}/low" --seed "${seed}"
  ) >"/mnt/tier2/users/u103576/logs/p1_weak_v2_low_seed${seed}_gpu_remainder_${SLURM_JOB_ID}.out" \
    2>"/mnt/tier2/users/u103576/logs/p1_weak_v2_low_seed${seed}_gpu_remainder_${SLURM_JOB_ID}.err" &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  wait "${pid}" || status=1
done
exit "${status}"
