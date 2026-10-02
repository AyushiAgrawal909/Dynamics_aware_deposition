#!/bin/bash -l
#SBATCH --job-name=landau-sigma-R
#SBATCH --account=p201103
#SBATCH --partition=cpu
#SBATCH --qos=short
#SBATCH --array=0-17%18
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=1G
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/tier2/users/u103576/logs/landau_sigma_R_%A_%a.out
#SBATCH --error=/mnt/tier2/users/u103576/logs/landau_sigma_R_%A_%a.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=ayushiphysics909@gmail.com

set -euo pipefail

PROJECT=/home/users/u103576/hexapic-main/pic_kde_2d
PYTHON_ROOT=/apps/USE/easybuild/release/2024.1/software/Python/3.11.10-GCCcore-13.3.0
PYTHON=${PYTHON_ROOT}/bin/python3
RUNNER=${PROJECT}/scripts/run_landau_sigma_refinement.py
PPCS=(32 1024)
SIGMAS=(0 0.3 0.5 1 2 3 5 8 12)
NSEEDS=10
NSIGMA=${#SIGMAS[@]}

combo=${SLURM_ARRAY_TASK_ID}
sigma_index=$((combo % NSIGMA))
ppc_index=$((combo / NSIGMA))
ppc=${PPCS[$ppc_index]}
sigma=${SIGMAS[$sigma_index]}
tag=${sigma/./p}
OUTPUT_ROOT=${OUTPUT_ROOT:-${PROJECT}/results/landau_sigma_refinement_${SLURM_ARRAY_JOB_ID}}

export LD_LIBRARY_PATH="${PYTHON_ROOT}/lib:${PYTHON_ROOT}/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${PROJECT}:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMBA_NUM_THREADS=1
pids=()
for seed in $(seq 0 $((NSEEDS - 1))); do
  SHARD=${OUTPUT_ROOT}/shards/ppc${ppc}/sigma${tag}/seed${seed}
  CACHE=${SHARD}/numba
  mkdir -p "${SHARD}" "${CACHE}"
  srun --exclusive -N 1 -n 1 -c 1 env NUMBA_CACHE_DIR="${CACHE}" "${PYTHON}" "${RUNNER}" simulate \
    --output "${SHARD}" \
    --ppc "${ppc}" \
    --sigma-dx "${sigma}" \
    --seed "${seed}" \
    --alpha 0.01 \
    --n-steps 1800 \
    >"${SHARD}/run.log" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done
if [ "${status}" -ne 0 ]; then
  exit "${status}"
fi

echo "complete ppc=${ppc} sigma=${sigma} seeds=0-$((NSEEDS - 1)) output=${OUTPUT_ROOT}"
