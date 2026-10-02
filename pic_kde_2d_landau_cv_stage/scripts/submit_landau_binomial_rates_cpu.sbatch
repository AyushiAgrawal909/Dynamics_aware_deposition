#!/bin/bash -l
#SBATCH --job-name=p1_L_binom
#SBATCH --account=p201103
#SBATCH --partition=cpu
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --array=0-7%4
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ayushiphysics909@gmail.com
#SBATCH --output=/mnt/tier2/users/u103576/logs/p1_L_binom_%A_%a.out
#SBATCH --error=/mnt/tier2/users/u103576/logs/p1_L_binom_%A_%a.err

set -euo pipefail
: "${BENCHMARK:?set BENCHMARK=weak or strong}"

PROJECT=/home/users/u103576/hexapic-main/pic_kde_2d
PYROOT=/apps/USE/easybuild/release/2024.1/software/Python/3.11.10-GCCcore-13.3.0
PYTHON=${PYROOT}/bin/python3
PASSES=(0 1 2 4 8 12 13 16)
N=${PASSES[${SLURM_ARRAY_TASK_ID}]}
if [[ "${BENCHMARK}" == "weak" ]]; then
  ALPHA=0.01
  NSTEPS=1800
elif [[ "${BENCHMARK}" == "strong" ]]; then
  ALPHA=0.5
  NSTEPS=900
else
  echo "unknown BENCHMARK=${BENCHMARK}" >&2
  exit 2
fi
ROOT=${OUTPUT_ROOT:-${PROJECT}/results/landau_binomial_${BENCHMARK}_${SLURM_ARRAY_JOB_ID}}

export LD_LIBRARY_PATH="${PYROOT}/lib:${PYROOT}/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${PROJECT}:${PYTHONPATH:-}"
export NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd "${PROJECT}"

pids=()
for seed in $(seq 0 9); do
  OUT=${ROOT}/shards/passes${N}/seed${seed}
  mkdir -p "${OUT}/numba"
  (
    export NUMBA_CACHE_DIR="${OUT}/numba"
    "${PYTHON}" scripts/run_landau_binomial_pass.py \
      --output "${OUT}" --passes "${N}" --alpha "${ALPHA}" \
      --ppc 1024 --seed "${seed}" --n-steps "${NSTEPS}"
  ) >"${OUT}/run.log" 2>&1 &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "${pid}" || status=1; done
exit "${status}"
