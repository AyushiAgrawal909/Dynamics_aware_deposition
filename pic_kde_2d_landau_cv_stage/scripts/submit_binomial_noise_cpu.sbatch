#!/bin/bash -l
#SBATCH --job-name=p1_bin_noise
#SBATCH --account=p201103
#SBATCH --partition=cpu
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --array=0-9%5
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ayushiphysics909@gmail.com
#SBATCH --output=/mnt/tier2/users/u103576/logs/p1_bin_noise_%A_%a.out
#SBATCH --error=/mnt/tier2/users/u103576/logs/p1_bin_noise_%A_%a.err

set -euo pipefail
: "${BENCHMARK:?set BENCHMARK=landau_weak, landau_strong, or e2}"
PROJECT=/home/users/u103576/hexapic-main/pic_kde_2d
PYROOT=/apps/USE/easybuild/release/2024.1/software/Python/3.11.10-GCCcore-13.3.0
PYTHON=${PYROOT}/bin/python3
if [[ "${BENCHMARK}" == "landau_weak" ]]; then
  PARAMS=(--ppc 1024 --reference-ppc 16384 --alpha 0.01 --n-steps 600 --eval-t-min 5 --eval-t-max 30)
elif [[ "${BENCHMARK}" == "landau_strong" ]]; then
  PARAMS=(--ppc 1024 --reference-ppc 16384 --alpha 0.5 --n-steps 144 --eval-t-min 5 --eval-t-max 7.2)
elif [[ "${BENCHMARK}" == "e2" ]]; then
  PARAMS=(--ppc 32 --reference-ppc 1024 --alpha 0.001 --n-steps 80 --eval-t-min 0.5 --eval-t-max 4)
else
  echo "unknown BENCHMARK=${BENCHMARK}" >&2
  exit 2
fi
ROOT=${OUTPUT_ROOT:-${PROJECT}/results/binomial_noise_${BENCHMARK}_${SLURM_ARRAY_JOB_ID}}
OUT=${ROOT}/shards/seed${SLURM_ARRAY_TASK_ID}
mkdir -p "${OUT}/numba"
export LD_LIBRARY_PATH="${PYROOT}/lib:${PYROOT}/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${PROJECT}:${PYTHONPATH:-}"
export NUMBA_CACHE_DIR="${OUT}/numba"
export NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd "${PROJECT}"
"${PYTHON}" scripts/run_binomial_noise_seed.py \
  --output "${OUT}" --benchmark "${BENCHMARK}" \
  --seed "${SLURM_ARRAY_TASK_ID}" "${PARAMS[@]}"
