#!/bin/bash -l
# CPU-quota slice of the frozen v2 campaign: exactly four low-PPC seeds.
# Prior campaign maximum: 2317 s/seed, so four seeds consume at most 2.574
# CPU-node hours under the observed runtime envelope.  All other low seeds are
# assigned to non-overlapping GPU-partition launchers.
#SBATCH --job-name=p1_weak_v2_low_cpu
#SBATCH --account=p201103
#SBATCH --partition=cpu
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=01:15:00
#SBATCH --array=0-3%4
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ayushiphysics909@gmail.com
#SBATCH --output=/mnt/tier2/users/u103576/logs/p1_weak_v2_low_cpu_%A_%a.out
#SBATCH --error=/mnt/tier2/users/u103576/logs/p1_weak_v2_low_cpu_%A_%a.err

set -euo pipefail

PROJECT=/home/users/u103576/hexapic-main/pic_kde_2d
CONFIRMATORY=/home/users/u103576/hexapic-main/paper1_source_package/reproducibility
CONFIG=${CONFIRMATORY}/weak_landau_confirmatory_v2_global_velocity.json
PYROOT=/apps/USE/easybuild/release/2024.1/software/Python/3.11.10-GCCcore-13.3.0
PYTHON=${PYROOT}/bin/python3
ROOT=/mnt/tier2/users/u103576/paper1_validation/weak_landau_confirmatory_v2_global_velocity
PREFLIGHT=${ROOT}/preflight/preflight.json
SEED=$((202609300 + SLURM_ARRAY_TASK_ID))
TARGET=${ROOT}/low/seed${SEED}
PARTIAL=${TARGET}.partial
CACHE=${ROOT}/numba/low_seed${SEED}_cpu

if [[ -e "${TARGET}" || -e "${PARTIAL}" ]]; then
  echo "Refusing duplicate v2 low seed ${SEED}: ${TARGET} or ${PARTIAL} exists" >&2
  exit 2
fi

export LD_LIBRARY_PATH="${PYROOT}/lib:${PYROOT}/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${PROJECT}:${PROJECT}/scripts:${PYTHONPATH:-}"
export NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mkdir -p "${CACHE}"
cd "${PROJECT}"

env NUMBA_CACHE_DIR="${CACHE}" \
  "${PYTHON}" scripts/run_weak_landau_confirmatory_v2.py --config "${CONFIG}" \
    --preflight "${PREFLIGHT}" simulate-low-seed \
    --output "${ROOT}/low" --seed "${SEED}"
