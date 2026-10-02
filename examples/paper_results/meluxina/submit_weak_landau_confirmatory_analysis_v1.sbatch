#!/bin/bash -l
# Stage 2: comparative rates and same-campaign noise metrics. Do not submit
# until stage-1 window_decisions.json has been retained and reported.
#SBATCH --job-name=p1_weak_confirm_reduce
#SBATCH --account=p201103
#SBATCH --partition=cpu
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --time=01:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ayushiphysics909@gmail.com
#SBATCH --output=/mnt/tier2/users/u103576/logs/p1_weak_confirm_reduce_%j.out
#SBATCH --error=/mnt/tier2/users/u103576/logs/p1_weak_confirm_reduce_%j.err

set -euo pipefail

PROJECT=/home/users/u103576/hexapic-main/pic_kde_2d
CONFIRMATORY=/home/users/u103576/hexapic-main/paper1_source_package/reproducibility
CONFIG=${CONFIRMATORY}/weak_landau_confirmatory_v1.json
ANALYZER=${CONFIRMATORY}/analyze_weak_landau_confirmatory.py
PYROOT=/apps/USE/easybuild/release/2024.1/software/Python/3.11.10-GCCcore-13.3.0
PYTHON=${PYROOT}/bin/python3
ROOT=/mnt/tier2/users/u103576/paper1_validation/weak_landau_confirmatory_v1

export LD_LIBRARY_PATH="${PYROOT}/lib:${PYROOT}/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${PROJECT}:${PROJECT}/scripts:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd "${PROJECT}"

"${PYTHON}" "${ANALYZER}" --config "${CONFIG}" analyze \
  --input-low "${ROOT}/low" \
  --input-reference "${ROOT}/reference" \
  --window-decisions "${ROOT}/stage1_windows/window_decisions.json" \
  --output "${ROOT}/stage2_analysis"
