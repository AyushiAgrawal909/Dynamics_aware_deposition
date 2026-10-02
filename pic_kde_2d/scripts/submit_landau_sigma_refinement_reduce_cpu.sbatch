#!/bin/bash -l
#SBATCH --job-name=landau-sigma-reduce
#SBATCH --account=p201103
#SBATCH --partition=cpu
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=/mnt/tier2/users/u103576/logs/landau_sigma_reduce_%j.out
#SBATCH --error=/mnt/tier2/users/u103576/logs/landau_sigma_reduce_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ayushiphysics909@gmail.com

set -euo pipefail
: "${INPUT_ROOT:?submit with INPUT_ROOT set to the array result root}"

PROJECT=/home/users/u103576/hexapic-main/pic_kde_2d
PYTHON_ROOT=/apps/USE/easybuild/release/2024.1/software/Python/3.11.10-GCCcore-13.3.0
PYTHON=${PYTHON_ROOT}/bin/python3
export LD_LIBRARY_PATH="${PYTHON_ROOT}/lib:${PYTHON_ROOT}/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${PROJECT}:${PYTHONPATH:-}"
export MPLCONFIGDIR=${INPUT_ROOT}/mpl-cache
mkdir -p "${MPLCONFIGDIR}" "${INPUT_ROOT}/analysis"

"${PYTHON}" "${PROJECT}/scripts/run_landau_sigma_refinement.py" analyze \
  --input "${INPUT_ROOT}/shards" \
  --output "${INPUT_ROOT}/analysis" \
  --ppcs 32,1024 \
  --sigma-grid 0,0.3,0.5,1,2,3,5,8,12 \
  --seeds 10 \
  --nboot 1000 \
  --eps-gamma 0.10

echo "Landau sigma refinement reduced: ${INPUT_ROOT}/analysis"
