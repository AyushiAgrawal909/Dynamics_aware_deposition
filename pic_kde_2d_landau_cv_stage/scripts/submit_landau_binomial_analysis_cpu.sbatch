#!/bin/bash -l
#SBATCH --job-name=p1_Lb_an
#SBATCH --account=p201103
#SBATCH --partition=cpu
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ayushiphysics909@gmail.com
#SBATCH --output=/mnt/tier2/users/u103576/logs/p1_Lb_an_%j.out
#SBATCH --error=/mnt/tier2/users/u103576/logs/p1_Lb_an_%j.err

set -euo pipefail
: "${INPUT_ROOT:?set INPUT_ROOT}"
: "${ALPHA:?set ALPHA}"
PROJECT=/home/users/u103576/hexapic-main/pic_kde_2d
PYROOT=/apps/USE/easybuild/release/2024.1/software/Python/3.11.10-GCCcore-13.3.0
export LD_LIBRARY_PATH="${PYROOT}/lib:${PYROOT}/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${PROJECT}:${PYTHONPATH:-}"
cd "${PROJECT}"
ANALYSIS_OUT=${OUTPUT_ROOT:-${INPUT_ROOT}/analysis}
"${PYROOT}/bin/python3" scripts/analyze_landau_binomial_passes.py \
  --input "${INPUT_ROOT}/shards" --output "${ANALYSIS_OUT}" \
  --alpha "${ALPHA}" --seeds 10 --nboot 2000
