#!/bin/bash -l
#SBATCH --job-name=df_band_reduce
#SBATCH --account=p201103
#SBATCH --partition=cpu
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ayushiphysics909@gmail.com
#SBATCH --output=/mnt/tier2/users/u103576/logs/df_band_reduce_%j.out
#SBATCH --error=/mnt/tier2/users/u103576/logs/df_band_reduce_%j.err

set -euo pipefail

PROJECT=/home/users/u103576/hexapic-main/pic_kde_2d
PYROOT=/apps/USE/easybuild/release/2024.1/software/Python/3.11.10-GCCcore-13.3.0
PYTHON=${PYROOT}/bin/python3
ROOT=${OUTPUT_ROOT:-/mnt/tier2/users/u103576/logs/df_band_1_14_4873221}

export LD_LIBRARY_PATH="${PYROOT}/lib:${PYROOT}/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${PROJECT}:${PROJECT}/scripts:${PYTHONPATH:-}"
cd "${PROJECT}"

"${PYTHON}" scripts/analyze_linearized_delta_f_band.py \
  --input "${ROOT}" --output "${ROOT}/analysis" \
  --modes 1-14 --seeds 10 --nboot 2000

echo "complete output=${ROOT}"
