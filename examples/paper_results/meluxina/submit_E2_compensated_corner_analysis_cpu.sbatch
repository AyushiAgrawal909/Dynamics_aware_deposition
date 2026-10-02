#!/bin/bash -l
#SBATCH --job-name=p1_E2_comp_reduce
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
#SBATCH --output=/mnt/tier2/users/u103576/logs/p1_E2_comp_reduce_%j.out
#SBATCH --error=/mnt/tier2/users/u103576/logs/p1_E2_comp_reduce_%j.err

set -euo pipefail
: "${RATE_JOB_ID:?set RATE_JOB_ID to the completed array job ID}"

PROJECT=/home/users/u103576/hexapic-main/pic_kde_2d
AUDIT=/home/users/u103576/paper1_reproducibility
PYROOT=/apps/USE/easybuild/release/2024.1/software/Python/3.11.10-GCCcore-13.3.0
PYTHON=${PYROOT}/bin/python3
ROOT=/mnt/tier2/users/u103576/paper1_validation/e2_compensated_corner_${RATE_JOB_ID}

export LD_LIBRARY_PATH="${PYROOT}/lib:${PYROOT}/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${PROJECT}:${PROJECT}/scripts:${PYTHONPATH:-}"
"${PYTHON}" "${AUDIT}/analyze_E2_compensated_corner.py" \
  --input "${ROOT}/rate" --output "${ROOT}/analysis" \
  --labels base,p1n1,p2n2,p2n4,p3n4 --seeds 10 --nboot 5000 \
  --fit-t-start 24.32 --fit-t-stop 59.68

