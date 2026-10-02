#!/bin/bash -l
#SBATCH --job-name=p1_E2_dense_reduce
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
#SBATCH --output=/mnt/tier2/users/u103576/logs/p1_E2_dense_reduce_%j.out
#SBATCH --error=/mnt/tier2/users/u103576/logs/p1_E2_dense_reduce_%j.err

set -euo pipefail
: "${NOISE_JOB_ID:?set NOISE_JOB_ID to the completed array job ID}"

AUDIT=/home/users/u103576/paper1_reproducibility
PROJECT=/home/users/u103576/hexapic-main/pic_kde_2d
PYROOT=/apps/USE/easybuild/release/2024.1/software/Python/3.11.10-GCCcore-13.3.0
PYTHON=${PYROOT}/bin/python3
ROOT=/mnt/tier2/users/u103576/paper1_validation/e2_anisotropic_dense_${NOISE_JOB_ID}

export LD_LIBRARY_PATH="${PYROOT}/lib:${PYROOT}/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${PROJECT}:${PROJECT}/scripts:${AUDIT}:${PYTHONPATH:-}"
export MPLCONFIGDIR=${ROOT}/matplotlib_reduce
mkdir -p "${MPLCONFIGDIR}"
"${PYTHON}" "${AUDIT}/analyze_E2_anisotropic_dense_noise.py" \
  --input "${ROOT}" --output "${ROOT}/analysis" \
  --seed-start 100 --seeds 10 --train-seeds 5 --nboot 5000

