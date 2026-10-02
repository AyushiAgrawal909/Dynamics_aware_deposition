#!/bin/bash -l
#SBATCH --job-name=H0_clip_audit
#SBATCH --account=p201103
#SBATCH --partition=gpu
#SBATCH --qos=default
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-node=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ayushiphysics909@gmail.com
#SBATCH --output=/mnt/tier2/users/u103576/logs/H0_clip_audit_%j.out
#SBATCH --error=/mnt/tier2/users/u103576/logs/H0_clip_audit_%j.err

set -euo pipefail

PROJECT=/home/users/u103576/hexapic-main/pic_kde_2d
PYROOT=/apps/USE/easybuild/release/2024.1/software/Python/3.11.10-GCCcore-13.3.0
PYTHON=${PYROOT}/bin/python3
ROOT=${OUTPUT_ROOT:-/mnt/tier2/users/u103576/logs/H0_clip_audit_${SLURM_JOB_ID}}

export LD_LIBRARY_PATH="${PYROOT}/lib:${PYROOT}/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${PROJECT}:${PYTHONPATH:-}"
export NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMBA_CACHE_DIR="${ROOT}/numba"
mkdir -p "${ROOT}" "${NUMBA_CACHE_DIR}"
cd "${PROJECT}"

"${PYTHON}" scripts/run_H0_analytic_ladder.py \
  --output "${ROOT}" \
  --dynamic-ranges 10,100,1000 \
  --seed-start 0 \
  --seeds 10 \
  --ppc 32 \
  --sigma-c4-dx 1.0 \
  --sigma-min-dx 0.1 \
  --width-bins 32 \
  --nboot 2000
