#!/bin/bash -l
#SBATCH --job-name=p1_ppc_ref
#SBATCH --account=p201103
#SBATCH --partition=cpu
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ayushiphysics909@gmail.com
#SBATCH --output=/mnt/tier2/users/u103576/logs/p1_ppc_ref_%j.out
#SBATCH --error=/mnt/tier2/users/u103576/logs/p1_ppc_ref_%j.err

set -euo pipefail
: "${SNAPSHOT_ROOT:?missing SNAPSHOT_ROOT}"
PROJECT=/home/users/u103576/hexapic-main/pic_kde_2d
PYROOT=/apps/USE/easybuild/release/2024.1/software/Python/3.11.10-GCCcore-13.3.0
PYTHON=${PYROOT}/bin/python3
OUT=${OUTPUT_ROOT:-${PROJECT}/results/landau_ppc_common_reference_${SLURM_JOB_ID}}
export LD_LIBRARY_PATH="${PYROOT}/lib:${PYROOT}/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${PROJECT}:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd "${PROJECT}"
"${PYTHON}" scripts/reduce_landau_ppc_common_reference.py \
  --input "${SNAPSHOT_ROOT}" --output "${OUT}"
