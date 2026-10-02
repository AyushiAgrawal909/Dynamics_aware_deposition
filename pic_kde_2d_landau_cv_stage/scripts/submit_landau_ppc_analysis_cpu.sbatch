#!/bin/bash -l
#SBATCH --job-name=p1_ppc_an
#SBATCH --account=p201103
#SBATCH --partition=cpu
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:45:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ayushiphysics909@gmail.com
#SBATCH --output=/mnt/tier2/users/u103576/logs/p1_ppc_an_%j.out
#SBATCH --error=/mnt/tier2/users/u103576/logs/p1_ppc_an_%j.err

set -euo pipefail
: "${SELECTOR_ROOT:?missing SELECTOR_ROOT}"
: "${DYNAMICS_ROOT:?missing DYNAMICS_ROOT}"
PROJECT=/home/users/u103576/hexapic-main/pic_kde_2d
PYROOT=/apps/USE/easybuild/release/2024.1/software/Python/3.11.10-GCCcore-13.3.0
PYTHON=${PYROOT}/bin/python3
OUT=${OUTPUT_ROOT:-${DYNAMICS_ROOT}/analysis}
export LD_LIBRARY_PATH="${PYROOT}/lib:${PYROOT}/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${PROJECT}:${PYTHONPATH:-}"
export MPLBACKEND=Agg OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd "${PROJECT}"
"${PYTHON}" scripts/reduce_landau_ppc_convergence.py \
  --input "${DYNAMICS_ROOT}" --selector-root "${SELECTOR_ROOT}" --output "${OUT}"
