#!/bin/bash -l
#SBATCH --job-name=p1_ppc_sel
#SBATCH --account=p201103
#SBATCH --partition=cpu
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --array=0-5%6
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ayushiphysics909@gmail.com
#SBATCH --output=/mnt/tier2/users/u103576/logs/p1_ppc_sel_%A_%a.out
#SBATCH --error=/mnt/tier2/users/u103576/logs/p1_ppc_sel_%A_%a.err

set -euo pipefail
PROJECT=/home/users/u103576/hexapic-main/pic_kde_2d
PYROOT=/apps/USE/easybuild/release/2024.1/software/Python/3.11.10-GCCcore-13.3.0
PYTHON=${PYROOT}/bin/python3
PPCS=(8 32 128 512 1024 4096)
PPC=${PPCS[${SLURM_ARRAY_TASK_ID}]}
ROOT=${OUTPUT_ROOT:-${PROJECT}/results/landau_ppc_selectors_${SLURM_ARRAY_JOB_ID}}
OUT=${ROOT}/ppc${PPC}
export LD_LIBRARY_PATH="${PYROOT}/lib:${PYROOT}/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${PROJECT}:${PYTHONPATH:-}"
export NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mkdir -p "${OUT}/numba"
export NUMBA_CACHE_DIR="${OUT}/numba"
cd "${PROJECT}"
"${PYTHON}" scripts/run_landau_ppc_selectors.py --output "${OUT}" --ppc "${PPC}"
