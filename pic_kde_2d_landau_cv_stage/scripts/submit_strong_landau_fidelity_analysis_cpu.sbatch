#!/bin/bash -l
#SBATCH --job-name=p1_SL_analyze
#SBATCH --account=p201103
#SBATCH --partition=cpu
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ayushiphysics909@gmail.com
#SBATCH --output=/mnt/tier2/users/u103576/logs/p1_SL_analyze_%j.out
#SBATCH --error=/mnt/tier2/users/u103576/logs/p1_SL_analyze_%j.err

set -euo pipefail

: "${INPUT_ROOT:?submit with INPUT_ROOT=/path/to/strong_landau_fidelity_JOBID}"
FIT_START=${FIT_START:-5.0}
OUTPUT_SUBDIR=${OUTPUT_SUBDIR:-analysis}
PROJECT=/home/users/u103576/hexapic-main/pic_kde_2d
PYROOT=/apps/USE/easybuild/release/2024.1/software/Python/3.11.10-GCCcore-13.3.0
PYTHON=${PYROOT}/bin/python3

export LD_LIBRARY_PATH="${PYROOT}/lib:${PYROOT}/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${PROJECT}:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

cd "${PROJECT}"
"${PYTHON}" scripts/analyze_strong_landau_fidelity.py \
  --input "${INPUT_ROOT}/shards" \
  --output "${INPUT_ROOT}/${OUTPUT_SUBDIR}" \
  --ppc 1024 --seeds 10 --nboot 2000 --fit-start "${FIT_START}"
