#!/bin/bash -l
#SBATCH --job-name=p1_cvt_an
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
#SBATCH --output=/mnt/tier2/users/u103576/logs/p1_cvt_an_%j.out
#SBATCH --error=/mnt/tier2/users/u103576/logs/p1_cvt_an_%j.err

set -euo pipefail
: "${INPUT_ROOT:?missing INPUT_ROOT}"
PROJECT=/home/users/u103576/hexapic-main/pic_kde_2d
PYROOT=/apps/USE/easybuild/release/2024.1/software/Python/3.11.10-GCCcore-13.3.0
PYTHON=${PYROOT}/bin/python3
OUT=${OUTPUT_ROOT:-${INPUT_ROOT}/analysis_corrected}
export LD_LIBRARY_PATH="${PYROOT}/lib:${PYROOT}/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${PROJECT}:${PYTHONPATH:-}"
export MPLBACKEND=Agg OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd "${PROJECT}"
"${PYTHON}" scripts/analyze_landau_cv_time.py \
  --input "${INPUT_ROOT}" \
  --trace-root "${PROJECT}/results/landau_sigma_refinement_4864819/shards/ppc1024/sigma0" \
  --output "${OUT}"
