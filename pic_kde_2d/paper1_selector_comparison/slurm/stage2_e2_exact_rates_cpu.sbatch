#!/bin/bash -l
#SBATCH --job-name=p1_E2_rates
#SBATCH --account=p201103
#SBATCH --partition=gpu
#SBATCH --qos=default
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=30
#SBATCH --gpus-per-node=1
#SBATCH --mem=120G
#SBATCH --time=08:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ayushiphysics909@gmail.com
#SBATCH --output=/mnt/tier2/users/u103576/logs/p1_E2_rates_%j.out
#SBATCH --error=/mnt/tier2/users/u103576/logs/p1_E2_rates_%j.err

set -euo pipefail
PROJECT=/home/users/u103576/hexapic-main/pic_kde_2d
PYROOT=/apps/USE/easybuild/release/2024.1/software/Python/3.11.10-GCCcore-13.3.0
PYTHON=${PYROOT}/bin/python3
ROOT=${OUTPUT_ROOT:-/mnt/tier2/users/u103576/logs/p1_e2_rates_${SLURM_JOB_ID}}
export LD_LIBRARY_PATH="${PYROOT}/lib:${PYROOT}/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${PROJECT}:${PYTHONPATH:-}"
export NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mkdir -p "${ROOT}/raw" "${ROOT}/analysis"
cd "${PROJECT}"
"${PYTHON}" scripts/run_E2_m8_sigma_sweep_highppc.py \
  --output "${ROOT}/raw" --sigma-dx 0.22,5.819326058515848,60 \
  --target-mode 8 --ppc 2048 --seeds 10 --workers 30 \
  --dt 0.02 --t-end 35 --diag-stride 2
"${PYTHON}" scripts/analyze_E2_m8_sigma_sweep_highppc.py \
  --input "${ROOT}/raw" --output "${ROOT}/analysis" \
  --sigma-dx 0.22,5.819326058515848,60 --target-mode 8 --nboot 2000
