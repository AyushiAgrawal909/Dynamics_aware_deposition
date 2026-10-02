#!/bin/bash -l
#SBATCH --job-name=p1_L_noise
#SBATCH --account=p201103
#SBATCH --partition=gpu
#SBATCH --qos=long
#SBATCH --array=0-9%10
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ayushiphysics909@gmail.com
#SBATCH --output=/mnt/tier2/users/u103576/logs/p1_L_noise_%A_%a.out
#SBATCH --error=/mnt/tier2/users/u103576/logs/p1_L_noise_%A_%a.err

set -euo pipefail
PROJECT=/home/users/u103576/hexapic-main/pic_kde_2d
PYROOT=/apps/USE/easybuild/release/2024.1/software/Python/3.11.10-GCCcore-13.3.0
PYTHON=${PYROOT}/bin/python3
ROOT=${OUTPUT_ROOT:-/mnt/tier2/users/u103576/logs/p1_landau_noise_${SLURM_ARRAY_JOB_ID}}
SEED=${SLURM_ARRAY_TASK_ID}
OUT=${ROOT}/seed${SEED}
SIGMAS='0,0.25,0.5,1,1.8329728493314705,2,3,4,6,8,12,16,20,24,30,40,60'

export LD_LIBRARY_PATH="${PYROOT}/lib:${PYROOT}/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${PROJECT}:${PYTHONPATH:-}"
export NUMBA_CACHE_DIR=${OUT}/numba
export NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mkdir -p "${OUT}/mise" "${NUMBA_CACHE_DIR}"
cd "${PROJECT}"
"${PYTHON}" scripts/run_landau_mise_gate.py \
  --output "${OUT}/mise" --sigma-dx "${SIGMAS}" \
  --seed-start "${SEED}" --seeds 1 --ppc 1024 --reference-ppc 16384 \
  --Nx 64 --Ny 16 --Ly 8 --kx-mode 1 --k-lambdaD 0.4 --alpha 0.01 \
  --live-sigma-dx 0.25 --dt 0.05 --n-steps 600 \
  --diag-stride 10 --eval-t-min 5 --eval-t-max 30 --nboot 200
