#!/bin/bash -l
#SBATCH --job-name=p1_e2_noise
#SBATCH --account=p201103
#SBATCH --partition=gpu
#SBATCH --qos=default
#SBATCH --array=0-9%10
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus-per-node=1
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ayushiphysics909@gmail.com
#SBATCH --output=/mnt/tier2/users/u103576/logs/p1_e2_noise_%A_%a.out
#SBATCH --error=/mnt/tier2/users/u103576/logs/p1_e2_noise_%A_%a.err

set -euo pipefail
PROJECT=/home/users/u103576/hexapic-main/pic_kde_2d
PYROOT=/apps/USE/easybuild/release/2024.1/software/Python/3.11.10-GCCcore-13.3.0
PYTHON=${PYROOT}/bin/python3
ROOT=${OUTPUT_ROOT:-/mnt/tier2/users/u103576/logs/p1_e2_noise_${SLURM_ARRAY_JOB_ID}}
SEED=${SLURM_ARRAY_TASK_ID}
OUT=${ROOT}/seed${SEED}
SIGMAS='0,0.1,0.2,0.22,0.25,0.35,0.5,0.68,1,2,3,4,5,5.819326058515848,6,8,12,16,20,30,40,60'

export LD_LIBRARY_PATH="${PYROOT}/lib:${PYROOT}/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${PROJECT}:${PYTHONPATH:-}"
export NUMBA_CACHE_DIR=${OUT}/numba
export NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mkdir -p "${OUT}/mise" "${NUMBA_CACHE_DIR}"
cd "${PROJECT}"
"${PYTHON}" scripts/run_E2_mise_gate.py \
  --output "${OUT}/mise" --sigma-dx "${SIGMAS}" \
  --seed-start "${SEED}" --seeds 1 --ppc 32 --reference-ppc 1024 \
  --Nx 128 --Ny 16 --Lx 64 --Ly 16 --kx-mode 12 \
  --v-beam 0.8 --v-thermal 0.05 --alpha 0.001 \
  --live-sigma-dx 0.25 --dt 0.05 --n-steps 80 \
  --diag-stride 2 --eval-t-min 0.5 --eval-t-max 4.0 --nboot 200
