#!/bin/bash -l
# Dense E2 anisotropic-noise design with held-out seed IDs 100--109.
# Each array element evaluates all 85 filters on one paired trajectory.
#SBATCH --job-name=p1_E2_dense_noise
#SBATCH --account=p201103
#SBATCH --partition=cpu
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --array=0-9%10
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ayushiphysics909@gmail.com
#SBATCH --output=/mnt/tier2/users/u103576/logs/p1_E2_dense_noise_%A_%a.out
#SBATCH --error=/mnt/tier2/users/u103576/logs/p1_E2_dense_noise_%A_%a.err

set -euo pipefail

PROJECT=/home/users/u103576/hexapic-main/pic_kde_2d
AUDIT=/home/users/u103576/paper1_reproducibility
PYROOT=/apps/USE/easybuild/release/2024.1/software/Python/3.11.10-GCCcore-13.3.0
PYTHON=${PYROOT}/bin/python3
ROOT=/mnt/tier2/users/u103576/paper1_validation/e2_anisotropic_dense_${SLURM_ARRAY_JOB_ID}
SEED=$((100 + SLURM_ARRAY_TASK_ID))
OUT=${ROOT}/seed${SEED}

export LD_LIBRARY_PATH="${PYROOT}/lib:${PYROOT}/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${PROJECT}:${PROJECT}/scripts:${AUDIT}:${PYTHONPATH:-}"
export NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export MPLCONFIGDIR=${OUT}/matplotlib
mkdir -p "${OUT}/numba" "${MPLCONFIGDIR}"
cd "${PROJECT}"

sha256sum "${AUDIT}/run_E2_anisotropic_dense_noise_seed.py" >"${OUT}/source.sha256"
env NUMBA_CACHE_DIR="${OUT}/numba" \
  "${PYTHON}" "${AUDIT}/run_E2_anisotropic_dense_noise_seed.py" \
    --output "${OUT}/result" --seed "${SEED}" \
    --q-budget 0.009086217804777913 \
    --ppc 32 --reference-ppc 1024 \
    --Nx 128 --Ny 16 --Lx 64 --Ly 16 --kx-mode 12 \
    --v-beam 0.8 --v-thermal 0.05 --alpha 0.001 \
    --live-sigma-dx 0.25 --dt 0.05 --n-steps 80 \
    --diag-stride 2 --eval-t-min 0.5 --eval-t-max 4.0 \
    >"${OUT}/run.log" 2>&1

