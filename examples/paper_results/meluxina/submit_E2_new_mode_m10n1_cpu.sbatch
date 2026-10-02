#!/bin/bash -l
# Independent Gaussian validation at new protected E2 mode (10,1).
#SBATCH --job-name=p1_E2_m10n1
#SBATCH --account=p201103
#SBATCH --partition=cpu
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=72G
#SBATCH --time=04:00:00
#SBATCH --array=0-1%2
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ayushiphysics909@gmail.com
#SBATCH --output=/mnt/tier2/users/u103576/logs/p1_E2_m10n1_%A_%a.out
#SBATCH --error=/mnt/tier2/users/u103576/logs/p1_E2_m10n1_%A_%a.err

set -euo pipefail

PROJECT=/home/users/u103576/hexapic-main/pic_kde_2d
PYROOT=/apps/USE/easybuild/release/2024.1/software/Python/3.11.10-GCCcore-13.3.0
PYTHON=${PYROOT}/bin/python3
ROOT=/mnt/tier2/users/u103576/paper1_validation/e2_new_mode_m10n1_${SLURM_ARRAY_JOB_ID}
LABELS=(base c4)
SIGMA_Y=(0 0.24273459979982182)
LABEL=${LABELS[${SLURM_ARRAY_TASK_ID}]}
WIDTH_Y=${SIGMA_Y[${SLURM_ARRAY_TASK_ID}]}

export LD_LIBRARY_PATH="${PYROOT}/lib:${PYROOT}/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${PROJECT}:${PROJECT}/scripts:${PYTHONPATH:-}"
export NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mkdir -p "${ROOT}/rate/${LABEL}"
cd "${PROJECT}"

pids=()
for seed in $(seq 100 109); do
  OUT=${ROOT}/rate/${LABEL}/seed${seed}
  mkdir -p "${OUT}/numba"
  env NUMBA_CACHE_DIR="${OUT}/numba" \
    "${PYTHON}" scripts/run_linearized_delta_f_mode.py \
      --output "${OUT}" --benchmark two_stream --mode 10 --ky-mode 1 \
      --seed "${seed}" --ppc 512 --alpha 1e-8 \
      --filter-kind gaussian --sigma-x-dx 0 --sigma-y-dy "${WIDTH_Y}" \
      --Nx 128 --Ny 16 --Lx 64 --Ly 16 \
      --v-beam 0.8 --v-thermal 0.05 --dt 0.02 --t-end 60 \
      --diag-stride 2 --project-target-mode >"${OUT}/run.log" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do wait "${pid}" || status=1; done
exit "${status}"

