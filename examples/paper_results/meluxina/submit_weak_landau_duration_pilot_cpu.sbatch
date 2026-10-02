#!/bin/bash -l
# Three-seed signal-strength pilot for the weak-Landau two-e-fold duration gate.
# Alpha=0.02 is tested against the preserved alpha=0.01 campaign; it is not
# assumed linear until the measured rate and pre-bounce behavior are checked.
#SBATCH --job-name=p1_weak_duration
#SBATCH --account=p201103
#SBATCH --partition=cpu
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --array=0-3%4
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ayushiphysics909@gmail.com
#SBATCH --output=/mnt/tier2/users/u103576/logs/p1_weak_duration_%A_%a.out
#SBATCH --error=/mnt/tier2/users/u103576/logs/p1_weak_duration_%A_%a.err

set -euo pipefail

PROJECT=/home/users/u103576/hexapic-main/pic_kde_2d
PYROOT=/apps/USE/easybuild/release/2024.1/software/Python/3.11.10-GCCcore-13.3.0
PYTHON=${PYROOT}/bin/python3
ROOT=/mnt/tier2/users/u103576/paper1_validation/weak_landau_duration_pilot_${SLURM_ARRAY_JOB_ID}
LABELS=(gaussian_cic gaussian_c4 compensated_cic compensated_p1n16)
LABEL=${LABELS[${SLURM_ARRAY_TASK_ID}]}

export LD_LIBRARY_PATH="${PYROOT}/lib:${PYROOT}/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${PROJECT}:${PROJECT}/scripts:${PYTHONPATH:-}"
export NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd "${PROJECT}"

pids=()
for seed in $(seq 0 2); do
  if [[ "${LABEL}" == "gaussian_cic" || "${LABEL}" == "gaussian_c4" ]]; then
    SIGMA=0
    if [[ "${LABEL}" == "gaussian_c4" ]]; then SIGMA=1; fi
    TAG=${SIGMA/./p}
    OUT=${ROOT}/gaussian/ppc1024/sigma${TAG}/seed${seed}
    mkdir -p "${OUT}/numba"
    env NUMBA_CACHE_DIR="${OUT}/numba" \
      "${PYTHON}" scripts/run_landau_sigma_refinement.py simulate \
        --output "${OUT}" --ppc 1024 --sigma-dx "${SIGMA}" --seed "${seed}" \
        --alpha 0.02 --dt 0.05 --n-steps 1800 --diag-stride 2 \
        >"${OUT}/run.log" 2>&1 &
  else
    N=0
    if [[ "${LABEL}" == "compensated_p1n16" ]]; then N=16; fi
    OUT=${ROOT}/compensated/passes${N}/seed${seed}
    mkdir -p "${OUT}/numba"
    env NUMBA_CACHE_DIR="${OUT}/numba" \
      "${PYTHON}" scripts/run_landau_binomial_pass.py \
        --output "${OUT}" --passes "${N}" \
        --filter-family compensated_binomial --compensation-order 1 \
        --alpha 0.02 --ppc 1024 --seed "${seed}" \
        --dt 0.05 --n-steps 1800 --diag-stride 2 \
        >"${OUT}/run.log" 2>&1 &
  fi
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do wait "${pid}" || status=1; done
exit "${status}"

