#!/bin/bash -l
#SBATCH --job-name=p1_hoc_rate
#SBATCH --account=p201103
#SBATCH --partition=cpu
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=72G
#SBATCH --time=04:00:00
#SBATCH --array=0-15%4
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=ayushiphysics909@gmail.com
#SBATCH --output=/mnt/tier2/users/u103576/logs/p1_hoc_rate_%A_%a.out
#SBATCH --error=/mnt/tier2/users/u103576/logs/p1_hoc_rate_%A_%a.err

set -euo pipefail
: "${OUTPUT_ROOT:?set OUTPUT_ROOT}"
PROJECT=/home/users/u103576/hexapic-main/pic_kde_2d
PYROOT=/apps/USE/easybuild/release/2024.1/software/Python/3.11.10-GCCcore-13.3.0
PYTHON=${PYROOT}/bin/python3
PASSES=(0 1 2 4)
ORDER=$((2 + SLURM_ARRAY_TASK_ID / 8))
WITHIN=$((SLURM_ARRAY_TASK_ID % 8))
BENCH_INDEX=$((WITHIN / 4))
N=${PASSES[$((WITHIN % 4))]}
BENCH=landau
if [[ "${BENCH_INDEX}" -eq 1 ]]; then BENCH=e2; fi

export LD_LIBRARY_PATH="${PYROOT}/lib:${PYROOT}/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${PROJECT}:${PROJECT}/scripts:${PYTHONPATH:-}"
export NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd "${PROJECT}"

pids=()
for seed in $(seq 0 9); do
  OUT=${OUTPUT_ROOT}/p${ORDER}/${BENCH}/shards/passes${N}/seed${seed}
  mkdir -p "${OUT}/numba"
  if [[ "${BENCH}" == "landau" ]]; then
    env NUMBA_CACHE_DIR="${OUT}/numba" \
      "${PYTHON}" scripts/run_landau_binomial_pass.py \
        --output "${OUT}" --passes "${N}" \
        --filter-family compensated_binomial --compensation-order "${ORDER}" \
        --alpha 0.01 --ppc 1024 --seed "${seed}" --n-steps 1800 \
        >"${OUT}/run.log" 2>&1 &
  else
    env NUMBA_CACHE_DIR="${OUT}/numba" \
      "${PYTHON}" scripts/run_linearized_delta_f_mode.py \
        --output "${OUT}" --benchmark two_stream --mode 12 \
        --seed "${seed}" --ppc 512 --alpha 1e-8 \
        --filter-kind compensated_binomial --compensation-order "${ORDER}" \
        --binomial-passes-x "${N}" --binomial-passes-y "${N}" \
        --Nx 128 --Ny 16 --Lx 64 --Ly 16 \
        --v-beam 0.8 --v-thermal 0.05 --dt 0.02 --t-end 60 --diag-stride 2 \
        >"${OUT}/run.log" 2>&1 &
  fi
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "${pid}" || status=1; done
exit "${status}"
