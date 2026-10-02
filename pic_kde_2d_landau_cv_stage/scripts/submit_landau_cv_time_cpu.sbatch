#!/bin/bash -l
#SBATCH --job-name=p1_cv_time
#SBATCH --account=p201103
#SBATCH --partition=cpu
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ayushiphysics909@gmail.com
#SBATCH --output=/mnt/tier2/users/u103576/logs/p1_cv_time_%j.out
#SBATCH --error=/mnt/tier2/users/u103576/logs/p1_cv_time_%j.err

set -euo pipefail
PROJECT=/home/users/u103576/hexapic-main/pic_kde_2d
PYROOT=/apps/USE/easybuild/release/2024.1/software/Python/3.11.10-GCCcore-13.3.0
PYTHON=${PYROOT}/bin/python3
OUT=${OUTPUT_ROOT:-${PROJECT}/results/landau_cv_time_${SLURM_JOB_ID}}
SIGMAS="0,0.25,0.5,1,2,3,4,5,6,6.5,7,7.5,8,9,10,11,12,14,16,20,30,45,60"
export LD_LIBRARY_PATH="${PYROOT}/lib:${PYROOT}/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${PROJECT}:${PYTHONPATH:-}"
export NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MPLBACKEND=Agg
cd "${PROJECT}"
pids=()
for seed in $(seq 0 9); do
  SEED_OUT=${OUT}/seed${seed}
  mkdir -p "${SEED_OUT}/numba"
  (
    export NUMBA_CACHE_DIR="${SEED_OUT}/numba"
    "${PYTHON}" scripts/run_landau_lscv_aligned.py scan \
      --output "${SEED_OUT}" --seed-start "${seed}" --seeds 1 \
      --ppc 1024 --alpha 0.01 --live-sigma-dx 0.25 \
      --n-steps 600 --diag-stride 10 --eval-t-min 0 --eval-t-max 30 \
      --sigma-dx "${SIGMAS}" --save-time-resolved
  ) >"${SEED_OUT}/run.log" 2>&1 &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "${pid}" || status=1; done
if (( status != 0 )); then exit "${status}"; fi
"${PYTHON}" scripts/analyze_landau_cv_time.py \
  --input "${OUT}" \
  --trace-root "${PROJECT}/results/landau_sigma_refinement_4864819/shards/ppc1024/sigma0" \
  --output "${OUT}/analysis"
