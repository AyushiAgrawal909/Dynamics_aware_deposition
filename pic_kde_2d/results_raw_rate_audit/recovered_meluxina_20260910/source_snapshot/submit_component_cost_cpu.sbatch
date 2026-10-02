#!/bin/bash -l
#SBATCH --job-name=p1_cost
#SBATCH --account=p201103
#SBATCH --partition=cpu
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ayushiphysics909@gmail.com
#SBATCH --output=/mnt/tier2/users/u103576/logs/p1_cost_%j.out
#SBATCH --error=/mnt/tier2/users/u103576/logs/p1_cost_%j.err

set -euo pipefail
PROJECT=/home/users/u103576/hexapic-main/pic_kde_2d
PYROOT=/apps/USE/easybuild/release/2024.1/software/Python/3.11.10-GCCcore-13.3.0
PYTHON=${PYROOT}/bin/python3
OUT=${OUTPUT_ROOT:-${PROJECT}/results/component_cost_${SLURM_JOB_ID}}
export LD_LIBRARY_PATH="${PYROOT}/lib:${PYROOT}/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${PROJECT}:${PYTHONPATH:-}"
export NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mkdir -p "${OUT}"
cd "${PROJECT}"
orders=("cic c4 deltaf" "c4 deltaf cic" "deltaf cic c4" "cic deltaf c4" "c4 cic deltaf")
for rep in 0 1 2 3 4; do
  REP_OUT=${OUT}/rep${rep}
  mkdir -p "${REP_OUT}"
  for method in ${orders[${rep}]}; do
    "${PYTHON}" scripts/benchmark_component_cost.py \
      --method "${method}" --output "${REP_OUT}" --ppc 1024 --steps 200 --seed "${rep}"
  done
done
"${PYTHON}" scripts/reduce_component_cost.py \
  --input "${OUT}" --output "${OUT}/analysis" --expected-repeats 5
