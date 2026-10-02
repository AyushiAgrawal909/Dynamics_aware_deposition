#!/bin/bash -l
# Resume only the seven PPC-4096 CIC reference seeds that remained pending in
# CPU array 5199342.  The NumPy/Numba solver is CPU-backed; the GPU request is
# for MeluXina GPU-partition placement, so all seven seeds share one GPU node.
# This script deliberately refuses any seed whose final or partial output
# already exists.
#SBATCH --job-name=p1_weak_ref_gpu33_39
#SBATCH --account=p201103
#SBATCH --partition=gpu
#SBATCH --qos=default
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ayushiphysics909@gmail.com
#SBATCH --output=/mnt/tier2/users/u103576/logs/p1_weak_ref_gpu33_39_%j.out
#SBATCH --error=/mnt/tier2/users/u103576/logs/p1_weak_ref_gpu33_39_%j.err

set -euo pipefail

PROJECT=/home/users/u103576/hexapic-main/pic_kde_2d
CONFIRMATORY=/home/users/u103576/hexapic-main/paper1_source_package/reproducibility
CONFIG=${CONFIRMATORY}/weak_landau_confirmatory_v1.json
PYROOT=/apps/USE/easybuild/release/2024.1/software/Python/3.11.10-GCCcore-13.3.0
PYTHON=${PYROOT}/bin/python3
ROOT=/mnt/tier2/users/u103576/paper1_validation/weak_landau_confirmatory_v1

export LD_LIBRARY_PATH="${PYROOT}/lib:${PYROOT}/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${PROJECT}:${PROJECT}/scripts:${PYTHONPATH:-}"
export NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

indices=(33 34 35 36 37 38 39)
for index in "${indices[@]}"; do
  seed=$((202609100 + index))
  target=${ROOT}/reference/seed${seed}
  partial=${target}.partial
  if [[ -e "${target}" || -e "${partial}" ]]; then
    echo "Refusing duplicate seed ${seed}: ${target} or ${partial} exists" >&2
    exit 2
  fi
done

cd "${PROJECT}"
pids=()
for index in "${indices[@]}"; do
  seed=$((202609100 + index))
  cache=${ROOT}/numba/reference_seed${seed}_gpu
  mkdir -p "${cache}"
  (
    env NUMBA_CACHE_DIR="${cache}" \
      "${PYTHON}" scripts/run_weak_landau_confirmatory.py --config "${CONFIG}" \
        --preflight "${ROOT}/preflight/preflight.json" \
        simulate-reference-seed --output "${ROOT}/reference" --seed "${seed}"
  ) >"/mnt/tier2/users/u103576/logs/p1_weak_ref_gpu_seed${seed}_${SLURM_JOB_ID}.out" \
    2>"/mnt/tier2/users/u103576/logs/p1_weak_ref_gpu_seed${seed}_${SLURM_JOB_ID}.err" &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  wait "${pid}" || status=1
done
exit "${status}"
