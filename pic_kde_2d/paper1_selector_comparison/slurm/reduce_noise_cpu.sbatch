#!/bin/bash -l
#SBATCH --job-name=p1_noise_reduce
#SBATCH --account=p201103
#SBATCH --partition=gpu
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-node=1
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ayushiphysics909@gmail.com
#SBATCH --output=/mnt/tier2/users/u103576/logs/p1_noise_reduce_%j.out
#SBATCH --error=/mnt/tier2/users/u103576/logs/p1_noise_reduce_%j.err

set -euo pipefail
: "${INPUT_ROOT:?INPUT_ROOT is required}"
: "${BENCHMARK:?BENCHMARK is required}"
PROJECT=/home/users/u103576/hexapic-main/pic_kde_2d
PYROOT=/apps/USE/easybuild/release/2024.1/software/Python/3.11.10-GCCcore-13.3.0
PYTHON=${PYROOT}/bin/python3
OUT=${OUTPUT_ROOT:-${INPUT_ROOT}/reduced}
export LD_LIBRARY_PATH="${PYROOT}/lib:${PYROOT}/lib64:${LD_LIBRARY_PATH:-}"
cd "${PROJECT}"
"${PYTHON}" paper1_selector_comparison/scripts/reduce_noise_shards.py \
  --input "${INPUT_ROOT}" --output "${OUT}" --benchmark "${BENCHMARK}" \
  --seeds 10 --nboot 10000
