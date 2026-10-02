#!/bin/bash -l
#SBATCH --job-name=p1_energy_an
#SBATCH --account=p201103
#SBATCH --partition=cpu
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ayushiphysics909@gmail.com
#SBATCH --output=/mnt/tier2/users/u103576/logs/p1_energy_an_%j.out
#SBATCH --error=/mnt/tier2/users/u103576/logs/p1_energy_an_%j.err

set -euo pipefail
: "${INPUT_ROOT:?set INPUT_ROOT}"
PROJECT=/home/users/u103576/hexapic-main/pic_kde_2d
PYROOT=/apps/USE/easybuild/release/2024.1/software/Python/3.11.10-GCCcore-13.3.0
export LD_LIBRARY_PATH="${PYROOT}/lib:${PYROOT}/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${PROJECT}:${PYTHONPATH:-}"
cd "${PROJECT}"
"${PYROOT}/bin/python3" scripts/reduce_energy_conservation.py \
  --input "${INPUT_ROOT}" --output "${INPUT_ROOT}/analysis" --seeds 10
