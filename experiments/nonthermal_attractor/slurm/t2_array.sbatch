#!/bin/bash -l
# T2 regulator scan as a SLURM array: one task per line of the job list $JOBS.
# Filled in for Meluxina, project p201103, GPU partition (64 cores/node; the
# code is CPU-only numba, the GPUs stay idle). Submit with slurm/t2_submit.sh,
# which sets --time per group from the timings in slurm/T2_SUBMISSION.md.
#
#   python3 scripts/t2_scan.py configs/t2_waterbag_dv020.yaml --out $OUT --list > slurm/t2_jobs.txt
#   JOBS=slurm/t2_jobs.txt sbatch --array=1-$(wc -l < slurm/t2_jobs.txt) --time=HH:MM:SS slurm/t2_array.sbatch
#SBATCH --job-name=nta-t2
#SBATCH --account=p201103
#SBATCH --partition=gpu
#SBATCH --qos=default
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=02:00:00
#SBATCH --mail-user=ayushi.ph@sric.iitr.ac.in
#SBATCH --mail-type=ALL
#SBATCH --output=/project/home/p201103/u103576/nta_t2/logs/%x_%A_%a.out

set -euo pipefail
module load env/release/2024.1
module load Python
BASE=/project/home/p201103/u103576
source "$BASE/venvs/nta/bin/activate"     # requirements-simulation.txt + pyyaml

export NUMBA_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
REPO=${REPO:-$BASE/Dynamics_aware_deposition}
OUT=${OUT:-$BASE/nta_t2/bundles}
JOBS=${JOBS:-slurm/t2_jobs.txt}
mkdir -p "$OUT"

cd "$REPO/experiments/nonthermal_attractor"
CMD=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$JOBS")
echo "task ${SLURM_ARRAY_TASK_ID}: ${CMD}  (repo $(git rev-parse HEAD))"
srun ${CMD}
