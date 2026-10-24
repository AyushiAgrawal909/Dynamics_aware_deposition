#!/bin/bash
# T2 regulator scan as a SLURM array: one task per (backend, level, seed).
# Lines marked PLACEHOLDER are environment-specific and MUST be edited.
#
#   python3 scripts/t2_scan.py configs/t2_waterbag_dv020.yaml --out $OUT --list > slurm/t2_jobs.txt
#   sbatch --array=1-$(wc -l < slurm/t2_jobs.txt) slurm/t2_array.sbatch
#
# Freeze protocol/T2_PROTOCOL.yaml in its own commit BEFORE submitting.
#SBATCH --job-name=nta-t2
#SBATCH --account=PLACEHOLDER_ACCOUNT
#SBATCH --partition=cpu
#SBATCH --qos=default
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --time=48:00:00           # PLACEHOLDER: finest PIC (16384 ppc) and finest Vlasov runs; see reports/COST_ESTIMATE.md
#SBATCH --output=logs/%x_%A_%a.out

set -euo pipefail
module load env/release/2023.1     # PLACEHOLDER
module load Python                  # PLACEHOLDER
source "$HOME/venvs/dynamics-aware-pic/bin/activate"   # PLACEHOLDER: env with requirements-simulation.txt

export NUMBA_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
REPO=${REPO:-$HOME/dynamics_aware_deposition}        # PLACEHOLDER
OUT=${OUT:-$SCRATCH/nta_t2/bundles}                   # PLACEHOLDER: large outputs outside the repo
mkdir -p "$OUT" logs

cd "$REPO/experiments/nonthermal_attractor"
CMD=$(sed -n "${SLURM_ARRAY_TASK_ID}p" slurm/t2_jobs.txt)
echo "task ${SLURM_ARRAY_TASK_ID}: ${CMD}"
srun ${CMD}
