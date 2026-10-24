#!/bin/bash
# Apply the frozen T1 analysis to the finished T2 bundles.
#SBATCH --job-name=nta-t2-analyze
#SBATCH --account=PLACEHOLDER_ACCOUNT
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
set -euo pipefail
module load Python                  # PLACEHOLDER
source "$HOME/venvs/dynamics-aware-pic/bin/activate"   # PLACEHOLDER
REPO=${REPO:-$HOME/dynamics_aware_deposition}        # PLACEHOLDER
OUT=${OUT:-$SCRATCH/nta_t2/bundles}                   # PLACEHOLDER
cd "$REPO/experiments/nonthermal_attractor"
python3 scripts/t2_analyze.py configs/t2_waterbag_dv020.yaml "$OUT" --out results/t2_production --label PRODUCTION
