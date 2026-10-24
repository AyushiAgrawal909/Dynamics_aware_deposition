#!/bin/bash -l
# Apply the frozen T1 analysis to the finished T2 bundles.
#SBATCH --job-name=nta-t2-analyze
#SBATCH --account=p201103
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=64
#SBATCH --time=02:00:00
#SBATCH --output=/project/home/p201103/u103576/nta_t2/logs/%x_%j.out
#SBATCH --mail-user=ayushi.ph@sric.iitr.ac.in
#SBATCH --mail-type=ALL
set -euo pipefail
module load env/release/2024.1; module load Python
source /project/home/p201103/u103576/venvs/nta/bin/activate
REPO=${REPO:-/project/home/p201103/u103576/Dynamics_aware_deposition}
OUT=${OUT:-/project/home/p201103/u103576/nta_t2/bundles}
cd "$REPO/experiments/nonthermal_attractor"
python3 scripts/t2_analyze.py configs/t2_waterbag_dv020.yaml "$OUT" --out results/t2_production --label PRODUCTION
