#!/bin/bash -l
# Reduce the completed frozen E2 drift scan. Submit with RAW_JOB exported and a
# dependency on that raw job; analysis cannot influence the frozen design.
#SBATCH --job-name=p1_E2_vb_reduce
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
#SBATCH --output=/mnt/tier2/users/u103576/logs/p1_E2_vb_reduce_%j.out
#SBATCH --error=/mnt/tier2/users/u103576/logs/p1_E2_vb_reduce_%j.err

set -euo pipefail
: "${RAW_JOB:?submit with --export=ALL,RAW_JOB=<raw job id>}"

PROJECT=/home/users/u103576/hexapic-main/pic_kde_2d
REPRO=${PROJECT}/reproducibility
PYROOT=/apps/USE/easybuild/release/2024.1/software/Python/3.11.10-GCCcore-13.3.0
PYTHON=${PYROOT}/bin/python3
ROOT=/mnt/tier2/users/u103576/paper1_validation/e2_beam_drift_scan_${RAW_JOB}
CONFIG=${ROOT}/frozen_campaign.json

export LD_LIBRARY_PATH="${PYROOT}/lib:${PYROOT}/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${PROJECT}:${PROJECT}/scripts:${PYTHONPATH:-}"
export NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

[[ -f "${ROOT}/COMPLETED" ]]
[[ "$(sha256sum "${CONFIG}" | awk '{print $1}')" == "e664460d392a216b953bc87d3859c3bce9e1cce2f1652ab9417c1a58b3baf446" ]]
"${PYTHON}" "${REPRO}/analyze_E2_beam_drift_scan.py" \
  --config "${CONFIG}" --input "${ROOT}/runs" --output "${ROOT}/analysis"
echo "complete output=${ROOT}/analysis"
