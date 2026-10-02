#!/bin/bash -l
# GPU-placement amendment for the frozen E2 scan. The numerical workload is
# CPU/Numba, packed on the GPU node's host cores; the scientific payload and
# four-hour hard cap are unchanged. The reducer runs only after all raw files.
#SBATCH --job-name=p1_E2_vb_scan_g
#SBATCH --account=p201103
#SBATCH --partition=gpu
#SBATCH --qos=long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --gpus-per-node=1
#SBATCH --mem=220G
#SBATCH --time=04:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ayushiphysics909@gmail.com
#SBATCH --output=/mnt/tier2/users/u103576/logs/p1_E2_vb_scan_g_%j.out
#SBATCH --error=/mnt/tier2/users/u103576/logs/p1_E2_vb_scan_g_%j.err

set -euo pipefail

PROJECT=/home/users/u103576/hexapic-main/pic_kde_2d
REPRO=${PROJECT}/reproducibility
PYROOT=/apps/USE/easybuild/release/2024.1/software/Python/3.11.10-GCCcore-13.3.0
PYTHON=${PYROOT}/bin/python3
ROOT=/mnt/tier2/users/u103576/paper1_validation/e2_beam_drift_scan_${SLURM_JOB_ID}

# SBATCH lines in the shared CPU file are comments when executed as a payload.
bash "${REPRO}/submit_E2_beam_drift_scan_cpu.sbatch"

export LD_LIBRARY_PATH="${PYROOT}/lib:${PYROOT}/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${PROJECT}:${PROJECT}/scripts:${PYTHONPATH:-}"
export NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
"${PYTHON}" "${REPRO}/analyze_E2_beam_drift_scan.py" \
  --config "${ROOT}/frozen_campaign.json" \
  --input "${ROOT}/runs" \
  --output "${ROOT}/analysis"
echo "complete output=${ROOT}"
