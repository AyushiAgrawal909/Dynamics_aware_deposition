#!/bin/bash -l
# Frozen exact-width E2 validation: archived CIC plus paired q050_x000, ten seeds.
# Prepared for review. This file has not been submitted.
#SBATCH --job-name=p1_q050_exact
#SBATCH --account=p201103
#SBATCH --partition=cpu
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=220G
#SBATCH --time=01:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ayushiphysics909@gmail.com
#SBATCH --output=/mnt/tier2/users/u103576/logs/p1_q050_exact_%j.out
#SBATCH --error=/mnt/tier2/users/u103576/logs/p1_q050_exact_%j.err

set -euo pipefail

PROJECT=/home/users/u103576/hexapic-main/pic_kde_2d
PACKAGE=${PROJECT}/reproducibility/exact_width_q050_validation_20260921
PYROOT=/apps/USE/easybuild/release/2024.1/software/Python/3.11.10-GCCcore-13.3.0
PYTHON=${PYROOT}/bin/python3
CONFIG=${PACKAGE}/config/q050_x000_vb080_exact_width.json
CONFIG_SHA=3b763155abe28c4b105a1824969ff411a6d0b1b6a5f3d9319b23ff3c39ec4800
ROOT=/mnt/tier2/users/u103576/paper1_validation/e2_q050_exact_${SLURM_JOB_ID}
LEGACY_CAMPAIGN=/mnt/tier2/users/u103576/paper1_validation/e2_beam_drift_scan_5201231

export LD_LIBRARY_PATH="${PYROOT}/lib:${PYROOT}/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${PROJECT}:${PROJECT}/scripts:${PYTHONPATH:-}"
export NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

[[ "$(sha256sum "${CONFIG}" | awk '{print $1}')" == "${CONFIG_SHA}" ]]
[[ "$(sha256sum "${PACKAGE}/code/run_E2_q050_exact_arm.py" | awk '{print $1}')" == "1121e06004aadf7081d023cdaf31c8b101e8dd103d28a00bc3a121e9bae393ba" ]]
[[ "$(sha256sum "${PACKAGE}/code/analyze_E2_q050_exact.py" | awk '{print $1}')" == "44b62c850c05e382196a13ecf01dbc9d52db0c6875cdb43a4d91a593f8598629" ]]
[[ "$(sha256sum "${PACKAGE}/code/audit_existing_cic_reuse.py" | awk '{print $1}')" == "7d2b1e9b876077f1d40008aa835d882ab96cbc68fe912b19d3d1659ef425af73" ]]
[[ "$(git -C "${PROJECT}" rev-parse HEAD)" == "ed6dd96d16d0b70a53bf9486fd78a270ab3486d6" ]]
[[ "$("${PYTHON}" -c 'import platform; print(platform.python_version())')" == "3.11.10" ]]
[[ "$("${PYTHON}" -c 'import numpy; print(numpy.__version__)')" == "2.3.5" ]]
[[ -d "${LEGACY_CAMPAIGN}/runs/vb0800/cic" ]]
[[ ! -e "${ROOT}" ]]
mkdir -p "${ROOT}/runs" "${ROOT}/runner_logs" "${ROOT}/numba_cache"
cp "${CONFIG}" "${ROOT}/frozen_config.json"
sha256sum \
  "${CONFIG}" \
  "${PACKAGE}/code/run_E2_q050_exact_arm.py" \
  "${PACKAGE}/code/analyze_E2_q050_exact.py" \
  "${PACKAGE}/code/audit_existing_cic_reuse.py" \
  "${PROJECT}/scripts/run_linearized_delta_f_mode.py" \
  "${PROJECT}/pic_kde_2d/conventions.py" \
  "${PROJECT}/pic_kde_2d/core/grid2d.py" \
  "${PROJECT}/pic_kde_2d/core/particles2d.py" \
  "${PROJECT}/pic_kde_2d/core/pic_loop_delta_f.py" \
  "${PROJECT}/pic_kde_2d/methods/cic.py" \
  "${PROJECT}/pic_kde_2d/methods/delta_f.py" \
  "${PROJECT}/pic_kde_2d/methods/spectral_W.py" \
  > "${ROOT}/input_sha256.txt"
git -C "${PROJECT}" status --short > "${ROOT}/git_status.txt"
git -C "${PROJECT}" rev-parse HEAD > "${ROOT}/git_revision.txt"

# Copy the immutable archived CIC evidence into the new result root, then audit
# it before starting the paired filtered arm. The original archive is untouched.
mkdir -p "${ROOT}/runs/cic"
cp -a "${LEGACY_CAMPAIGN}/runs/vb0800/cic/." "${ROOT}/runs/cic/"
"${PYTHON}" "${PACKAGE}/code/audit_existing_cic_reuse.py" \
  --config "${CONFIG}" \
  --config-sha256 "${CONFIG_SHA}" \
  --project "${PROJECT}" \
  --input "${ROOT}/runs/cic" \
  --legacy-campaign-root "${LEGACY_CAMPAIGN}" \
  --output "${ROOT}/runs/cic/reuse_audit.json"

SEEDS=(200 201 202 203 204 205 206 207 208 209)
pids=()
status=0
for seed in "${SEEDS[@]}"; do
  out=${ROOT}/runs/q050_x000/seed${seed}
  key=q050_x000_seed${seed}
  cache=${ROOT}/numba_cache/${key}
  log=${ROOT}/runner_logs/${key}.log
  mkdir -p "${cache}"
  env NUMBA_CACHE_DIR="${cache}" \
    "${PYTHON}" "${PACKAGE}/code/run_E2_q050_exact_arm.py" \
      --config "${CONFIG}" \
      --config-sha256 "${CONFIG_SHA}" \
      --project "${PROJECT}" \
      --output "${out}" \
      --arm q050_x000 \
      --seed "${seed}" \
      > "${log}" 2>&1 &
  pids+=("$!")
done
for pid in "${pids[@]}"; do
  wait "${pid}" || status=1
done
if [[ "${status}" -ne 0 ]]; then
  echo "one or more frozen histories failed; preserving partial outputs" >&2
  exit "${status}"
fi

find "${ROOT}/runs" -name mode_history.npz -type f | sort > "${ROOT}/history_files.txt"
[[ "$(wc -l < "${ROOT}/history_files.txt")" -eq 20 ]]
[[ "$(find "${ROOT}/runs/q050_x000" -name mode_history.npz -type f | wc -l)" -eq 10 ]]
"${PYTHON}" "${PACKAGE}/code/analyze_E2_q050_exact.py" \
  --config "${CONFIG}" \
  --config-sha256 "${CONFIG_SHA}" \
  --input "${ROOT}/runs" \
  --output "${ROOT}/analysis"
date -u +%Y-%m-%dT%H:%M:%SZ > "${ROOT}/COMPLETED"
echo "complete output=${ROOT}"
