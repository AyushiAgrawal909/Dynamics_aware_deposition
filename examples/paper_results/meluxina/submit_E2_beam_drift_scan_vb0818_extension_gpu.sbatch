#!/bin/bash -l
# Post-timeout extension of the frozen E2 (12,1) scan. This runs exactly the 30
# v_b=0.818 histories that job 5201231 did not begin. The original four-hour
# campaign remains incomplete; these outputs live in a separate extension tree.
#SBATCH --job-name=p1_E2_vb818_ext
#SBATCH --account=p201103
#SBATCH --partition=gpu
#SBATCH --qos=long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --gpus-per-node=1
#SBATCH --mem=220G
#SBATCH --time=08:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ayushiphysics909@gmail.com
#SBATCH --output=/mnt/tier2/users/u103576/logs/p1_E2_vb818_ext_%j.out
#SBATCH --error=/mnt/tier2/users/u103576/logs/p1_E2_vb818_ext_%j.err

set -euo pipefail

PROJECT=/home/users/u103576/hexapic-main/pic_kde_2d
REPRO=${PROJECT}/reproducibility
PYROOT=/apps/USE/easybuild/release/2024.1/software/Python/3.11.10-GCCcore-13.3.0
PYTHON=${PYROOT}/bin/python3
CONFIG=${REPRO}/e2_beam_drift_scan_v1.json
AMENDMENT=${REPRO}/e2_beam_drift_scan_extension_budget_20260911.json
CONFIG_SHA=e664460d392a216b953bc87d3859c3bce9e1cce2f1652ab9417c1a58b3baf446
AMENDMENT_SHA=fc9ae62fcdd7321c546718852582a4759c977e55d3791673cbdd8274f4b15700
ROOT=/mnt/tier2/users/u103576/paper1_validation/e2_beam_drift_scan_extension_${SLURM_JOB_ID}

export LD_LIBRARY_PATH="${PYROOT}/lib:${PYROOT}/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${PROJECT}:${PROJECT}/scripts:${PYTHONPATH:-}"
export NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

[[ "$(sha256sum "${CONFIG}" | awk '{print $1}')" == "${CONFIG_SHA}" ]]
[[ "$(sha256sum "${AMENDMENT}" | awk '{print $1}')" == "${AMENDMENT_SHA}" ]]
[[ "$(sha256sum "${REPRO}/run_E2_beam_drift_seed.py" | awk '{print $1}')" == "9935058260947f6af08cb801a9200508d7e040da6c79864214c5e49452a75279" ]]
[[ "$(sha256sum "${PROJECT}/scripts/run_linearized_delta_f_mode.py" | awk '{print $1}')" == "c0e849a96c4822d07263cb6de137296766f1ef531d182a10c9a3d39236143599" ]]
[[ "$(sha256sum "${PROJECT}/pic_kde_2d/core/particles2d.py" | awk '{print $1}')" == "7bc3e88101d9071048c85eff204694c71e96f2a55f32b68082ff055f171bdb4e" ]]
[[ "$(sha256sum "${PROJECT}/pic_kde_2d/core/pic_loop_delta_f.py" | awk '{print $1}')" == "cc2231527ba0e725d4929bd0b12c3ffb04b52cb6c69d6fc20c925e16d831c814" ]]
[[ "$(sha256sum "${PROJECT}/pic_kde_2d/methods/delta_f.py" | awk '{print $1}')" == "e5c75624ae09cb94381fa3c519a82828528ca511ec3e73f9cf6dce6799f9e074" ]]
[[ "$(sha256sum "${PROJECT}/pic_kde_2d/methods/spectral_W.py" | awk '{print $1}')" == "fc9d4091d53873be2519bac4c1a6edd7bcbd047913de56ee4ec7777b03053ca2" ]]
[[ "$(sha256sum "${PROJECT}/pic_kde_2d/core/grid2d.py" | awk '{print $1}')" == "f3b3c04fb7bbb06ee67cf299ca2e2b0c7d4ae58f18660d2e083fae1381826299" ]]
[[ ! -e "${ROOT}" ]]
mkdir -p "${ROOT}/runs" "${ROOT}/runner_logs" "${ROOT}/numba_cache"
cp "${CONFIG}" "${ROOT}/frozen_campaign.json"
cp "${AMENDMENT}" "${ROOT}/extension_budget_amendment.json"
sha256sum \
  "${CONFIG}" \
  "${AMENDMENT}" \
  "${REPRO}/run_E2_beam_drift_seed.py" \
  "${PROJECT}/scripts/run_linearized_delta_f_mode.py" \
  "${PROJECT}/pic_kde_2d/core/particles2d.py" \
  "${PROJECT}/pic_kde_2d/core/pic_loop_delta_f.py" \
  "${PROJECT}/pic_kde_2d/methods/delta_f.py" \
  "${PROJECT}/pic_kde_2d/methods/spectral_W.py" \
  "${PROJECT}/pic_kde_2d/core/grid2d.py" \
  > "${ROOT}/input_sha256.txt"
git -C "${PROJECT}" rev-parse HEAD > "${ROOT}/git_revision.txt"

ARMS=(cic fixed_gaussian constrained_gaussian)
SEEDS=(200 201 202 203 204 205 206 207 208 209)
pids=()
status=0
for arm in "${ARMS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    out=${ROOT}/runs/vb0818/${arm}/seed${seed}
    work_key=vb0818_${arm}_seed${seed}
    cache=${ROOT}/numba_cache/${work_key}
    log=${ROOT}/runner_logs/${work_key}.log
    mkdir -p "${cache}"
    env NUMBA_CACHE_DIR="${cache}" \
      "${PYTHON}" "${REPRO}/run_E2_beam_drift_seed.py" \
        --config "${CONFIG}" --config-sha256 "${CONFIG_SHA}" \
        --output "${out}" --drift-index 4 \
        --arm "${arm}" --seed "${seed}" \
        >"${log}" 2>&1 &
    pids+=("$!")
  done
done
[[ "${#pids[@]}" -eq 30 ]]
for pid in "${pids[@]}"; do
  wait "${pid}" || status=1
done
if [[ "${status}" -ne 0 ]]; then
  echo "one or more extension histories failed" >&2
  exit "${status}"
fi

find "${ROOT}/runs" -name mode_history.npz -type f | sort > "${ROOT}/history_files.txt"
[[ "$(wc -l < "${ROOT}/history_files.txt")" -eq 30 ]]
date -u +%Y-%m-%dT%H:%M:%SZ > "${ROOT}/COMPLETED"
echo "extension complete output=${ROOT}"
