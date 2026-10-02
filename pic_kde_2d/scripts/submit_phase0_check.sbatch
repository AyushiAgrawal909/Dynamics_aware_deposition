#!/bin/bash
#SBATCH --job-name=pic-kde-2d-p0
#SBATCH --account=p201103
#SBATCH --partition=cpu
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:15:00
#SBATCH --output=/mnt/tier2/users/u103576/logs/pic_kde_2d_phase0_%j.out
#SBATCH --error=/mnt/tier2/users/u103576/logs/pic_kde_2d_phase0_%j.err

set -euo pipefail

PROJECT_ROOT="/home/users/u103576/hexapic-main/pic_kde_2d"
PYTHON_ROOT="/apps/USE/easybuild/release/2024.1/software/Python/3.11.10-GCCcore-13.3.0"
PYTHON_BIN="${PYTHON_ROOT}/bin/python3"
export LD_LIBRARY_PATH="${PYTHON_ROOT}/lib:${PYTHON_ROOT}/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

cd "${PROJECT_ROOT}"

"${PYTHON_BIN}" -m pip install --user -e . --no-deps
"${PYTHON_BIN}" -c "from pic_kde_2d.core import grid2d, particles2d, pic_loop"

if "${PYTHON_BIN}" -m pytest --version >/dev/null 2>&1; then
  "${PYTHON_BIN}" -m pytest tests/test_phase0_setup.py -q
else
  "${PYTHON_BIN}" - <<'PY'
import pathlib
import numpy as np

import pic_kde_2d.core.grid2d
import pic_kde_2d.core.particles2d
import pic_kde_2d.core.pic_loop
import pic_kde_2d.methods.cic
import pic_kde_2d.methods.gaussian
import pic_kde_2d.methods.spectral_W
import pic_kde_2d.methods.adaptive_sigma
import pic_kde_2d.methods.delta_f
import pic_kde_2d.benchmarks.landau_2d
import pic_kde_2d.benchmarks.two_stream_2d
import pic_kde_2d.benchmarks.theory
import pic_kde_2d.validation.self_force
import pic_kde_2d.validation.conservation
import pic_kde_2d.validation.spectral_rule

Nx, Ny = 16, 16
a = np.random.rand(Nx, Ny)
a_hat = np.fft.rfft2(a)
assert a_hat.shape == (Nx, Ny // 2 + 1)
a_recovered = np.fft.irfft2(a_hat, s=(Nx, Ny))
np.testing.assert_allclose(a, a_recovered, atol=1e-12)

readme = pathlib.Path("README.md").read_text(encoding="utf-8")
assert "lambda_D = 1" in readme or "Debye lengths" in readme
assert "omega_p" in readme
assert "q_e = -1" in readme or "electron" in readme.lower()
print("Phase 0 fallback checks passed")
PY
fi
