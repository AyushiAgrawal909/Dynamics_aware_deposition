"""
validation/run_phase1_validation.py
===================================

Meluxina-friendly Phase 1 validation runner.

This script does not require pytest or SciPy. It executes the core acceptance
checks directly and writes a JSON summary plus a compact CSV time history.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import pathlib
import sys

import numpy as np

PROJECT_PARENT = pathlib.Path(__file__).resolve().parents[2]
if str(PROJECT_PARENT) not in sys.path:
    sys.path.insert(0, str(PROJECT_PARENT))

from pic_kde_2d.benchmarks.landau_2d import fit_mode_gamma, run_landau_baseline
from pic_kde_2d.benchmarks.theory import landau_gamma_theory
from pic_kde_2d.conventions import CHARGE_TOL, SELF_FORCE_TOL
from pic_kde_2d.core.grid2d import Grid2D
from pic_kde_2d.methods.cic import deposit_cic_2d
from pic_kde_2d.methods.cic_wrappers import cic_deposit_method, cic_gather_method
from pic_kde_2d.validation.conservation_2d import check_charge_conservation, compute_energy_drift
from pic_kde_2d.validation.self_force_2d import test_self_force


def write_history_csv(history, path: str) -> None:
    arrays = history.to_arrays()
    keys = list(arrays.keys())
    n_rows = len(arrays["times"])
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(keys)
        for i in range(n_rows):
            row = []
            for key in keys:
                value = arrays[key][i]
                row.append(value)
            writer.writerow(row)


def validate_phase1(outdir: str, Nx: int, Ny: int, ppc: int, dt: float, n_steps: int, diag_stride: int) -> dict:
    os.makedirs(outdir, exist_ok=True)

    g = Grid2D(Nx=64, Ny=64, Lx=10.0, Ly=10.0)
    rng = np.random.default_rng(42)
    x = rng.uniform(0.0, g.Lx, 1000)
    y = rng.uniform(0.0, g.Ly, 1000)
    w = rng.uniform(-1.0, -0.5, 1000)
    rho = deposit_cic_2d(x, y, w, g.Nx, g.Ny, g.dx, g.dy)
    charge_test = check_charge_conservation(rho, np.sum(w), g.dx, g.dy, tol=CHARGE_TOL)

    self_force_grid = Grid2D(Nx=64, Ny=64, Lx=10.0, Ly=10.0)
    self_force_test = test_self_force(cic_deposit_method, cic_gather_method, self_force_grid)

    history, summary = run_landau_baseline(
        Nx=Nx,
        Ny=Ny,
        ppc=ppc,
        dt=dt,
        n_steps=n_steps,
        diag_stride=diag_stride,
    )
    energy = compute_energy_drift(history)
    history_arrays = history.to_arrays()
    amplitudes = np.abs(history_arrays["Ex_mode_1_0"])
    gamma_measured = fit_mode_gamma(history_arrays["times"], amplitudes, 2.0, min(12.0, history_arrays["times"][-1]), amp_floor=1.0e-8)
    gamma_theory = landau_gamma_theory(0.4)
    gamma_relative_error = abs(gamma_measured - gamma_theory) / abs(gamma_theory)

    result = {
        "charge_test": charge_test,
        "self_force_test": self_force_test,
        "landau_summary": summary.__dict__,
        "energy_drift": energy,
        "phase1_gates": {
            "charge_ok": bool(charge_test["passes"]),
            "self_force_ok": bool(self_force_test["magnitude"] < SELF_FORCE_TOL),
            "landau_gamma_ok": bool(gamma_relative_error < 0.25),
            "energy_reasonable": bool(energy["max_relative_drift"] < 0.05),
        },
    }

    with open(os.path.join(outdir, "phase1_validation_summary.json"), "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
    write_history_csv(history, os.path.join(outdir, "landau_history.csv"))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phase 1 validation for the 2D CIC baseline.")
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--Nx", type=int, default=32)
    parser.add_argument("--Ny", type=int, default=8)
    parser.add_argument("--ppc", type=int, default=64)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--n-steps", dest="n_steps", type=int, default=200)
    parser.add_argument("--diag-stride", dest="diag_stride", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = validate_phase1(
        outdir=args.outdir,
        Nx=args.Nx,
        Ny=args.Ny,
        ppc=args.ppc,
        dt=args.dt,
        n_steps=args.n_steps,
        diag_stride=args.diag_stride,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
