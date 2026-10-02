"""
validation/run_phase2_validation.py
===================================

Meluxina-friendly Phase 2 validation runner for anisotropic matched Gaussian kernels.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

import numpy as np

PROJECT_PARENT = pathlib.Path(__file__).resolve().parents[2]
if str(PROJECT_PARENT) not in sys.path:
    sys.path.insert(0, str(PROJECT_PARENT))

from pic_kde_2d.benchmarks.landau_2d import fit_mode_gamma
from pic_kde_2d.benchmarks.theory import landau_gamma_theory
from pic_kde_2d.conventions import CHARGE_TOL, SELF_FORCE_TOL, landau_box_length, marker_grid_shape, thermal_velocity
from pic_kde_2d.core.grid2d import Grid2D
from pic_kde_2d.core.particles2d import init_landau_2d
from pic_kde_2d.core.pic_loop import run_simulation
from pic_kde_2d.methods.gaussian import deposit_gaussian_2d_aniso
from pic_kde_2d.methods.gaussian_wrappers import make_gaussian_methods
from pic_kde_2d.methods.sigma_selection import select_sigma_anisotropic_2d
from pic_kde_2d.validation.conservation_2d import check_charge_conservation, compute_energy_drift
from pic_kde_2d.validation.self_force_2d import test_self_force
from pic_kde_2d.validation.transfer_function_test import compare_mode_transfer_to_analytical, measure_transfer_function_deposit


def _random_charge_case(grid: Grid2D, seed: int = 42):
    rng = np.random.default_rng(seed)
    N_p = 1000
    x = rng.uniform(0.0, grid.Lx, N_p)
    y = rng.uniform(0.0, grid.Ly, N_p)
    w = rng.uniform(-1.0, -0.5, N_p)
    return x, y, w


def validate_phase2(outdir: str) -> dict:
    os.makedirs(outdir, exist_ok=True)

    charge_grid = Grid2D(Nx=64, Ny=64, Lx=10.0, Ly=10.0)
    x, y, w = _random_charge_case(charge_grid)
    charge_results = []
    for sigma_dx in [0.5, 1.0, 2.0, 3.0]:
        sigma = sigma_dx * charge_grid.dx
        rho = deposit_gaussian_2d_aniso(x, y, w, sigma, sigma, charge_grid.Nx, charge_grid.Ny, charge_grid.dx, charge_grid.dy)
        charge_results.append(
            {
                "sigma_dx": sigma_dx,
                **check_charge_conservation(rho, np.sum(w), charge_grid.dx, charge_grid.dy, tol=CHARGE_TOL),
            }
        )

    self_force_grid = Grid2D(Nx=64, Ny=64, Lx=10.0, Ly=10.0)
    self_force_iso = []
    for sigma_dx in [0.5, 1.0, 1.5, 2.0, 3.0]:
        sigma = sigma_dx * self_force_grid.dx
        deposit, gather = make_gaussian_methods(sigma, sigma)
        self_force_iso.append({"sigma_dx": sigma_dx, **test_self_force(deposit, gather, self_force_grid)})

    self_force_aniso = []
    for sx, sy in [(0.5, 1.0), (1.0, 2.0), (3.0, 1.0)]:
        sigma_x = sx * self_force_grid.dx
        sigma_y = sy * self_force_grid.dy
        deposit, gather = make_gaussian_methods(sigma_x, sigma_y)
        self_force_aniso.append({"sigma_x_dx": sx, "sigma_y_dy": sy, **test_self_force(deposit, gather, self_force_grid)})

    transfer_grid = Grid2D(Nx=32, Ny=32, Lx=10.0, Ly=10.0)
    sigma_transfer = 1.0 * transfer_grid.dx
    transfer_deposit, _ = make_gaussian_methods(sigma_transfer, sigma_transfer)
    uniform_transfer = measure_transfer_function_deposit(transfer_deposit, transfer_grid, Nx_marker=64, Ny_marker=64)
    transfer_modes = compare_mode_transfer_to_analytical(
        transfer_deposit,
        transfer_grid,
        sigma_x=sigma_transfer,
        sigma_y=sigma_transfer,
        modes=[(1, 0), (2, 0), (0, 1), (1, 1)],
    )

    kx_mode = 1
    Nx, Ny = 32, 8
    Lx = landau_box_length(kx_mode)
    Ly = 8.0
    grid = Grid2D(Nx=Nx, Ny=Ny, Lx=Lx, Ly=Ly)
    v_th = thermal_velocity()
    sigma_x_cells, _ = select_sigma_anisotropic_2d(Nx=Nx, Ny=Ny, mx_max=1, my_max=1, A_min=0.85)
    sigma_x = sigma_x_cells * grid.dx
    sigma_y = 3.0 * grid.dy
    deposit, gather = make_gaussian_methods(sigma_x, sigma_y)
    Nx_marker, Ny_marker = marker_grid_shape(grid.Nx, grid.Ny, 64)
    particles = init_landau_2d(
        Nx_marker=Nx_marker,
        Ny_marker=Ny_marker,
        Lx=Lx,
        Ly=Ly,
        v_thermal=v_th,
        alpha=0.02,
        kx_mode=kx_mode,
        ky_mode=0,
        rng=np.random.default_rng(123),
    )
    history = run_simulation(
        particles,
        grid,
        0.05,
        200,
        deposit,
        gather,
        diag_stride=2,
        tracked_modes=[(kx_mode, 0)],
        verbose=False,
    )
    energy = compute_energy_drift(history)
    amps = np.abs(np.asarray(history.mode_amplitudes_Ex[(kx_mode, 0)], dtype=np.complex128))
    times = np.asarray(history.times, dtype=np.float64)
    gamma_measured = fit_mode_gamma(times, amps, 2.0, 10.0, amp_floor=1.0e-8)
    gamma_theory = landau_gamma_theory(0.4, v_thermal=v_th)
    gamma_relative_error = abs(gamma_measured - gamma_theory) / abs(gamma_theory)

    result = {
        "charge_results": charge_results,
        "self_force_isotropic": self_force_iso,
        "self_force_anisotropic": self_force_aniso,
        "transfer_uniform": {
            "rms_nonzero_modes": uniform_transfer["rms_nonzero_modes"],
        },
        "transfer_modes": transfer_modes,
        "landau_gaussian": {
            "sigma_x_cells": sigma_x_cells,
            "sigma_y_cells": sigma_y / grid.dy,
            "gamma_measured": gamma_measured,
            "gamma_theory": gamma_theory,
            "gamma_relative_error": gamma_relative_error,
            "energy": energy,
        },
        "phase2_gates": {
            "charge_ok": all(row["passes"] for row in charge_results),
            "self_force_isotropic_ok": all(row["magnitude"] < SELF_FORCE_TOL for row in self_force_iso),
            "self_force_anisotropic_ok": all(row["magnitude"] < SELF_FORCE_TOL for row in self_force_aniso),
            "transfer_uniform_ok": bool(uniform_transfer["rms_nonzero_modes"] < 1.0e-3),
            "transfer_modes_ok": all(row["relative_error"] < 0.05 for row in transfer_modes),
            "landau_gamma_ok": bool(gamma_relative_error < 0.25),
        },
    }

    with open(os.path.join(outdir, "phase2_validation_summary.json"), "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phase 2 validation for anisotropic Gaussian PIC smoothing.")
    parser.add_argument("--outdir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = validate_phase2(args.outdir)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
