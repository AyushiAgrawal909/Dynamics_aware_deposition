"""
validation/run_phase3_validation.py
===================================

Meluxina-friendly Phase 3 validation runner for spectral-W filtering.
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

from pic_kde_2d.benchmarks.cost_comparison import benchmark_deposit_cost
from pic_kde_2d.benchmarks.landau_2d import fit_mode_gamma
from pic_kde_2d.benchmarks.theory import landau_gamma_theory
from pic_kde_2d.conventions import CHARGE_TOL, SELF_FORCE_TOL, landau_box_length, marker_grid_shape, thermal_velocity
from pic_kde_2d.core.grid2d import Grid2D
from pic_kde_2d.core.particles2d import Particles2D, init_landau_2d
from pic_kde_2d.core.pic_loop import run_simulation
from pic_kde_2d.methods.spectral_W import make_W_gaussian
from pic_kde_2d.methods.spectral_W_wrappers import make_spectral_W_gaussian_methods
from pic_kde_2d.validation.conservation_2d import check_charge_conservation, compute_energy_drift
from pic_kde_2d.validation.self_force_2d import test_self_force
from pic_kde_2d.validation.spectral_W_validation import compare_phase2_vs_phase3_density


def _random_particles(grid: Grid2D, seed: int = 42, n_particles: int = 512) -> Particles2D:
    rng = np.random.default_rng(seed)
    return Particles2D(
        x=rng.uniform(0.0, grid.Lx, n_particles),
        y=rng.uniform(0.0, grid.Ly, n_particles),
        vx=np.zeros(n_particles, dtype=np.float64),
        vy=np.zeros(n_particles, dtype=np.float64),
        w=rng.uniform(-1.0, -0.5, n_particles),
        Lx=grid.Lx,
        Ly=grid.Ly,
    )


def validate_phase3(outdir: str) -> dict:
    os.makedirs(outdir, exist_ok=True)

    basic_grid = Grid2D(Nx=32, Ny=32, Lx=10.0, Ly=10.0)
    sigma_basic = basic_grid.dx
    W = make_W_gaussian(basic_grid, sigma_basic, sigma_basic)
    basic_checks = {
        "W00": float(W[0, 0]),
        "high_k_sample": float(W[basic_grid.Nx // 2, 0]),
    }

    charge_grid = Grid2D(Nx=64, Ny=64, Lx=10.0, Ly=10.0)
    charge_particles = _random_particles(charge_grid)
    sigma_charge = 2.0 * charge_grid.dx
    deposit_charge, _ = make_spectral_W_gaussian_methods(charge_grid, sigma_charge, sigma_charge)
    rho_charge = deposit_charge(charge_particles, charge_grid)
    charge_result = check_charge_conservation(rho_charge, np.sum(charge_particles.w), charge_grid.dx, charge_grid.dy, tol=CHARGE_TOL)

    self_force_grid = Grid2D(Nx=64, Ny=64, Lx=10.0, Ly=10.0)
    self_force_iso = []
    for sigma_dx in [0.5, 1.0, 2.0, 3.0, 5.0]:
        sigma = sigma_dx * self_force_grid.dx
        deposit, gather = make_spectral_W_gaussian_methods(self_force_grid, sigma, sigma)
        self_force_iso.append({"sigma_dx": sigma_dx, **test_self_force(deposit, gather, self_force_grid)})

    self_force_aniso = []
    for sx_dx, sy_dx in [(0.5, 2.0), (3.0, 0.5), (1.0, 1.0)]:
        deposit, gather = make_spectral_W_gaussian_methods(
            self_force_grid,
            sx_dx * self_force_grid.dx,
            sy_dx * self_force_grid.dy,
        )
        self_force_aniso.append(
            {
                "sigma_x_dx": sx_dx,
                "sigma_y_dy": sy_dx,
                **test_self_force(deposit, gather, self_force_grid),
            }
        )

    compare_grid = Grid2D(Nx=48, Ny=48, Lx=10.0, Ly=10.0)
    compare_particles = init_landau_2d(
        Nx_marker=24,
        Ny_marker=24,
        Lx=compare_grid.Lx,
        Ly=compare_grid.Ly,
        v_thermal=thermal_velocity(),
        alpha=0.02,
        kx_mode=1,
        ky_mode=0,
        rng=np.random.default_rng(123),
    )
    phase23_compare = []
    for sigma_dx in [0.5, 1.0, 2.0]:
        sigma_x = sigma_dx * compare_grid.dx
        sigma_y = sigma_dx * compare_grid.dy
        row = compare_phase2_vs_phase3_density(compare_grid, compare_particles, sigma_x, sigma_y)
        phase23_compare.append(
            {
                "sigma_dx": sigma_dx,
                "relative_rms": row["relative_rms"],
                "rms_diff": row["rms_diff"],
                "rms_signal": row["rms_signal"],
            }
        )

    kx_mode = 1
    Nx, Ny = 64, 16
    Lx = landau_box_length(kx_mode)
    Ly = 8.0
    landau_grid = Grid2D(Nx=Nx, Ny=Ny, Lx=Lx, Ly=Ly)
    sigma_landau = 3.0 * landau_grid.dx
    deposit_landau, gather_landau = make_spectral_W_gaussian_methods(landau_grid, sigma_landau, sigma_landau)
    Nx_marker, Ny_marker = marker_grid_shape(landau_grid.Nx, landau_grid.Ny, 16)
    landau_particles = init_landau_2d(
        Nx_marker=Nx_marker,
        Ny_marker=Ny_marker,
        Lx=Lx,
        Ly=Ly,
        v_thermal=thermal_velocity(),
        alpha=0.02,
        kx_mode=kx_mode,
        ky_mode=0,
        rng=np.random.default_rng(123),
    )
    history = run_simulation(
        landau_particles,
        landau_grid,
        0.05,
        200,
        deposit_landau,
        gather_landau,
        diag_stride=2,
        tracked_modes=[(kx_mode, 0)],
        verbose=False,
    )
    energy = compute_energy_drift(history)
    amps = np.abs(np.asarray(history.mode_amplitudes_Ex[(kx_mode, 0)], dtype=np.complex128))
    times = np.asarray(history.times, dtype=np.float64)
    gamma_measured = fit_mode_gamma(times, amps, 2.0, 10.0, amp_floor=1.0e-8)
    gamma_theory = landau_gamma_theory(0.4, v_thermal=thermal_velocity())
    gamma_relative_error = abs(gamma_measured - gamma_theory) / abs(gamma_theory)

    cost_grid = Grid2D(Nx=96, Ny=96, Lx=15.0, Ly=15.0)
    cost_results = benchmark_deposit_cost(cost_grid, ppc=8, sigma_dx_list=[2.0, 3.0], n_warmup=1, n_trials=2)
    cost_ratio_sigma3 = cost_results["t_phase2"][1] / max(cost_results["t_phase3"][1], 1.0e-30)

    result = {
        "basic_checks": basic_checks,
        "charge_result": charge_result,
        "self_force_isotropic": self_force_iso,
        "self_force_anisotropic": self_force_aniso,
        "phase2_vs_phase3": phase23_compare,
        "landau_spectral_W": {
            "Nx": Nx,
            "Ny": Ny,
            "sigma_dx": 3.0,
            "gamma_measured": gamma_measured,
            "gamma_theory": gamma_theory,
            "gamma_relative_error": gamma_relative_error,
            "energy": energy,
        },
        "cost_comparison": {
            "sigma_dx": cost_results["sigma_dx"],
            "t_phase2": cost_results["t_phase2"],
            "t_phase3": cost_results["t_phase3"],
            "ratio_sigma_3dx": cost_ratio_sigma3,
        },
        "phase3_gates": {
            "W00_ok": abs(basic_checks["W00"] - 1.0) < 1.0e-12,
            "charge_ok": charge_result["passes"],
            "self_force_isotropic_ok": all(row["magnitude"] < SELF_FORCE_TOL for row in self_force_iso),
            "self_force_anisotropic_ok": all(row["magnitude"] < SELF_FORCE_TOL for row in self_force_aniso),
            "phase2_match_ok": all(
                (row["sigma_dx"] < 1.0) or (row["relative_rms"] < 0.05) for row in phase23_compare
            ),
            "landau_gamma_ok": gamma_relative_error < 0.30,
            "cost_ok": cost_ratio_sigma3 > 2.0,
        },
    }

    with open(os.path.join(outdir, "phase3_validation_summary.json"), "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phase 3 validation for spectral-W PIC smoothing.")
    parser.add_argument("--outdir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = validate_phase3(args.outdir)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
