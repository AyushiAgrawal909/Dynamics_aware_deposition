#!/usr/bin/env python3
"""Dependency-light numerical audit of the frozen Landau refinement chain."""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pic_kde_2d.benchmarks.theory import solve_landau_dispersion
from pic_kde_2d.conventions import landau_box_length
from pic_kde_2d.core.grid2d import Grid2D
from pic_kde_2d.core.particles2d import init_landau_2d_eigenmode
from pic_kde_2d.methods.cic import deposit_cic_2d
from pic_kde_2d.methods.spectral_W import apply_W_to_grid, make_W_gaussian


def mode(field: np.ndarray, m: int, n: int = 0) -> complex:
    return complex(np.fft.rfft2(field)[m, n]) / float(field.shape[0] * field.shape[1])


def main() -> None:
    Nx, Ny = 64, 16
    Lx, Ly = landau_box_length(1, 0.4), 8.0
    grid = Grid2D(Nx=Nx, Ny=Ny, Lx=Lx, Ly=Ly)
    omega = solve_landau_dispersion(0.4)
    particles = init_landau_2d_eigenmode(
        Nx_marker=Nx * 8,
        Ny_marker=Ny * 2,
        Lx=Lx,
        Ly=Ly,
        v_thermal=0.0,
        alpha=1.0e-3,
        kx_mode=1,
        omega_r=omega.real,
        rng=np.random.default_rng(7),
    )
    rho = deposit_cic_2d(
        particles.x, particles.y, particles.w,
        grid.Nx, grid.Ny, grid.dx, grid.dy,
    )
    current = deposit_cic_2d(
        particles.x, particles.y, particles.w * particles.vx,
        grid.Nx, grid.Ny, grid.dx, grid.dy,
    )
    rho_mode = mode(rho, 1)
    current_mode = mode(current, 1)
    k = 2.0 * np.pi / Lx
    continuity_ratio = current_mode / rho_mode
    expected_continuity_ratio = omega / k

    Ex, _ = grid.poisson_solve(rho)
    poisson_ratio = mode(Ex, 1) / rho_mode
    expected_poisson_ratio = -1j / k

    sigma_dx = 3.0
    W = make_W_gaussian(grid, sigma_dx * grid.dx, sigma_dx * grid.dy)
    rho_filtered = apply_W_to_grid(rho, W, grid)
    Ex_filtered, Ey_filtered = grid.poisson_solve(rho_filtered)
    Ex_matched = apply_W_to_grid(Ex_filtered, W, grid)
    _ = apply_W_to_grid(Ey_filtered, W, grid)
    measured_W2 = abs(mode(Ex_matched, 1) / mode(Ex, 1))
    expected_W2 = float(W[1, 0] ** 2)

    checks = {
        "k_lambdaD": k,
        "theory_omega": [omega.real, omega.imag],
        "density_mode_real_positive": bool(rho_mode.real > 0.0 and abs(rho_mode.imag) < 1.0e-10),
        "continuity_relative_error": abs(continuity_ratio - expected_continuity_ratio) / abs(expected_continuity_ratio),
        "poisson_relative_error": abs(poisson_ratio - expected_poisson_ratio) / abs(expected_poisson_ratio),
        "matched_W2_absolute_error": abs(measured_W2 - expected_W2),
        "rho_mean": float(np.mean(rho)),
        "rho_mode": [rho_mode.real, rho_mode.imag],
        "continuity_ratio": [continuity_ratio.real, continuity_ratio.imag],
        "expected_continuity_ratio": [expected_continuity_ratio.real, expected_continuity_ratio.imag],
        "poisson_ratio": [poisson_ratio.real, poisson_ratio.imag],
        "expected_poisson_ratio": [expected_poisson_ratio.real, expected_poisson_ratio.imag],
        "measured_W2": measured_W2,
        "expected_W2": expected_W2,
        "initializer_scope": (
            "density/current moment-matched traveling perturbation; "
            "not the exact velocity-dependent Vlasov eigenfunction"
        ),
    }
    checks["passes"] = bool(
        abs(k - 0.4) < 1.0e-14
        and checks["density_mode_real_positive"]
        and checks["continuity_relative_error"] < 5.0e-3
        and checks["poisson_relative_error"] < 1.0e-12
        and checks["matched_W2_absolute_error"] < 1.0e-12
    )
    print(json.dumps(checks, indent=2, sort_keys=True))
    if not checks["passes"]:
        raise SystemExit("LANDAU PHYSICS AUDIT FAILED")
    print("LANDAU PHYSICS AUDIT PASS")


if __name__ == "__main__":
    main()
