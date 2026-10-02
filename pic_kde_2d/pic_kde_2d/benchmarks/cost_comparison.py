"""
benchmarks/cost_comparison.py
=============================

Timing helpers for direct Gaussian deposition versus CIC + spectral W.
"""

from __future__ import annotations

import time

import numpy as np

from pic_kde_2d.conventions import marker_grid_shape, thermal_velocity
from pic_kde_2d.core.grid2d import Grid2D
from pic_kde_2d.core.particles2d import init_landau_2d
from pic_kde_2d.methods.gaussian_wrappers import make_gaussian_methods
from pic_kde_2d.methods.spectral_W_wrappers import make_spectral_W_gaussian_methods


def benchmark_deposit_cost(
    grid: Grid2D,
    ppc: int,
    sigma_dx_list: list[float],
    n_warmup: int = 2,
    n_trials: int = 3,
) -> dict:
    """Benchmark direct Gaussian deposit versus CIC + spectral Gaussian W."""
    Nx_marker, Ny_marker = marker_grid_shape(grid.Nx, grid.Ny, ppc)
    particles = init_landau_2d(
        Nx_marker=Nx_marker,
        Ny_marker=Ny_marker,
        Lx=grid.Lx,
        Ly=grid.Ly,
        v_thermal=thermal_velocity(),
        alpha=0.02,
        kx_mode=1,
        ky_mode=0,
        rng=np.random.default_rng(42),
    )

    results = {"sigma_dx": [], "t_phase2": [], "t_phase3": []}

    for sigma_dx in sigma_dx_list:
        sigma_x = sigma_dx * grid.dx
        sigma_y = sigma_dx * grid.dy
        deposit2, _ = make_gaussian_methods(sigma_x, sigma_y)
        deposit3, _ = make_spectral_W_gaussian_methods(grid, sigma_x, sigma_y)

        for _ in range(n_warmup):
            deposit2(particles, grid)
            deposit3(particles, grid)

        t0 = time.perf_counter()
        for _ in range(n_trials):
            deposit2(particles, grid)
        t_phase2 = (time.perf_counter() - t0) / float(n_trials)

        t0 = time.perf_counter()
        for _ in range(n_trials):
            deposit3(particles, grid)
        t_phase3 = (time.perf_counter() - t0) / float(n_trials)

        results["sigma_dx"].append(float(sigma_dx))
        results["t_phase2"].append(float(t_phase2))
        results["t_phase3"].append(float(t_phase3))

    return results
