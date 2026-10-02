"""Regression tests for exact CIC-plus-spectral-W leave-one-out CV."""

from __future__ import annotations

import numpy as np

from pic_kde_2d.core.grid2d import Grid2D
from pic_kde_2d.methods.cic import deposit_cic_2d
from pic_kde_2d.methods.lscv import (
    cic_filter_self_matrix,
    cic_weight_second_moment,
    lscv_score,
    mean_cic_filter_self,
)
from pic_kde_2d.methods.spectral_W import apply_W_to_grid, make_W_gaussian


def _one_particle_density(x: float, y: float, grid: Grid2D) -> np.ndarray:
    return deposit_cic_2d(
        np.asarray([x]),
        np.asarray([y]),
        np.asarray([1.0]),
        grid.Nx,
        grid.Ny,
        grid.dx,
        grid.dy,
    )


def test_mean_self_matches_particle_by_particle_evaluation() -> None:
    grid = Grid2D(Nx=12, Ny=8, Lx=3.0, Ly=2.0)
    rng = np.random.default_rng(1729)
    x = rng.uniform(0.0, grid.Lx, size=23)
    y = rng.uniform(0.0, grid.Ly, size=23)
    W = make_W_gaussian(grid, 0.7 * grid.dx, 1.1 * grid.dy)

    exact = mean_cic_filter_self(
        cic_weight_second_moment(x, y, grid),
        cic_filter_self_matrix(grid, W),
    )
    brute_force = []
    for xp, yp in zip(x, y):
        rho = _one_particle_density(float(xp), float(yp), grid)
        smoothed = apply_W_to_grid(rho, W, grid)
        brute_force.append(float(np.sum(rho * smoothed) * grid.dx * grid.dy))

    np.testing.assert_allclose(exact, np.mean(brute_force), rtol=0.0, atol=2.0e-15)


def test_lscv_matches_explicit_leave_one_out_densities() -> None:
    grid = Grid2D(Nx=10, Ny=6, Lx=5.0, Ly=3.0)
    rng = np.random.default_rng(991)
    x = rng.uniform(0.0, grid.Lx, size=11)
    y = rng.uniform(0.0, grid.Ly, size=11)
    n = x.size
    W = make_W_gaussian(grid, 0.8 * grid.dx, 0.6 * grid.dy)
    rho_probability = deposit_cic_2d(
        x,
        y,
        np.full(n, 1.0 / n),
        grid.Nx,
        grid.Ny,
        grid.dx,
        grid.dy,
    )
    score = lscv_score(
        rho_probability,
        W,
        mean_cic_filter_self(
            cic_weight_second_moment(x, y, grid),
            cic_filter_self_matrix(grid, W),
        ),
        n,
        grid,
    )

    smoothed = apply_W_to_grid(rho_probability, W, grid)
    integrated_square = float(np.sum(smoothed**2) * grid.dx * grid.dy)
    evaluations = []
    for particle in range(n):
        mask = np.arange(n) != particle
        rho_loo = deposit_cic_2d(
            x[mask],
            y[mask],
            np.full(n - 1, 1.0 / (n - 1)),
            grid.Nx,
            grid.Ny,
            grid.dx,
            grid.dy,
        )
        smooth_loo = apply_W_to_grid(rho_loo, W, grid)
        rho_one = _one_particle_density(float(x[particle]), float(y[particle]), grid)
        evaluations.append(
            float(np.sum(smooth_loo * rho_one) * grid.dx * grid.dy)
        )
    explicit = integrated_square - 2.0 * float(np.mean(evaluations))

    np.testing.assert_allclose(score, explicit, rtol=0.0, atol=3.0e-15)
