"""Momentum conservation for unequal matched per-particle kernels."""

import numpy as np

from pic_kde_2d.core.grid2d import Grid2D
from pic_kde_2d.validation.momentum_conservation import (
    force_from_shapes,
    pair_force_matrices,
    per_particle_cic_gaussian_shapes,
)


def make_case():
    grid = Grid2D(Nx=24, Ny=20, Lx=9.0, Ly=7.0)
    rng = np.random.default_rng(314159)
    n_particles = 7
    x = rng.uniform(0.0, grid.Lx, n_particles)
    y = rng.uniform(0.0, grid.Ly, n_particles)
    charges = rng.uniform(-1.2, 0.9, n_particles)
    sigma_x = rng.uniform(0.35, 2.1, n_particles) * grid.dx
    sigma_y = rng.uniform(0.4, 1.8, n_particles) * grid.dy
    shapes = per_particle_cic_gaussian_shapes(grid, x, y, sigma_x, sigma_y)
    return grid, charges, shapes


def test_unequal_matched_shapes_conserve_total_momentum():
    grid, charges, shapes = make_case()
    result = force_from_shapes(grid, charges, shapes)
    force_scale = float(np.sum(np.hypot(result["Fx"], result["Fy"])))
    assert result["total_force"] <= 2.0e-13 * max(force_scale, 1.0)


def test_unequal_matched_shapes_have_antisymmetric_pair_forces():
    grid, charges, shapes = make_case()
    Fx, Fy = pair_force_matrices(grid, charges, shapes)
    scale = max(float(np.max(np.abs(Fx))), float(np.max(np.abs(Fy))), 1.0)
    assert np.max(np.abs(Fx + Fx.T)) <= 2.0e-13 * scale
    assert np.max(np.abs(Fy + Fy.T)) <= 2.0e-13 * scale
    assert np.max(np.abs(np.diag(Fx))) <= 2.0e-13 * scale
    assert np.max(np.abs(np.diag(Fy))) <= 2.0e-13 * scale


def test_mismatched_gather_breaks_exact_momentum_conservation():
    grid, charges, shapes = make_case()
    gather_shapes = shapes.copy()
    gather_shapes[:, 0] = np.roll(gather_shapes[:, 0], shift=1)
    result = force_from_shapes(grid, charges, shapes, gather_shapes)
    assert result["total_force"] > 1.0e-8

