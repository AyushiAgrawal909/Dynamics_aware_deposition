"""Momentum checks for matched arbitrary per-particle grid shapes."""

from __future__ import annotations

import numpy as np

from pic_kde_2d.core.grid2d import Grid2D
from pic_kde_2d.methods.cic import deposit_cic_2d
from pic_kde_2d.methods.spectral_W import apply_W_to_grid, make_W_gaussian


def per_particle_cic_gaussian_shapes(
    grid: Grid2D,
    x: np.ndarray,
    y: np.ndarray,
    sigma_x: np.ndarray,
    sigma_y: np.ndarray,
) -> np.ndarray:
    """Return columns containing normalized ``G_Hi * W_CIC`` grid weights."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    sigma_x = np.asarray(sigma_x, dtype=np.float64)
    sigma_y = np.asarray(sigma_y, dtype=np.float64)
    if not (x.shape == y.shape == sigma_x.shape == sigma_y.shape):
        raise ValueError("particle position and sigma arrays must have equal shapes")

    cell_volume = grid.dx * grid.dy
    shapes = np.empty((grid.Nx * grid.Ny, len(x)), dtype=np.float64)
    for particle in range(len(x)):
        rho_cic = deposit_cic_2d(
            np.asarray([x[particle]]),
            np.asarray([y[particle]]),
            np.asarray([1.0]),
            grid.Nx,
            grid.Ny,
            grid.dx,
            grid.dy,
        )
        W = make_W_gaussian(grid, sigma_x[particle], sigma_y[particle])
        shape = apply_W_to_grid(rho_cic, W, grid) * cell_volume
        shapes[:, particle] = shape.ravel()
    return shapes


def force_from_shapes(
    grid: Grid2D,
    charges: np.ndarray,
    deposit_shapes: np.ndarray,
    gather_shapes: np.ndarray | None = None,
) -> dict:
    """Deposit, solve, gather, and return particle and total forces."""
    charges = np.asarray(charges, dtype=np.float64)
    deposit_shapes = np.asarray(deposit_shapes, dtype=np.float64)
    if gather_shapes is None:
        gather_shapes = deposit_shapes
    gather_shapes = np.asarray(gather_shapes, dtype=np.float64)
    expected = (grid.Nx * grid.Ny, len(charges))
    if deposit_shapes.shape != expected or gather_shapes.shape != expected:
        raise ValueError(f"shape matrices must have shape {expected}")

    cell_volume = grid.dx * grid.dy
    rho = (deposit_shapes @ charges).reshape(grid.Nx, grid.Ny) / cell_volume
    Ex, Ey = grid.poisson_solve(rho)
    Ex_p = gather_shapes.T @ Ex.ravel()
    Ey_p = gather_shapes.T @ Ey.ravel()
    Fx = charges * Ex_p
    Fy = charges * Ey_p
    return {
        "Fx": Fx,
        "Fy": Fy,
        "total_Fx": float(np.sum(Fx)),
        "total_Fy": float(np.sum(Fy)),
        "total_force": float(np.hypot(np.sum(Fx), np.sum(Fy))),
    }


def pair_force_matrices(
    grid: Grid2D,
    charges: np.ndarray,
    shapes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return matrices with entry ``[i,j]`` equal to force on i from j."""
    charges = np.asarray(charges, dtype=np.float64)
    n_particles = len(charges)
    cell_volume = grid.dx * grid.dy
    Fx = np.zeros((n_particles, n_particles), dtype=np.float64)
    Fy = np.zeros((n_particles, n_particles), dtype=np.float64)
    for source in range(n_particles):
        rho = (charges[source] * shapes[:, source]).reshape(grid.Nx, grid.Ny) / cell_volume
        Ex, Ey = grid.poisson_solve(rho)
        Fx[:, source] = charges * (shapes.T @ Ex.ravel())
        Fy[:, source] = charges * (shapes.T @ Ey.ravel())
    return Fx, Fy

