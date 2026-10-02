"""
methods/spectral_W_wrappers.py
==============================

Method-pluggable wrappers for CIC deposition with post-deposition spectral
filtering. The same filter W is applied to rho and to the grid electric field
before CIC gather.
"""

from __future__ import annotations

import numpy as np

from pic_kde_2d.methods.cic import deposit_cic_2d, gather_cic_2d
from pic_kde_2d.methods.spectral_W import (
    apply_W_to_grid,
    make_W_binomial,
    make_W_box,
    make_W_compensated_binomial,
    make_W_gaussian,
)


def make_spectral_W_methods(W: np.ndarray):
    """Return (deposit_fn, gather_fn) with matched spectral smoothing baked in."""
    W = np.asarray(W, dtype=np.float64)

    def deposit(particles, grid):
        rho_cic = deposit_cic_2d(
            particles.x,
            particles.y,
            particles.w,
            grid.Nx,
            grid.Ny,
            grid.dx,
            grid.dy,
        )
        return apply_W_to_grid(rho_cic, W, grid)

    def gather(Ex, Ey, particles, grid):
        Ex_smooth = apply_W_to_grid(Ex, W, grid)
        Ey_smooth = apply_W_to_grid(Ey, W, grid)
        return gather_cic_2d(
            Ex_smooth,
            Ey_smooth,
            particles.x,
            particles.y,
            grid.Nx,
            grid.Ny,
            grid.dx,
            grid.dy,
        )

    return deposit, gather


def make_spectral_W_gaussian_methods(grid, sigma_x: float, sigma_y: float):
    """Convenience wrapper for a Gaussian spectral filter."""
    W = make_W_gaussian(grid, sigma_x, sigma_y)
    return make_spectral_W_methods(W)


def make_spectral_W_binomial_methods(
    grid,
    passes_x: int,
    passes_y: int | None = None,
):
    """Convenience wrapper for matched repeated binomial filtering."""
    W = make_W_binomial(grid, passes_x, passes_y)
    return make_spectral_W_methods(W)


def make_spectral_W_compensated_binomial_methods(
    grid,
    passes_x: int,
    passes_y: int | None = None,
    compensation_order: int = 1,
):
    """Convenience wrapper for matched compensated-binomial filtering."""
    W = make_W_compensated_binomial(
        grid,
        passes_x,
        passes_y,
        compensation_order=compensation_order,
    )
    return make_spectral_W_methods(W)


def make_binned_adaptive_spectral_W_methods(
    grid,
    sigma_x: np.ndarray,
    sigma_y: np.ndarray,
    particle_weights: np.ndarray,
    n_bins: int = 32,
):
    """Return a matched binned approximation to per-particle Gaussian shapes.

    Widths are frozen particle properties for the run. Each bin deposits with
    ``W_bin`` and gathers through the transpose chain ``W_bin -> CIC``. This is
    the analytic H1 rung; live controller refresh and grad-h are deliberately
    outside this fixed-pilot locality gate.
    """
    sx = np.asarray(sigma_x, dtype=np.float64)
    sy = np.asarray(sigma_y, dtype=np.float64)
    weights = np.asarray(particle_weights, dtype=np.float64)
    if sx.ndim != 1 or sy.shape != sx.shape or weights.shape != sx.shape:
        raise ValueError("width and particle-weight arrays must share a 1D shape")
    if np.any(~np.isfinite(sx)) or np.any(~np.isfinite(sy)):
        raise ValueError("adaptive widths must be finite")
    if np.any(sx < 0.0) or np.any(sy < 0.0):
        raise ValueError("adaptive widths must be nonnegative")
    if n_bins <= 0:
        raise ValueError("n_bins must be positive")

    scale = np.hypot(sx / grid.dx, sy / grid.dy)
    if np.ptp(scale) == 0.0:
        assignments = np.zeros(scale.size, dtype=np.int64)
        n_groups = 1
    else:
        edges = np.linspace(float(np.min(scale)), float(np.max(scale)), n_bins + 1)
        assignments = np.clip(
            np.searchsorted(edges, scale, side="right") - 1,
            0,
            n_bins - 1,
        )
        n_groups = n_bins

    groups = []
    for index in range(n_groups):
        mask = assignments == index
        if not np.any(mask):
            continue
        charge_weight = np.abs(weights[mask])
        sx_bin = float(np.average(sx[mask], weights=charge_weight))
        sy_bin = float(np.average(sy[mask], weights=charge_weight))
        W = make_W_gaussian(grid, sx_bin, sy_bin)
        groups.append((mask, W))

    def deposit(particles, grid_here):
        if grid_here is not grid or particles.x.shape != sx.shape:
            raise ValueError("adaptive deposit received a different grid or particle set")
        result = np.zeros((grid.Nx, grid.Ny), dtype=np.float64)
        for mask, W in groups:
            rho = deposit_cic_2d(
                particles.x[mask],
                particles.y[mask],
                particles.w[mask],
                grid.Nx,
                grid.Ny,
                grid.dx,
                grid.dy,
            )
            result += apply_W_to_grid(rho, W, grid)
        return result

    def gather(Ex, Ey, particles, grid_here):
        if grid_here is not grid or particles.x.shape != sx.shape:
            raise ValueError("adaptive gather received a different grid or particle set")
        Ex_p = np.empty_like(particles.x, dtype=np.float64)
        Ey_p = np.empty_like(particles.y, dtype=np.float64)
        for mask, W in groups:
            Ex_smooth = apply_W_to_grid(Ex, W, grid)
            Ey_smooth = apply_W_to_grid(Ey, W, grid)
            gathered_x, gathered_y = gather_cic_2d(
                Ex_smooth,
                Ey_smooth,
                particles.x[mask],
                particles.y[mask],
                grid.Nx,
                grid.Ny,
                grid.dx,
                grid.dy,
            )
            Ex_p[mask] = gathered_x
            Ey_p[mask] = gathered_y
        return Ex_p, Ey_p

    return deposit, gather


def make_spectral_W_box_methods(grid, kx_cut: float, ky_cut: float):
    """Convenience wrapper for a sharp box spectral filter."""
    W = make_W_box(grid, kx_cut, ky_cut)
    return make_spectral_W_methods(W)
