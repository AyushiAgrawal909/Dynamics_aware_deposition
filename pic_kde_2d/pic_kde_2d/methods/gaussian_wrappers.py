"""
methods/gaussian_wrappers.py
============================

Method-pluggable wrappers for the anisotropic Gaussian deposit/gather.
"""

from __future__ import annotations

from pic_kde_2d.methods.gaussian import deposit_gaussian_2d_aniso, gather_gaussian_2d_aniso


def make_gaussian_methods(sigma_x: float, sigma_y: float):
    """Return (deposit_fn, gather_fn) with fixed sigma_x and sigma_y baked in."""

    def deposit(particles, grid):
        return deposit_gaussian_2d_aniso(
            particles.x,
            particles.y,
            particles.w,
            sigma_x,
            sigma_y,
            grid.Nx,
            grid.Ny,
            grid.dx,
            grid.dy,
        )

    def gather(Ex, Ey, particles, grid):
        return gather_gaussian_2d_aniso(
            Ex,
            Ey,
            particles.x,
            particles.y,
            sigma_x,
            sigma_y,
            grid.Nx,
            grid.Ny,
            grid.dx,
            grid.dy,
        )

    return deposit, gather
