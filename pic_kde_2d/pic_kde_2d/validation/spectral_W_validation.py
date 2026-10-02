"""
validation/spectral_W_validation.py
===================================

Validation helpers for Phase 3 spectral filtering.
"""

from __future__ import annotations

import numpy as np

from pic_kde_2d.core.grid2d import Grid2D
from pic_kde_2d.core.particles2d import Particles2D
from pic_kde_2d.methods.gaussian import deposit_gaussian_2d_aniso
from pic_kde_2d.methods.spectral_W_wrappers import make_spectral_W_gaussian_methods


def compare_phase2_vs_phase3_density(
    grid: Grid2D,
    particles: Particles2D,
    sigma_x: float,
    sigma_y: float,
) -> dict:
    """Compare direct Gaussian deposition against CIC followed by Gaussian W."""
    rho_direct = deposit_gaussian_2d_aniso(
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
    deposit_W, _ = make_spectral_W_gaussian_methods(grid, sigma_x, sigma_y)
    rho_W = deposit_W(particles, grid)
    diff = rho_direct - rho_W
    rms = float(np.sqrt(np.mean(diff**2)))
    rms_signal = float(np.sqrt(np.mean(rho_direct**2)))
    return {
        "rho_direct": rho_direct,
        "rho_W": rho_W,
        "rms_diff": rms,
        "rms_signal": rms_signal,
        "relative_rms": float(rms / (rms_signal + 1.0e-30)),
    }
