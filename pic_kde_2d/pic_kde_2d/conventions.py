"""
conventions.py
==============

Shared normalized-unit conventions and helper utilities for 2D PIC.

This module centralizes the baseline constants used by Phase 1 validation.
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np


Q_OVER_M = -1.0
CHARGE_TOL = 1.0e-10
SELF_FORCE_TOL = 1.0e-8
DEFAULT_LANDAU_K_LAMBDA_D = 0.4
DEFAULT_LANDAU_ALPHA = 0.02


def thermal_velocity() -> float:
    """Thermal velocity in the normalized lambda_D = 1, omega_p = 1 system."""
    return 1.0


def landau_box_length(kx_mode: int, k_lambdaD: float = DEFAULT_LANDAU_K_LAMBDA_D) -> float:
    """Return Lx so that mode index `kx_mode` has the requested k * lambda_D."""
    if kx_mode <= 0:
        raise ValueError("kx_mode must be positive")
    if k_lambdaD <= 0.0:
        raise ValueError("k_lambdaD must be positive")
    return 2.0 * math.pi * float(kx_mode) / float(k_lambdaD)


def factor_ppc_2d(ppc: int) -> Tuple[int, int]:
    """Split particles-per-cell into x/y marker counts per cell."""
    if ppc <= 0:
        raise ValueError("ppc must be positive")
    root = int(math.sqrt(ppc))
    for px in range(root, 0, -1):
        if ppc % px == 0:
            return px, ppc // px
    return 1, ppc


def marker_grid_shape(Nx: int, Ny: int, ppc: int) -> Tuple[int, int]:
    """Return total marker counts along x and y for a quiet-start marker lattice."""
    px, py = factor_ppc_2d(ppc)
    return Nx * px, Ny * py


def gaussian_transfer_mode_2d(
    Nx: int,
    Ny: int,
    mx: int,
    my: int,
    sigma_x_cells: float,
    sigma_y_cells: float,
) -> float:
    """Continuous separable Gaussian transfer for one Fourier mode."""
    kx = 2.0 * math.pi * float(mx) / float(Nx)
    ky = 2.0 * math.pi * float(my) / float(Ny)
    return float(np.exp(-0.5 * (kx * sigma_x_cells) ** 2 - 0.5 * (ky * sigma_y_cells) ** 2))


def gaussian_transfer_2d(Nx: int, Ny: int, sigma_x_cells: float, sigma_y_cells: float) -> np.ndarray:
    """Continuous separable Gaussian transfer on the full rfft2 spectral grid."""
    kx = 2.0 * np.pi * np.fft.fftfreq(Nx)
    ky = 2.0 * np.pi * np.fft.rfftfreq(Ny)
    KX, KY = np.meshgrid(kx, ky, indexing="ij")
    return np.exp(-0.5 * (KX * sigma_x_cells) ** 2 - 0.5 * (KY * sigma_y_cells) ** 2)
