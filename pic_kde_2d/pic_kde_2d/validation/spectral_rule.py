"""
validation/spectral_rule.py
===========================

Spectral transfer-function helpers for the 2D baseline and later kernels.
"""

from __future__ import annotations

import numpy as np

from pic_kde_2d.conventions import gaussian_transfer_mode_2d


def cic_transfer_1d(mode_index: int, n_cells: int) -> float:
    arg = np.pi * float(mode_index) / float(n_cells)
    if abs(arg) < 1.0e-15:
        return 1.0
    return float((np.sin(arg) / arg) ** 2)


def cic_transfer_2d(mx: int, my: int, Nx: int, Ny: int) -> float:
    return cic_transfer_1d(mx, Nx) * cic_transfer_1d(my, Ny)


def gaussian_transfer_2d_mode(
    mx: int,
    my: int,
    Nx: int,
    Ny: int,
    sigma_x_cells: float,
    sigma_y_cells: float,
    matched: bool = False,
) -> float:
    """Return the continuous Gaussian transfer at one mode, optionally squared for matched gather."""
    transfer = gaussian_transfer_mode_2d(Nx, Ny, mx, my, sigma_x_cells, sigma_y_cells)
    return transfer * transfer if matched else transfer
