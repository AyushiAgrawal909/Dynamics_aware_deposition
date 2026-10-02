"""
methods/sigma_selection.py
==========================

Analytical sigma-selection rules based on spectral preservation.
"""

from __future__ import annotations

import numpy as np


def sigma_max_matched(N: int, m_max: int, A_min: float = 0.85) -> float:
    """Maximum sigma in cell units for matched deposit+gather preserving mode m_max."""
    return float(np.sqrt(-np.log(A_min)) * N / (2.0 * np.pi * m_max))


def sigma_max_deposit_only(N: int, m_max: int, A_min: float = 0.85) -> float:
    """Maximum sigma in cell units if only deposition is smoothed."""
    return float(np.sqrt(-2.0 * np.log(A_min)) * N / (2.0 * np.pi * m_max))


def select_sigma_anisotropic_2d(
    Nx: int,
    Ny: int,
    mx_max: int,
    my_max: int,
    A_min: float = 0.85,
    safety_factor: float = 1.0,
) -> tuple[float, float]:
    """Select anisotropic sigma_x, sigma_y in cell units for matched-gather smoothing."""
    sx = sigma_max_matched(Nx, mx_max, A_min) * safety_factor
    sy = sigma_max_matched(Ny, my_max, A_min) * safety_factor
    return float(sx), float(sy)
