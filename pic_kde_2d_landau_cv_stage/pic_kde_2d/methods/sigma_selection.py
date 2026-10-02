"""
methods/sigma_selection.py
==========================

Analytical sigma-selection rules based on spectral preservation.
"""

from __future__ import annotations

import numpy as np


def equivalent_gaussian_sigma_cells(binomial_passes: int) -> float:
    """Leading-order Gaussian width of repeated ``(1, 2, 1) / 4`` filtering."""
    if isinstance(binomial_passes, bool) or int(binomial_passes) != binomial_passes:
        raise ValueError("binomial_passes must be an integer")
    if binomial_passes < 0:
        raise ValueError("binomial_passes must be nonnegative")
    return float(np.sqrt(float(binomial_passes) / 2.0))


def binomial_pass_budget_for_gaussian_ceiling(
    N: int,
    mode: int,
    sigma_ceiling_cells: float,
) -> int:
    """Largest pass count no more attenuating than a Gaussian ceiling.

    Exact matched responses are compared at the protected mode:
    ``exp[-(2*pi*m*sigma/N)^2]`` for Gaussian smoothing and
    ``cos(pi*m/N)^(4*n)`` for ``n`` binomial passes.
    """
    if N <= 0 or mode <= 0 or mode >= N / 2:
        raise ValueError("require 0 < mode < N/2")
    if not np.isfinite(sigma_ceiling_cells) or sigma_ceiling_cells < 0.0:
        raise ValueError("sigma_ceiling_cells must be finite and nonnegative")
    angle = np.pi * float(mode) / float(N)
    log_single_matched = 4.0 * np.log(np.cos(angle))
    log_gaussian_matched = -(
        2.0 * np.pi * float(mode) * sigma_ceiling_cells / float(N)
    ) ** 2
    return max(
        0,
        int(np.floor(log_gaussian_matched / log_single_matched + 1.0e-12)),
    )


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
