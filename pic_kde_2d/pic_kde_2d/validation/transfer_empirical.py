"""
validation/transfer_empirical.py
================================

Empirical and analytic transfer-function helpers for revised Phase 4.
"""

from __future__ import annotations

import numpy as np


def cic_transfer_1d(m: int, N: int) -> float:
    """CIC shape-function transfer factor for integer mode m on N cells."""
    x = np.pi * float(m) / float(N)
    if x == 0.0:
        return 1.0
    sinc = np.sin(x) / x
    return float(sinc**2)


def sigma_max_pure_gaussian(N: int, m_max: int, A_min: float = 0.85) -> float:
    """Matched Gaussian rule in cell units: exp(-k^2 sigma^2) = A_min."""
    return float(np.sqrt(-np.log(A_min)) * N / (2.0 * np.pi * m_max))


def sigma_max_cic_corrected(
    N: int,
    m_max: int,
    A_min: float = 0.85,
    cic_power: float = 4.0,
) -> float:
    """CIC-corrected matched rule with T_CIC^cic_power attenuation budget."""
    T_cic = cic_transfer_1d(m_max, N)
    budget = A_min / (T_cic**cic_power)
    if budget >= 1.0:
        return 0.0
    k_cells = 2.0 * np.pi * float(m_max) / float(N)
    return float(np.sqrt(-np.log(budget)) / k_cells)


def measure_effective_transfer(
    rho_smoothed_hat: np.ndarray,
    rho_cic_hat: np.ndarray,
    m_target: tuple[int, int],
) -> float:
    """Return |rho_smoothed_hat[m]| / |rho_cic_hat[m]| for one rfft2 mode."""
    mx, my = m_target
    numerator = abs(rho_smoothed_hat[mx, my])
    denominator = abs(rho_cic_hat[mx, my]) + 1.0e-30
    return float(numerator / denominator)
