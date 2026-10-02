"""
validation/conservation_2d.py
=============================

Charge and energy conservation diagnostics.
"""

from __future__ import annotations

import numpy as np

from pic_kde_2d.core.pic_loop import DiagnosticHistory


def check_charge_conservation(rho: np.ndarray, expected_total: float, dx: float, dy: float, tol: float = 1.0e-10) -> dict:
    """Check that the deposited density sums to the expected total charge."""
    measured = float(np.sum(rho) * dx * dy)
    relative_error = abs(measured - expected_total) / abs(expected_total + 1.0e-30)
    return {
        "measured": measured,
        "expected": float(expected_total),
        "relative_error": float(relative_error),
        "passes": bool(relative_error < tol),
    }


def compute_energy_drift(history: DiagnosticHistory) -> dict:
    """Compute relative total-energy drift over the simulation."""
    arrays = history.to_arrays()
    E_total = arrays["total_energy"]
    E_0 = float(E_total[0])
    relative_drift = np.abs(E_total - E_0) / abs(E_0 + 1.0e-30)
    return {
        "E_initial": E_0,
        "E_final": float(E_total[-1]),
        "max_relative_drift": float(np.max(relative_drift)),
        "final_relative_drift": float(relative_drift[-1]),
    }
