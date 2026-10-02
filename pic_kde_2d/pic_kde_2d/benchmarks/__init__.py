"""Benchmark entry points and theory helpers for 2D PIC validation."""

from .landau_2d import fit_mode_gamma, run_landau_baseline
from .theory import landau_gamma_theory, landau_omega_theory

__all__ = [
    "fit_mode_gamma",
    "landau_gamma_theory",
    "landau_omega_theory",
    "run_landau_baseline",
]
