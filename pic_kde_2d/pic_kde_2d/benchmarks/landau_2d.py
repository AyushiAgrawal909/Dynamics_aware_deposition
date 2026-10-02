"""
benchmarks/landau_2d.py
=======================

2D Landau damping benchmark driver for the CIC Phase 1 baseline.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np

from pic_kde_2d.conventions import (
    DEFAULT_LANDAU_ALPHA,
    DEFAULT_LANDAU_K_LAMBDA_D,
    factor_ppc_2d,
    landau_box_length,
    marker_grid_shape,
    thermal_velocity,
)
from pic_kde_2d.core.grid2d import Grid2D
from pic_kde_2d.core.particles2d import init_landau_2d
from pic_kde_2d.core.pic_loop import DiagnosticHistory, run_simulation
from pic_kde_2d.methods.cic_wrappers import cic_deposit_method, cic_gather_method
from pic_kde_2d.benchmarks.theory import landau_gamma_theory, landau_omega_theory


def fit_mode_gamma(times: np.ndarray, amplitudes: np.ndarray, t_min: float, t_max: float, amp_floor: float = 1.0e-12) -> float:
    """Fit log(|E_k|) = a + gamma * t over the requested linear window."""
    times = np.asarray(times, dtype=np.float64)
    amplitudes = np.asarray(amplitudes, dtype=np.float64)
    mask = (times >= t_min) & (times <= t_max) & np.isfinite(amplitudes) & (amplitudes > amp_floor)
    if np.count_nonzero(mask) < 2:
        return float("nan")
    coeffs = np.polyfit(times[mask], np.log(amplitudes[mask]), 1)
    return float(coeffs[0])


@dataclass
class LandauRunSummary:
    Nx: int
    Ny: int
    ppc: int
    dt: float
    n_steps: int
    diag_stride: int
    Lx: float
    Ly: float
    kx_mode: int
    ky_mode: int
    alpha: float
    v_thermal: float
    gamma_measured: float
    gamma_theory: float
    gamma_relative_error: float
    omega_theory: float


def run_landau_baseline(
    Nx: int = 32,
    Ny: int = 8,
    ppc: int = 64,
    dt: float = 0.05,
    n_steps: int = 200,
    diag_stride: int = 2,
    kx_mode: int = 1,
    ky_mode: int = 0,
    alpha: float = DEFAULT_LANDAU_ALPHA,
    v_thermal: Optional[float] = None,
    Ly: float = 8.0,
    seed: int = 123,
    fit_t_min: float = 2.0,
    fit_t_max: float = 12.0,
) -> tuple[DiagnosticHistory, LandauRunSummary]:
    """Run the 2D CIC Landau baseline and return diagnostics plus a fit summary."""
    if v_thermal is None:
        v_thermal = thermal_velocity()
    Lx = landau_box_length(kx_mode, DEFAULT_LANDAU_K_LAMBDA_D)
    grid = Grid2D(Nx=Nx, Ny=Ny, Lx=Lx, Ly=Ly)
    Nx_marker, Ny_marker = marker_grid_shape(Nx, Ny, ppc)
    particles = init_landau_2d(
        Nx_marker=Nx_marker,
        Ny_marker=Ny_marker,
        Lx=Lx,
        Ly=Ly,
        v_thermal=v_thermal,
        alpha=alpha,
        kx_mode=kx_mode,
        ky_mode=ky_mode,
        rng=np.random.default_rng(seed),
    )
    history = run_simulation(
        particles,
        grid,
        dt,
        n_steps,
        cic_deposit_method,
        cic_gather_method,
        diag_stride=diag_stride,
        tracked_modes=[(kx_mode, ky_mode)],
        verbose=False,
    )
    amps = np.abs(np.asarray(history.mode_amplitudes_Ex[(kx_mode, ky_mode)], dtype=np.complex128))
    times = np.asarray(history.times, dtype=np.float64)
    gamma_measured = fit_mode_gamma(times, amps, fit_t_min, fit_t_max)
    gamma_theory = landau_gamma_theory(DEFAULT_LANDAU_K_LAMBDA_D, v_thermal=v_thermal)
    omega_theory = landau_omega_theory(DEFAULT_LANDAU_K_LAMBDA_D, v_thermal=v_thermal)
    summary = LandauRunSummary(
        Nx=Nx,
        Ny=Ny,
        ppc=ppc,
        dt=dt,
        n_steps=n_steps,
        diag_stride=diag_stride,
        Lx=Lx,
        Ly=Ly,
        kx_mode=kx_mode,
        ky_mode=ky_mode,
        alpha=alpha,
        v_thermal=v_thermal,
        gamma_measured=gamma_measured,
        gamma_theory=gamma_theory,
        gamma_relative_error=abs(gamma_measured - gamma_theory) / abs(gamma_theory),
        omega_theory=omega_theory,
    )
    return history, summary
