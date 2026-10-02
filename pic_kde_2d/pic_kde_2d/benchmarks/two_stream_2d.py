"""
benchmarks/two_stream_2d.py
===========================

2D two-stream baseline driver.

Phase 1 only needs the initialization path and a minimal executable baseline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from pic_kde_2d.conventions import marker_grid_shape, thermal_velocity
from pic_kde_2d.core.grid2d import Grid2D
from pic_kde_2d.core.particles2d import init_two_stream_2d
from pic_kde_2d.core.pic_loop import DiagnosticHistory, run_simulation
from pic_kde_2d.methods.cic_wrappers import cic_deposit_method, cic_gather_method


@dataclass
class TwoStreamRunSummary:
    Nx: int
    Ny: int
    ppc: int
    dt: float
    n_steps: int
    initial_mode_amplitude: float
    final_mode_amplitude: float


def run_two_stream_baseline(
    Nx: int = 64,
    Ny: int = 16,
    ppc: int = 64,
    dt: float = 0.05,
    n_steps: int = 200,
    Lx: float = 2.0 * np.pi,
    Ly: float = 8.0,
    v_beam: float = 1.0,
    v_thermal: Optional[float] = None,
    alpha: float = 0.01,
    kx_mode: int = 1,
    ky_mode: int = 0,
    seed: int = 123,
) -> tuple[DiagnosticHistory, TwoStreamRunSummary]:
    if v_thermal is None:
        v_thermal = thermal_velocity()
    grid = Grid2D(Nx=Nx, Ny=Ny, Lx=Lx, Ly=Ly)
    Nx_marker, Ny_marker = marker_grid_shape(Nx, Ny, ppc)
    particles = init_two_stream_2d(
        Nx_marker=Nx_marker,
        Ny_marker=Ny_marker,
        Lx=Lx,
        Ly=Ly,
        v_beam=v_beam,
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
        diag_stride=2,
        tracked_modes=[(kx_mode, ky_mode)],
        verbose=False,
    )
    amps = np.abs(np.asarray(history.mode_amplitudes_Ex[(kx_mode, ky_mode)], dtype=np.complex128))
    summary = TwoStreamRunSummary(
        Nx=Nx,
        Ny=Ny,
        ppc=ppc,
        dt=dt,
        n_steps=n_steps,
        initial_mode_amplitude=float(amps[0]),
        final_mode_amplitude=float(amps[-1]),
    )
    return history, summary
