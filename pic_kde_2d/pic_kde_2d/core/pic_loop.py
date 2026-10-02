"""
core/pic_loop.py
================

Main time-stepping driver for 2D electrostatic PIC.
Method-agnostic: deposit and gather callables are provided by the caller.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from pic_kde_2d.conventions import Q_OVER_M
from pic_kde_2d.core.grid2d import Grid2D
from pic_kde_2d.core.particles2d import Particles2D


@dataclass
class DiagnosticHistory:
    """Time series of diagnostics collected during a simulation."""

    times: List[float] = field(default_factory=list)
    field_energy: List[float] = field(default_factory=list)
    kinetic_energy: List[float] = field(default_factory=list)
    total_energy: List[float] = field(default_factory=list)
    total_charge: List[float] = field(default_factory=list)
    mode_amplitudes_Ex: Dict[Tuple[int, int], List[complex]] = field(default_factory=dict)
    delta_f_ratio: List[float] = field(default_factory=list)
    delta_f_fallback: List[bool] = field(default_factory=list)

    def to_arrays(self):
        """Convert lists to NumPy arrays for analysis."""
        result = {
            "times": np.asarray(self.times, dtype=np.float64),
            "field_energy": np.asarray(self.field_energy, dtype=np.float64),
            "kinetic_energy": np.asarray(self.kinetic_energy, dtype=np.float64),
            "total_energy": np.asarray(self.total_energy, dtype=np.float64),
            "total_charge": np.asarray(self.total_charge, dtype=np.float64),
            "delta_f_ratio": np.asarray(self.delta_f_ratio, dtype=np.float64),
            "delta_f_fallback": np.asarray(self.delta_f_fallback, dtype=bool),
        }
        for key, values in self.mode_amplitudes_Ex.items():
            result[f"Ex_mode_{key[0]}_{key[1]}"] = np.asarray(values, dtype=np.complex128)
        return result


def pic_step(
    particles: Particles2D,
    grid: Grid2D,
    dt: float,
    deposit_fn: Callable,
    gather_fn: Callable,
):
    """Single velocity-Verlet PIC step: deposit -> Poisson -> gather -> push."""
    rho = deposit_fn(particles, grid)
    Ex, Ey = grid.poisson_solve(rho)
    Ex_p, Ey_p = gather_fn(Ex, Ey, particles, grid)

    particles.vx += 0.5 * dt * Q_OVER_M * Ex_p
    particles.vy += 0.5 * dt * Q_OVER_M * Ey_p
    particles.x = (particles.x + dt * particles.vx) % particles.Lx
    particles.y = (particles.y + dt * particles.vy) % particles.Ly

    rho = deposit_fn(particles, grid)
    Ex, Ey = grid.poisson_solve(rho)
    Ex_p, Ey_p = gather_fn(Ex, Ey, particles, grid)
    particles.vx += 0.5 * dt * Q_OVER_M * Ex_p
    particles.vy += 0.5 * dt * Q_OVER_M * Ey_p
    return rho, Ex, Ey


def run_simulation(
    particles: Particles2D,
    grid: Grid2D,
    dt: float,
    n_steps: int,
    deposit_fn: Callable,
    gather_fn: Callable,
    diag_stride: int = 10,
    tracked_modes: Optional[List[tuple]] = None,
    verbose: bool = False,
) -> DiagnosticHistory:
    """Run a complete PIC simulation, collecting diagnostics."""
    if tracked_modes is None:
        tracked_modes = [(1, 0)]

    history = DiagnosticHistory()
    for key in tracked_modes:
        history.mode_amplitudes_Ex[key] = []

    for step in range(n_steps + 1):
        if step % diag_stride == 0:
            rho = deposit_fn(particles, grid)
            Ex, Ey = grid.poisson_solve(rho)
            t = step * dt
            history.times.append(t)
            history.field_energy.append(grid.field_energy(Ex, Ey))
            history.kinetic_energy.append(particles.kinetic_energy())
            history.total_energy.append(history.field_energy[-1] + history.kinetic_energy[-1])
            history.total_charge.append(particles.total_charge())
            for key in tracked_modes:
                m, n = key
                amp = grid.mode_amplitude(Ex, m, n)
                history.mode_amplitudes_Ex[key].append(amp)
            if verbose and step % max(diag_stride * 10, 1) == 0:
                print(
                    f"step {step}/{n_steps}, t={t:.2f}, "
                    f"W_field={history.field_energy[-1]:.3e}"
                )

        if step == n_steps:
            break

        pic_step(particles, grid, dt, deposit_fn, gather_fn)

    return history
