"""
core/pic_loop_delta_f.py
=========================

Velocity-Verlet time stepping for two-weight delta-f simulations.
"""

from __future__ import annotations

from typing import Callable, List, Optional

import numpy as np

from pic_kde_2d.conventions import Q_OVER_M
from pic_kde_2d.core.grid2d import Grid2D
from pic_kde_2d.core.pic_loop import DiagnosticHistory
from pic_kde_2d.methods.delta_f import (
    ParticlesDeltaF,
    activate_full_f_fallback,
    check_delta_f_validity,
    delta_weight_rhs,
    evolve_delta_weights,
)


def pic_step_delta_f(
    p: ParticlesDeltaF,
    grid: Grid2D,
    dt: float,
    deposit_fn: Callable,
    gather_fn: Callable,
    enforce_neutrality: bool = True,
) -> tuple:
    """Advance markers and perturbation weights by one symmetric step."""
    rho = deposit_fn(p, grid)
    Ex, Ey = grid.poisson_solve(rho)
    Ex_p, Ey_p = gather_fn(Ex, Ey, p, grid)

    evolve_delta_weights(p, Ex_p, Ey_p, Q_OVER_M, 0.5 * dt, enforce_neutrality)
    p.vx += 0.5 * dt * Q_OVER_M * Ex_p
    p.vy += 0.5 * dt * Q_OVER_M * Ey_p
    p.x = (p.x + dt * p.vx) % p.Lx
    p.y = (p.y + dt * p.vy) % p.Ly

    rho = deposit_fn(p, grid)
    Ex, Ey = grid.poisson_solve(rho)
    Ex_p, Ey_p = gather_fn(Ex, Ey, p, grid)
    p.vx += 0.5 * dt * Q_OVER_M * Ex_p
    p.vy += 0.5 * dt * Q_OVER_M * Ey_p
    evolve_delta_weights(p, Ex_p, Ey_p, Q_OVER_M, 0.5 * dt, enforce_neutrality)
    return rho, Ex, Ey


def run_simulation_delta_f(
    p: ParticlesDeltaF,
    grid: Grid2D,
    dt: float,
    n_steps: int,
    deposit_fn: Callable,
    gather_fn: Callable,
    diag_stride: int = 10,
    tracked_modes: Optional[List[tuple]] = None,
    validity_threshold: float = 0.5,
    fallback_to_full_f: bool = False,
    enforce_neutrality: bool = True,
    verbose: bool = False,
) -> DiagnosticHistory:
    """Run delta-f PIC while recording weight validity and optional fallback."""
    if tracked_modes is None:
        tracked_modes = [(1, 0)]
    history = DiagnosticHistory()
    for key in tracked_modes:
        history.mode_amplitudes_Ex[key] = []

    for step in range(n_steps + 1):
        if step % diag_stride == 0:
            rho = deposit_fn(p, grid)
            Ex, Ey = grid.poisson_solve(rho)
            validity = check_delta_f_validity(p, threshold=validity_threshold)
            history.times.append(step * dt)
            history.field_energy.append(grid.field_energy(Ex, Ey))
            history.kinetic_energy.append(p.kinetic_energy_perturbation())
            history.total_energy.append(history.field_energy[-1] + history.kinetic_energy[-1])
            history.total_charge.append(p.deposited_charge())
            history.delta_f_ratio.append(validity["ratio"])
            history.delta_f_fallback.append(p.use_full_f)
            for key in tracked_modes:
                history.mode_amplitudes_Ex[key].append(grid.mode_amplitude(Ex, key[0], key[1]))
            if not validity["valid"] and verbose:
                print(f"delta-f validity ratio={validity['ratio']:.3f} at t={step * dt:.2f}")

        if step == n_steps:
            break
        pic_step_delta_f(p, grid, dt, deposit_fn, gather_fn, enforce_neutrality)
        if fallback_to_full_f and not p.use_full_f and not check_delta_f_validity(p, validity_threshold)["valid"]:
            activate_full_f_fallback(p)

    return history


def pic_step_linearized_delta_f(
    p: ParticlesDeltaF,
    grid: Grid2D,
    dt: float,
    deposit_fn: Callable,
    gather_fn: Callable,
    enforce_neutrality: bool = True,
) -> tuple:
    """Advance linearized delta-f on frozen equilibrium characteristics.

    Marker velocities remain fixed and positions free-stream. An explicit
    midpoint update advances the perturbation weights without feeding the
    perturbation field back into marker trajectories.
    """
    x_initial = p.x.copy()
    y_initial = p.y.copy()
    w_initial = p.w_delta.copy()

    rho = deposit_fn(p, grid)
    Ex, Ey = grid.poisson_solve(rho)
    Ex_p, Ey_p = gather_fn(Ex, Ey, p, grid)
    rhs_initial = delta_weight_rhs(p, Ex_p, Ey_p, Q_OVER_M)

    p.x = (x_initial + 0.5 * dt * p.vx) % p.Lx
    p.y = (y_initial + 0.5 * dt * p.vy) % p.Ly
    p.w_delta = w_initial + 0.5 * dt * rhs_initial
    if enforce_neutrality:
        p.w_delta -= np.mean(p.w_delta)

    rho_mid = deposit_fn(p, grid)
    Ex_mid, Ey_mid = grid.poisson_solve(rho_mid)
    Ex_p_mid, Ey_p_mid = gather_fn(Ex_mid, Ey_mid, p, grid)
    rhs_mid = delta_weight_rhs(p, Ex_p_mid, Ey_p_mid, Q_OVER_M)

    p.x = (x_initial + dt * p.vx) % p.Lx
    p.y = (y_initial + dt * p.vy) % p.Ly
    p.w_delta = w_initial + dt * rhs_mid
    if enforce_neutrality:
        p.w_delta -= np.mean(p.w_delta)
    return rho_mid, Ex_mid, Ey_mid


def run_simulation_linearized_delta_f(
    p: ParticlesDeltaF,
    grid: Grid2D,
    dt: float,
    n_steps: int,
    deposit_fn: Callable,
    gather_fn: Callable,
    diag_stride: int = 10,
    tracked_modes: Optional[List[tuple]] = None,
    validity_threshold: float = 0.5,
    enforce_neutrality: bool = True,
    verbose: bool = False,
) -> DiagnosticHistory:
    """Run linearized delta-f with per-mode and validity diagnostics."""
    if tracked_modes is None:
        tracked_modes = [(1, 0)]
    history = DiagnosticHistory()
    for key in tracked_modes:
        history.mode_amplitudes_Ex[key] = []

    for step in range(n_steps + 1):
        if step % diag_stride == 0:
            rho = deposit_fn(p, grid)
            Ex, Ey = grid.poisson_solve(rho)
            validity = check_delta_f_validity(p, threshold=validity_threshold)
            history.times.append(step * dt)
            history.field_energy.append(grid.field_energy(Ex, Ey))
            history.kinetic_energy.append(p.kinetic_energy_perturbation())
            history.total_energy.append(history.field_energy[-1] + history.kinetic_energy[-1])
            history.total_charge.append(p.deposited_charge())
            history.delta_f_ratio.append(validity["ratio"])
            history.delta_f_fallback.append(False)
            for key in tracked_modes:
                history.mode_amplitudes_Ex[key].append(
                    grid.mode_amplitude(Ex, key[0], key[1])
                )
            if not validity["valid"] and verbose:
                print(
                    f"linearized delta-f validity ratio={validity['ratio']:.3f} "
                    f"at t={step * dt:.2f}"
                )

        if step == n_steps:
            break
        pic_step_linearized_delta_f(
            p,
            grid,
            dt,
            deposit_fn,
            gather_fn,
            enforce_neutrality,
        )

    return history
