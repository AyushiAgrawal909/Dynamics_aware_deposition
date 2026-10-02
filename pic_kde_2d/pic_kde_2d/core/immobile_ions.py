"""Explicit immobile-ion backgrounds for nonuniform periodic plasmas."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from pic_kde_2d.core.grid2d import Grid2D


@dataclass(frozen=True)
class ImmobileIonBackground:
    """Grid-resolved positive charge density held fixed during a PIC run."""

    rho: np.ndarray
    grid: Grid2D

    def __post_init__(self) -> None:
        density = np.asarray(self.rho, dtype=np.float64)
        if density.shape != (self.grid.Nx, self.grid.Ny):
            raise ValueError("ion background shape must match the grid")
        if np.any(~np.isfinite(density)) or np.any(density < 0.0):
            raise ValueError("ion background must be finite and nonnegative")
        object.__setattr__(self, "rho", density.copy())

    @property
    def total_charge(self) -> float:
        return float(np.sum(self.rho) * self.grid.dx * self.grid.dy)

    def add(self, electron_rho: np.ndarray) -> np.ndarray:
        electron = np.asarray(electron_rho, dtype=np.float64)
        if electron.shape != self.rho.shape:
            raise ValueError("electron density shape must match the ion background")
        return electron + self.rho


def with_immobile_ions(
    electron_deposit: Callable,
    background: ImmobileIonBackground,
) -> Callable:
    """Wrap an electron deposition callable with an explicit ion background."""

    def deposit_total(particles, grid):
        if grid is not background.grid:
            raise ValueError("deposit called with a grid different from the background grid")
        return background.add(electron_deposit(particles, grid))

    return deposit_total


def neutrality_diagnostics(total_rho: np.ndarray, grid: Grid2D) -> dict:
    """Return integral and relative mean-charge defects before the Poisson solve."""
    rho = np.asarray(total_rho, dtype=np.float64)
    if rho.shape != (grid.Nx, grid.Ny):
        raise ValueError("rho shape must match grid")
    integrated = float(np.sum(rho) * grid.dx * grid.dy)
    scale = float(np.sum(np.abs(rho)) * grid.dx * grid.dy)
    return {
        "integrated_charge": integrated,
        "relative_integrated_charge": abs(integrated) / max(scale, 1.0e-300),
        "mean_charge_density": float(np.mean(rho)),
    }
