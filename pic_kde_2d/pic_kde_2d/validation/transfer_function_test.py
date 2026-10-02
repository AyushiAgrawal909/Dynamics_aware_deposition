"""
validation/transfer_function_test.py
====================================

Empirical transfer-function measurements for Gaussian deposition.
"""

from __future__ import annotations

import numpy as np

from pic_kde_2d.conventions import gaussian_transfer_2d, gaussian_transfer_mode_2d
from pic_kde_2d.core.grid2d import Grid2D
from pic_kde_2d.core.particles2d import Particles2D


def measure_transfer_function_deposit(deposit_fn, grid: Grid2D, Nx_marker: int = 256, Ny_marker: int = 256) -> dict:
    """Deposit a uniform quiet particle set and measure residual non-zero spectral power."""
    xs = (np.arange(Nx_marker, dtype=np.float64) + 0.5) * grid.Lx / Nx_marker
    ys = (np.arange(Ny_marker, dtype=np.float64) + 0.5) * grid.Ly / Ny_marker
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    particles = Particles2D(
        x=X.ravel(),
        y=Y.ravel(),
        vx=np.zeros(Nx_marker * Ny_marker, dtype=np.float64),
        vy=np.zeros(Nx_marker * Ny_marker, dtype=np.float64),
        w=np.full(Nx_marker * Ny_marker, 1.0, dtype=np.float64),
        Lx=grid.Lx,
        Ly=grid.Ly,
    )
    rho = deposit_fn(particles, grid)
    rho_hat = np.fft.rfft2(rho - np.mean(rho))
    spectrum = np.abs(rho_hat) ** 2 / (grid.Nx * grid.Ny) ** 2
    return {
        "rho": rho,
        "spectrum": spectrum,
        "rms_nonzero_modes": float(np.sqrt(np.sum(spectrum))),
    }


def measure_mode_transfer(deposit_fn, grid: Grid2D, mx: int, my: int, amplitude: float = 1.0) -> dict:
    """Measure the deposit transfer for a single sinusoidal density mode."""
    X, Y = np.meshgrid(grid.x, grid.y, indexing="ij")
    phase = 2.0 * np.pi * mx * X / grid.Lx + 2.0 * np.pi * my * Y / grid.Ly
    rho_target = amplitude * np.cos(phase)
    particles = Particles2D(
        x=X.ravel(),
        y=Y.ravel(),
        vx=np.zeros(grid.Nx * grid.Ny, dtype=np.float64),
        vy=np.zeros(grid.Nx * grid.Ny, dtype=np.float64),
        w=(rho_target * grid.dx * grid.dy).ravel(),
        Lx=grid.Lx,
        Ly=grid.Ly,
    )
    rho_measured = deposit_fn(particles, grid)
    input_amp = grid.mode_amplitude(rho_target, mx, my)
    output_amp = grid.mode_amplitude(rho_measured, mx, my)
    transfer = abs(output_amp) / max(abs(input_amp), 1.0e-30)
    return {
        "mode": (mx, my),
        "input_amplitude": complex(input_amp),
        "output_amplitude": complex(output_amp),
        "transfer_measured": float(transfer),
    }


def compare_transfer_to_analytical(grid: Grid2D, sigma_x: float, sigma_y: float):
    """Compute the analytical continuous Gaussian transfer grid for this domain."""
    sigma_x_cells = sigma_x / grid.dx
    sigma_y_cells = sigma_y / grid.dy
    return gaussian_transfer_2d(grid.Nx, grid.Ny, sigma_x_cells, sigma_y_cells)


def compare_mode_transfer_to_analytical(
    deposit_fn,
    grid: Grid2D,
    sigma_x: float,
    sigma_y: float,
    modes: list[tuple[int, int]],
) -> list[dict]:
    """Compare empirical and analytical transfer for selected low-k modes."""
    sigma_x_cells = sigma_x / grid.dx
    sigma_y_cells = sigma_y / grid.dy
    rows = []
    for mx, my in modes:
        measured = measure_mode_transfer(deposit_fn, grid, mx, my)
        analytical = gaussian_transfer_mode_2d(grid.Nx, grid.Ny, mx, my, sigma_x_cells, sigma_y_cells)
        rows.append(
            {
                "mode": (mx, my),
                "transfer_measured": measured["transfer_measured"],
                "transfer_analytical": float(analytical),
                "relative_error": float(abs(measured["transfer_measured"] - analytical) / max(abs(analytical), 1.0e-30)),
            }
        )
    return rows
