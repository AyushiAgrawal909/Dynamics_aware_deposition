"""
core/grid2d.py
==============

2D periodic grid with FFT-based Poisson solver.
Provides field storage, Fourier-space utilities, and energy diagnostics.

All shapes follow the convention: (Nx, Ny) with i indexing x and j indexing y.
Wavenumbers use rfft2 convention: kx is fftfreq (signed), ky is rfftfreq (non-negative).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple

import numpy as np


@dataclass
class Grid2D:
    """2D periodic grid for electrostatic PIC."""

    Nx: int
    Ny: int
    Lx: float
    Ly: float

    dx: float = field(init=False)
    dy: float = field(init=False)
    x: np.ndarray = field(init=False)
    y: np.ndarray = field(init=False)
    kx: np.ndarray = field(init=False)
    ky: np.ndarray = field(init=False)
    KX: np.ndarray = field(init=False)
    KY: np.ndarray = field(init=False)
    k2: np.ndarray = field(init=False)
    k2_inv: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.dx = self.Lx / self.Nx
        self.dy = self.Ly / self.Ny
        self.x = np.arange(self.Nx, dtype=np.float64) * self.dx
        self.y = np.arange(self.Ny, dtype=np.float64) * self.dy
        self.kx = 2.0 * np.pi * np.fft.fftfreq(self.Nx, d=self.dx)
        self.ky = 2.0 * np.pi * np.fft.rfftfreq(self.Ny, d=self.dy)
        self.KX, self.KY = np.meshgrid(self.kx, self.ky, indexing="ij")
        self.k2 = self.KX**2 + self.KY**2
        self.k2_inv = np.zeros_like(self.k2)
        mask = self.k2 > 0.0
        self.k2_inv[mask] = 1.0 / self.k2[mask]

    def poisson_solve(self, rho: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Solve -laplacian phi = rho, return E = -grad phi."""
        if rho.shape != (self.Nx, self.Ny):
            raise AssertionError(f"rho shape {rho.shape} != grid shape ({self.Nx}, {self.Ny})")
        rho_hat = np.fft.rfft2(rho)
        phi_hat = rho_hat * self.k2_inv
        Ex_hat = -1j * self.KX * phi_hat
        Ey_hat = -1j * self.KY * phi_hat
        Ex = np.fft.irfft2(Ex_hat, s=(self.Nx, self.Ny))
        Ey = np.fft.irfft2(Ey_hat, s=(self.Nx, self.Ny))
        return Ex.real.astype(np.float64), Ey.real.astype(np.float64)

    def field_energy(self, Ex: np.ndarray, Ey: np.ndarray) -> float:
        """Electrostatic field energy: (1/2) integral (Ex^2 + Ey^2) dxdy."""
        return 0.5 * float(np.sum(Ex**2 + Ey**2)) * self.dx * self.dy

    def density_spectrum(self, rho: np.ndarray) -> np.ndarray:
        """Return |rho_hat|^2 power spectrum, normalized for plotting."""
        rho_hat = np.fft.rfft2(rho)
        return np.abs(rho_hat) ** 2 / (self.Nx * self.Ny) ** 2

    def mode_amplitude(self, field: np.ndarray, m: int, n: int) -> complex:
        """Return the complex amplitude of Fourier mode (m, n) of a scalar field."""
        f_hat = np.fft.rfft2(field)
        m_idx = m % self.Nx
        n_idx = n
        if not 0 <= n_idx <= self.Ny // 2:
            raise AssertionError(f"n={n} out of rfft2 range")
        return complex(f_hat[m_idx, n_idx]) / float(self.Nx * self.Ny)
