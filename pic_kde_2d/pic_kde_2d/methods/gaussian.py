"""
methods/gaussian.py
===================

Anisotropic separable 2D Gaussian deposit and gather.

Kernel:
    K(x - x_p, y - y_p; sigma_x, sigma_y) =
        exp(-0.5*(dx/sigma_x)^2 - 0.5*(dy/sigma_y)^2)

We truncate at radius 3*sigma in each direction and normalize each
particle stencil so the weights sum to 1 exactly.
"""

from __future__ import annotations

import numpy as np


def _gaussian_radii(sigma_x: float, sigma_y: float, dx: float, dy: float) -> tuple[int, int]:
    if sigma_x <= 0.0 or sigma_y <= 0.0:
        raise ValueError("sigma_x and sigma_y must be positive")
    Rx = max(1, int(np.ceil(3.0 * sigma_x / dx)))
    Ry = max(1, int(np.ceil(3.0 * sigma_y / dy)))
    return Rx, Ry


def _gaussian_chunk_stencil(
    x_chunk: np.ndarray,
    y_chunk: np.ndarray,
    sigma_x: float,
    sigma_y: float,
    Nx: int,
    Ny: int,
    dx: float,
    dy: float,
    Rx: int,
    Ry: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    offsets_x = np.arange(-Rx, Rx + 1, dtype=np.int64)
    offsets_y = np.arange(-Ry, Ry + 1, dtype=np.int64)

    i_c = np.floor(x_chunk / dx).astype(np.int64)
    j_c = np.floor(y_chunk / dy).astype(np.int64)

    ii_unwrapped = i_c[:, None] + offsets_x[None, :]
    jj_unwrapped = j_c[:, None] + offsets_y[None, :]

    x_nodes = ii_unwrapped.astype(np.float64) * dx
    y_nodes = jj_unwrapped.astype(np.float64) * dy

    wx = np.exp(-0.5 * ((x_nodes - x_chunk[:, None]) / sigma_x) ** 2)
    wy = np.exp(-0.5 * ((y_nodes - y_chunk[:, None]) / sigma_y) ** 2)
    weights = wx[:, :, None] * wy[:, None, :]
    weights /= np.sum(weights, axis=(1, 2), keepdims=True)

    ii = ii_unwrapped % Nx
    jj = jj_unwrapped % Ny
    flat_indices = ii[:, :, None] * Ny + jj[:, None, :]
    return flat_indices.astype(np.int64), ii.astype(np.int64), weights.astype(np.float64)


def _gaussian_indices_and_weights(
    x_chunk: np.ndarray,
    y_chunk: np.ndarray,
    sigma_x: float,
    sigma_y: float,
    Nx: int,
    Ny: int,
    dx: float,
    dy: float,
    Rx: int,
    Ry: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    offsets_x = np.arange(-Rx, Rx + 1, dtype=np.int64)
    offsets_y = np.arange(-Ry, Ry + 1, dtype=np.int64)

    i_c = np.floor(x_chunk / dx).astype(np.int64)
    j_c = np.floor(y_chunk / dy).astype(np.int64)
    ii_unwrapped = i_c[:, None] + offsets_x[None, :]
    jj_unwrapped = j_c[:, None] + offsets_y[None, :]
    x_nodes = ii_unwrapped.astype(np.float64) * dx
    y_nodes = jj_unwrapped.astype(np.float64) * dy
    wx = np.exp(-0.5 * ((x_nodes - x_chunk[:, None]) / sigma_x) ** 2)
    wy = np.exp(-0.5 * ((y_nodes - y_chunk[:, None]) / sigma_y) ** 2)
    weights = wx[:, :, None] * wy[:, None, :]
    weights /= np.sum(weights, axis=(1, 2), keepdims=True)
    ii = ii_unwrapped % Nx
    jj = jj_unwrapped % Ny
    return ii.astype(np.int64), jj.astype(np.int64), weights.astype(np.float64)


def deposit_gaussian_2d_aniso(
    x,
    y,
    w,
    sigma_x,
    sigma_y,
    Nx,
    Ny,
    dx,
    dy,
    chunk_size: int = 2048,
):
    """Anisotropic Gaussian deposit returning density per area."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    Rx, Ry = _gaussian_radii(sigma_x, sigma_y, dx, dy)
    rho_flat = np.zeros(Nx * Ny, dtype=np.float64)
    n_particles = x.shape[0]
    for start in range(0, n_particles, chunk_size):
        stop = min(start + chunk_size, n_particles)
        flat_indices, _, weights = _gaussian_chunk_stencil(
            x[start:stop],
            y[start:stop],
            sigma_x,
            sigma_y,
            Nx,
            Ny,
            dx,
            dy,
            Rx,
            Ry,
        )
        contrib = (w[start:stop, None, None] * weights / (dx * dy)).ravel()
        np.add.at(rho_flat, flat_indices.ravel(), contrib)
    return rho_flat.reshape((Nx, Ny))


def gather_gaussian_2d_aniso(
    Ex,
    Ey,
    x,
    y,
    sigma_x,
    sigma_y,
    Nx,
    Ny,
    dx,
    dy,
    chunk_size: int = 2048,
):
    """Anisotropic Gaussian gather matched to the deposit weights."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    Ex = np.asarray(Ex, dtype=np.float64)
    Ey = np.asarray(Ey, dtype=np.float64)
    Rx, Ry = _gaussian_radii(sigma_x, sigma_y, dx, dy)
    Ex_p = np.zeros_like(x)
    Ey_p = np.zeros_like(y)
    n_particles = x.shape[0]
    for start in range(0, n_particles, chunk_size):
        stop = min(start + chunk_size, n_particles)
        ii, jj, weights = _gaussian_indices_and_weights(
            x[start:stop],
            y[start:stop],
            sigma_x,
            sigma_y,
            Nx,
            Ny,
            dx,
            dy,
            Rx,
            Ry,
        )
        ex_vals = Ex[ii[:, :, None], jj[:, None, :]]
        ey_vals = Ey[ii[:, :, None], jj[:, None, :]]
        Ex_p[start:stop] = np.sum(ex_vals * weights, axis=(1, 2))
        Ey_p[start:stop] = np.sum(ey_vals * weights, axis=(1, 2))
    return Ex_p.astype(np.float64), Ey_p.astype(np.float64)
