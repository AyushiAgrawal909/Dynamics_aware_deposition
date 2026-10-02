"""
methods/cic.py
==============

2D Cloud-in-Cell (bilinear) deposition and gather.
This is the baseline against which all other methods are compared.

Mathematical specification:
    Particle at (x_p, y_p) deposits to 4 surrounding grid nodes with bilinear weights.
    Charge is exactly conserved: sum of the 4 weights equals 1.
    Matched gather uses the same 4 weights.
"""

from __future__ import annotations

import numpy as np


def deposit_cic_2d(x, y, w, Nx, Ny, dx, dy):
    """CIC deposit: particles -> density on grid."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)

    xp_dx = x / dx
    yp_dy = y / dy
    i = np.floor(xp_dx).astype(np.int64)
    j = np.floor(yp_dy).astype(np.int64)
    fx = xp_dx - i
    fy = yp_dy - j
    i0 = i % Nx
    j0 = j % Ny
    i1 = (i0 + 1) % Nx
    j1 = (j0 + 1) % Ny

    c00 = w * (1.0 - fx) * (1.0 - fy)
    c10 = w * fx * (1.0 - fy)
    c01 = w * (1.0 - fx) * fy
    c11 = w * fx * fy

    idx00 = i0 * Ny + j0
    idx10 = i1 * Ny + j0
    idx01 = i0 * Ny + j1
    idx11 = i1 * Ny + j1

    flat_idx = np.concatenate([idx00, idx10, idx01, idx11])
    flat_w = np.concatenate([c00, c10, c01, c11])
    rho = np.bincount(flat_idx, weights=flat_w, minlength=Nx * Ny).reshape((Nx, Ny))
    return rho / (dx * dy)


def gather_cic_2d(Ex, Ey, x, y, Nx, Ny, dx, dy):
    """CIC gather: grid field -> field at particle positions using matched weights."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    xp_dx = x / dx
    yp_dy = y / dy
    i = np.floor(xp_dx).astype(np.int64)
    j = np.floor(yp_dy).astype(np.int64)
    fx = xp_dx - i
    fy = yp_dy - j
    i0 = i % Nx
    j0 = j % Ny
    i1 = (i0 + 1) % Nx
    j1 = (j0 + 1) % Ny

    w00 = (1.0 - fx) * (1.0 - fy)
    w10 = fx * (1.0 - fy)
    w01 = (1.0 - fx) * fy
    w11 = fx * fy

    Ex_p = Ex[i0, j0] * w00 + Ex[i1, j0] * w10 + Ex[i0, j1] * w01 + Ex[i1, j1] * w11
    Ey_p = Ey[i0, j0] * w00 + Ey[i1, j0] * w10 + Ey[i0, j1] * w01 + Ey[i1, j1] * w11
    return Ex_p.astype(np.float64), Ey_p.astype(np.float64)
