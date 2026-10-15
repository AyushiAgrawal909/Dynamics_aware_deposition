"""Periodic B-spline charge deposition (NGP, CIC, TSC, PQS) in 1-3D.

Use it to turn saved particle states into mesh densities for the snapshot
objective when your code does not save the deposited density itself.  Nodes
sit at x_g = g dx; a shift of this origin only changes Fourier phases and
leaves the snapshot objective unchanged as long as ``rho`` and ``rho_ref``
are deposited the same way.
"""

from __future__ import annotations

import numpy as np

from .grid import Grid, shape_order


def _weights_1d(x: np.ndarray, order: int):
    """Node offsets (relative to a base index) and weights, shape (order+1, Np)."""
    if order == 0:
        i = np.floor(x + 0.5).astype(np.int64)
        return i, np.array([0]), np.ones((1, x.size))
    if order == 1:
        i = np.floor(x).astype(np.int64)
        f = x - i
        return i, np.array([0, 1]), np.stack([1.0 - f, f])
    if order == 2:
        i = np.floor(x + 0.5).astype(np.int64)
        d = x - i
        return i, np.array([-1, 0, 1]), np.stack([0.5 * (0.5 - d) ** 2, 0.75 - d * d,
                                                  0.5 * (0.5 + d) ** 2])
    if order == 3:
        i = np.floor(x).astype(np.int64)
        f = x - i
        return i, np.array([-1, 0, 1, 2]), np.stack([
            (1.0 - f) ** 3 / 6.0, (4.0 - 6.0 * f**2 + 3.0 * f**3) / 6.0,
            (1.0 + 3.0 * f + 3.0 * f**2 - 3.0 * f**3) / 6.0, f**3 / 6.0])
    raise ValueError("order must be 0..3")


def deposit(positions: np.ndarray, charges: np.ndarray | float, grid: Grid,
            shape: str | int | None = None) -> np.ndarray:
    """Charge density on the periodic mesh of ``grid``.

    Parameters
    ----------
    positions
        Marker positions, shape (Np, dim) (or (Np,) in 1D).
    charges
        Charge of each marker (array of length Np or a scalar).
    shape
        Particle shape; default: the grid's shape.
    """
    order = grid.order if shape is None else shape_order(shape)
    x = np.asarray(positions, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    if x.shape[1] != grid.dim:
        raise ValueError(f"positions have {x.shape[1]} components for a {grid.dim}D grid")
    q = np.broadcast_to(np.asarray(charges, dtype=float), (x.shape[0],))
    flat = np.zeros(grid.n_cells)
    per_axis = [_weights_1d(x[:, d] / grid.dx[d], order) for d in range(grid.dim)]
    strides = np.cumprod((1,) + tuple(grid.N[::-1]))[:-1][::-1]
    from itertools import product
    for combo in product(*[range(order + 1)] * grid.dim):
        idx = np.zeros(x.shape[0], dtype=np.int64)
        w = q.copy()
        for d, c in enumerate(combo):
            base, offs, wts = per_axis[d]
            idx += ((base + offs[c]) % grid.N[d]) * strides[d]
            w = w * wts[c]
        flat += np.bincount(idx, weights=w, minlength=grid.n_cells)
    return flat.reshape(grid.N) / float(np.prod(grid.dx))
