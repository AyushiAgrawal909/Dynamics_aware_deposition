"""Least-squares leave-one-out CV for the binned CIC plus spectral-W KDE.

The CIC self contribution depends weakly on a particle's sub-cell phase.  A
fixed set of representative offsets is not accurate enough for shallow LSCV
curves, so this module averages the exact quadratic CIC self term over the
particles in each snapshot.
"""

from __future__ import annotations

import numpy as np

from pic_kde_2d.core.grid2d import Grid2D
from pic_kde_2d.methods.spectral_W import apply_W_to_grid


_CIC_NODE_OFFSETS = np.asarray(
    [(0, 0), (1, 0), (0, 1), (1, 1)],
    dtype=np.int64,
)


def cic_weight_second_moment(
    x: np.ndarray,
    y: np.ndarray,
    grid: Grid2D,
) -> np.ndarray:
    """Return ``mean_p[c_p c_p^T]`` for the four bilinear CIC weights."""
    fx = np.mod(np.asarray(x, dtype=np.float64) / grid.dx, 1.0)
    fy = np.mod(np.asarray(y, dtype=np.float64) / grid.dy, 1.0)
    weights = np.stack(
        (
            (1.0 - fx) * (1.0 - fy),
            fx * (1.0 - fy),
            (1.0 - fx) * fy,
            fx * fy,
        ),
        axis=1,
    )
    return (weights.T @ weights) / float(weights.shape[0])


def cic_filter_self_matrix(
    grid: Grid2D,
    W: np.ndarray | None,
) -> np.ndarray:
    """Return the four-by-four self-interaction matrix for ``CIC -> W``."""
    node_density = np.zeros((grid.Nx, grid.Ny), dtype=np.float64)
    node_density[0, 0] = 1.0 / (grid.dx * grid.dy)
    kernel = (
        node_density
        if W is None
        else apply_W_to_grid(node_density, W, grid)
    )
    matrix = np.empty((4, 4), dtype=np.float64)
    for a, offset_a in enumerate(_CIC_NODE_OFFSETS):
        for b, offset_b in enumerate(_CIC_NODE_OFFSETS):
            delta = offset_a - offset_b
            matrix[a, b] = kernel[delta[0] % grid.Nx, delta[1] % grid.Ny]
    return matrix


def mean_cic_filter_self(
    weight_second_moment: np.ndarray,
    self_matrix: np.ndarray,
) -> float:
    """Average exact per-particle ``CIC -> W -> CIC^T`` self contribution."""
    moment = np.asarray(weight_second_moment, dtype=np.float64)
    matrix = np.asarray(self_matrix, dtype=np.float64)
    if moment.shape != (4, 4) or matrix.shape != (4, 4):
        raise ValueError("CIC self-term inputs must both have shape (4, 4)")
    return float(np.sum(moment * matrix))


def lscv_score(
    rho_probability_cic: np.ndarray,
    W: np.ndarray | None,
    mean_self: float,
    n_particles: int,
    grid: Grid2D,
) -> float:
    """Evaluate the exact binned-estimator LSCV objective for one snapshot."""
    if n_particles < 2:
        raise ValueError("LSCV requires at least two particles")
    rho_smooth = (
        np.asarray(rho_probability_cic, dtype=np.float64)
        if W is None
        else apply_W_to_grid(rho_probability_cic, W, grid)
    )
    cell_area = grid.dx * grid.dy
    integrated_square = float(np.sum(rho_smooth * rho_smooth) * cell_area)
    empirical_evaluation = float(
        np.sum(rho_smooth * rho_probability_cic) * cell_area
    )
    leave_one_out_mean = (
        empirical_evaluation - mean_self / float(n_particles)
    ) * (float(n_particles) / float(n_particles - 1))
    return integrated_square - 2.0 * leave_one_out_mean
