"""
validation/self_force_2d.py
===========================

Single-particle self-force test for matched-gather verification.
"""

from __future__ import annotations

import numpy as np

from pic_kde_2d.conventions import SELF_FORCE_TOL
from pic_kde_2d.core.grid2d import Grid2D
from pic_kde_2d.core.particles2d import Particles2D


def test_self_force(deposit_fn, gather_fn, grid: Grid2D, x_p=None, y_p=None, q=1.0) -> dict:
    """Deposit one particle, solve Poisson, gather, and measure the self-force."""
    if x_p is None:
        x_p = grid.Lx / 2.0 + 0.3 * grid.dx
    if y_p is None:
        y_p = grid.Ly / 2.0 + 0.7 * grid.dy
    particles = Particles2D(
        x=np.array([x_p], dtype=np.float64),
        y=np.array([y_p], dtype=np.float64),
        vx=np.array([0.0], dtype=np.float64),
        vy=np.array([0.0], dtype=np.float64),
        w=np.array([q], dtype=np.float64),
        Lx=grid.Lx,
        Ly=grid.Ly,
    )
    rho = deposit_fn(particles, grid)
    Ex, Ey = grid.poisson_solve(rho)
    Ex_p, Ey_p = gather_fn(Ex, Ey, particles, grid)
    magnitude = float(np.hypot(Ex_p[0], Ey_p[0]))
    return {
        "Ex_p": float(Ex_p[0]),
        "Ey_p": float(Ey_p[0]),
        "magnitude": magnitude,
        "passes": bool(magnitude < SELF_FORCE_TOL),
    }


# This is a reusable diagnostic function, not a standalone pytest test.  It is
# imported into several test modules, where pytest would otherwise collect it
# and try to resolve its function arguments as fixtures.
test_self_force.__test__ = False
