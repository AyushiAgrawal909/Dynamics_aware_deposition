"""
methods/cic_wrappers.py
=======================

Method-pluggable wrappers around the CIC kernels.
The wrappers use a uniform signature so they can be passed to the PIC driver.
"""

from __future__ import annotations

from pic_kde_2d.methods.cic import deposit_cic_2d, gather_cic_2d


def cic_deposit_method(particles, grid):
    """Wrapper matching the signature expected by run_simulation."""
    return deposit_cic_2d(
        particles.x,
        particles.y,
        particles.w,
        grid.Nx,
        grid.Ny,
        grid.dx,
        grid.dy,
    )


def cic_gather_method(Ex, Ey, particles, grid):
    """Wrapper matching the signature expected by run_simulation."""
    return gather_cic_2d(
        Ex,
        Ey,
        particles.x,
        particles.y,
        grid.Nx,
        grid.Ny,
        grid.dx,
        grid.dy,
    )
