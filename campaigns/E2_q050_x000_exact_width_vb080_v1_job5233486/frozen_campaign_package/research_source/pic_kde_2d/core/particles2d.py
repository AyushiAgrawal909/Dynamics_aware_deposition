"""
core/particles2d.py
===================

2D particle storage and initialization helpers.

Structure-of-arrays layout: x, y, vx, vy, w are separate 1D arrays.
This is required for efficient numerical kernels.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


def _inverse_normal_cdf(u: np.ndarray) -> np.ndarray:
    """
    Fast inverse CDF for the standard normal using Acklam's rational approximation.

    This avoids a hard SciPy dependency in the meluxina runtime.
    """
    a = np.array(
        [
            -3.969683028665376e01,
            2.209460984245205e02,
            -2.759285104469687e02,
            1.383577518672690e02,
            -3.066479806614716e01,
            2.506628277459239e00,
        ]
    )
    b = np.array(
        [
            -5.447609879822406e01,
            1.615858368580409e02,
            -1.556989798598866e02,
            6.680131188771972e01,
            -1.328068155288572e01,
        ]
    )
    c = np.array(
        [
            -7.784894002430293e-03,
            -3.223964580411365e-01,
            -2.400758277161838e00,
            -2.549732539343734e00,
            4.374664141464968e00,
            2.938163982698783e00,
        ]
    )
    d = np.array(
        [
            7.784695709041462e-03,
            3.224671290700398e-01,
            2.445134137142996e00,
            3.754408661907416e00,
        ]
    )
    plow = 0.02425
    phigh = 1.0 - plow

    u = np.asarray(u, dtype=np.float64)
    if np.any(u <= 0.0) or np.any(u >= 1.0):
        raise ValueError("inverse normal CDF requires 0 < u < 1")

    x = np.empty_like(u)
    low = u < plow
    high = u > phigh
    mid = ~(low | high)

    if np.any(low):
        q = np.sqrt(-2.0 * np.log(u[low]))
        num = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
        den = ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
        x[low] = num / den
    if np.any(mid):
        q = u[mid] - 0.5
        r = q * q
        num = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
        den = (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
        x[mid] = num / den
    if np.any(high):
        q = np.sqrt(-2.0 * np.log(1.0 - u[high]))
        num = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
        den = ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
        x[high] = -(num / den)
    return x


@dataclass
class Particles2D:
    """2D electron particles (SoA layout)."""

    x: np.ndarray
    y: np.ndarray
    vx: np.ndarray
    vy: np.ndarray
    w: np.ndarray
    Lx: float
    Ly: float

    @property
    def n_particles(self) -> int:
        return len(self.x)

    def wrap(self) -> None:
        """Wrap positions into periodic box."""
        self.x = self.x % self.Lx
        self.y = self.y % self.Ly

    def kinetic_energy(self) -> float:
        """Total kinetic energy = sum (1/2) |w_p| v_p^2."""
        return 0.5 * float(np.sum(np.abs(self.w) * (self.vx**2 + self.vy**2)))

    def total_charge(self) -> float:
        return float(np.sum(self.w))


def quiet_start_maxwellian_2d(
    Nx_marker: int,
    Ny_marker: int,
    Lx: float,
    Ly: float,
    v_thermal: float,
    rng: np.random.Generator,
    grid_shape: Optional[Tuple[int, int]] = None,
    position_jitter: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Quiet-start Maxwellian in 2D using a stratified phase-space template.

    ``position_jitter=False`` retains the deterministic cell-centered marker
    lattice used by production runs. ``True`` draws one independent position
    inside each marker stratum, which is useful for estimating sampling
    uncertainty of position-only diagnostics such as static LSCV.
    """
    N_p = Nx_marker * Ny_marker
    marker_dx = Lx / Nx_marker
    marker_dy = Ly / Ny_marker
    base_x, base_y = np.meshgrid(
        np.arange(Nx_marker, dtype=np.float64),
        np.arange(Ny_marker, dtype=np.float64),
        indexing="ij",
    )
    if position_jitter:
        x = ((base_x + rng.random(base_x.shape)) * marker_dx).ravel()
        y = ((base_y + rng.random(base_y.shape)) * marker_dy).ravel()
    else:
        x = ((base_x + 0.5) * marker_dx).ravel()
        y = ((base_y + 0.5) * marker_dy).ravel()

    if grid_shape is not None:
        grid_nx, grid_ny = grid_shape
        if Nx_marker % grid_nx != 0 or Ny_marker % grid_ny != 0:
            raise ValueError("marker grid must be an integer refinement of grid_shape")
        px = Nx_marker // grid_nx
        py = Ny_marker // grid_ny
        ux = (np.arange(px, dtype=np.float64) + 0.5) / float(px)
        uy = (np.arange(py, dtype=np.float64) + 0.5) / float(py)
        vx_local, vy_local = np.meshgrid(
            v_thermal * _inverse_normal_cdf(ux),
            v_thermal * _inverse_normal_cdf(uy),
            indexing="ij",
        )
        vx_template = vx_local.ravel()
        vy_template = vy_local.ravel()
        vx_grid = np.empty((Nx_marker, Ny_marker), dtype=np.float64)
        vy_grid = np.empty((Nx_marker, Ny_marker), dtype=np.float64)

        # Keep exact per-cell quadrature while varying its sub-cell assignment.
        for cell_x in range(grid_nx):
            x_slice = slice(cell_x * px, (cell_x + 1) * px)
            for cell_y in range(grid_ny):
                y_slice = slice(cell_y * py, (cell_y + 1) * py)
                permutation = rng.permutation(px * py)
                vx_grid[x_slice, y_slice] = vx_template[permutation].reshape(px, py)
                vy_grid[x_slice, y_slice] = vy_template[permutation].reshape(px, py)
        return x, y, vx_grid.ravel(), vy_grid.ravel()

    u = (np.arange(N_p, dtype=np.float64) + 0.5) / N_p
    v_template = v_thermal * _inverse_normal_cdf(u)
    vx = v_template[rng.permutation(N_p)]
    vy = v_template[rng.permutation(N_p)]
    return x, y, vx, vy


def init_landau_2d(
    Nx_marker: int,
    Ny_marker: int,
    Lx: float,
    Ly: float,
    v_thermal: float,
    alpha: float,
    kx_mode: int,
    ky_mode: int = 0,
    rng: Optional[np.random.Generator] = None,
    velocity_perturbation: bool = True,
    omega_real: Optional[float] = None,
    gamma: Optional[float] = None,
    grid_shape: Optional[Tuple[int, int]] = None,
    position_jitter: bool = False,
) -> Particles2D:
    """Initialize Landau damping with a moment-matched density/current perturbation.

    The position and bulk-velocity perturbations satisfy the linear continuity
    relation for the requested complex Langmuir frequency. They are not the
    full velocity-dependent Vlasov eigenfunction, so a short projection
    transient remains and must be excluded from damping-rate fits.
    """
    if rng is None:
        rng = np.random.default_rng(0)

    x, y, vx, vy = quiet_start_maxwellian_2d(
        Nx_marker,
        Ny_marker,
        Lx,
        Ly,
        v_thermal,
        rng,
        grid_shape=grid_shape,
        position_jitter=position_jitter,
    )
    kx_phys = 2.0 * np.pi * kx_mode / Lx
    ky_phys = 2.0 * np.pi * ky_mode / Ly
    if kx_mode != 0 or ky_mode != 0:
        phase = kx_phys * x + ky_phys * y
        k_sq = kx_phys**2 + ky_phys**2
        x = (x + alpha * kx_phys * np.sin(phase) / k_sq) % Lx
        y = (y + alpha * ky_phys * np.sin(phase) / k_sq) % Ly

        if velocity_perturbation:
            if omega_real is None or gamma is None:
                from pic_kde_2d.benchmarks.theory import landau_gamma_theory, landau_omega_theory

                k_lambdaD = float(np.sqrt(k_sq))
                if omega_real is None:
                    omega_real = landau_omega_theory(k_lambdaD=k_lambdaD, v_thermal=v_thermal)
                if gamma is None:
                    gamma = landau_gamma_theory(k_lambdaD=k_lambdaD, v_thermal=v_thermal)

            # Match both first-order moments at the same Lagrangian marker
            # position. Re-evaluating the phase after displacement would add
            # an avoidable O(alpha**2) inconsistency.
            phase_velocity = phase
            k_mag = float(np.sqrt(k_sq))
            u_parallel = alpha * (float(gamma) * np.sin(phase_velocity) - float(omega_real) * np.cos(phase_velocity)) / k_mag
            vx = vx + u_parallel * (kx_phys / k_mag)
            vy = vy + u_parallel * (ky_phys / k_mag)

    N_p = Nx_marker * Ny_marker
    w = np.full(N_p, -Lx * Ly / N_p, dtype=np.float64)
    return Particles2D(x=x, y=y, vx=vx, vy=vy, w=w, Lx=Lx, Ly=Ly)


def init_landau_2d_eigenmode(
    Nx_marker: int,
    Ny_marker: int,
    Lx: float,
    Ly: float,
    v_thermal: float,
    alpha: float,
    kx_mode: int,
    omega_r: float,
    ky_mode: int = 0,
    rng: Optional[np.random.Generator] = None,
    grid_shape: Optional[Tuple[int, int]] = None,
    position_jitter: bool = False,
) -> Particles2D:
    """Initialize a moment-matched traveling Langmuir perturbation.

    The historical function name is retained for API compatibility. Its phase
    convention includes the damping component and is consistent with the
    signed electron push, but this is not an exact kinetic eigenfunction.
    """
    return init_landau_2d(
        Nx_marker=Nx_marker,
        Ny_marker=Ny_marker,
        Lx=Lx,
        Ly=Ly,
        v_thermal=v_thermal,
        alpha=alpha,
        kx_mode=kx_mode,
        ky_mode=ky_mode,
        rng=rng,
        velocity_perturbation=True,
        omega_real=omega_r,
        gamma=None,
        grid_shape=grid_shape,
        position_jitter=position_jitter,
    )


def init_two_stream_2d(
    Nx_marker: int,
    Ny_marker: int,
    Lx: float,
    Ly: float,
    v_beam: float,
    v_thermal: float,
    alpha: float,
    kx_mode: int,
    ky_mode: int = 0,
    rng: Optional[np.random.Generator] = None,
) -> Particles2D:
    """Initialize 2D two-stream instability with equal-density counter-streaming beams."""
    if rng is None:
        rng = np.random.default_rng(0)

    x, y, vx, vy = quiet_start_maxwellian_2d(Nx_marker, Ny_marker, Lx, Ly, v_thermal, rng)
    beam_sign = np.ones_like(vx)
    beam_sign[: beam_sign.size // 2] = -1.0
    beam_sign = beam_sign[rng.permutation(beam_sign.size)]
    vx = vx + beam_sign * v_beam

    kx_phys = 2.0 * np.pi * kx_mode / Lx
    ky_phys = 2.0 * np.pi * ky_mode / Ly
    if kx_mode != 0 or ky_mode != 0:
        phase = kx_phys * x + ky_phys * y
        k_sq = kx_phys**2 + ky_phys**2
        x = (x + alpha * kx_phys * np.sin(phase) / k_sq) % Lx
        y = (y + alpha * ky_phys * np.sin(phase) / k_sq) % Ly

    N_p = x.size
    w = np.full(N_p, -Lx * Ly / N_p, dtype=np.float64)
    return Particles2D(x=x, y=y, vx=vx, vy=vy, w=w, Lx=Lx, Ly=Ly)
