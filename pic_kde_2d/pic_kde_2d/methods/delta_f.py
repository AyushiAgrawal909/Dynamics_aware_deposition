"""
methods/delta_f.py
==================

Two-weight delta-f control variates for reduced-noise electrostatic PIC.

The uniform Maxwellian equilibrium is handled analytically and cancels the
immobile ion background. Particles therefore deposit only their perturbation
charge, ``w_delta``, while their signed marker charge, ``w_full``, is retained
for weight evolution and optional fallback to noisy full-f deposition.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from pic_kde_2d.methods.cic import deposit_cic_2d, gather_cic_2d
from pic_kde_2d.methods.spectral_W import apply_W_to_grid


@dataclass
class ParticlesDeltaF:
    """Structure-of-arrays marker storage for a delta-f simulation."""

    x: np.ndarray
    y: np.ndarray
    vx: np.ndarray
    vy: np.ndarray
    w_full: np.ndarray
    w_delta: np.ndarray
    Lx: float
    Ly: float
    f0_v_thermal: float
    f0_density: float
    use_full_f: bool = False
    background_kind: str = "maxwellian"
    beam_velocity: float = 0.0

    @property
    def n_particles(self) -> int:
        return len(self.x)

    def wrap(self) -> None:
        self.x = self.x % self.Lx
        self.y = self.y % self.Ly

    def kinetic_energy_perturbation(self) -> float:
        """Signed perturbation kinetic-energy moment."""
        return 0.5 * float(np.sum(self.w_delta * (self.vx**2 + self.vy**2)))

    def deposited_charge(self) -> float:
        """Charge represented by the live deposited particle weights."""
        if self.use_full_f:
            return float(np.sum(self.w_full + self.w_delta))
        return float(np.sum(self.w_delta))


def init_delta_f_from_particles(
    p,
    f0_v_thermal: float,
    f0_density: float,
    alpha: float = 0.0,
    kx_phys: float = 0.0,
    ky_phys: float = 0.0,
    omega_real: Optional[float] = None,
    gamma: Optional[float] = None,
    background_kind: str = "maxwellian",
    beam_velocity: float = 0.0,
) -> ParticlesDeltaF:
    """Create delta-f markers from uniform equilibrium particles.

    The input particles must be loaded from ``f0`` without a positional
    perturbation. For consistency with the existing signed-electron baseline,
    ``alpha*cos(phase)`` creates the same positive charge perturbation produced
    by the full-f Landau displacement.

    When ``omega_real`` and ``gamma`` are supplied, the initial weights also
    encode the linear velocity-shift component of the validated eigenmode
    initializer; markers themselves remain sampled from ``f0``.
    """
    if f0_v_thermal <= 0.0:
        raise ValueError("f0_v_thermal must be positive")
    if background_kind not in {"maxwellian", "two_stream"}:
        raise ValueError(f"unsupported delta-f background: {background_kind}")
    if background_kind == "two_stream" and beam_velocity <= 0.0:
        raise ValueError("two_stream background requires a positive beam_velocity")
    phase = kx_phys * p.x + ky_phys * p.y
    marker_charge = np.asarray(p.w, dtype=np.float64)

    # Number-density perturbation is -alpha*cos; multiply by negative electron
    # marker charge to obtain the positive perturbed charge convention used here.
    delta_ratio = -float(alpha) * np.cos(phase)
    k_mag = float(np.hypot(kx_phys, ky_phys))
    if omega_real is not None:
        if k_mag == 0.0:
            raise ValueError("nonzero wave number is required for eigenmode initialization")
        damping = 0.0 if gamma is None else float(gamma)
        u_parallel = float(alpha) * (
            damping * np.sin(phase) - float(omega_real) * np.cos(phase)
        ) / k_mag
        v_parallel = (kx_phys * p.vx + ky_phys * p.vy) / k_mag
        delta_ratio += u_parallel * v_parallel / (f0_v_thermal**2)

    w_delta = marker_charge * delta_ratio
    # The perturbation is charge neutral analytically; remove finite-marker
    # quadrature residue before the first diagnostic sample.
    w_delta -= np.mean(w_delta)

    return ParticlesDeltaF(
        x=np.asarray(p.x, dtype=np.float64).copy(),
        y=np.asarray(p.y, dtype=np.float64).copy(),
        vx=np.asarray(p.vx, dtype=np.float64).copy(),
        vy=np.asarray(p.vy, dtype=np.float64).copy(),
        w_full=marker_charge.copy(),
        w_delta=w_delta,
        Lx=float(p.Lx),
        Ly=float(p.Ly),
        f0_v_thermal=float(f0_v_thermal),
        f0_density=float(f0_density),
        background_kind=background_kind,
        beam_velocity=float(beam_velocity),
    )


def background_velocity_score(p: ParticlesDeltaF) -> tuple[np.ndarray, np.ndarray]:
    """Return the exact stable velocity score ``grad_v log(f0)``."""
    inv_vth_sq = 1.0 / (p.f0_v_thermal**2)
    if p.background_kind == "maxwellian":
        return -p.vx * inv_vth_sq, -p.vy * inv_vth_sq
    if p.background_kind == "two_stream":
        # Equal bi-Maxwellian mixture centered at +/- beam_velocity. The tanh
        # form is algebraically identical to a log-sum-exp responsibility and
        # remains finite where direct division by f0 would underflow.
        conditional_center = p.beam_velocity * np.tanh(
            p.vx * p.beam_velocity * inv_vth_sq
        )
        return (
            -(p.vx - conditional_center) * inv_vth_sq,
            -p.vy * inv_vth_sq,
        )
    raise ValueError(f"unsupported delta-f background: {p.background_kind}")


def delta_weight_rhs(
    p: ParticlesDeltaF,
    Ex_p: np.ndarray,
    Ey_p: np.ndarray,
    q_over_m: float,
) -> np.ndarray:
    """Linearized Vlasov source for signed perturbation weights."""
    score_x, score_y = background_velocity_score(p)
    field_dot_score = np.asarray(Ex_p) * score_x + np.asarray(Ey_p) * score_y
    return -float(q_over_m) * field_dot_score * p.w_full


def evolve_delta_weights(
    p: ParticlesDeltaF,
    Ex_p: np.ndarray,
    Ey_p: np.ndarray,
    q_over_m: float,
    dt: float,
    enforce_neutrality: bool = True,
) -> None:
    """Advance Maxwellian delta weights for one time increment."""
    if p.use_full_f:
        return
    p.w_delta += float(dt) * delta_weight_rhs(p, Ex_p, Ey_p, q_over_m)
    if enforce_neutrality:
        # Total perturbed charge is an exact invariant in the periodic neutral
        # system; remove roundoff and finite-marker violation of its DC mode.
        p.w_delta -= np.mean(p.w_delta)


def deposit_delta_f_2d(p: ParticlesDeltaF, grid) -> np.ndarray:
    """Deposit perturbation charge, or reconstructed full-f after fallback."""
    weights = p.w_full + p.w_delta if p.use_full_f else p.w_delta
    return deposit_cic_2d(p.x, p.y, weights, grid.Nx, grid.Ny, grid.dx, grid.dy)


def make_delta_f_methods(W: Optional[np.ndarray] = None):
    """Build matched deposit/gather callables for delta-f markers."""
    filter_array = None if W is None else np.asarray(W, dtype=np.float64)

    def deposit(particles: ParticlesDeltaF, grid):
        rho = deposit_delta_f_2d(particles, grid)
        return apply_W_to_grid(rho, filter_array, grid) if filter_array is not None else rho

    def gather(Ex, Ey, particles: ParticlesDeltaF, grid):
        if filter_array is not None:
            Ex = apply_W_to_grid(Ex, filter_array, grid)
            Ey = apply_W_to_grid(Ey, filter_array, grid)
        return gather_cic_2d(Ex, Ey, particles.x, particles.y, grid.Nx, grid.Ny, grid.dx, grid.dy)

    return deposit, gather


def check_delta_f_validity(p: ParticlesDeltaF, threshold: float = 0.5) -> dict:
    """Return the largest perturbation-to-marker weight ratio and validity."""
    max_delta = float(np.max(np.abs(p.w_delta))) if p.w_delta.size else 0.0
    mean_full = float(np.mean(np.abs(p.w_full))) if p.w_full.size else 0.0
    ratio = max_delta / (mean_full + 1.0e-30)
    return {
        "max_w_delta": max_delta,
        "mean_w_full": mean_full,
        "ratio": ratio,
        "valid": bool(ratio < threshold),
    }


def activate_full_f_fallback(p: ParticlesDeltaF) -> None:
    """Switch subsequent deposition to the noisy reconstructed full-f weights."""
    p.use_full_f = True
