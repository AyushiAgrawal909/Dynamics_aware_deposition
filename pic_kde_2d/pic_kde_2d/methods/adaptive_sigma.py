"""
Continuous anisotropic bandwidth fields for learned particle shapes.

The production controller predicts diagonal coefficients ``(a, c)`` and
inserts ``b=0``. This module retains the full three-coefficient tensor utility
for research and finite-difference verification. Coefficients, rather than
covariance matrices, are interpolated to a particle position. For
``ell = (a, b, c)``::

    L = [[exp(a), 0], [b, exp(c)]]
    H = L @ L.T

``H`` is therefore symmetric positive definite at every position.  Periodic
bilinear interpolation is C0 and has an analytic, piecewise-continuous spatial
derivative.  The derivative is the one consumed by the variational grad-h
force; a cellwise lookup with zero derivatives is deliberately not supported.
Rotated ``H`` is outside the current separable production stencil.

All coordinates and covariance entries use grid-cell units unless a caller
chooses another consistent coordinate system.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class BandwidthEvaluation:
    """Bandwidth tensor and spatial derivatives at one or more positions."""

    coefficients: np.ndarray
    cholesky: np.ndarray
    H: np.ndarray
    dH_dx: np.ndarray
    dH_dy: np.ndarray


@dataclass(frozen=True)
class ShapeDerivatives:
    """Normalized particle shape and its two force-side derivatives."""

    weights: np.ndarray
    dweights_main: np.ndarray
    dweights_grad_h: np.ndarray

    @property
    def dweights_total(self) -> np.ndarray:
        return self.dweights_main + self.dweights_grad_h


def diagonal_log_cholesky_coefficients(coefficients: np.ndarray) -> np.ndarray:
    """Promote production ``(..., 2)`` diagonal heads to ``(..., 3)`` form."""
    diagonal = np.asarray(coefficients, dtype=np.float64)
    if diagonal.shape[-1:] != (2,) or np.any(~np.isfinite(diagonal)):
        raise ValueError("diagonal coefficients must have trailing shape (2,) and be finite")
    result = np.zeros(diagonal.shape[:-1] + (3,), dtype=np.float64)
    result[..., 0] = diagonal[..., 0]
    result[..., 2] = diagonal[..., 1]
    return result


def abramson_c4_bandwidth(
    marker_density: np.ndarray,
    sigma_pilot: np.ndarray,
    sigma_c4_max: np.ndarray,
    sigma_min: np.ndarray = 0.0,
    density_floor: Optional[float] = None,
) -> np.ndarray:
    """Abramson square-root bandwidth with a hard C4 dynamics ceiling.

    ``h(x) = h_pilot * sqrt(g / n_marker(x))``, where ``g`` is the
    geometric mean marker density. The result is clipped componentwise to
    ``[sigma_min, sigma_c4_max]``. Scalar widths are promoted to both axes;
    two-component inputs return an anisotropic diagonal bandwidth pair.
    """
    density = np.asarray(marker_density, dtype=np.float64)
    if density.size == 0 or np.any(~np.isfinite(density)) or np.any(density < 0.0):
        raise ValueError("marker_density must be a nonempty, finite, nonnegative array")
    positive = density[density > 0.0]
    if positive.size == 0:
        raise ValueError("marker_density must contain a positive value")
    floor = (
        max(float(np.min(positive)) * 1.0e-6, np.finfo(np.float64).tiny)
        if density_floor is None
        else float(density_floor)
    )
    if not np.isfinite(floor) or floor <= 0.0:
        raise ValueError("density_floor must be finite and positive")
    safe_density = np.maximum(density, floor)
    geometric_mean = float(np.exp(np.mean(np.log(safe_density))))
    scale = np.sqrt(geometric_mean / safe_density)

    def pair(value: np.ndarray, name: str) -> np.ndarray:
        array = np.asarray(value, dtype=np.float64)
        if array.ndim == 0:
            array = np.repeat(array, 2)
        if array.shape != (2,) or np.any(~np.isfinite(array)):
            raise ValueError(f"{name} must be a finite scalar or length-two vector")
        return array

    pilot = pair(sigma_pilot, "sigma_pilot")
    cap = pair(sigma_c4_max, "sigma_c4_max")
    lower = pair(sigma_min, "sigma_min")
    if np.any(pilot <= 0.0) or np.any(cap <= 0.0) or np.any(lower < 0.0):
        raise ValueError("pilot/cap must be positive and sigma_min nonnegative")
    if np.any(lower > cap):
        raise ValueError("sigma_min cannot exceed the C4 ceiling")
    raw = scale[..., None] * pilot
    return np.minimum(np.maximum(raw, lower), cap)


def log_cholesky_to_matrix(coefficients: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Convert ``(..., 3)`` log-Cholesky coefficients to ``L`` and ``H``."""
    ell = np.asarray(coefficients, dtype=np.float64)
    if ell.shape[-1:] != (3,):
        raise ValueError(f"coefficients must have trailing shape (3,), got {ell.shape}")
    if not np.all(np.isfinite(ell)):
        raise ValueError("log-Cholesky coefficients must be finite")

    a, b, c = np.moveaxis(ell, -1, 0)
    ea = np.exp(a)
    ec = np.exp(c)
    L = np.zeros(ell.shape[:-1] + (2, 2), dtype=np.float64)
    L[..., 0, 0] = ea
    L[..., 1, 0] = b
    L[..., 1, 1] = ec
    H = L @ np.swapaxes(L, -1, -2)
    return L, H


def log_cholesky_jacobian(coefficients: np.ndarray) -> np.ndarray:
    """Return ``dH/d(a,b,c)`` with shape ``(..., 2, 2, 3)``."""
    ell = np.asarray(coefficients, dtype=np.float64)
    if ell.shape[-1:] != (3,):
        raise ValueError(f"coefficients must have trailing shape (3,), got {ell.shape}")
    a, b, c = np.moveaxis(ell, -1, 0)
    ea = np.exp(a)
    ec = np.exp(c)
    jac = np.zeros(ell.shape[:-1] + (2, 2, 3), dtype=np.float64)

    jac[..., 0, 0, 0] = 2.0 * ea * ea
    jac[..., 0, 1, 0] = b * ea
    jac[..., 1, 0, 0] = b * ea

    jac[..., 0, 1, 1] = ea
    jac[..., 1, 0, 1] = ea
    jac[..., 1, 1, 1] = 2.0 * b

    jac[..., 1, 1, 2] = 2.0 * ec * ec
    return jac


class PeriodicLogCholeskyField:
    """Periodic C0 field formed by bilinear interpolation of nodal coefficients."""

    def __init__(self, nodal_coefficients: np.ndarray, Lx: float, Ly: float):
        coefficients = np.asarray(nodal_coefficients, dtype=np.float64)
        if coefficients.ndim != 3 or coefficients.shape[-1] != 3:
            raise ValueError(
                "nodal_coefficients must have shape (controller_nx, controller_ny, 3)"
            )
        if coefficients.shape[0] < 2 or coefficients.shape[1] < 2:
            raise ValueError("controller mesh requires at least 2 nodes per direction")
        if not np.all(np.isfinite(coefficients)):
            raise ValueError("nodal_coefficients must be finite")
        if not np.isfinite(Lx) or not np.isfinite(Ly) or Lx <= 0.0 or Ly <= 0.0:
            raise ValueError("Lx and Ly must be finite and positive")

        self.nodal_coefficients = coefficients.copy()
        self.Lx = float(Lx)
        self.Ly = float(Ly)
        self.Nx, self.Ny = coefficients.shape[:2]
        self.dx = self.Lx / self.Nx
        self.dy = self.Ly / self.Ny

    def evaluate(self, x: np.ndarray, y: np.ndarray) -> BandwidthEvaluation:
        """Evaluate ``H`` and its analytic spatial derivative at ``(x, y)``."""
        x_arr, y_arr = np.broadcast_arrays(
            np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
        )
        if not np.all(np.isfinite(x_arr)) or not np.all(np.isfinite(y_arr)):
            raise ValueError("particle positions must be finite")

        ux = np.mod(x_arr, self.Lx) / self.dx
        uy = np.mod(y_arr, self.Ly) / self.dy
        i0_unwrapped = np.floor(ux).astype(np.int64)
        j0_unwrapped = np.floor(uy).astype(np.int64)
        fx = ux - i0_unwrapped
        fy = uy - j0_unwrapped
        i0 = i0_unwrapped % self.Nx
        j0 = j0_unwrapped % self.Ny
        i1 = (i0 + 1) % self.Nx
        j1 = (j0 + 1) % self.Ny

        c00 = self.nodal_coefficients[i0, j0]
        c10 = self.nodal_coefficients[i1, j0]
        c01 = self.nodal_coefficients[i0, j1]
        c11 = self.nodal_coefficients[i1, j1]
        wx0 = (1.0 - fx)[..., None]
        wx1 = fx[..., None]
        wy0 = (1.0 - fy)[..., None]
        wy1 = fy[..., None]

        ell = wx0 * wy0 * c00 + wx1 * wy0 * c10 + wx0 * wy1 * c01 + wx1 * wy1 * c11
        dell_dx = (wy0 * (c10 - c00) + wy1 * (c11 - c01)) / self.dx
        dell_dy = (wx0 * (c01 - c00) + wx1 * (c11 - c10)) / self.dy

        L, H = log_cholesky_to_matrix(ell)
        jac = log_cholesky_jacobian(ell)
        dH_dx = np.einsum("...ijp,...p->...ij", jac, dell_dx)
        dH_dy = np.einsum("...ijp,...p->...ij", jac, dell_dy)
        return BandwidthEvaluation(ell, L, H, dH_dx, dH_dy)


def diagonal_bandwidth_adapter(
    evaluation: BandwidthEvaluation,
    off_diagonal_tolerance: float = 1.0e-12,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Map a diagonal ``H`` to the separable HEXAPIC ``h``/gradient interface.

    A rotated covariance cannot be represented by the current separable KDE
    kernel.  Refuse it instead of silently throwing away ``Hxy``.
    """
    H = np.asarray(evaluation.H)
    scale = np.maximum(np.sqrt(H[..., 0, 0] * H[..., 1, 1]), 1.0)
    if np.any(np.abs(H[..., 0, 1]) > off_diagonal_tolerance * scale):
        raise ValueError(
            "the production separable kernel cannot represent nonzero Hxy; "
            "use a diagonal controller or implement the rotated Gaussian stencil"
        )
    hx = np.sqrt(H[..., 0, 0])
    hy = np.sqrt(H[..., 1, 1])
    dhx_dx = evaluation.dH_dx[..., 0, 0] / (2.0 * hx)
    dhx_dy = evaluation.dH_dy[..., 0, 0] / (2.0 * hx)
    dhy_dx = evaluation.dH_dx[..., 1, 1] / (2.0 * hy)
    dhy_dy = evaluation.dH_dy[..., 1, 1] / (2.0 * hy)
    return hx, hy, dhx_dx, dhx_dy, dhy_dx, dhy_dy


def _minimum_image(displacements: np.ndarray, box_lengths: np.ndarray) -> np.ndarray:
    return displacements - np.round(displacements / box_lengths) * box_lengths


def normalized_gaussian_shape_derivatives(
    node_positions: np.ndarray,
    particle_position: np.ndarray,
    evaluation: BandwidthEvaluation,
    box_lengths: Optional[np.ndarray] = None,
) -> ShapeDerivatives:
    """Evaluate a normalized Gaussian stencil and its full position derivative.

    ``dweights_main`` differentiates the translated kernel while holding ``H``
    fixed. ``dweights_grad_h`` is the additional chain-rule contribution from
    ``H(x_p)``. Their sum is the derivative required by the variational force.
    Per-particle normalization and its derivative are included exactly.
    """
    nodes = np.asarray(node_positions, dtype=np.float64)
    position = np.asarray(particle_position, dtype=np.float64)
    if nodes.ndim != 2 or nodes.shape[1] != 2 or position.shape != (2,):
        raise ValueError("node_positions must be (n,2) and particle_position must be (2,)")
    H = np.asarray(evaluation.H, dtype=np.float64)
    dH_dx = np.asarray(evaluation.dH_dx, dtype=np.float64)
    dH_dy = np.asarray(evaluation.dH_dy, dtype=np.float64)
    if H.shape != (2, 2) or dH_dx.shape != (2, 2) or dH_dy.shape != (2, 2):
        raise ValueError("evaluation must describe exactly one particle")

    r = nodes - position[None, :]
    if box_lengths is not None:
        lengths = np.asarray(box_lengths, dtype=np.float64)
        if lengths.shape != (2,) or np.any(lengths <= 0.0):
            raise ValueError("box_lengths must contain two positive values")
        r = _minimum_image(r, lengths)

    H_inv = np.linalg.inv(H)
    projected = r @ H_inv.T
    exponent = -0.5 * np.einsum("ni,ni->n", r, projected)
    raw = np.exp(exponent - np.max(exponent)) / np.sqrt(np.linalg.det(H))
    norm = float(np.sum(raw))
    if not np.isfinite(norm) or norm <= 0.0:
        raise FloatingPointError("Gaussian stencil normalization is not positive")

    # r = x_g - x_p, so the fixed-H derivative wrt x_p is +H^-1 r.
    raw_main = raw[:, None] * projected
    dG_dH = 0.5 * raw[:, None, None] * (
        projected[:, :, None] * projected[:, None, :] - H_inv[None, :, :]
    )
    raw_grad_h = np.column_stack(
        [
            np.einsum("nij,ij->n", dG_dH, dH_dx),
            np.einsum("nij,ij->n", dG_dH, dH_dy),
        ]
    )

    weights = raw / norm
    dnorm_main = np.sum(raw_main, axis=0)
    dnorm_grad_h = np.sum(raw_grad_h, axis=0)
    dweights_main = (raw_main - weights[:, None] * dnorm_main[None, :]) / norm
    dweights_grad_h = (
        raw_grad_h - weights[:, None] * dnorm_grad_h[None, :]
    ) / norm
    return ShapeDerivatives(weights, dweights_main, dweights_grad_h)
