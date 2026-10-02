"""
methods/spectral_W.py
=====================

Post-deposition spectral filters W(kx, ky) applied to grid quantities.

Phase 3 uses cheap CIC deposition followed by FFT-space smoothing on the grid:
    rho_smooth = F^-1[ W * F[rho_cic] ]

The same filter W is also applied to the electric field before gather:
    E_smooth = F^-1[ W * F[E] ]

Using the same W for density and field preserves the matched-smoothing symmetry
needed for low self-force and clean momentum behavior.
"""

from __future__ import annotations

import numpy as np

from pic_kde_2d.core.grid2d import Grid2D


def make_W_gaussian(grid: Grid2D, sigma_x: float, sigma_y: float) -> np.ndarray:
    """Return a separable Gaussian spectral filter on the grid rfft2 layout."""
    if sigma_x < 0.0 or sigma_y < 0.0:
        raise ValueError("sigma_x and sigma_y must be non-negative")
    return np.exp(-0.5 * (grid.KX**2 * sigma_x**2 + grid.KY**2 * sigma_y**2)).astype(np.float64)


def make_W_box(grid: Grid2D, kx_cut: float, ky_cut: float) -> np.ndarray:
    """Return a sharp spectral cutoff filter."""
    if kx_cut < 0.0 or ky_cut < 0.0:
        raise ValueError("kx_cut and ky_cut must be non-negative")
    return ((np.abs(grid.KX) <= kx_cut) & (np.abs(grid.KY) <= ky_cut)).astype(np.float64)


def make_W_binomial(
    grid: Grid2D,
    passes_x: int,
    passes_y: int | None = None,
) -> np.ndarray:
    """Return the exact response of repeated ``(1, 2, 1) / 4`` filters.

    One pass has response ``cos(k*dx/2)^2``. For resolved low modes, ``n``
    passes are equivalent at leading order to a Gaussian with
    ``sigma/dx = sqrt(n/2)``.
    """
    if passes_y is None:
        passes_y = passes_x
    if (
        isinstance(passes_x, bool)
        or isinstance(passes_y, bool)
        or int(passes_x) != passes_x
        or int(passes_y) != passes_y
        or passes_x < 0
        or passes_y < 0
    ):
        raise ValueError("binomial pass counts must be nonnegative integers")
    response_x = np.cos(0.5 * grid.KX * grid.dx) ** (2 * int(passes_x))
    response_y = np.cos(0.5 * grid.KY * grid.dy) ** (2 * int(passes_y))
    return (response_x * response_y).astype(np.float64)


def make_W_compensated_binomial(
    grid: Grid2D,
    passes_x: int,
    passes_y: int | None = None,
    compensation_order: int = 1,
) -> np.ndarray:
    """Return repeated binomial filtering with truncated inverse compensation.

    For ``n`` bilinear passes, order ``p`` uses the first ``p + 1`` terms of
    the inverse-binomial series. Its spectral response in one direction is

    ``(1-s)**n * sum(comb(n+j-1,j)*s**j, j=0..p)``,

    where ``s = sin(k*dx/2)**2``. Order one is the standard compensation pass
    ``1 + n*s``. Order ``p`` cancels attenuation through ``O(s**p)`` while
    preserving the binomial filter's zero at Nyquist. Applying this same real,
    even response on deposit and gather retains the matched force chain.
    """
    if passes_y is None:
        passes_y = passes_x
    if (
        isinstance(passes_x, bool)
        or isinstance(passes_y, bool)
        or int(passes_x) != passes_x
        or int(passes_y) != passes_y
        or passes_x < 0
        or passes_y < 0
    ):
        raise ValueError("binomial pass counts must be nonnegative integers")
    if (
        isinstance(compensation_order, bool)
        or int(compensation_order) != compensation_order
        or compensation_order < 1
    ):
        raise ValueError("compensation_order must be a positive integer")

    def response(k: np.ndarray, spacing: float, passes: int) -> np.ndarray:
        phase = 0.5 * k * spacing
        s = np.sin(phase) ** 2
        base = (1.0 - s) ** int(passes)
        compensation = np.ones_like(s)
        coefficient = 1.0
        for order in range(1, int(compensation_order) + 1):
            coefficient *= (int(passes) + order - 1) / order
            compensation += coefficient * s**order
        return base * compensation

    response_x = response(grid.KX, grid.dx, int(passes_x))
    response_y = response(grid.KY, grid.dy, int(passes_y))
    return (response_x * response_y).astype(np.float64)


def make_W_wiener(P_signal: np.ndarray, P_noise: np.ndarray, eps: float = 1.0e-30) -> np.ndarray:
    """Construct a Wiener filter from signal/noise power estimates."""
    P_signal = np.asarray(P_signal, dtype=np.float64)
    P_noise = np.asarray(P_noise, dtype=np.float64)
    if P_signal.shape != P_noise.shape:
        raise ValueError("P_signal and P_noise must have the same shape")
    return P_signal / (P_signal + P_noise + eps)


def make_W_wiener_estimated(
    grid: Grid2D,
    rho_realization: np.ndarray,
    ppc: float,
    kc_signal: float | None = None,
) -> np.ndarray:
    """Estimate a Wiener filter from one density realization.

    The model is intentionally simple for Phase 3:
    - high-k edge power estimates a flat particle noise floor
    - the residual low-k power is treated as signal
    - an optional `kc_signal` can clamp the signal region
    """
    del ppc  # reserved for later refinements
    rho_hat = np.fft.rfft2(np.asarray(rho_realization, dtype=np.float64))
    P_total = np.abs(rho_hat) ** 2

    edge_mask = (np.abs(grid.KX) > 0.7 * np.max(np.abs(grid.kx))) | (grid.KY > 0.7 * np.max(grid.ky))
    if kc_signal is not None:
        edge_mask &= np.sqrt(grid.KX**2 + grid.KY**2) >= kc_signal
    if not np.any(edge_mask):
        edge_mask = np.ones_like(P_total, dtype=bool)

    P_noise_level = float(np.mean(P_total[edge_mask]))
    P_noise = np.full_like(P_total, P_noise_level, dtype=np.float64)
    P_signal = np.maximum(P_total - P_noise, max(P_noise_level, 1.0) * 1.0e-30)
    return make_W_wiener(P_signal, P_noise)


def apply_W_to_grid(field: np.ndarray, W: np.ndarray, grid: Grid2D) -> np.ndarray:
    """Apply a precomputed spectral filter to a real grid quantity."""
    field = np.asarray(field, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    expected_shape = (grid.Nx, grid.Ny // 2 + 1)
    if W.shape != expected_shape:
        raise ValueError(f"W shape {W.shape} != expected rfft2 shape {expected_shape}")
    filtered_hat = np.fft.rfft2(field) * W
    return np.fft.irfft2(filtered_hat, s=(grid.Nx, grid.Ny)).real.astype(np.float64)


def deposit_binned_adaptive_spectral_W(
    particles,
    grid: Grid2D,
    sigma_x: np.ndarray,
    sigma_y: np.ndarray,
    n_bins: int = 32,
) -> np.ndarray:
    """Approximate per-particle ``CIC*W`` deposition with width bins.

    Every bin is deposited with CIC and filtered at its charge-weighted mean
    width. Summing bins preserves total charge. This static estimator supports
    H0; adaptive dynamics additionally require matched gather and grad-h.
    """
    from pic_kde_2d.methods.cic import deposit_cic_2d

    sx = np.asarray(sigma_x, dtype=np.float64)
    sy = np.asarray(sigma_y, dtype=np.float64)
    if sx.shape != particles.x.shape or sy.shape != particles.y.shape:
        raise ValueError("adaptive sigma arrays must match particle arrays")
    if np.any(~np.isfinite(sx)) or np.any(~np.isfinite(sy)):
        raise ValueError("adaptive sigma arrays must be finite")
    if np.any(sx < 0.0) or np.any(sy < 0.0):
        raise ValueError("adaptive sigma arrays must be nonnegative")
    if n_bins <= 0:
        raise ValueError("n_bins must be positive")

    scale = np.hypot(sx / grid.dx, sy / grid.dy)
    if np.ptp(scale) == 0.0:
        rho = deposit_cic_2d(
            particles.x,
            particles.y,
            particles.w,
            grid.Nx,
            grid.Ny,
            grid.dx,
            grid.dy,
        )
        if sx[0] == 0.0 and sy[0] == 0.0:
            return rho
        return apply_W_to_grid(rho, make_W_gaussian(grid, sx[0], sy[0]), grid)

    edges = np.linspace(float(np.min(scale)), float(np.max(scale)), n_bins + 1)
    assignments = np.minimum(
        np.searchsorted(edges, scale, side="right") - 1,
        n_bins - 1,
    )
    assignments = np.maximum(assignments, 0)
    result = np.zeros((grid.Nx, grid.Ny), dtype=np.float64)
    for bin_index in range(n_bins):
        mask = assignments == bin_index
        if not np.any(mask):
            continue
        rho = deposit_cic_2d(
            particles.x[mask],
            particles.y[mask],
            particles.w[mask],
            grid.Nx,
            grid.Ny,
            grid.dx,
            grid.dy,
        )
        charge_weight = np.abs(particles.w[mask])
        sx_bin = float(np.average(sx[mask], weights=charge_weight))
        sy_bin = float(np.average(sy[mask], weights=charge_weight))
        if sx_bin > 0.0 or sy_bin > 0.0:
            rho = apply_W_to_grid(
                rho,
                make_W_gaussian(grid, sx_bin, sy_bin),
                grid,
            )
        result += rho
    return result
