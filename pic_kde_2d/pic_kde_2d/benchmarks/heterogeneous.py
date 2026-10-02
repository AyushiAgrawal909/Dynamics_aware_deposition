"""Periodic heterogeneous benchmarks for adaptive-bandwidth validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from pic_kde_2d.core.particles2d import Particles2D, quiet_start_maxwellian_2d


@dataclass(frozen=True)
class PeriodicVonMisesProfile:
    """Smooth periodic marker density with a controlled peak-to-trough ratio."""

    Lx: float
    kappa: float
    phase: float = 0.0

    @property
    def dynamic_range(self) -> float:
        return float(np.exp(2.0 * self.kappa))

    def pdf(self, x: np.ndarray) -> np.ndarray:
        theta = 2.0 * np.pi * np.asarray(x) / self.Lx - self.phase
        return np.exp(self.kappa * np.cos(theta)) / (self.Lx * np.i0(self.kappa))

    def quiet_quantiles(self, n: int, quadrature_points: int = 65536) -> np.ndarray:
        """Deterministic inverse-CDF samples, avoiding seed noise in marker density."""
        if n <= 0:
            raise ValueError("n must be positive")
        grid = np.linspace(0.0, self.Lx, quadrature_points + 1)
        density = self.pdf(grid)
        increments = 0.5 * (density[:-1] + density[1:]) * np.diff(grid)
        cdf = np.concatenate([[0.0], np.cumsum(increments)])
        cdf /= cdf[-1]
        quantiles = (np.arange(n) + 0.5) / n
        return np.interp(quantiles, cdf, grid)

    def randomized_stratified_quantiles(
        self,
        n: int,
        rng: np.random.Generator,
        quadrature_points: int = 65536,
    ) -> np.ndarray:
        """Independent stratified inverse-CDF samples for ensemble statistics."""
        if n <= 0:
            raise ValueError("n must be positive")
        grid = np.linspace(0.0, self.Lx, quadrature_points + 1)
        density = self.pdf(grid)
        increments = 0.5 * (density[:-1] + density[1:]) * np.diff(grid)
        cdf = np.concatenate([[0.0], np.cumsum(increments)])
        cdf /= cdf[-1]
        quantiles = (np.arange(n) + rng.random(n)) / n
        return np.interp(quantiles, cdf, grid)


@dataclass(frozen=True)
class PeriodicFloorVonMisesProfile:
    """Mean-one physical density with a finite floor and periodic peak."""

    Lx: float
    peak_to_floor: float
    kappa: float = 4.0
    phase: float = 0.0

    def __post_init__(self) -> None:
        if not np.isfinite(self.Lx) or self.Lx <= 0.0:
            raise ValueError("Lx must be finite and positive")
        if not np.isfinite(self.peak_to_floor) or self.peak_to_floor <= 1.0:
            raise ValueError("peak_to_floor must exceed one")
        if not np.isfinite(self.kappa) or self.kappa <= 0.0:
            raise ValueError("kappa must be finite and positive")
        if self.peak_to_floor >= np.exp(2.0 * self.kappa):
            raise ValueError("peak_to_floor must be below exp(2*kappa)")

    @property
    def peak_amplitude(self) -> float:
        tail = np.exp(-2.0 * self.kappa)
        return float((self.peak_to_floor - 1.0) / (1.0 - self.peak_to_floor * tail))

    @property
    def normalization(self) -> float:
        mean_peak = np.exp(-self.kappa) * np.i0(self.kappa)
        return float(1.0 + self.peak_amplitude * mean_peak)

    def density(self, x: np.ndarray) -> np.ndarray:
        """Return the mean-one physical number density."""
        theta = 2.0 * np.pi * np.asarray(x) / self.Lx - self.phase
        peak = np.exp(self.kappa * (np.cos(theta) - 1.0))
        return (1.0 + self.peak_amplitude * peak) / self.normalization

    def pdf(self, x: np.ndarray) -> np.ndarray:
        """Return the normalized x-marker PDF corresponding to the density."""
        return self.density(x) / self.Lx

    def quiet_quantiles(self, n: int, quadrature_points: int = 65536) -> np.ndarray:
        if n <= 0:
            raise ValueError("n must be positive")
        grid = np.linspace(0.0, self.Lx, quadrature_points + 1)
        density = self.pdf(grid)
        increments = 0.5 * (density[:-1] + density[1:]) * np.diff(grid)
        cdf = np.concatenate([[0.0], np.cumsum(increments)])
        cdf /= cdf[-1]
        quantiles = (np.arange(n) + 0.5) / n
        return np.interp(quantiles, cdf, grid)

    def randomized_stratified_quantiles(
        self,
        n: int,
        rng: np.random.Generator,
        quadrature_points: int = 65536,
    ) -> np.ndarray:
        if n <= 0:
            raise ValueError("n must be positive")
        grid = np.linspace(0.0, self.Lx, quadrature_points + 1)
        density = self.pdf(grid)
        increments = 0.5 * (density[:-1] + density[1:]) * np.diff(grid)
        cdf = np.concatenate([[0.0], np.cumsum(increments)])
        cdf /= cdf[-1]
        quantiles = (np.arange(n) + rng.random(n)) / n
        return np.interp(quantiles, cdf, grid)

    @property
    def shoulder_x(self) -> float:
        """Right shoulder where the peak component is halfway between extrema."""
        target = 0.5 * (1.0 + np.exp(-2.0 * self.kappa))
        cosine = np.clip(1.0 + np.log(target) / self.kappa, -1.0, 1.0)
        theta = self.phase + np.arccos(cosine)
        return float(np.mod(theta, 2.0 * np.pi) * self.Lx / (2.0 * np.pi))


def periodic_displacement(x: np.ndarray, center: float, period: float) -> np.ndarray:
    """Signed minimum-image displacement from ``center``."""
    return (np.asarray(x) - center + 0.5 * period) % period - 0.5 * period


def init_profiled_plasma_2d(
    Nx_marker: int,
    Ny_marker: int,
    Lx: float,
    Ly: float,
    v_thermal: float,
    profile: PeriodicFloorVonMisesProfile,
    rng: np.random.Generator,
    *,
    alpha: float = 0.0,
    carrier_mode: int = 4,
    shoulder_width: float | None = None,
    randomized_positions: bool = True,
) -> Particles2D:
    """Equal-weight electrons following H1's physical density profile.

    A localized displacement seeds a finite-k perturbation on the profile
    shoulder while preserving total electron charge exactly.
    """
    if profile.Lx != Lx:
        raise ValueError("profile and particle box must use the same Lx")
    if abs(alpha) >= 0.25:
        raise ValueError("H1 perturbation alpha must remain weak")
    xs = (
        profile.randomized_stratified_quantiles(Nx_marker, rng)
        if randomized_positions
        else profile.quiet_quantiles(Nx_marker)
    )
    ys = (np.arange(Ny_marker) + 0.5) * Ly / Ny_marker
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    x = X.ravel()
    y = Y.ravel()
    _, _, vx, vy = quiet_start_maxwellian_2d(
        Nx_marker,
        Ny_marker,
        Lx,
        Ly,
        v_thermal,
        rng,
    )
    if alpha != 0.0:
        k = 2.0 * np.pi * carrier_mode / Lx
        width = Lx / 10.0 if shoulder_width is None else float(shoulder_width)
        if width <= 0.0:
            raise ValueError("shoulder_width must be positive")
        displacement = periodic_displacement(x, profile.shoulder_x, Lx)
        envelope = np.exp(-0.5 * (displacement / width) ** 2)
        x = (x + alpha * envelope * np.sin(k * displacement) / k) % Lx

    n_particles = x.size
    weights = np.full(n_particles, -Lx * Ly / n_particles, dtype=np.float64)
    return Particles2D(x, y, vx, vy, weights, Lx, Ly)


def ion_background_on_grid(
    profile: PeriodicFloorVonMisesProfile,
    x_nodes: np.ndarray,
    Ny: int,
    *,
    cic_matched: bool = True,
) -> np.ndarray:
    """Explicit immobile-ion density represented on the electron charge grid.

    The continuous ion profile is passed through the CIC assignment transfer
    by default. This is the grid representation of the physical background,
    not an adaptive smoothing operation.
    """
    ion_x = np.asarray(profile.density(x_nodes), dtype=np.float64)
    if cic_matched:
        n = ion_x.size
        modes = np.fft.fftfreq(n) * n
        argument = np.pi * modes / n
        sinc = np.ones_like(argument)
        nonzero = argument != 0.0
        sinc[nonzero] = np.sin(argument[nonzero]) / argument[nonzero]
        ion_x = np.fft.ifft(np.fft.fft(ion_x) * sinc**2).real
    ion_x /= np.mean(ion_x)
    return ion_x[:, None] * np.ones((1, Ny), dtype=np.float64)


def init_importance_sampled_landau_2d(
    Nx_marker: int,
    Ny_marker: int,
    Lx: float,
    Ly: float,
    v_thermal: float,
    alpha: float,
    kx_mode: int,
    marker_kappa: float,
    rng: np.random.Generator,
    randomized_positions: bool = False,
) -> Tuple[Particles2D, PeriodicVonMisesProfile]:
    """Uniform physical plasma represented by a strongly nonuniform marker load.

    The marker PDF is a smooth von Mises profile while particle weights divide
    by that PDF.  Consequently the expected physical density is
    ``1 + alpha*cos(kx*x)``. This benchmark has an analytic density reference
    and isolates whether a local bandwidth handles spatially varying sampling
    noise better than every global bandwidth.
    """
    if abs(alpha) >= 1.0:
        raise ValueError("alpha must keep the physical density positive")
    profile = PeriodicVonMisesProfile(Lx=Lx, kappa=marker_kappa)
    xs = (
        profile.randomized_stratified_quantiles(Nx_marker, rng)
        if randomized_positions
        else profile.quiet_quantiles(Nx_marker)
    )
    ys = (np.arange(Ny_marker) + 0.5) * Ly / Ny_marker
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    x = X.ravel()
    y = Y.ravel()

    # Reuse the validated quiet velocity construction and shuffle only velocity.
    _, _, vx, vy = quiet_start_maxwellian_2d(
        Nx_marker, Ny_marker, Lx, Ly, v_thermal, rng
    )
    kx = 2.0 * np.pi * kx_mode / Lx
    target_density = 1.0 + alpha * np.cos(kx * x)
    joint_marker_pdf = profile.pdf(x) / Ly
    n_particles = x.size
    w = -target_density / (n_particles * joint_marker_pdf)
    # Remove finite inverse-CDF quadrature error without changing relative
    # importance weights. The cosine mode has zero box mean for integer k.
    w *= (-Lx * Ly) / np.sum(w)
    return Particles2D(x, y, vx, vy, w, Lx, Ly), profile


def analytic_importance_sampled_density(
    x_nodes: np.ndarray, alpha: float, kx_mode: int, Lx: float
) -> np.ndarray:
    """Expected electron charge density for the estimator-isolation benchmark."""
    kx = 2.0 * np.pi * kx_mode / Lx
    return -(1.0 + alpha * np.cos(kx * np.asarray(x_nodes)))


def importance_sampling_noise_profile(
    x: np.ndarray,
    profile: PeriodicVonMisesProfile,
    alpha: float,
    kx_mode: int,
) -> dict:
    """Analytic relative weight and aggregate-variance profiles for H0.

    Squared individual weights scale as ``target^2 / p_marker^2``. The
    expected local sum of squared weights additionally includes the marker
    count and therefore scales as ``target^2 / p_marker``. Both are returned
    relative to their minima; H0 decile diagnostics use the aggregate proxy.
    """
    x = np.asarray(x, dtype=np.float64)
    kx = 2.0 * np.pi * kx_mode / profile.Lx
    target = 1.0 + alpha * np.cos(kx * x)
    pdf = profile.pdf(x)
    squared_weight = target * target / (pdf * pdf)
    aggregate_variance = target * target / pdf
    return {
        "squared_weight_amplification": squared_weight / np.min(squared_weight),
        "aggregate_variance_amplification": aggregate_variance
        / np.min(aggregate_variance),
    }
