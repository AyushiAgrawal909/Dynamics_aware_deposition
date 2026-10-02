"""
benchmarks/theory.py
====================

Analytical dispersion relations for benchmark physics.
Used to provide ground-truth gamma_theory and omega_theory.
"""

from __future__ import annotations

import math

import numpy as np

try:  # pragma: no cover - SciPy is optional and broken on the remote target.
    from scipy.special import wofz  # type: ignore
except Exception:  # pragma: no cover
    wofz = None

try:
    import mpmath
except Exception:  # pragma: no cover
    mpmath = None


def _complex_erf_series(z: complex, tol: float = 1.0e-15, max_terms: int = 200) -> complex:
    """Dependency-free fallback for complex erf(z) in the moderate-|z| regime."""
    term = complex(z)
    series = complex(z)
    zz = complex(z) * complex(z)
    for n in range(1, max_terms):
        term *= -zz / float(n)
        increment = term / float(2 * n + 1)
        series += increment
        if abs(increment) < tol * max(1.0, abs(series)):
            break
    return (2.0 / np.sqrt(np.pi)) * series


def Z_function(zeta: complex) -> complex:
    """Plasma dispersion function Z(zeta)."""
    if wofz is not None:
        return 1j * np.sqrt(np.pi) * wofz(zeta)
    if mpmath is not None:
        z = mpmath.mpc(zeta.real, zeta.imag)
        return complex(1j * np.sqrt(np.pi) * (mpmath.e ** (-z * z)) * mpmath.erfc(-1j * z))
    z = complex(zeta)
    faddeeva = np.exp(-(z * z)) * (1.0 - _complex_erf_series(-1j * z))
    return complex(1j * np.sqrt(np.pi) * faddeeva)


def Z_prime(zeta: complex) -> complex:
    """Derivative of the plasma dispersion function."""
    return -2.0 * (1.0 + zeta * Z_function(zeta))


def _landau_dielectric(omega: complex, k_lambdaD: float) -> complex:
    zeta = omega / (np.sqrt(2.0) * k_lambdaD)
    return 1.0 + (1.0 / (k_lambdaD * k_lambdaD)) * (1.0 + zeta * Z_function(zeta))


def solve_landau_dispersion(k_lambdaD: float = 0.4, max_iter: int = 50, tol: float = 1.0e-12) -> complex:
    """Solve the electrostatic Landau dispersion relation for a Maxwellian plasma."""
    omega = complex(np.sqrt(1.0 + 3.0 * k_lambdaD * k_lambdaD), -0.05)
    for _ in range(max_iter):
        residual = _landau_dielectric(omega, k_lambdaD)
        if abs(residual) < tol:
            return omega
        eps = 1.0e-6 + 1.0e-6j
        derivative = (_landau_dielectric(omega + eps, k_lambdaD) - residual) / eps
        omega = omega - residual / derivative
    return omega


def landau_gamma_theory(k_lambdaD: float = 0.4, v_thermal: float = 1.0) -> float:
    """Return the theoretical Landau damping rate gamma."""
    _ = v_thermal  # normalized theory depends on k * lambda_D only
    return float(np.imag(solve_landau_dispersion(k_lambdaD=k_lambdaD)))


def landau_omega_theory(k_lambdaD: float = 0.4, v_thermal: float = 1.0) -> float:
    """Return the theoretical real frequency omega_r."""
    _ = v_thermal
    return float(np.real(solve_landau_dispersion(k_lambdaD=k_lambdaD)))


def langmuir_omega_r(k_lambdaD: float = 0.4, v_thermal: float = 1.0) -> float:
    """Return Re(omega) for the kinetic Langmuir dispersion relation."""
    return landau_omega_theory(k_lambdaD=k_lambdaD, v_thermal=v_thermal)


def two_stream_growth(k: float, v_beam: float) -> float:
    """Return the normalized cold symmetric two-stream growth rate.

    The two beams have equal density, drift velocities ``+/-v_beam``, and
    ``omega_p = 1``. The unstable branch is

    ``gamma = sqrt(-y)``,
    ``y = ((2*u**2 + 1) - sqrt(8*u**2 + 1)) / 2``,
    ``u = k*v_beam``.

    Modes at and beyond the cold marginal point ``|u| = 1`` return zero.
    """
    if not np.isfinite(k) or not np.isfinite(v_beam) or v_beam <= 0.0:
        raise ValueError("k must be finite and v_beam must be finite and positive")
    u = float(k) * float(v_beam)
    y = ((2.0 * u * u + 1.0) - np.sqrt(8.0 * u * u + 1.0)) / 2.0
    return float(np.sqrt(-y)) if y < 0.0 else 0.0


def matched_cic_force_coupling(mode: int, n_cells: int, sigma_cells: float) -> float:
    """Return the matched CIC-plus-Gaussian force coupling at one grid mode."""
    if mode < 0 or n_cells <= 0 or sigma_cells < 0.0:
        raise ValueError("mode and sigma_cells must be nonnegative; n_cells must be positive")
    x = math.pi * float(mode) / float(n_cells)
    sinc = 1.0 if x == 0.0 else math.sin(x) / x
    q = (2.0 * math.pi * float(mode) * float(sigma_cells) / float(n_cells)) ** 2
    return float(sinc**4 * math.exp(-q))


def solve_landau_dispersion_coupling(
    k_lambdaD: float,
    coupling: float,
    *,
    initial: complex | None = None,
    max_iter: int = 80,
    tol: float = 1.0e-12,
) -> complex:
    """Solve the Maxwellian dispersion relation with reduced field coupling."""
    if k_lambdaD <= 0.0 or coupling <= 0.0:
        raise ValueError("k_lambdaD and coupling must be positive")

    def dielectric(omega: complex) -> complex:
        zeta = omega / (np.sqrt(2.0) * k_lambdaD)
        return 1.0 + coupling * (1.0 + zeta * Z_function(zeta)) / (k_lambdaD * k_lambdaD)

    omega = initial
    if omega is None:
        omega = complex(np.sqrt(coupling + 3.0 * k_lambdaD * k_lambdaD), -0.05)
    for _ in range(max_iter):
        residual = dielectric(omega)
        if abs(residual) < tol:
            return omega
        step = 1.0e-6 + 1.0e-6j
        derivative = (dielectric(omega + step) - residual) / step
        if derivative == 0.0:
            break
        omega -= residual / derivative
    return omega


def _stable_plasma_Z(zeta: complex) -> complex:
    """Evaluate Z with an asymptotic branch for the narrow E2 beams."""
    z = complex(zeta)
    if abs(z) <= 8.0:
        return Z_function(z)
    inv = 1.0 / z
    inv2 = inv * inv
    series = 1.0 + 0.5 * inv2 + 0.75 * inv2**2 + 1.875 * inv2**3 + 6.5625 * inv2**4
    return -inv * series


def warm_two_stream_growth(
    k: float,
    v_beam: float,
    v_thermal: float,
    coupling: float = 1.0,
    *,
    gamma_max: float = 2.0,
    n_scan: int = 500,
) -> float:
    """Return the purely growing root of the symmetric warm-beam dielectric."""
    return warm_two_stream_oblique_growth(
        k,
        0.0,
        v_beam,
        v_thermal,
        coupling,
        gamma_max=gamma_max,
        n_scan=n_scan,
    )


def warm_two_stream_oblique_growth(
    kx: float,
    ky: float,
    v_beam: float,
    v_thermal: float,
    coupling: float = 1.0,
    *,
    gamma_max: float = 2.0,
    n_scan: int = 500,
) -> float:
    """Return the growing root for symmetric x-directed beams at ``(kx, ky)``.

    Thermal broadening depends on ``|k| v_thermal``, whereas the beam Doppler
    shifts depend only on ``kx v_beam``. In the cold limit the ``|k|`` factors
    cancel from the longitudinal susceptibility, leaving oblique stripes of
    equal growth at fixed ``kx``.
    """
    k = float(np.hypot(kx, ky))
    if k <= 0.0 or v_beam <= 0.0 or v_thermal <= 0.0 or coupling < 0.0:
        raise ValueError("k and velocities must be positive; coupling must be nonnegative")
    if coupling == 0.0:
        return 0.0

    denom = math.sqrt(2.0) * k * v_thermal

    def residual(gamma: float) -> float:
        omega = 1j * gamma
        zeta_plus = (omega - kx * v_beam) / denom
        zeta_minus = (omega + kx * v_beam) / denom
        susceptibility = (
            1.0
            + zeta_plus * _stable_plasma_Z(zeta_plus)
            + 1.0
            + zeta_minus * _stable_plasma_Z(zeta_minus)
        )
        dielectric = 1.0 + coupling * susceptibility / (2.0 * k * k * v_thermal * v_thermal)
        return float(np.real(dielectric))

    grid = np.concatenate(([1.0e-9], np.geomspace(1.0e-7, gamma_max, n_scan)))
    values = np.asarray([residual(float(gamma)) for gamma in grid])
    intervals: list[tuple[float, float]] = []
    for index in range(len(grid) - 1):
        left_value = values[index]
        right_value = values[index + 1]
        if not (np.isfinite(left_value) and np.isfinite(right_value)):
            continue
        if left_value == 0.0:
            intervals.append((float(grid[index]), float(grid[index])))
        elif left_value * right_value < 0.0:
            intervals.append((float(grid[index]), float(grid[index + 1])))
    if not intervals:
        return 0.0

    roots: list[float] = []
    for lower, upper in intervals:
        if lower == upper:
            roots.append(lower)
            continue
        lower_value = residual(lower)
        for _ in range(90):
            midpoint = 0.5 * (lower + upper)
            midpoint_value = residual(midpoint)
            if abs(midpoint_value) < 1.0e-12 or upper - lower < 1.0e-10:
                lower = upper = midpoint
                break
            if lower_value * midpoint_value <= 0.0:
                upper = midpoint
            else:
                lower = midpoint
                lower_value = midpoint_value
        roots.append(0.5 * (lower + upper))
    return float(max(roots))


def rate_tolerance_crossing(
    rate_at_sigma,
    *,
    tolerance: float = 0.10,
    sigma_upper: float = 20.0,
    increasing_magnitude: bool = False,
) -> float:
    """Find the first bandwidth where a signed rate changes by ``tolerance``."""
    if not 0.0 < tolerance < 1.0 or sigma_upper <= 0.0:
        raise ValueError("tolerance must lie in (0, 1) and sigma_upper must be positive")
    baseline = float(rate_at_sigma(0.0))
    if baseline == 0.0 or not np.isfinite(baseline):
        raise ValueError("the zero-bandwidth rate must be finite and nonzero")
    target = 1.0 + tolerance if increasing_magnitude else 1.0 - tolerance

    def ratio(sigma: float) -> float:
        return abs(float(rate_at_sigma(sigma))) / abs(baseline)

    if (ratio(sigma_upper) < target) == increasing_magnitude:
        raise ValueError("sigma_upper does not bracket the tolerance crossing")
    lower = 0.0
    upper = sigma_upper
    for _ in range(90):
        midpoint = 0.5 * (lower + upper)
        if (ratio(midpoint) >= target) == increasing_magnitude:
            upper = midpoint
        else:
            lower = midpoint
    return float(0.5 * (lower + upper))
