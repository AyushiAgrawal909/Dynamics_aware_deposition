#!/usr/bin/env python3
"""Recompute Paper 1 dispersion references from the recovered implementation.

This is a diagnostic calculation, not a production PIC run.  It separates four
levels of theory:

1. continuum Vlasov--Poisson;
2. the represented (principal) mode with the matched CIC deposit/gather factor;
3. the grid response including spatial aliases; and
4. the grid/time-discrete response of the explicit-midpoint linearized-delta-f
   update, with a continuum velocity integral.

It also evaluates the actual quiet-start velocity samples used by the E2 runs.
No measured value is used as a fit parameter or as a root constraint.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import pathlib
import sys
from dataclasses import dataclass
from typing import Callable

import numpy as np


HERE = pathlib.Path(__file__).resolve().parent
DOWNLOADS = HERE.parents[1]
CAMPAIGN_ROOT = DOWNLOADS / "pic_kde_2d_landau_cv_stage"
RESULTS_ROOT = DOWNLOADS / "pic_kde_2d" / "results"
if not CAMPAIGN_ROOT.is_dir():
    raise FileNotFoundError(f"recovered campaign source not found: {CAMPAIGN_ROOT}")
if not RESULTS_ROOT.is_dir():
    raise FileNotFoundError(f"result directory not found: {RESULTS_ROOT}")
sys.path.insert(0, str(CAMPAIGN_ROOT))

from pic_kde_2d.benchmarks.theory import (  # noqa: E402
    _stable_plasma_Z,
    solve_landau_dispersion,
    warm_two_stream_oblique_growth,
)
from pic_kde_2d.conventions import marker_grid_shape  # noqa: E402
from pic_kde_2d.core.particles2d import init_two_stream_2d  # noqa: E402


@dataclass(frozen=True)
class ModeCase:
    name: str
    benchmark: str
    nx: int
    ny: int
    lx: float
    ly: float
    mx: int
    my: int
    vthermal: float
    vbeam: float
    dt: float

    @property
    def dx(self) -> float:
        return self.lx / self.nx

    @property
    def dy(self) -> float:
        return self.ly / self.ny

    @property
    def kx(self) -> float:
        return 2.0 * math.pi * self.mx / self.lx

    @property
    def ky(self) -> float:
        return 2.0 * math.pi * self.my / self.ly


LANDAU = ModeCase(
    "weak_landau_m1",
    "landau",
    64,
    16,
    2.0 * math.pi / 0.4,
    8.0,
    1,
    0,
    1.0,
    0.0,
    0.05,
)
E2_M8 = ModeCase("E2_m8", "two_stream", 128, 16, 64.0, 16.0, 8, 0, 0.05, 0.8, 0.02)
E2_M12 = ModeCase("E2_m12", "two_stream", 128, 16, 64.0, 16.0, 12, 0, 0.05, 0.8, 0.02)
E2_M11N1 = ModeCase("E2_m11_n1", "two_stream", 128, 16, 64.0, 16.0, 11, 1, 0.05, 0.8, 0.02)
E2_M12N1 = ModeCase("E2_m12_n1", "two_stream", 128, 16, 64.0, 16.0, 12, 1, 0.05, 0.8, 0.02)
CASES = (LANDAU, E2_M8, E2_M12, E2_M11N1, E2_M12N1)


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(relative: str) -> dict:
    path = RESULTS_ROOT / relative
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"expected object in {path}")
    return value


def stable_z_array(values: np.ndarray) -> np.ndarray:
    """Vector wrapper around the recovered Z implementation.

    Temporal aliases are normally far into the asymptotic regime.  Evaluating
    them as arrays avoids calling the scalar power series thousands of times.
    """
    z = np.asarray(values, dtype=np.complex128)
    result = np.empty_like(z)
    moderate = np.abs(z) <= 8.0
    for index in np.flatnonzero(moderate):
        result.flat[index] = _stable_plasma_Z(complex(z.flat[index]))
    large = z[~moderate]
    if large.size:
        inv = 1.0 / large
        inv2 = inv * inv
        series = (
            1.0
            + 0.5 * inv2
            + 0.75 * inv2**2
            + 1.875 * inv2**3
            + 6.5625 * inv2**4
        )
        result[~moderate] = -inv * series
    return result


def cic_shape(qx: float, qy: float, case: ModeCase) -> float:
    """One CIC transform; the matched chain uses its square."""
    return float(
        np.sinc(qx * case.dx / (2.0 * math.pi)) ** 2
        * np.sinc(qy * case.dy / (2.0 * math.pi)) ** 2
    )


def spatial_aliases(case: ModeCase, qmax: int) -> list[tuple[float, float, float]]:
    """Return (qx, qy, S_CIC**2) for aliases visible to one grid mode."""
    rows: list[tuple[float, float, float]] = []
    for alias_x in range(-qmax, qmax + 1):
        for alias_y in range(-qmax, qmax + 1):
            qx = case.kx + 2.0 * math.pi * alias_x / case.dx
            qy = case.ky + 2.0 * math.pi * alias_y / case.dy
            shape_squared = cic_shape(qx, qy, case) ** 2
            # Exact axis aliases of a mode with m_axis=0 have zero CIC shape.
            # Drop only floating-point representations of those exact zeros.
            if shape_squared > 1.0e-28:
                rows.append((qx, qy, shape_squared))
    return rows


def response_by_temporal_alias(
    lam: complex,
    case: ModeCase,
    *,
    qmax: int,
    lmax: int,
    filter_force_transfer: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return temporal indices and J_l, where epsilon = 1 - sum(J_l)."""
    if filter_force_transfer <= 0.0 or filter_force_transfer > 1.0 + 1.0e-12:
        raise ValueError("filter force transfer must lie in (0, 1]")
    temporal = np.arange(-lmax, lmax + 1, dtype=np.int64)
    omega = 1j * complex(lam) - 2.0 * math.pi * temporal / case.dt
    response = np.zeros(temporal.size, dtype=np.complex128)
    d2 = case.kx**2 + case.ky**2
    for qx, qy, shape_squared in spatial_aliases(case, qmax):
        q2 = qx**2 + qy**2
        q = math.sqrt(q2)
        coefficient = (
            filter_force_transfer
            * shape_squared
            * (case.kx * qx + case.ky * qy)
            / (d2 * q2 * case.vthermal**2)
        )
        if case.benchmark == "two_stream":
            zeta_plus = (omega - qx * case.vbeam) / (
                math.sqrt(2.0) * q * case.vthermal
            )
            zeta_minus = (omega + qx * case.vbeam) / (
                math.sqrt(2.0) * q * case.vthermal
            )
            susceptibility = 0.5 * (
                1.0
                + zeta_plus * stable_z_array(zeta_plus)
                + 1.0
                + zeta_minus * stable_z_array(zeta_minus)
            )
        else:
            zeta = omega / (math.sqrt(2.0) * q * case.vthermal)
            susceptibility = 1.0 + zeta * stable_z_array(zeta)
        # J is minus the conventional susceptibility contribution.
        response -= coefficient * susceptibility
    return temporal, response


def continuous_residual(
    lam: complex,
    case: ModeCase,
    *,
    qmax: int,
    filter_force_transfer: float = 1.0,
) -> complex:
    _, response = response_by_temporal_alias(
        lam,
        case,
        qmax=qmax,
        lmax=0,
        filter_force_transfer=filter_force_transfer,
    )
    return complex(1.0 - np.sum(response))


def midpoint_residual(
    lam: complex,
    case: ModeCase,
    *,
    qmax: int,
    lmax: int,
    filter_force_transfer: float = 1.0,
) -> complex:
    """Grid/time-discrete dielectric of the explicit-midpoint weight update."""
    temporal, response = response_by_temporal_alias(
        lam,
        case,
        qmax=qmax,
        lmax=lmax,
        filter_force_transfer=filter_force_transfer,
    )
    tau = 0.5 * case.dt
    d2 = case.kx**2 + case.ky**2
    c0 = 0.0j
    for qx, qy, shape_squared in spatial_aliases(case, qmax):
        q2 = qx**2 + qy**2
        characteristic = math.exp(-0.5 * case.vthermal**2 * q2 * tau**2)
        if case.benchmark == "two_stream":
            characteristic *= math.cos(qx * case.vbeam * tau)
        c0 -= (
            filter_force_transfer
            * shape_squared
            * (case.kx * qx + case.ky * qy)
            / d2
            * tau
            * characteristic
        )
    alternating = np.where(temporal % 2 == 0, 1.0, -1.0)
    return complex(
        1.0
        - np.sum(response)
        - 0.5
        * case.dt
        * c0
        * np.exp(-complex(lam) * case.dt / 2.0)
        * np.sum(alternating * response)
    )


def complex_newton(
    residual: Callable[[complex], complex],
    initial: complex,
    *,
    tolerance: float = 2.0e-11,
    max_iterations: int = 50,
) -> complex:
    value = complex(initial)
    for _ in range(max_iterations):
        error = complex(residual(value))
        if abs(error) < tolerance:
            return value
        step = 1.0e-6 * max(1.0, abs(value))
        derivative = (residual(value + step) - residual(value - step)) / (2.0 * step)
        if abs(derivative) < 1.0e-14:
            raise RuntimeError(f"zero derivative at {value}; residual={error}")
        update = error / derivative
        if abs(update) > 0.5:
            update *= 0.5 / abs(update)
        value -= update
    raise RuntimeError(f"root did not converge: value={value}, residual={residual(value)}")


def initial_root(case: ModeCase) -> complex:
    if case.benchmark == "landau":
        return -1j * solve_landau_dispersion(0.4)
    value = warm_two_stream_oblique_growth(
        case.kx,
        case.ky,
        case.vbeam,
        case.vthermal,
        1.0,
    )
    return complex(value if value > 0.0 else 0.05, 0.0)


def solve_level(
    case: ModeCase,
    level: str,
    *,
    filter_force_transfer: float = 1.0,
    initial: complex | None = None,
    qmax: int = 8,
    lmax: int = 1024,
) -> complex:
    guess = initial_root(case) if initial is None else complex(initial)
    if level == "continuum":
        # No CIC and no aliases.  This is evaluated by setting the principal
        # shape to unity explicitly through the recovered closed-form solvers.
        if case.benchmark == "landau":
            if abs(filter_force_transfer - 1.0) < 1.0e-15:
                return -1j * solve_landau_dispersion(0.4)

            def dielectric(omega: complex) -> complex:
                zeta = omega / (math.sqrt(2.0) * 0.4)
                return 1.0 + filter_force_transfer * (
                    1.0 + zeta * _stable_plasma_Z(zeta)
                ) / 0.4**2

            omega_guess = 1j * guess
            omega = complex_newton(dielectric, omega_guess)
            return -1j * omega
        growth = warm_two_stream_oblique_growth(
            case.kx,
            case.ky,
            case.vbeam,
            case.vthermal,
            filter_force_transfer,
        )
        if growth <= 0.0:
            return 0.0j
        return complex(growth, 0.0)
    if level == "principal_cic":
        return complex_newton(
            lambda value: continuous_residual(
                value,
                case,
                qmax=0,
                filter_force_transfer=filter_force_transfer,
            ),
            guess,
        )
    if level == "spatial_discrete":
        return complex_newton(
            lambda value: continuous_residual(
                value,
                case,
                qmax=qmax,
                filter_force_transfer=filter_force_transfer,
            ),
            guess,
        )
    if level == "grid_time_discrete":
        return complex_newton(
            lambda value: midpoint_residual(
                value,
                case,
                qmax=qmax,
                lmax=lmax,
                filter_force_transfer=filter_force_transfer,
            ),
            guess,
        )
    raise ValueError(level)


def loaded_measurements() -> dict[str, dict]:
    landau_ref = read_json(
        "linearized_delta_f_reference_4872599/linearized_delta_f_reference.json"
    )
    band = read_json(
        "linearized_delta_f_band_4873221_full/analysis_final/linearized_delta_f_band.json"
    )
    oblique = read_json("e2_oblique_df_4902570/oblique_growth.json")
    m11 = read_json("e2_oblique_m11_4902583/oblique_growth.json")

    def row_by(items: list[dict], **match: int | str) -> dict:
        selected = [row for row in items if all(row.get(key) == value for key, value in match.items())]
        if len(selected) != 1:
            raise AssertionError(f"expected one row for {match}, found {len(selected)}")
        return selected[0]

    return {
        LANDAU.name: {
            **row_by(landau_ref["rows"], benchmark="landau"),
            "source": "linearized_delta_f_reference_4872599",
        },
        E2_M8.name: {
            **row_by(band["rows"], mode=8),
            "source": "linearized_delta_f_band_4873221_full/analysis_final",
        },
        E2_M12.name: {
            **row_by(band["rows"], mode=12),
            "source": "linearized_delta_f_band_4873221_full/analysis_final",
        },
        E2_M11N1.name: {
            **row_by(m11["rows"], mode_x=11, mode_y=1),
            "source": "e2_oblique_m11_4902583",
        },
        E2_M12N1.name: {
            **row_by(oblique["rows"], mode_x=12, mode_y=1),
            "source": "e2_oblique_df_4902570",
        },
    }


def prediction_table(qmax: int, lmax: int) -> list[dict]:
    measured = loaded_measurements()
    rows: list[dict] = []
    for case in CASES:
        previous: complex | None = None
        for level in ("continuum", "principal_cic", "spatial_discrete", "grid_time_discrete"):
            root = solve_level(case, level, initial=previous, qmax=qmax, lmax=lmax)
            previous = root
            datum = measured[case.name]
            measured_rate = float(datum["gamma"])
            rows.append(
                {
                    "case": case.name,
                    "level": level,
                    "lambda_real_rate": float(root.real),
                    "lambda_imag_frequency": float(root.imag),
                    "measured_rate": measured_rate,
                    "measured_ci_lo": float(datum["gamma_ci_lo"]),
                    "measured_ci_hi": float(datum["gamma_ci_hi"]),
                    "relative_difference_measured_minus_prediction": (
                        measured_rate / float(root.real) - 1.0 if root.real != 0.0 else None
                    ),
                    "prediction_inside_measured_ci": bool(
                        float(datum["gamma_ci_lo"])
                        <= root.real
                        <= float(datum["gamma_ci_hi"])
                    ),
                    "measurement_source": datum["source"],
                    "qmax": qmax if level in {"spatial_discrete", "grid_time_discrete"} else 0,
                    "lmax": lmax if level == "grid_time_discrete" else 0,
                }
            )
    return rows


def convergence_table(qmax_final: int, lmax_final: int) -> list[dict]:
    rows: list[dict] = []
    for case in CASES:
        guess = solve_level(case, "principal_cic")
        for qmax in (0, 1, 2, 4, qmax_final):
            root = solve_level(
                case,
                "spatial_discrete",
                qmax=qmax,
                initial=guess,
            )
            rows.append(
                {
                    "case": case.name,
                    "convergence_kind": "spatial_alias_qmax",
                    "truncation": qmax,
                    "lambda_real_rate": float(root.real),
                    "lambda_imag_frequency": float(root.imag),
                }
            )
        guess = solve_level(case, "spatial_discrete", qmax=qmax_final, initial=guess)
        for lmax in (16, 32, 64, 128, 256, 512, lmax_final):
            root = solve_level(
                case,
                "grid_time_discrete",
                qmax=qmax_final,
                lmax=lmax,
                initial=guess,
            )
            rows.append(
                {
                    "case": case.name,
                    "convergence_kind": "temporal_alias_lmax",
                    "truncation": lmax,
                    "lambda_real_rate": float(root.real),
                    "lambda_imag_frequency": float(root.imag),
                }
            )
    return rows


def rate_magnitude(case: ModeCase, root: complex) -> float:
    return abs(root.real)


def gaussian_force_transfer(case: ModeCase, sigma_x: float, sigma_y: float) -> float:
    return float(math.exp(-((case.kx * case.dx * sigma_x) ** 2 + (case.ky * case.dy * sigma_y) ** 2)))


def find_ceiling(
    case: ModeCase,
    level: str,
    direction: str,
    *,
    qmax: int,
    lmax: int,
) -> float:
    baseline = solve_level(case, level, qmax=qmax, lmax=lmax)
    increasing = case.benchmark == "landau"
    target = 1.10 if increasing else 0.90

    def ratio(sigma: float) -> float:
        sx = sigma if direction in {"isotropic", "x"} else 0.0
        sy = sigma if direction in {"isotropic", "y"} else 0.0
        transfer = gaussian_force_transfer(case, sx, sy)
        root = solve_level(
            case,
            level,
            filter_force_transfer=transfer,
            initial=baseline,
            qmax=qmax,
            lmax=lmax,
        )
        return rate_magnitude(case, root) / rate_magnitude(case, baseline)

    lower = 0.0
    upper = 0.25
    while upper < 64.0:
        value = ratio(upper)
        if (increasing and value >= target) or (not increasing and value <= target):
            break
        upper *= 2.0
    else:
        raise RuntimeError(f"failed to bracket ceiling for {case.name}/{level}/{direction}")
    for _ in range(55):
        midpoint = 0.5 * (lower + upper)
        value = ratio(midpoint)
        crossed = value >= target if increasing else value <= target
        if crossed:
            upper = midpoint
        else:
            lower = midpoint
    return 0.5 * (lower + upper)


def ceiling_table(qmax: int, lmax: int) -> list[dict]:
    specifications = (
        (LANDAU, "isotropic"),
        (E2_M8, "isotropic"),
        (E2_M12, "isotropic"),
        (E2_M12N1, "y"),
    )
    rows: list[dict] = []
    for case, direction in specifications:
        for level in ("continuum", "principal_cic", "grid_time_discrete"):
            ceiling = find_ceiling(
                case,
                level,
                direction,
                qmax=qmax,
                lmax=lmax,
            )
            rows.append(
                {
                    "case": case.name,
                    "direction": direction,
                    "level": level,
                    "sigma_cells_at_10pct": ceiling,
                    "criterion": "1.10 x |gamma0|" if case.benchmark == "landau" else "0.90 x gamma0",
                }
            )
    return rows


def compensated_axis(theta: float, passes: int, order: int) -> float:
    s = math.sin(0.5 * theta) ** 2
    polynomial = sum(math.comb(passes + j - 1, j) * s**j for j in range(order + 1))
    return float((1.0 - s) ** passes * polynomial)


def filter_rate_table(qmax: int, lmax: int) -> list[dict]:
    first_order = read_json(
        "compensated_binomial_20260811/analysis/merged/compensated_binomial_rows.json"
    )
    higher_order = read_json(
        "higher_order_compensation_20260811/analysis/merged/higher_order_compensation.json"
    )
    combined = first_order["rows"] + higher_order["rows"]
    wanted = {
        ("weak_landau", 1, 16),
        ("near_marginal_two_stream_m12", 1, 1),
        ("near_marginal_two_stream_m12", 2, 2),
        ("near_marginal_two_stream_m12", 2, 4),
        ("near_marginal_two_stream_m12", 3, 4),
    }
    rows: list[dict] = []
    for benchmark, order, passes in sorted(wanted):
        matches = [
            row
            for row in combined
            if row["benchmark"] == benchmark
            and int(row["passes"]) == passes
            and int(row.get("compensation_order", 1)) == order
        ]
        if len(matches) != 1:
            raise AssertionError(
                f"expected one measured filter row for {(benchmark, order, passes)}, found {len(matches)}"
            )
        measured = matches[0]
        measured_case = LANDAU if benchmark == "weak_landau" else E2_M12
        evaluation_cases = (measured_case,) if measured_case is LANDAU else (E2_M12, E2_M12N1)
        for case in evaluation_cases:
            wx = compensated_axis(2.0 * math.pi * case.mx / case.nx, passes, order)
            wy = compensated_axis(2.0 * math.pi * case.my / case.ny, passes, order)
            filter_force = (wx * wy) ** 2
            for level in ("principal_cic", "grid_time_discrete"):
                baseline = solve_level(case, level, qmax=qmax, lmax=lmax)
                filtered = solve_level(
                    case,
                    level,
                    filter_force_transfer=filter_force,
                    initial=baseline,
                    qmax=qmax,
                    lmax=lmax,
                )
                rows.append(
                    {
                        "benchmark": benchmark,
                        "evaluated_mode": f"({case.mx},{case.my})",
                        "measured_mode": "(1,0)" if case is LANDAU else "(12,0)",
                        "compensation_order": order,
                        "passes": passes,
                        "filter_W": wx * wy,
                        "filter_force_W2": filter_force,
                        "theory_level": level,
                        "predicted_rate_ratio": rate_magnitude(case, filtered) / rate_magnitude(case, baseline),
                        "measured_rate_ratio": (
                            float(measured["gamma_ratio_to_cic"])
                            if case is measured_case
                            else None
                        ),
                        "measured_ratio_ci_lo": (
                            float(measured["ratio_ci_lo"]) if case is measured_case else None
                        ),
                        "measured_ratio_ci_hi": (
                            float(measured["ratio_ci_hi"]) if case is measured_case else None
                        ),
                        "status": (
                            "measured"
                            if case is measured_case
                            else "prediction only; no recovered corner-mode filter run"
                        ),
                    }
                )
    return rows


def finite_marker_residual(
    lam: complex,
    case: ModeCase,
    vx: np.ndarray,
    vy: np.ndarray,
    *,
    time_discrete: bool,
) -> complex:
    """Principal-mode, finite-marker diagonal response of the actual sample."""
    d2 = case.kx**2 + case.ky**2
    shape_squared = cic_shape(case.kx, case.ky, case) ** 2
    conditional_center = case.vbeam * np.tanh(
        vx * case.vbeam / case.vthermal**2
    )
    score_x = -(vx - conditional_center) / case.vthermal**2
    score_y = -vy / case.vthermal**2
    phase_rate = case.kx * vx + case.ky * vy
    source = (
        1j
        * shape_squared
        * (case.kx * score_x + case.ky * score_y)
        / d2
        / vx.size
    )
    if not time_discrete:
        return complex(1.0 - np.sum(source / (complex(lam) + 1j * phase_rate)))
    streaming = np.exp(-1j * phase_rate * case.dt)
    half_streaming = np.exp(-0.5j * phase_rate * case.dt)
    denominator = np.exp(complex(lam) * case.dt) - streaming
    c0 = np.sum(half_streaming * source)
    return complex(
        1.0
        - case.dt * np.sum(streaming * source / denominator)
        - 0.5
        * case.dt**2
        * c0
        * np.sum(half_streaming * source / denominator)
    )


def equilibrium_and_marker_table() -> tuple[list[dict], list[dict]]:
    nx_marker, ny_marker = marker_grid_shape(128, 16, 512)
    if (nx_marker, ny_marker) != (2048, 512):
        raise AssertionError((nx_marker, ny_marker))
    moment_rows: list[dict] = []
    root_rows: list[dict] = []
    marker_cases = (E2_M12, E2_M11N1, E2_M12N1)
    for seed in range(10):
        particles = init_two_stream_2d(
            nx_marker,
            ny_marker,
            64.0,
            16.0,
            0.8,
            0.05,
            0.0,
            12,
            1,
            np.random.default_rng(seed),
        )
        conditional_center = 0.8 * np.tanh(particles.vx * 0.8 / 0.05**2)
        moment_rows.append(
            {
                "seed": seed,
                "particles": int(particles.vx.size),
                "vx_mean": float(np.mean(particles.vx)),
                "vy_mean": float(np.mean(particles.vy)),
                "vx_std_total": float(np.std(particles.vx)),
                "vx_std_about_conditional_beam": float(
                    np.std(particles.vx - conditional_center)
                ),
                "vy_std": float(np.std(particles.vy)),
                "negative_vx_count": int(np.count_nonzero(particles.vx < 0.0)),
                "positive_vx_count": int(np.count_nonzero(particles.vx > 0.0)),
            }
        )
        for case in marker_cases:
            continuum_guess = solve_level(case, "principal_cic")
            continuous_root = complex_newton(
                lambda value: finite_marker_residual(
                    value,
                    case,
                    particles.vx,
                    particles.vy,
                    time_discrete=False,
                ),
                continuum_guess,
            )
            discrete_root = complex_newton(
                lambda value: finite_marker_residual(
                    value,
                    case,
                    particles.vx,
                    particles.vy,
                    time_discrete=True,
                ),
                continuous_root,
            )
            root_rows.append(
                {
                    "seed": seed,
                    "case": case.name,
                    "approximation": "principal-mode finite-marker diagonal response",
                    "continuous_lambda_real": float(continuous_root.real),
                    "continuous_lambda_imag": float(continuous_root.imag),
                    "time_discrete_lambda_real": float(discrete_root.real),
                    "time_discrete_lambda_imag": float(discrete_root.imag),
                }
            )
        del particles
        gc.collect()
    return moment_rows, root_rows


def aggregate_marker_roots(rows: list[dict]) -> list[dict]:
    summary: list[dict] = []
    for case in (E2_M12, E2_M11N1, E2_M12N1):
        selected = [row for row in rows if row["case"] == case.name]
        for prefix in ("continuous", "time_discrete"):
            real = np.asarray([row[f"{prefix}_lambda_real"] for row in selected])
            imag = np.asarray([row[f"{prefix}_lambda_imag"] for row in selected])
            summary.append(
                {
                    "case": case.name,
                    "level": f"finite_marker_{prefix}",
                    "n_seeds": len(selected),
                    "mean_lambda_real": float(np.mean(real)),
                    "seed_std_lambda_real": float(np.std(real, ddof=1)),
                    "min_lambda_real": float(np.min(real)),
                    "max_lambda_real": float(np.max(real)),
                    "mean_lambda_imag": float(np.mean(imag)),
                    "max_abs_lambda_imag": float(np.max(np.abs(imag))),
                }
            )
    return summary


def write_csv(path: pathlib.Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty table: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(qmax: int, lmax: int, include_markers: bool) -> dict:
    if qmax < 1 or lmax < 32:
        raise ValueError("qmax must be >=1 and lmax must be >=32")
    predictions = prediction_table(qmax, lmax)
    convergence = convergence_table(qmax, lmax)
    ceilings = ceiling_table(qmax, lmax)
    filter_rates = filter_rate_table(qmax, lmax)
    if include_markers:
        moments, marker_roots = equilibrium_and_marker_table()
        marker_summary = aggregate_marker_roots(marker_roots)
    else:
        moments, marker_roots, marker_summary = [], [], []

    continuum_landau = next(
        row
        for row in predictions
        if row["case"] == LANDAU.name and row["level"] == "continuum"
    )
    principal_e2 = next(
        row
        for row in predictions
        if row["case"] == E2_M12.name and row["level"] == "principal_cic"
    )
    if not math.isclose(continuum_landau["lambda_real_rate"], -0.0661279586906626, abs_tol=2e-12):
        raise AssertionError(continuum_landau)
    if not math.isclose(principal_e2["lambda_real_rate"], 0.1446708324317, abs_tol=5e-10):
        raise AssertionError(principal_e2)
    if include_markers:
        if not all(row["particles"] == 1_048_576 for row in moments):
            raise AssertionError("E2 particle count changed")
        if max(abs(row["vy_std"] - 0.05) for row in moments) > 1.0e-6:
            raise AssertionError("E2 transverse temperature does not match vthermal=0.05")

    source_files = (
        CAMPAIGN_ROOT / "pic_kde_2d/benchmarks/theory.py",
        CAMPAIGN_ROOT / "pic_kde_2d/core/particles2d.py",
        CAMPAIGN_ROOT / "pic_kde_2d/core/pic_loop_delta_f.py",
        CAMPAIGN_ROOT / "pic_kde_2d/methods/delta_f.py",
        CAMPAIGN_ROOT / "pic_kde_2d/methods/cic.py",
        DOWNLOADS / "pic_kde_2d/scripts/submit_linearized_delta_f_m12_sigma_cpu.sbatch",
        DOWNLOADS / "pic_kde_2d/scripts/submit_E2_oblique_delta_f_gpu.sbatch",
        DOWNLOADS / "pic_kde_2d/scripts/submit_E2_oblique_corner_m11_gpu.sbatch",
        CAMPAIGN_ROOT / "scripts/submit_compensated_binomial_rates_cpu.sbatch",
        CAMPAIGN_ROOT / "scripts/submit_higher_order_comp_rates_cpu.sbatch",
    )
    return {
        "audit": "continuum, principal-CIC, spatial-alias, and grid/time-discrete dispersion",
        "normalization": "omega_pe=1; lambda convention exp(lambda*t), so lambda=-i*omega",
        "settings": {
            "spatial_alias_qmax": qmax,
            "temporal_alias_lmax": lmax,
            "e2": {
                "Nx": 128,
                "Ny": 16,
                "Lx": 64.0,
                "Ly": 16.0,
                "dx": 0.5,
                "dy": 1.0,
                "dt": 0.02,
                "vbeam": 0.8,
                "vthermal_x": 0.05,
                "vthermal_y": 0.05,
                "delta_f_ppc": 512,
                "delta_f_alpha": 1.0e-8,
                "marker_grid": [2048, 512],
            },
            "landau": {
                "Nx": 64,
                "Ny": 16,
                "k_lambda_D": 0.4,
                "dt": 0.05,
                "vthermal": 1.0,
            },
        },
        "method_limits": {
            "grid_time_discrete": "continuum velocity integral; finite-marker effects are tabulated separately",
            "finite_marker": "principal grid mode and diagonal +k response; spatial aliases and finite-sample +/-k mixing omitted",
            "measured_values": "loaded only for comparison, never used to fit theory",
        },
        "source_sha256": {
            str(path.relative_to(DOWNLOADS)): sha256(path) for path in source_files
        },
        "predictions": predictions,
        "convergence": convergence,
        "bandwidth_ceilings": ceilings,
        "filter_rate_predictions": filter_rates,
        "e2_equilibrium_seeds": moments,
        "finite_marker_roots": marker_roots,
        "finite_marker_summary": marker_summary,
    }


def write_outputs(result: dict, output: pathlib.Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "dispersion_audit.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    for filename, key in (
        ("dispersion_predictions.csv", "predictions"),
        ("dispersion_convergence.csv", "convergence"),
        ("bandwidth_ceilings.csv", "bandwidth_ceilings"),
        ("filter_rate_predictions.csv", "filter_rate_predictions"),
    ):
        write_csv(output / filename, result[key])
    if result["e2_equilibrium_seeds"]:
        write_csv(output / "e2_equilibrium_seeds.csv", result["e2_equilibrium_seeds"])
        write_csv(output / "finite_marker_roots.csv", result["finite_marker_roots"])
        write_csv(output / "finite_marker_summary.csv", result["finite_marker_summary"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=HERE / "results/dispersion_audit",
    )
    parser.add_argument("--qmax", type=int, default=8)
    parser.add_argument("--lmax", type=int, default=1024)
    parser.add_argument(
        "--skip-finite-markers",
        action="store_true",
        help="skip rebuilding the ten actual E2 velocity samples",
    )
    args = parser.parse_args()
    result = run(args.qmax, args.lmax, not args.skip_finite_markers)
    write_outputs(result, args.output)
    print(json.dumps(result["settings"], indent=2, sort_keys=True))
    print(f"wrote {args.output / 'dispersion_audit.json'}")


if __name__ == "__main__":
    main()
