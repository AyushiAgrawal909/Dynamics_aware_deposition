#!/usr/bin/env python3
"""Generate Paper 1 Figure 3: the unified smoothing mechanism.

Caption
-------
Smoothing reduces the effective plasma coupling by T_eff^2, moving every mode
toward greater stability. A growing mode's gamma falls toward marginality (b);
a damped mode's |gamma| rises (a). The opposite sign of gamma_sigma/gamma_0
between panels is the same physics read through the opposite sign of gamma.
"""

from __future__ import annotations

import csv
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


FIGURE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = FIGURE_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from paperstyle import (  # noqa: E402
    COLORS,
    DOUBLE_COL,
    MARKERS,
    UNSAFE_THRESHOLD,
    apply_style,
    unsafe_band,
)
from pic_kde_2d.benchmarks.theory import (  # noqa: E402
    Z_function,
    matched_cic_force_coupling,
    warm_two_stream_growth,
)


LANDAU_REFINEMENT = (
    PROJECT_ROOT
    / "results"
    / "landau_sigma_refinement_4864819"
    / "analysis"
    / "landau_sigma_refinement.csv"
)
LANDAU_SIGMA6 = (
    PROJECT_ROOT
    / "results"
    / "landau_mise6_rate_4872322"
    / "analysis"
    / "landau_sigma_refinement.csv"
)
E2_M8_SWEEP = (
    PROJECT_ROOT
    / "results"
    / "E2_m8_sigma_sweep_4863559"
    / "E2_m8_sigma_sweep.csv"
)

LANDAU_K = 0.4
LANDAU_MODE = 1
LANDAU_NX = 64
E2_MODE = 8
E2_NX = 128
E2_LX = 64.0
E2_V_BEAM = 0.8
E2_V_THERMAL = 0.05


def read_numeric_rows(path: Path) -> list[dict[str, float | str]]:
    """Read a CSV while preserving nonnumeric status fields."""
    rows: list[dict[str, float | str]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            row: dict[str, float | str] = {}
            for key, value in raw.items():
                if value is None:
                    row[key] = ""
                    continue
                try:
                    row[key] = float(value)
                except ValueError:
                    row[key] = value
            rows.append(row)
    return rows


def load_landau_measurements() -> dict[str, np.ndarray]:
    """Merge the main PPC-1024 sweep with the dedicated sigma=6 result."""
    merged: dict[float, dict[str, float | str]] = {}
    for path in (LANDAU_REFINEMENT, LANDAU_SIGMA6):
        for row in read_numeric_rows(path):
            if int(row["ppc"]) == 1024:
                merged[float(row["sigma_dx"])] = row

    positive = [merged[sigma] for sigma in sorted(merged) if sigma > 0.0]
    return {
        "sigma": np.asarray([float(row["sigma_dx"]) for row in positive]),
        "ratio": np.asarray(
            [float(row["gamma_ratio_to_cic"]) for row in positive]
        ),
        "ci_lo": np.asarray([float(row["ratio_ci_lo"]) for row in positive]),
        "ci_hi": np.asarray([float(row["ratio_ci_hi"]) for row in positive]),
    }


def load_e2_measurements() -> tuple[dict[str, np.ndarray], float]:
    """Load resolved isolated-m=8 measurements and their paired intervals."""
    rows = read_numeric_rows(E2_M8_SWEEP)
    baseline = next(row for row in rows if float(row["sigma_dx"]) == 0.0)
    clean = [
        row
        for row in rows
        if float(row["sigma_dx"]) > 0.0
        and row["status"] == "clean"
        and np.isfinite(float(row["gamma_ratio"]))
    ]
    clean.sort(key=lambda row: float(row["sigma_dx"]))
    return (
        {
            "sigma": np.asarray([float(row["sigma_dx"]) for row in clean]),
            "ratio": np.asarray([float(row["gamma_ratio"]) for row in clean]),
            "ci_lo": np.asarray([float(row["ratio_ci_lo"]) for row in clean]),
            "ci_hi": np.asarray([float(row["ratio_ci_hi"]) for row in clean]),
        },
        float(baseline["gamma_sigma"]),
    )


def landau_dielectric(omega: complex, coupling: float) -> complex:
    """Kinetic dielectric with Gaussian-reduced plasma coupling."""
    zeta = omega / (np.sqrt(2.0) * LANDAU_K)
    return 1.0 + coupling * (1.0 + zeta * Z_function(zeta)) / LANDAU_K**2


def newton_landau_root(
    coupling: float,
    initial: complex,
    *,
    max_iter: int = 80,
    tolerance: float = 1.0e-13,
) -> complex:
    """Follow the physical Landau root with a complex Newton iteration."""
    omega = complex(initial)
    for _ in range(max_iter):
        residual = landau_dielectric(omega, coupling)
        if abs(residual) < tolerance:
            return omega
        step = 1.0e-6 + 1.0e-6j
        derivative = (
            landau_dielectric(omega + step, coupling) - residual
        ) / step
        if abs(derivative) == 0.0:
            raise RuntimeError("zero derivative in Landau Newton iteration")
        omega -= residual / derivative
    raise RuntimeError(
        f"Landau root did not converge at coupling={coupling:.8g}"
    )


def solve_landau_curve(sigmas: np.ndarray) -> tuple[np.ndarray, complex]:
    """Solve in increasing sigma, seeding every root from its predecessor."""
    requested = np.asarray(sigmas, dtype=np.float64)
    ordered = np.unique(np.concatenate(([0.0], requested)))
    root = newton_landau_root(
        1.0,
        complex(np.sqrt(1.0 + 3.0 * LANDAU_K**2), -0.05),
    )
    baseline = root
    ratios: dict[float, float] = {0.0: 1.0}
    for sigma in ordered[1:]:
        # CIC is sigma-independent and cancels from the measured ratio. The
        # quoted kinetic anchors therefore use the normalized Gaussian W^2.
        coupling = float(
            np.exp(
                -(2.0 * np.pi * LANDAU_MODE * sigma / LANDAU_NX) ** 2
            )
        )
        root = newton_landau_root(coupling, root)
        ratios[float(sigma)] = abs(root.imag) / abs(baseline.imag)
    return np.asarray([ratios[float(sigma)] for sigma in requested]), baseline


def solve_warm_two_stream_curve(sigmas: np.ndarray) -> np.ndarray:
    """Evaluate the validated matched-CIC warm-beam dispersion response."""
    k = 2.0 * np.pi * E2_MODE / E2_LX
    baseline_coupling = matched_cic_force_coupling(E2_MODE, E2_NX, 0.0)
    baseline = warm_two_stream_growth(
        k,
        E2_V_BEAM,
        E2_V_THERMAL,
        baseline_coupling,
    )
    ratios = []
    for sigma in np.asarray(sigmas, dtype=np.float64):
        coupling = matched_cic_force_coupling(E2_MODE, E2_NX, float(sigma))
        gamma = warm_two_stream_growth(
            k,
            E2_V_BEAM,
            E2_V_THERMAL,
            coupling,
        )
        ratios.append(gamma / baseline)
    return np.asarray(ratios)


def value_at(
    sigma: float,
    sigmas: np.ndarray,
    values: np.ndarray,
) -> float:
    """Return an exact sampled value at sigma."""
    matches = np.flatnonzero(np.isclose(sigmas, sigma, rtol=0.0, atol=1.0e-12))
    if len(matches) != 1:
        raise AssertionError(f"expected exactly one row at sigma={sigma}")
    return float(values[matches[0]])


def verify_anchors(
    landau: dict[str, np.ndarray],
    e2: dict[str, np.ndarray],
    e2_gamma0: float,
) -> None:
    """Fail fast if source data or theory conventions drift."""
    landau_expected = {5.0: 1.43, 6.0: 1.568, 8.0: 2.06, 12.0: 2.93}
    landau_tolerances = {5.0: 0.03, 6.0: 0.002, 8.0: 0.02, 12.0: 0.01}
    for sigma, expected in landau_expected.items():
        measured = value_at(sigma, landau["sigma"], landau["ratio"])
        assert np.isclose(
            measured,
            expected,
            rtol=0.0,
            atol=landau_tolerances[sigma],
        ), (sigma, measured, expected)

    e2_expected = {
        0.1: 0.99657,
        0.2: 0.99544,
        0.35: 0.98373,
        0.5: 0.97401,
        1.0: 0.89853,
    }
    for sigma, expected in e2_expected.items():
        measured = value_at(sigma, e2["sigma"], e2["ratio"])
        assert f"{measured:.5f}" == f"{expected:.5f}"
    assert f"{e2_gamma0:.5f}" == "0.33646"


def main() -> None:
    apply_style()
    landau = load_landau_measurements()
    e2, e2_gamma0 = load_e2_measurements()
    verify_anchors(landau, e2, e2_gamma0)

    landau_sigma_curve = np.unique(
        np.concatenate(
            (
                np.geomspace(0.25, 12.5, 220),
                np.asarray([5.0, 6.0, 8.0, 12.0]),
            )
        )
    )
    landau_prediction, unsmoothed_root = solve_landau_curve(
        landau_sigma_curve
    )
    assert np.isclose(unsmoothed_root.real, 1.285057, atol=5.0e-7)
    assert np.isclose(unsmoothed_root.imag, -0.066128, atol=5.0e-7)
    landau_prediction_expected = {
        5.0: 1.439,
        6.0: 1.650,
        8.0: 2.224,
        12.0: 4.046,
    }
    for sigma, expected in landau_prediction_expected.items():
        predicted = value_at(
            sigma,
            landau_sigma_curve,
            landau_prediction,
        )
        assert f"{predicted:.3f}" == f"{expected:.3f}"

    e2_sigma_curve = np.geomspace(0.08, 1.02, 180)
    e2_dispersion = solve_warm_two_stream_curve(e2_sigma_curve)
    e2_q = (2.0 * np.pi * E2_MODE * e2_sigma_curve / E2_NX) ** 2
    e2_transfer_only = np.exp(-0.5 * e2_q)
    for sigma, expected in ((0.35, 0.9906), (0.5, 0.9809), (1.0, 0.9258)):
        q = (2.0 * np.pi * E2_MODE * sigma / E2_NX) ** 2
        assert f"{np.exp(-0.5 * q):.4f}" == f"{expected:.4f}"
    measured_transfer = np.exp(
        -0.5 * (2.0 * np.pi * E2_MODE * e2["sigma"] / E2_NX) ** 2
    )
    assert np.all(e2["ratio"] < measured_transfer)

    fig, (ax_landau, ax_e2) = plt.subplots(
        1,
        2,
        figsize=(DOUBLE_COL, 3.35),
        constrained_layout=True,
    )

    ax_landau.set_xscale("log")
    ax_landau.set_xlim(0.25, 13.5)
    ax_landau.set_ylim(0.75, 8.5)
    unsafe_band(
        ax_landau,
        1.0 + UNSAFE_THRESHOLD,
        above=True,
        label=r"unsafe: $|\Delta\gamma|/|\gamma_0|>10\%$",
        zorder=0,
    )
    ax_landau.axvspan(
        8.0,
        13.5,
        color=COLORS["WuQin"],
        alpha=0.10,
        linewidth=0.0,
        zorder=1,
    )
    ax_landau.axhline(1.0, color="0.35", linewidth=0.8, zorder=2)
    ax_landau.plot(
        landau_sigma_curve,
        landau_prediction,
        color=COLORS["theory"],
        linewidth=1.35,
        label="smoothed dispersion",
        zorder=4,
    )
    ax_landau.errorbar(
        landau["sigma"],
        landau["ratio"],
        yerr=np.vstack(
            (
                landau["ratio"] - landau["ci_lo"],
                landau["ci_hi"] - landau["ratio"],
            )
        ),
        fmt="o",
        color=COLORS["CIC"],
        markerfacecolor=COLORS["CIC"],
        markeredgecolor=COLORS["CIC"],
        markeredgewidth=MARKERS["PASS"]["markeredgewidth"],
        markersize=4.5,
        capsize=2.0,
        elinewidth=0.8,
        label="measured, PPC 1024",
        zorder=6,
    )
    landau_12 = value_at(12.0, landau["sigma"], landau["ratio"])
    ax_landau.annotate(
        "erasure ceiling:\nmeasurement capped by\nmeasurability",
        xy=(12.0, landau_12),
        xytext=(5.1, 5.25),
        arrowprops={
            "arrowstyle": "->",
            "color": COLORS["MISE"],
            "linewidth": 0.8,
        },
        color=COLORS["MISE"],
        fontsize=8,
        ha="left",
        va="center",
    )
    ax_landau.text(
        10.0,
        7.65,
        "erasure regime",
        color=COLORS["WuQin"],
        fontsize=8,
        ha="center",
    )
    ax_landau.set_xlabel(r"Gaussian width, $\sigma/\Delta x$")
    ax_landau.set_ylabel(r"Damping-rate ratio, $|\gamma_\sigma|/|\gamma_{\rm CIC}|$")
    ax_landau.set_title(r"(a) Landau: $|\gamma|$ rises")
    ax_landau.legend(loc="upper left", frameon=False)

    ax_e2.set_xscale("log")
    ax_e2.set_xlim(0.08, 1.08)
    ax_e2.set_ylim(0.85, 1.015)
    unsafe_band(
        ax_e2,
        1.0 - UNSAFE_THRESHOLD,
        above=False,
        label=r"unsafe: $|\Delta\gamma|/|\gamma_0|>10\%$",
        zorder=0,
    )
    ax_e2.axhline(1.0, color="0.35", linewidth=0.8, zorder=2)
    ax_e2.plot(
        e2_sigma_curve,
        e2_transfer_only,
        color=COLORS["WuQin"],
        linestyle="--",
        linewidth=1.15,
        label="transfer only",
        zorder=3,
    )
    ax_e2.plot(
        e2_sigma_curve,
        e2_dispersion,
        color=COLORS["theory"],
        linewidth=1.35,
        label="smoothed dispersion",
        zorder=4,
    )
    ax_e2.errorbar(
        e2["sigma"],
        e2["ratio"],
        yerr=np.vstack(
            (
                e2["ratio"] - e2["ci_lo"],
                e2["ci_hi"] - e2["ratio"],
            )
        ),
        fmt="o",
        color=COLORS["CIC"],
        markerfacecolor=COLORS["CIC"],
        markeredgecolor=COLORS["CIC"],
        markeredgewidth=MARKERS["PASS"]["markeredgewidth"],
        markersize=4.5,
        capsize=2.0,
        elinewidth=0.8,
        label="measured, PPC 2048",
        zorder=6,
    )
    ax_e2.annotate(
        r"threshold shift: $u_{\rm eff}=u\,e^{q/2}$",
        xy=(0.92, 0.907),
        xytext=(0.18, 0.925),
        arrowprops={
            "arrowstyle": "->",
            "color": COLORS["MISE"],
            "linewidth": 0.8,
        },
        color=COLORS["MISE"],
        fontsize=8,
        ha="left",
    )
    ax_e2.set_xlabel(r"Gaussian width, $\sigma/\Delta x$")
    ax_e2.set_ylabel(r"Growth-rate ratio, $\gamma_\sigma/\gamma_0$")
    ax_e2.set_title(r"(b) Two-stream: $\gamma$ falls")
    ax_e2.legend(loc="lower left", frameon=False)

    fig.savefig(FIGURE_DIR / "fig03_unified_mechanism.pdf")
    fig.savefig(FIGURE_DIR / "fig03_unified_mechanism.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
