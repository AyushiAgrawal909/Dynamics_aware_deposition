#!/usr/bin/env python3
"""Generate Paper 1 Figure 1 at the frozen selector widths."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from paperstyle import (
    COLORS,
    DOUBLE_COL,
    MARKERS,
    SIGMA_C4_E2,
    SIGMA_CV_BOUND,
    SIGMA_SILV_E2,
    UNSAFE_THRESHOLD,
    apply_style,
    mark_censored,
    unsafe_band,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PAPER_RESULTS = PROJECT_ROOT / "paper1_selector_comparison" / "results"
SELECTED_NOISE = (
    PROJECT_ROOT
    / "results"
    / "e2_selected_noise_4872840"
    / "e2_noise_summary.csv"
)
M12_RATE = (
    PROJECT_ROOT
    / "results"
    / "linearized_delta_f_m12_sigma_4872745"
    / "m12_sigma_sweep.csv"
)
NOISE_CURVE = PROJECT_ROOT / "results" / "part2" / "noise_model.csv"
OUTPUT_DIR = Path(__file__).resolve().parent

FORBIDDEN_PRODUCTION_WIDTHS = (0.22, 0.30, 0.35)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def row_at(
    rows: list[dict[str, str]],
    column: str,
    value: float,
    *,
    atol: float = 5.0e-7,
) -> dict[str, str]:
    matches = [
        row for row in rows if np.isclose(float(row[column]), value, rtol=0.0, atol=atol)
    ]
    if len(matches) != 1:
        raise AssertionError(
            f"expected one row with {column}={value}, found {len(matches)}"
        )
    return matches[0]


def selector_width(
    rows: list[dict[str, str]],
    selector_id: str,
    benchmark_id: str = "two_stream_e2",
) -> float:
    matches = [
        row
        for row in rows
        if row["selector_id"] == selector_id
        and row["benchmark_id"] == benchmark_id
    ]
    if len(matches) != 1:
        raise AssertionError(f"missing matrix row for {selector_id}/{benchmark_id}")
    return float(matches[0]["sigma_dx"])


def interpolate_noise(
    sigma: float,
    noise_sigma: np.ndarray,
    noise_reduction: np.ndarray,
) -> float:
    if sigma < noise_sigma[0] or sigma > noise_sigma[-1]:
        raise AssertionError(f"sigma={sigma} is outside the measured noise scan")
    return float(np.interp(sigma, noise_sigma, noise_reduction))


def main() -> None:
    apply_style()

    matrix = read_csv(PAPER_RESULTS / "comparison_cells.csv")
    selected_noise = read_csv(SELECTED_NOISE)
    rate_rows = read_csv(M12_RATE)
    noise_rows = read_csv(NOISE_CURVE)

    # The matrix is the frozen source for non-C4 selector definitions. C4 is
    # intentionally imported from paperstyle because the old matrix row still
    # records the superseded preregistration value 0.22.
    matrix_silverman = selector_width(matrix, "silverman")
    matrix_cv_bound = selector_width(matrix, "loo_cv")
    assert np.isclose(matrix_silverman, SIGMA_SILV_E2, atol=5.0e-4)
    assert np.isclose(matrix_cv_bound, SIGMA_CV_BOUND, atol=1.0e-12)

    c4_rate = row_at(rate_rows, "sigma_dx", SIGMA_C4_E2)
    c4_noise = row_at(selected_noise, "sigma_dx", SIGMA_C4_E2)
    c4_fidelity = float(c4_rate["gamma_ratio"])
    c4_fidelity_lo = float(c4_rate["ratio_ci_lo"])
    c4_fidelity_hi = float(c4_rate["ratio_ci_hi"])
    c4_noise_reduction = float(c4_noise["noise_reduction"])
    c4_noise_lo = float(c4_noise["noise_reduction_ci_lo"])
    c4_noise_hi = float(c4_noise["noise_reduction_ci_hi"])

    assert f"{c4_fidelity:.5f}" == "0.92727"
    assert f"{c4_noise_reduction:.5f}" == "1.20050"

    noise_sigma = np.asarray([float(row["sigma_dx"]) for row in noise_rows])
    noise_reduction = np.asarray(
        [float(row["noise_reduction"]) for row in noise_rows]
    )
    order = np.argsort(noise_sigma)
    noise_sigma = noise_sigma[order]
    noise_reduction = noise_reduction[order]

    frontier_rows = [
        row
        for row in rate_rows
        if float(row["sigma_dx"]) <= 0.2
        and not any(
            np.isclose(float(row["sigma_dx"]), forbidden, atol=1.0e-12)
            for forbidden in FORBIDDEN_PRODUCTION_WIDTHS
        )
    ]
    frontier_rows.sort(key=lambda row: float(row["sigma_dx"]))
    frontier_sigma = np.asarray(
        [float(row["sigma_dx"]) for row in frontier_rows], dtype=np.float64
    )
    frontier_fidelity = np.asarray(
        [float(row["gamma_ratio"]) for row in frontier_rows], dtype=np.float64
    )
    frontier_noise = np.asarray(
        [
            (
                c4_noise_reduction
                if np.isclose(sigma, SIGMA_C4_E2, atol=5.0e-7)
                else interpolate_noise(sigma, noise_sigma, noise_reduction)
            )
            for sigma in frontier_sigma
        ],
        dtype=np.float64,
    )
    # The sigma=0 endpoint is the unsmoothed CIC definition, not an
    # interpolation through the density scan.
    frontier_noise[np.isclose(frontier_sigma, 0.0)] = 1.0
    frontier_fidelity[np.isclose(frontier_sigma, 0.0)] = 1.0

    silverman_noise = interpolate_noise(
        SIGMA_SILV_E2, noise_sigma, noise_reduction
    )
    density_noise_bound = float(noise_reduction[-1])

    plotted_sigmas = [
        *frontier_sigma.tolist(),
        SIGMA_C4_E2,
        SIGMA_SILV_E2,
        SIGMA_CV_BOUND,
    ]
    assert not any(
        np.isclose(sigma, forbidden, rtol=0.0, atol=1.0e-12)
        for sigma in plotted_sigmas
        for forbidden in FORBIDDEN_PRODUCTION_WIDTHS
    )

    fig, ax = plt.subplots(figsize=(DOUBLE_COL, 3.7))
    ax.set_xscale("log")
    ax.set_xlim(0.92, 42.0)
    ax.set_ylim(0.0, 1.05)
    unsafe_band(
        ax,
        1.0 - UNSAFE_THRESHOLD,
        above=False,
        label=r"dynamically unsafe: $\gamma_\sigma/\gamma_{\rm CIC}<0.90$",
    )

    ax.plot(
        frontier_noise,
        frontier_fidelity,
        color="#3F4A56",
        linewidth=1.25,
        marker=".",
        markersize=3.5,
        label="measured smoothing frontier",
        zorder=3,
    )
    ax.plot(
        1.0,
        1.0,
        linestyle="none",
        color=COLORS["CIC"],
        markerfacecolor=COLORS["CIC"],
        markeredgecolor=COLORS["CIC"],
        markersize=6,
        label="CIC",
        zorder=6,
        **MARKERS["PASS"],
    )
    ax.errorbar(
        c4_noise_reduction,
        c4_fidelity,
        xerr=np.asarray(
            [
                [c4_noise_reduction - c4_noise_lo],
                [c4_noise_hi - c4_noise_reduction],
            ]
        ),
        yerr=np.asarray(
            [
                [c4_fidelity - c4_fidelity_lo],
                [c4_fidelity_hi - c4_fidelity],
            ]
        ),
        linestyle="none",
        color=COLORS["C4"],
        marker=MARKERS["PASS"]["marker"],
        markerfacecolor=COLORS["C4"],
        markeredgecolor=COLORS["C4"],
        markeredgewidth=MARKERS["PASS"]["markeredgewidth"],
        markersize=6,
        capsize=2.5,
        linewidth=0.9,
        label=r"dynamics selector, $\sigma=0.184\,\Delta x$",
        zorder=7,
    )

    ax.plot(
        silverman_noise,
        0.0,
        linestyle="none",
        color=COLORS["Silverman"],
        markerfacecolor="none",
        markeredgecolor=COLORS["Silverman"],
        markersize=6,
        label=r"Silverman, $\sigma=5.819\,\Delta x$",
        zorder=6,
        clip_on=False,
        **MARKERS["FAIL"],
    )
    ax.annotate(
        "growth window eliminated",
        xy=(silverman_noise, 0.0),
        xytext=(13.5, 0.19),
        arrowprops={
            "arrowstyle": "->",
            "color": COLORS["Silverman"],
            "linewidth": 0.8,
        },
        color=COLORS["Silverman"],
        fontsize=8,
        ha="center",
    )

    mark_censored(
        ax,
        density_noise_bound,
        0.035,
        direction="right",
        color=COLORS["MISE"],
        markersize=6,
        label="MISE, right-censored",
        zorder=7,
    )
    mark_censored(
        ax,
        density_noise_bound,
        0.075,
        direction="right",
        color=COLORS["LOO_CV"],
        markersize=6,
        label="LOO-CV, right-censored",
        zorder=7,
    )
    ax.annotate(
        r"mode killed; $\sigma\geq60\,\Delta x$"
        "\nMISE and LOO-CV coincide",
        xy=(density_noise_bound, 0.055),
        xytext=(8.5, 0.34),
        arrowprops={
            "arrowstyle": "->",
            "color": "#4B2635",
            "linewidth": 0.8,
        },
        color="#4B2635",
        fontsize=8,
        ha="left",
    )

    ax.set_xlabel(r"Noise reduction, $\mathrm{MISE}(0)/\mathrm{MISE}(\sigma)$")
    ax.set_ylabel(r"Growth-rate fidelity, $\gamma_\sigma/\gamma_{\rm CIC}$")
    ax.set_title(
        "Density-optimal and dynamics-safe selectors occupy different regimes"
    )
    ax.legend(
        loc="upper right",
        ncol=2,
        frameon=False,
        columnspacing=1.2,
        handletextpad=0.5,
    )
    fig.savefig(OUTPUT_DIR / "fig01_tradeoff_frontier.pdf")
    fig.savefig(OUTPUT_DIR / "fig01_tradeoff_frontier.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
