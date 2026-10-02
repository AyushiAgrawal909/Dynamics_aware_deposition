"""
Build Figure 6: closed-form C4 prediction and its binding-edge validation.

Panel (a) compares the zero-parameter warm-dispersion rate ceiling with measured
widths. Panel (b) shows that the raw m=12 prediction outperforms the attempted
m=8 bias correction, while distinguishing central-value and CI-supported gates.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np


THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pic_kde_2d.benchmarks.theory import (  # noqa: E402
    matched_cic_force_coupling,
    rate_tolerance_crossing,
    warm_two_stream_growth,
)
from paperstyle import (  # noqa: E402
    COLORS,
    DOUBLE_COL,
    MARKERS,
    SIGMA_C4_E2,
    UNSAFE_THRESHOLD,
    apply_style,
)


NX = 128
LX = 64.0
V0 = 0.8
VTH = 0.05
RATE_TOLERANCE = UNSAFE_THRESHOLD

M8_ANALYTIC = 1.0622934850688703
M8_MEASURED = 0.99
M12_ANALYTIC = 0.19872036040909125
M12_PRODUCTION = SIGMA_C4_E2
M12_CENTRAL_CROSSING = 0.21481178588733243


def two_stream_rate(mode: int, sigma_dx: float) -> float:
    """Warm two-stream growth rate using the audited matched-CIC coupling."""
    coupling = matched_cic_force_coupling(mode, NX, sigma_dx)
    k = 2.0 * np.pi * mode / LX
    return warm_two_stream_growth(k, V0, VTH, coupling=coupling)


def analytic_crossing(mode: int) -> float:
    """Return the 10% rate-damage crossing using the validated driver logic."""
    return rate_tolerance_crossing(
        lambda sigma: two_stream_rate(mode, sigma),
        tolerance=RATE_TOLERANCE,
        sigma_upper=20.0,
    )


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _load_inputs() -> tuple[dict, dict, list[dict[str, str]]]:
    m12_dir = PROJECT_ROOT / "results" / "linearized_delta_f_m12_sigma_4872745"
    with (m12_dir / "m12_sigma_sweep.json").open() as handle:
        m12 = json.load(handle)

    with (
        PROJECT_ROOT
        / "results"
        / "c4_closed_form"
        / "c4_closed_form_validation.json"
    ).open() as handle:
        validation = json.load(handle)

    band = _read_csv(
        PROJECT_ROOT
        / "results"
        / "linearized_delta_f_band_4873221_full"
        / "analysis_final"
        / "linearized_delta_f_band.csv"
    )
    return m12, validation, band


def _assert_close(actual: float, expected: float, decimals: int = 5) -> None:
    np.testing.assert_almost_equal(actual, expected, decimal=decimals)


def _validate_inputs(
    m12: dict, validation: dict, band: list[dict[str, str]]
) -> None:
    rows = {round(float(row["sigma_dx"]), 7): row for row in m12["rows"]}

    for sigma, ratio, lo, hi in (
        (0.1844549, 0.92727, 0.91428, 0.93337),
        (0.1987204, 0.91515, 0.89984, 0.92228),
        (0.22, 0.89512, 0.87592, 0.90397),
    ):
        row = rows[round(sigma, 7)]
        _assert_close(float(row["gamma_ratio"]), ratio)
        _assert_close(float(row["ratio_ci_lo"]), lo)
        _assert_close(float(row["ratio_ci_hi"]), hi)

    expected_raw = {
        0.175: 0.92338,
        0.1844549: 0.91448,
        0.1987204: 0.90000,
        0.22: 0.87588,
    }
    for sigma, expected in expected_raw.items():
        row = rows[round(sigma, 7)]
        _assert_close(float(row["raw_closed_form_ratio"]), expected)

    _assert_close(float(m12["raw_curve_rmse"]), 0.01493)
    _assert_close(float(m12["bias_corrected_curve_rmse"]), 0.03071)
    np.testing.assert_allclose(
        m12["rmse_difference_corrected_minus_raw_ci"],
        [0.01548, 0.01578],
        atol=5e-6,
        rtol=0.0,
    )
    _assert_close(
        float(m12["largest_ci_supported_sigma_dx"]), M12_PRODUCTION, decimals=7
    )
    _assert_close(
        float(m12["central_10pct_crossing_dx"]), M12_CENTRAL_CROSSING, decimals=7
    )
    validation_rows = {
        (row["benchmark"], int(row["mode"])): row for row in validation["rows"]
    }
    _assert_close(
        float(validation_rows[("two_stream", 8)]["sigma_R_10pct_dx"]),
        M8_ANALYTIC,
        decimals=7,
    )
    _assert_close(
        float(validation_rows[("two_stream", 12)]["sigma_R_10pct_dx"]),
        M12_ANALYTIC,
        decimals=7,
    )

    edge = [row for row in band if int(row["mode"]) in (13, 14)]
    assert len(edge) == 2
    assert all(row["classification"] == "no_resolved_growth" for row in edge)
    assert all(
        not np.isfinite(float(row["fit_r2"])) or float(row["fit_r2"]) < 0.01
        for row in edge
    )


def _crossing_error_bars() -> tuple[float, np.ndarray]:
    """Return documented measured-width uncertainty proxies for panel (a)."""
    m8_csv = _read_csv(
        PROJECT_ROOT
        / "results"
        / "E2_m8_sigma_sweep_4863559"
        / "E2_m8_sigma_sweep.csv"
    )
    half = {}
    ratios = {}
    for sigma in (0.5, 1.0):
        row = next(
            item for item in m8_csv if np.isclose(float(item["sigma_dx"]), sigma)
        )
        ratios[sigma] = float(row["gamma_ratio"])
        half[sigma] = 0.5 * (
            float(row["ratio_ci_hi"]) - float(row["ratio_ci_lo"])
        )
    local_slope = abs((ratios[1.0] - ratios[0.5]) / 0.5)
    m8_sigma_error = half[1.0] / local_slope

    m12_tested_bracket = np.array(
        [[M12_PRODUCTION - 0.175], [M12_ANALYTIC - M12_PRODUCTION]]
    )
    return m8_sigma_error, m12_tested_bracket


def build_figure() -> tuple[Path, Path]:
    apply_style()
    m12, validation, band = _load_inputs()
    _validate_inputs(m12, validation, band)

    modes = np.arange(2, 13)
    crossings = np.array([analytic_crossing(int(mode)) for mode in modes])
    _assert_close(crossings[modes == 8][0], M8_ANALYTIC, decimals=7)
    _assert_close(crossings[modes == 12][0], M12_ANALYTIC, decimals=7)

    naive = M8_MEASURED * 8.0 / modes
    _assert_close(naive[modes == 12][0], 0.66, decimals=7)

    fig, (ax_a, ax_b) = plt.subplots(
        1,
        2,
        figsize=(DOUBLE_COL, 3.35),
        gridspec_kw={"wspace": 0.30},
    )

    # Panel (a): analytic ceiling over the protected band.
    ax_a.plot(
        modes,
        crossings,
        color=COLORS["theory"],
        lw=1.6,
        label=r"warm+CIC dispersion, $\epsilon_\gamma=0.10$",
        zorder=3,
    )
    ax_a.plot(
        modes,
        naive,
        color=COLORS["WuQin"],
        lw=1.25,
        ls="--",
        label=r"naive $q\sim m^2$ scaling from $m=8$",
        zorder=2,
    )

    m8_error, m12_error = _crossing_error_bars()
    ax_a.errorbar(
        8,
        M8_MEASURED,
        yerr=m8_error,
        fmt=MARKERS["PASS"]["marker"],
        ms=5.7,
        color=COLORS["C4"],
        mec=COLORS["C4"],
        mfc=COLORS["C4"],
        capsize=2.5,
        lw=1.0,
        label="measured (bars: CI / tested bracket)",
        zorder=5,
    )
    ax_a.errorbar(
        12,
        M12_PRODUCTION,
        yerr=m12_error,
        fmt=MARKERS["PASS"]["marker"],
        ms=5.7,
        color=COLORS["C4"],
        mec=COLORS["C4"],
        mfc=COLORS["C4"],
        capsize=2.5,
        lw=1.0,
        zorder=5,
    )
    ax_a.annotate(
        r"$m=8$: analytic 1.062, measured 0.99",
        xy=(8, M8_MEASURED),
        xytext=(4.5, 0.72),
        arrowprops={"arrowstyle": "-", "color": COLORS["C4"], "lw": 0.7},
        fontsize=7.2,
        color=COLORS["C4"],
    )
    ax_a.annotate(
        "$m=12$: analytic 0.199\nproduction 0.184",
        xy=(12, M12_PRODUCTION),
        xytext=(8.0, 0.205),
        arrowprops={"arrowstyle": "-", "color": COLORS["C4"], "lw": 0.7},
        fontsize=7.2,
        color=COLORS["C4"],
    )

    ax_a.axvline(12, color=COLORS["C4"], lw=0.9, ls=":", zorder=1)
    ax_a.text(
        11.77,
        1.48,
        "binding protected edge",
        ha="right",
        va="center",
        rotation=90,
        fontsize=7.0,
        color=COLORS["C4"],
    )
    ax_a.axvspan(12.5, 14.4, color=COLORS["WuQin"], alpha=0.12, zorder=0)
    ax_a.text(
        13.12,
        3.15,
        "no resolved growth\n"
        r"($\delta f$ band: $R^2<0.01$)",
        ha="center",
        va="center",
        fontsize=6.9,
        color="0.25",
    )

    ax_a.annotate(
        "",
        xy=(12, naive[modes == 12][0]),
        xytext=(12, crossings[modes == 12][0]),
        arrowprops={
            "arrowstyle": "<->",
            "color": COLORS["MISE"],
            "lw": 1.0,
        },
    )
    ax_a.text(
        11.55,
        0.39,
        "marginality amplification:\n"
        r"$0.66\,\Delta x\ \rightarrow\ 0.20\,\Delta x$",
        ha="right",
        va="center",
        fontsize=7.0,
        color=COLORS["MISE"],
    )
    ax_a.set_yscale("log")
    ax_a.set_xlim(1.8, 14.25)
    ax_a.set_ylim(0.145, 14.0)
    ax_a.set_xticks(np.arange(2, 15, 2))
    ax_a.set_xlabel("Fourier mode $m$")
    ax_a.set_ylabel(r"Rate-safe ceiling $\sigma_R^\star/\Delta x$")
    ax_a.set_title("(a) Marginality tightens the safe width")
    ax_a.legend(loc="upper right", fontsize=6.7, frameon=False)

    # Panel (b): direct binding-edge model test.
    rows = sorted(m12["rows"], key=lambda row: float(row["sigma_dx"]))
    sigma = np.array([float(row["sigma_dx"]) for row in rows])
    measured = np.array([float(row["gamma_ratio"]) for row in rows])
    ci_lo = np.array([float(row["ratio_ci_lo"]) for row in rows])
    ci_hi = np.array([float(row["ratio_ci_hi"]) for row in rows])

    smooth_sigma = np.linspace(0.0, 0.23, 240)
    gamma0 = two_stream_rate(12, 0.0)
    raw_curve = np.array(
        [two_stream_rate(12, value) / gamma0 for value in smooth_sigma]
    )
    damage_multiplier = float(validation["m8_bias_calibration"]["damage_multiplier"])
    corrected_curve = 1.0 - damage_multiplier * (1.0 - raw_curve)

    ax_b.plot(
        smooth_sigma,
        raw_curve,
        color=COLORS["theory"],
        lw=1.6,
        label="raw closed form",
        zorder=2,
    )
    ax_b.plot(
        smooth_sigma,
        corrected_curve,
        color=COLORS["WuQin"],
        lw=1.35,
        ls="--",
        label=r"$m=8$-bias-corrected",
        zorder=2,
    )

    for x, y, low, high in zip(sigma, measured, ci_lo, ci_hi):
        if low >= 1.0 - RATE_TOLERANCE:
            color = COLORS["C4"]
            marker = MARKERS["PASS"]["marker"]
            mfc = color
        elif y >= 1.0 - RATE_TOLERANCE:
            color = COLORS["Silverman"]
            marker = MARKERS["FAIL"]["marker"]
            mfc = "white"
        else:
            color = COLORS["MISE"]
            marker = MARKERS["CENSORED"]["marker"]
            mfc = color

        ax_b.errorbar(
            x,
            y,
            yerr=[[y - low], [high - y]],
            fmt=marker,
            ms=5.4,
            color=color,
            mec=color,
            mfc=mfc,
            mew=1.1,
            capsize=2.3,
            lw=0.9,
            zorder=5,
        )

    tolerance_ratio = 1.0 - RATE_TOLERANCE
    tolerance_line = ax_b.axhline(
        tolerance_ratio,
        color=COLORS["MISE"],
        lw=1.0,
        ls=":",
        label="10% rate tolerance",
        zorder=1,
    )
    ax_b.text(
        0.168,
        0.938,
        "PASS",
        ha="right",
        va="center",
        fontsize=6.8,
        color=COLORS["C4"],
        weight="bold",
    )
    ax_b.text(
        0.201,
        0.923,
        "narrow fail",
        ha="left",
        va="bottom",
        fontsize=6.6,
        color=COLORS["Silverman"],
    )
    ax_b.text(
        0.225,
        0.895,
        "FAIL",
        ha="left",
        va="center",
        fontsize=6.8,
        color=COLORS["MISE"],
        weight="bold",
    )
    ax_b.axvline(
        M12_PRODUCTION,
        color=COLORS["C4"],
        lw=1.0,
        ls="-.",
        zorder=1,
    )
    ax_b.axvline(
        M12_CENTRAL_CROSSING,
        color="0.45",
        lw=0.8,
        ls=":",
        alpha=0.7,
        zorder=1,
    )
    ax_b.text(
        M12_PRODUCTION - 0.002,
        0.995,
        r"CI-supported $0.184$",
        ha="right",
        va="top",
        rotation=90,
        fontsize=6.7,
        color=COLORS["C4"],
    )
    ax_b.text(
        M12_CENTRAL_CROSSING + 0.002,
        0.995,
        r"central crossing $0.215$",
        ha="left",
        va="top",
        rotation=90,
        fontsize=6.5,
        color="0.4",
    )

    model_text = (
        "raw RMSE 0.01493 vs bias-corrected 0.03071;\n"
        "paired RMSE-difference CI [0.01548, 0.01578]\n"
        "excludes zero. The $m=8$ correction does not transfer:\n"
        "C4 predicts $m=12$ with zero fitted parameters."
    )
    ax_b.text(
        0.012,
        0.858,
        model_text,
        ha="left",
        va="bottom",
        fontsize=6.55,
        bbox={
            "boxstyle": "round,pad=0.3",
            "facecolor": "white",
            "edgecolor": "0.7",
            "alpha": 0.94,
        },
        zorder=8,
    )
    ax_b.set_xlim(-0.006, 0.232)
    ax_b.set_ylim(0.85, 1.01)
    ax_b.set_xlabel(r"Smoothing width $\sigma/\Delta x$")
    ax_b.set_ylabel(r"Growth-rate fidelity $\gamma_\sigma/\gamma_0$")
    ax_b.set_title(r"(b) Binding edge: linearized $\delta f$, $m=12$")
    handles, labels = ax_b.get_legend_handles_labels()
    model_handles = [
        handle
        for handle, label in zip(handles, labels)
        if label in {"raw closed form", r"$m=8$-bias-corrected", "10% rate tolerance"}
    ]
    model_labels = [
        label
        for label in labels
        if label in {"raw closed form", r"$m=8$-bias-corrected", "10% rate tolerance"}
    ]
    ax_b.legend(
        model_handles,
        model_labels,
        loc="upper left",
        fontsize=6.55,
        frameon=False,
    )

    fig.subplots_adjust(left=0.085, right=0.985, bottom=0.17, top=0.90)
    output_pdf = THIS_DIR / "fig06_closed_form_c4.pdf"
    output_png = THIS_DIR / "fig06_closed_form_c4.png"
    fig.savefig(output_pdf)
    fig.savefig(output_png, dpi=300)
    plt.close(fig)
    return output_pdf, output_png


if __name__ == "__main__":
    pdf, png = build_figure()
    print(f"Wrote {pdf}")
    print(f"Wrote {png}")
