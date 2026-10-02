#!/usr/bin/env python3
"""Generate Paper 1 Figure 4: mode erasure and floor collision.

The main panel reproduces the PPC-128 weak-Landau CIC and MISE-anchor traces.
The inset tests the matched-filter amplitude prediction against every
bandwidth available in that paired PPC-128 campaign.
"""

from __future__ import annotations

import json
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
    apply_style,
)


DATA_DIR = PROJECT_ROOT / "results" / "weak_landau_alias"
SUMMARY_PATH = DATA_DIR / "ppc128_cic_vs_mise16_logE.json"
TRACE_DIR = DATA_DIR / "ppc128_traces"

TRACE_PATHS = {
    0.0: TRACE_DIR / "sigma_0p0.npz",
    0.3: TRACE_DIR / "sigma_0p3.npz",
    16.0: TRACE_DIR / "sigma_16p0.npz",
    30.0: TRACE_DIR / "sigma_30p0.npz",
}

FIT_START = 2.0
FIT_STOP = 15.0
FLOOR_START = 70.0
MODE = 1
NX = 64
MISE_SIGMA = 16.0


def load_trace(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return time, per-seed complex modes, and coherent amplitude."""
    with np.load(path) as data:
        times = np.asarray(data["times"], dtype=np.float64)
        modes = np.asarray(data["mode_histories"], dtype=np.complex128)
    coherent = np.abs(np.mean(modes, axis=0))
    return times, modes, coherent


def fit_log_slope(
    times: np.ndarray,
    amplitude: np.ndarray,
) -> tuple[float, float]:
    """Fit natural-log amplitude over the frozen common window."""
    mask = (
        (times >= FIT_START)
        & (times <= FIT_STOP)
        & np.isfinite(amplitude)
        & (amplitude > 0.0)
    )
    slope, intercept = np.polyfit(times[mask], np.log(amplitude[mask]), 1)
    return float(slope), float(intercept)


def analytic_amplitude_transfer(sigma_cells: float) -> float:
    """Return sqrt(T_eff^2(sigma)/T_eff^2(0)); CIC cancels in the ratio."""
    q = (2.0 * np.pi * MODE * sigma_cells / NX) ** 2
    return float(np.exp(-0.5 * q))


def first_floor_crossing(
    times: np.ndarray,
    amplitude: np.ndarray,
    floor: float,
) -> tuple[float, float]:
    """Locate the first floor collision inside the common fit window."""
    indices = np.flatnonzero(
        (times >= FIT_START)
        & (times <= FIT_STOP)
        & (amplitude <= floor)
    )
    if len(indices) == 0:
        raise AssertionError("MISE trace does not cross its floor in fit window")
    index = int(indices[0])
    return float(times[index]), float(amplitude[index])


def build_erasure_regression() -> tuple[np.ndarray, np.ndarray, float, float]:
    """Regress measured initial erasure against matched-filter prediction."""
    sigmas = np.asarray(sorted(TRACE_PATHS), dtype=np.float64)
    initial_amplitudes = []
    for sigma in sigmas:
        _, _, coherent = load_trace(TRACE_PATHS[float(sigma)])
        initial_amplitudes.append(float(coherent[0]))
    measured = np.asarray(initial_amplitudes)
    measured /= measured[0]
    predicted = np.asarray(
        [analytic_amplitude_transfer(float(sigma)) for sigma in sigmas]
    )

    # The physical model has zero intercept: complete mode erasure predicts
    # zero measured coherent amplitude.
    slope = float(np.dot(predicted, measured) / np.dot(predicted, predicted))
    fitted = slope * predicted
    residual_sum = float(np.sum((measured - fitted) ** 2))
    total_sum = float(np.sum((measured - np.mean(measured)) ** 2))
    r_squared = 1.0 - residual_sum / total_sum
    return predicted, measured, slope, r_squared


def main() -> None:
    apply_style()
    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))

    times_cic, modes_cic, cic = load_trace(TRACE_PATHS[0.0])
    times_mise, modes_mise, mise = load_trace(TRACE_PATHS[MISE_SIGMA])
    np.testing.assert_allclose(times_cic, times_mise, rtol=0.0, atol=0.0)
    times = times_cic

    gamma_cic, intercept_cic = fit_log_slope(times, cic)
    gamma_mise, intercept_mise = fit_log_slope(times, mise)
    floor_cic = float(np.median(cic[times >= FLOOR_START]))
    floor_mise = float(np.median(mise[times >= FLOOR_START]))
    initial_ratio = float(mise[0] / cic[0])
    floor_ratio = floor_mise / floor_cic
    predicted_mise_ratio = analytic_amplitude_transfer(MISE_SIGMA)

    assert np.isclose(
        summary["initial_amplitude_ratio_mise_over_cic"],
        0.2912129,
        atol=5.0e-8,
    )
    assert np.isclose(initial_ratio, 0.2912129, atol=5.0e-8)
    assert np.isclose(initial_ratio, predicted_mise_ratio, atol=1.0e-14)
    assert np.isclose(summary["gamma_cic"], -0.0796175, atol=5.0e-8)
    assert np.isclose(summary["gamma_mise"], -0.2107110, atol=5.0e-8)
    assert np.isclose(gamma_cic, -0.0796175, atol=5.0e-8)
    assert np.isclose(gamma_mise, -0.2107110, atol=5.0e-8)
    assert np.isclose(
        summary["late_floor_ratio_mise_over_cic"],
        0.2221569,
        atol=5.0e-8,
    )
    assert np.isclose(floor_ratio, 0.2221569, atol=5.0e-8)
    assert int(summary["n_seeds"]) == modes_cic.shape[0] == modes_mise.shape[0]
    assert np.isclose(times[0], 0.0) and np.isclose(times[-1], 90.0)

    predicted, measured, erasure_slope, erasure_r2 = (
        build_erasure_regression()
    )
    assert np.isclose(erasure_slope, 1.0, atol=5.0e-13)
    assert np.isclose(erasure_r2, 1.0, atol=5.0e-13)

    censor_time, censor_amplitude = first_floor_crossing(
        times,
        mise,
        floor_mise,
    )
    assert FIT_START < censor_time < FIT_STOP

    fig, ax = plt.subplots(figsize=(DOUBLE_COL, 4.7))
    ax.axvspan(
        FIT_START,
        FIT_STOP,
        color=COLORS["WuQin"],
        alpha=0.10,
        linewidth=0.0,
        label="common fit window",
        zorder=0,
    )

    for index, mode in enumerate(modes_cic):
        ax.plot(
            times,
            np.log(np.maximum(np.abs(mode), 1.0e-300)),
            color=COLORS["CIC"],
            alpha=0.075,
            linewidth=0.55,
            label=(
                r"seed traces ($\alpha=0.15$)"
                if index == 0
                else "_nolegend_"
            ),
            zorder=1,
        )
    for mode in modes_mise:
        ax.plot(
            times,
            np.log(np.maximum(np.abs(mode), 1.0e-300)),
            color=COLORS["MISE"],
            alpha=0.075,
            linewidth=0.55,
            label="_nolegend_",
            zorder=1,
        )

    ax.plot(
        times,
        np.log(np.maximum(cic, 1.0e-300)),
        color=COLORS["CIC"],
        linewidth=1.7,
        label="CIC mean",
        zorder=4,
    )
    ax.plot(
        times,
        np.log(np.maximum(mise, 1.0e-300)),
        color=COLORS["MISE"],
        linewidth=1.7,
        label="MISE-optimal mean",
        zorder=4,
    )

    fit_times = np.asarray([FIT_START, FIT_STOP])
    ax.plot(
        fit_times,
        gamma_cic * fit_times + intercept_cic,
        color=COLORS["CIC"],
        linestyle="--",
        linewidth=1.25,
        label=rf"CIC fit: $\gamma={gamma_cic:.3f}$",
        zorder=5,
    )
    ax.plot(
        fit_times,
        gamma_mise * fit_times + intercept_mise,
        color=COLORS["MISE"],
        linestyle="--",
        linewidth=1.25,
        label=rf"MISE fit: $\gamma={gamma_mise:.3f}$",
        zorder=5,
    )
    ax.axhline(
        np.log(floor_cic),
        color=COLORS["CIC"],
        linestyle=":",
        linewidth=1.0,
        label="late floors",
        zorder=2,
    )
    ax.axhline(
        np.log(floor_mise),
        color=COLORS["MISE"],
        linestyle=":",
        linewidth=1.0,
        label="_nolegend_",
        zorder=2,
    )

    # Open circle plus cross marks the first collision with the late floor.
    ax.plot(
        censor_time,
        np.log(censor_amplitude),
        linestyle="none",
        marker="o",
        markerfacecolor="white",
        markeredgecolor=COLORS["MISE"],
        markeredgewidth=1.1,
        markersize=6.5,
        zorder=7,
    )
    ax.plot(
        censor_time,
        np.log(censor_amplitude),
        linestyle="none",
        marker=MARKERS["CENSORED"]["marker"],
        color=COLORS["MISE"],
        markeredgewidth=MARKERS["CENSORED"]["markeredgewidth"],
        markersize=6.0,
        zorder=8,
    )
    ax.annotate(
        "CENSORED: mode reaches its floor inside the fit window;\n"
        "slope is an erasure-collision artifact, not a physical rate",
        xy=(censor_time, np.log(censor_amplitude)),
        xytext=(19.0, -10.35),
        arrowprops={
            "arrowstyle": "->",
            "color": COLORS["MISE"],
            "linewidth": 0.85,
        },
        color=COLORS["MISE"],
        fontsize=8,
        ha="left",
        va="bottom",
        zorder=9,
    )
    ax.annotate(
        r"$\mathbf{predicted\ }\sqrt{T_{\rm eff}^2}=0.29,$"
        "\n"
        r"$\mathbf{measured}=0.291$",
        xy=(0.0, np.log(mise[0])),
        xytext=(20.0, -4.25),
        arrowprops={
            "arrowstyle": "->",
            "color": COLORS["C4"],
            "linewidth": 1.0,
        },
        color=COLORS["C4"],
        fontsize=8.5,
        fontweight="bold",
        ha="left",
        va="center",
        zorder=9,
    )
    ax.text(
        66.0,
        np.log(floor_mise) - 0.22,
        rf"floor ratio $={floor_ratio:.3f}$",
        color=COLORS["MISE"],
        fontsize=7.5,
        ha="left",
        va="top",
    )

    inset = ax.inset_axes([0.64, 0.59, 0.33, 0.35])
    bounds = np.linspace(0.0, 1.03, 100)
    inset.plot(
        bounds,
        bounds,
        color=COLORS["theory"],
        linestyle="--",
        linewidth=0.9,
        label=r"$y=x$",
        zorder=2,
    )
    inset.scatter(
        predicted,
        measured,
        color=COLORS["C4"],
        edgecolor="white",
        linewidth=0.45,
        s=21,
        zorder=4,
    )
    inset.set_xlim(0.0, 1.03)
    inset.set_ylim(0.0, 1.03)
    inset.set_xlabel("analytic ratio", fontsize=7, labelpad=1.0)
    inset.set_ylabel("measured ratio", fontsize=7, labelpad=1.0)
    inset.tick_params(labelsize=6.5)
    inset.set_title(
        rf"erasure: slope $={erasure_slope:.4f}$, $R^2={erasure_r2:.4f}$",
        fontsize=7.2,
        pad=3.0,
    )
    inset.grid(alpha=0.20)

    ax.set_xlim(0.0, 90.0)
    ax.set_ylim(-11.2, -3.7)
    ax.set_xlabel(r"time, $\omega_p t$")
    ax.set_ylabel(r"$\ln|\widehat{E}_x(m=1,t)|$")
    ax.set_title(
        "Density-optimal smoothing erases a weak mode into its noise floor"
    )
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.005),
        frameon=False,
        fontsize=7.0,
        ncol=4,
        columnspacing=1.0,
        handlelength=2.2,
    )
    fig.subplots_adjust(left=0.105, right=0.985, top=0.90, bottom=0.20)

    fig.savefig(FIGURE_DIR / "fig04_erasure.pdf")
    fig.savefig(FIGURE_DIR / "fig04_erasure.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
