#!/usr/bin/env python3
"""Make the two-panel erasure validation figure for Paper 1."""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


FIGURE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = FIGURE_DIR.parents[1]
DATA_DIR = PROJECT_ROOT / "results" / "weak_landau_alias"
SUMMARY_PATH = DATA_DIR / "ppc128_cic_vs_mise16_logE.json"
TRACE_DIR = DATA_DIR / "ppc128_traces"
PAPER_FIGURE_DIR = PROJECT_ROOT.parent / "paper1_source_package" / "figs"

NX = 64
MODE = 1
REFERENCE_SIGMA_CELLS = 16.0
FLOOR_START = 70.0
OUTPUT_STEM = "fig_erasure"


try:
    from paperstyle import COLORS, DOUBLE_COL, apply_style
except ImportError:
    DOUBLE_COL = 7.0
    COLORS = {
        "CIC": "#14213D",
        "MISE": "#8B1E1E",
        "C4": "#2A7F62",
        "WuQin": "#737373",
    }

    def apply_style() -> None:
        """Apply the requested fallback publication style."""
        matplotlib.rcParams.update(
            {
                "font.family": "serif",
                "font.size": 9,
                "axes.labelsize": 9,
                "axes.titlesize": 9,
                "xtick.labelsize": 8,
                "ytick.labelsize": 8,
                "legend.fontsize": 8,
                "axes.grid": True,
                "axes.axisbelow": True,
                "grid.alpha": 0.25,
                "axes.spines.top": False,
                "axes.spines.right": False,
            }
        )


def load_summary() -> dict[str, object]:
    """Load the JSON and explicitly inspect its top-level schema."""
    with SUMMARY_PATH.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)

    keys = summary.keys()
    print(f"JSON keys: {list(keys)}")
    required = {
        "fit_window",
        "gamma_cic",
        "gamma_mise",
        "initial_amplitude_ratio_mise_over_cic",
        "late_floor_cic",
        "late_floor_mise",
        "late_floor_ratio_mise_over_cic",
        "n_seeds",
    }
    missing = required.difference(keys)
    if missing:
        raise KeyError(f"Missing required JSON fields: {sorted(missing)}")
    return summary


def sigma_from_path(path: Path) -> float:
    """Decode sigma_<integer>p<fraction>.npz without displaying it in the plot."""
    encoded = path.stem.removeprefix("sigma_")
    if not re.fullmatch(r"\d+(?:p\d+)?", encoded):
        raise ValueError(f"Cannot decode bandwidth from {path.name}")
    return float(encoded.replace("p", "."))


def locate_seed_arrays(summary: dict[str, object]) -> dict[float, Path]:
    """Locate and report the per-seed arrays when absent from the JSON."""
    per_seed_keys = [key for key in summary if "seed" in key.lower()]
    has_embedded_arrays = any(
        isinstance(summary[key], list) and key != "fit_window"
        for key in per_seed_keys
    )
    if has_embedded_arrays:
        raise NotImplementedError(
            "Embedded per-seed JSON arrays were found; this script expects the "
            "campaign's NPZ trace products."
        )

    paths = sorted(TRACE_DIR.glob("sigma_*.npz"))
    if not paths:
        raise FileNotFoundError(
            f"No per-seed arrays were embedded in {SUMMARY_PATH.name}, and no "
            f"sigma_*.npz files were found in {TRACE_DIR}."
        )

    located: dict[float, Path] = {}
    print("No per-seed traces are stored in the JSON; found:")
    for path in paths:
        sigma_cells = sigma_from_path(path)
        with np.load(path) as arrays:
            fields = {
                key: (arrays[key].shape, str(arrays[key].dtype))
                for key in arrays.files
            }
        print(f"  {path.name}: {fields}")
        located[sigma_cells] = path
    return located


def load_trace(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return time, per-seed complex modes, and coherent-ensemble amplitude."""
    with np.load(path) as arrays:
        required = {"times", "mode_histories", "amplitude_coherent_mean"}
        missing = required.difference(arrays.files)
        if missing:
            raise KeyError(f"{path.name} is missing arrays {sorted(missing)}")
        times = np.asarray(arrays["times"], dtype=np.float64)
        modes = np.asarray(arrays["mode_histories"], dtype=np.complex128)
        coherent = np.asarray(
            arrays["amplitude_coherent_mean"], dtype=np.float64
        )

    if modes.ndim != 2 or modes.shape[1] != times.size:
        raise ValueError(f"Unexpected per-seed trace shape in {path.name}")
    np.testing.assert_allclose(
        coherent,
        np.abs(np.mean(modes, axis=0)),
        rtol=1.0e-13,
        atol=1.0e-15,
        err_msg=f"Stored coherent mean disagrees with modes in {path.name}",
    )
    return times, modes, coherent


def fit_natural_log_slope(
    times: np.ndarray,
    amplitude: np.ndarray,
    fit_start: float,
    fit_stop: float,
) -> tuple[float, float]:
    """Fit ln(amplitude) in the frozen shared window."""
    mask = (
        (times >= fit_start)
        & (times <= fit_stop)
        & np.isfinite(amplitude)
        & (amplitude > 0.0)
    )
    if np.count_nonzero(mask) < 2:
        raise ValueError("The shared fit window has too few valid samples")
    slope, intercept = np.polyfit(times[mask], np.log(amplitude[mask]), 1)
    return float(slope), float(intercept)


def analytic_erasure(sigma_cells: np.ndarray | float) -> np.ndarray:
    """Return sqrt(T_eff^2) for mode one on the 64-cell mesh."""
    sigma = np.asarray(sigma_cells, dtype=np.float64)
    cic_power = np.sinc(MODE / NX) ** 4
    gaussian_power = np.exp(-(2.0 * np.pi * MODE * sigma / NX) ** 2)
    return np.sqrt(cic_power * gaussian_power)


def first_downward_crossing(
    times: np.ndarray,
    amplitude: np.ndarray,
    threshold: float,
) -> tuple[float, float]:
    """Linearly interpolate the first downward crossing of a threshold."""
    indices = np.flatnonzero(
        (amplitude[:-1] > threshold) & (amplitude[1:] <= threshold)
    )
    if indices.size == 0:
        raise AssertionError("The reference-MISE curve never crosses 5x its floor")
    index = int(indices[0])
    fraction = (amplitude[index] - threshold) / (
        amplitude[index] - amplitude[index + 1]
    )
    crossing_time = times[index] + fraction * (times[index + 1] - times[index])
    return float(crossing_time), float(threshold)


def assert_no_forbidden_sigma_labels(fig: matplotlib.figure.Figure) -> None:
    """Fail if a superseded numeric value is presented as a sigma label."""
    forbidden = ("0.22", "0.30", "0.35")
    sigma_patterns = (r"\bsigma\b", r"\\sigma", "σ")
    for artist in fig.findobj(match=matplotlib.text.Text):
        label = artist.get_text()
        identifies_sigma = any(
            re.search(pattern, label, flags=re.IGNORECASE)
            if pattern.startswith("\\")
            else pattern.lower() in label.lower()
            for pattern in sigma_patterns
        )
        if identifies_sigma:
            for value in forbidden:
                assert value not in label, (
                    f"Forbidden sigma value {value} appears in figure text: {label!r}"
                )


def main() -> None:
    summary = load_summary()
    trace_paths = locate_seed_arrays(summary)

    if 0.0 not in trace_paths or REFERENCE_SIGMA_CELLS not in trace_paths:
        raise FileNotFoundError("CIC or reference-MISE trace file is missing")

    times_cic, modes_cic, cic = load_trace(trace_paths[0.0])
    times_mise, modes_mise, mise = load_trace(
        trace_paths[REFERENCE_SIGMA_CELLS]
    )
    np.testing.assert_array_equal(times_cic, times_mise)
    times = times_cic

    fit_start, fit_stop = map(float, summary["fit_window"])
    gamma_cic, intercept_cic = fit_natural_log_slope(
        times, cic, fit_start, fit_stop
    )
    gamma_mise, intercept_mise = fit_natural_log_slope(
        times, mise, fit_start, fit_stop
    )
    floor_cic = float(summary["late_floor_cic"])
    floor_mise = float(summary["late_floor_mise"])
    initial_ratio = float(summary["initial_amplitude_ratio_mise_over_cic"])
    floor_ratio = float(summary["late_floor_ratio_mise_over_cic"])

    # Requested numerical guardrails: fail before creating an output on mismatch.
    assert abs(np.sinc(1 / 64) ** 4 - 0.9983947798) < 1.0e-9
    assert round(initial_ratio, 6) == round(0.2912129, 6)
    assert round(float(mise[0] / cic[0]), 6) == round(0.2912129, 6)
    assert np.isclose(float(summary["gamma_cic"]), -0.0796175, atol=5.0e-4)
    assert np.isclose(float(summary["gamma_mise"]), -0.2107110, atol=5.0e-4)
    assert np.isclose(gamma_cic, -0.0796175, atol=5.0e-4)
    assert np.isclose(gamma_mise, -0.2107110, atol=5.0e-4)
    assert np.isclose(floor_ratio, 0.2221569, atol=5.0e-6)
    assert np.isclose(floor_mise / floor_cic, 0.2221569, atol=5.0e-6)
    assert int(summary["n_seeds"]) == modes_cic.shape[0] == modes_mise.shape[0]
    assert np.isclose(times[0], 0.0) and np.isclose(times[-1], 90.0)

    sigmas = np.asarray(sorted(trace_paths), dtype=np.float64)
    initial_amplitudes = []
    for sigma_cells in sigmas:
        _, _, coherent = load_trace(trace_paths[float(sigma_cells)])
        initial_amplitudes.append(float(coherent[0]))
    measured = np.asarray(initial_amplitudes, dtype=np.float64)
    measured /= measured[sigmas.tolist().index(0.0)]
    predicted = analytic_erasure(sigmas)
    erasure_slope, erasure_intercept = np.polyfit(predicted, measured, 1)
    fitted = erasure_slope * predicted + erasure_intercept
    residual_sum = float(np.sum((measured - fitted) ** 2))
    total_sum = float(np.sum((measured - measured.mean()) ** 2))
    erasure_r2 = 1.0 - residual_sum / total_sum

    censor_time, censor_amplitude = first_downward_crossing(
        times, mise, 5.0 * floor_mise
    )
    assert fit_start <= censor_time <= fit_stop

    apply_style()
    # Exact physical output size takes precedence over paperstyle's tight crop.
    matplotlib.rcParams["savefig.bbox"] = None
    fig, (ax_a, ax_b) = plt.subplots(
        1,
        2,
        figsize=(DOUBLE_COL, 2.8),
        gridspec_kw={"width_ratios": (1.72, 1.0)},
    )
    fig.subplots_adjust(
        left=0.085,
        right=0.985,
        bottom=0.19,
        top=0.86,
        wspace=0.33,
    )

    # Panel (a): all seeds, coherent means, frozen fits, and late floors.
    ax_a.axvspan(
        fit_start,
        fit_stop,
        color="#B0B0B0",
        alpha=0.15,
        linewidth=0.0,
        zorder=0,
    )
    for mode_history in modes_cic:
        ax_a.plot(
            times,
            np.log10(np.maximum(np.abs(mode_history), 1.0e-300)),
            color=COLORS["CIC"],
            alpha=0.12,
            linewidth=0.45,
            label="_nolegend_",
            zorder=1,
        )
    for mode_history in modes_mise:
        ax_a.plot(
            times,
            np.log10(np.maximum(np.abs(mode_history), 1.0e-300)),
            color=COLORS["MISE"],
            alpha=0.12,
            linewidth=0.45,
            label="_nolegend_",
            zorder=1,
        )

    ax_a.plot(
        times,
        np.log10(cic),
        color=COLORS["CIC"],
        linewidth=1.55,
        label="CIC",
        zorder=4,
    )
    ax_a.plot(
        times,
        np.log10(mise),
        color=COLORS["MISE"],
        linewidth=1.55,
        label="reference-MISE width",
        zorder=4,
    )

    fit_times = np.asarray([fit_start, fit_stop])
    ax_a.plot(
        fit_times,
        (gamma_cic * fit_times + intercept_cic) / np.log(10.0),
        color=COLORS["CIC"],
        linestyle="--",
        linewidth=1.1,
        label="_nolegend_",
        zorder=5,
    )
    ax_a.plot(
        fit_times,
        (gamma_mise * fit_times + intercept_mise) / np.log(10.0),
        color=COLORS["MISE"],
        linestyle="--",
        linewidth=1.1,
        label="_nolegend_",
        zorder=5,
    )
    log_floor_cic = np.log10(floor_cic)
    log_floor_mise = np.log10(floor_mise)
    ax_a.axhline(
        log_floor_cic,
        color=COLORS["CIC"],
        linestyle=":",
        linewidth=1.0,
        zorder=2,
    )
    ax_a.axhline(
        log_floor_mise,
        color=COLORS["MISE"],
        linestyle=":",
        linewidth=1.0,
        zorder=2,
    )

    initial_midpoint = 0.5 * (np.log10(cic[0]) + np.log10(mise[0]))
    ax_a.annotate(
        "predicted sqrt(T_eff^2) = 0.29\nmeasured 0.291",
        xy=(0.8, initial_midpoint),
        xytext=(5.0, -2.12),
        color="#1F1F1F",
        fontsize=7.7,
        fontweight="bold",
        ha="left",
        va="center",
        arrowprops={
            "arrowstyle": "-",
            "color": "#404040",
            "linewidth": 0.8,
        },
        bbox={
            "boxstyle": "round,pad=0.16",
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.82,
        },
        zorder=9,
    )

    cic_fit_y = (gamma_cic * 13.0 + intercept_cic) / np.log(10.0)
    ax_a.annotate(
        r"$\gamma_{\rm CIC}=-0.0796$",
        xy=(13.0, cic_fit_y),
        xytext=(28.0, -2.56),
        color=COLORS["CIC"],
        fontsize=7.4,
        ha="left",
        va="center",
        arrowprops={
            "arrowstyle": "-",
            "color": COLORS["CIC"],
            "linewidth": 0.75,
        },
        zorder=9,
    )

    log_censor = np.log10(censor_amplitude)
    ax_a.plot(
        censor_time,
        log_censor,
        linestyle="none",
        marker="o",
        markerfacecolor="white",
        markeredgecolor=COLORS["MISE"],
        markeredgewidth=1.1,
        markersize=5.8,
        zorder=8,
    )
    ax_a.annotate(
        "censor",
        xy=(censor_time, log_censor),
        xytext=(8.5, -3.34),
        color=COLORS["MISE"],
        fontsize=7.1,
        ha="left",
        va="center",
        arrowprops={
            "arrowstyle": "-",
            "color": COLORS["MISE"],
            "linewidth": 0.7,
        },
        zorder=9,
    )
    ax_a.annotate(
        r"$\gamma_{\rm MISE}=-0.211$" "\n"
        "CENSORED: mode meets its floor inside the\n"
        "window; slope is an erasure-collision artefact,\n"
        "not a physical rate",
        xy=(censor_time, log_censor),
        xytext=(14.0, -4.28),
        color=COLORS["MISE"],
        fontsize=6.6,
        ha="left",
        va="center",
        arrowprops={
            "arrowstyle": "->",
            "color": COLORS["MISE"],
            "linewidth": 0.75,
        },
        bbox={
            "boxstyle": "round,pad=0.13",
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.78,
        },
        zorder=9,
    )

    floor_midpoint = 0.5 * (log_floor_cic + log_floor_mise)
    floor_text_position = (57.0, floor_midpoint)
    ax_a.annotate(
        f"late_floor_ratio = {floor_ratio:.7f}",
        xy=(82.0, log_floor_cic),
        xytext=floor_text_position,
        color="#303030",
        fontsize=6.7,
        ha="left",
        va="center",
        arrowprops={
            "arrowstyle": "-",
            "color": COLORS["CIC"],
            "linewidth": 0.7,
        },
        bbox={
            "boxstyle": "round,pad=0.1",
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.76,
        },
        zorder=9,
    )
    ax_a.annotate(
        "",
        xy=(82.0, log_floor_mise),
        xytext=floor_text_position,
        arrowprops={
            "arrowstyle": "-",
            "color": COLORS["MISE"],
            "linewidth": 0.7,
        },
        zorder=8,
    )

    ax_a.set_xlim(0.0, 90.0)
    ax_a.set_ylim(-4.9, -1.6)
    ax_a.set_xticks(np.arange(0.0, 91.0, 15.0))
    ax_a.set_xlabel(r"Time, $t\omega_{\rm pe}$")
    ax_a.set_ylabel(r"$\log_{10}|\widehat{E}_x(m=1,t)|$")
    ax_a.set_title("(a) Coherent mode erasure and floor collision", loc="left")
    ax_a.legend(
        loc="upper right",
        frameon=False,
        fontsize=7.0,
        handlelength=2.2,
        borderaxespad=0.3,
    )

    # Panel (b): no bandwidth labels are shown, only the analytic prediction.
    bounds = np.linspace(0.0, 1.04, 200)
    ax_b.plot(
        bounds,
        bounds,
        color="#777777",
        linestyle="--",
        linewidth=1.0,
        zorder=1,
    )
    ax_b.scatter(
        predicted,
        measured,
        s=28,
        marker="o",
        facecolor=COLORS["C4"],
        edgecolor="white",
        linewidth=0.55,
        zorder=4,
    )
    ax_b.text(
        0.73,
        0.79,
        r"$y=x$",
        color="#666666",
        fontsize=7.2,
        rotation=39,
        ha="left",
        va="bottom",
    )
    if abs(erasure_slope - 1.0) <= 0.05 and erasure_r2 > 0.99:
        ax_b.text(
            0.06,
            0.67,
            "erasure predicted and measured\n"
            "across bandwidths,\n"
            "zero fitted parameters",
            fontsize=7.0,
            ha="left",
            va="top",
            bbox={
                "boxstyle": "round,pad=0.18",
                "facecolor": "white",
                "edgecolor": "#B5B5B5",
                "linewidth": 0.5,
                "alpha": 0.9,
            },
            zorder=6,
        )
    ax_b.set_xlim(0.0, 1.04)
    ax_b.set_ylim(0.0, 1.04)
    ax_b.set_aspect("equal", adjustable="box")
    ax_b.set_xticks([0.0, 0.25, 0.50, 0.75, 1.0])
    ax_b.set_yticks([0.0, 0.25, 0.50, 0.75, 1.0])
    ax_b.set_xlabel(r"Predicted $\sqrt{T_{\rm eff}^2}$")
    ax_b.set_ylabel("Measured initial-amplitude ratio")
    ax_b.set_title(
        "(b) Erasure across bandwidths\n"
        rf"slope $={erasure_slope:.4f}$, $R^2={erasure_r2:.4f}$",
        loc="left",
        fontsize=8.5,
    )

    assert_no_forbidden_sigma_labels(fig)

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = FIGURE_DIR / f"{OUTPUT_STEM}.pdf"
    png_path = FIGURE_DIR / f"{OUTPUT_STEM}.png"
    fig.savefig(pdf_path, dpi=300, bbox_inches=None)
    fig.savefig(png_path, dpi=300, bbox_inches=None)
    plt.close(fig)

    PAPER_FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(pdf_path, PAPER_FIGURE_DIR / pdf_path.name)
    shutil.copy2(png_path, PAPER_FIGURE_DIR / png_path.name)
    print(
        f"Wrote {pdf_path} and {png_path}; copied both to {PAPER_FIGURE_DIR}."
    )


if __name__ == "__main__":
    main()
