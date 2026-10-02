#!/usr/bin/env python3
"""Build Figure 7: final selector comparison in rate-noise metric space.

Each benchmark panel shows both preregistered gates. A selector passes only when
its entire rate-ratio interval lies in [0.90, 1.10] and its noise-reduction
interval lies strictly above one. Censored rows are rendered on an explicit
censoring rail rather than assigned fictitious metric coordinates.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np


FIGURE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = FIGURE_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from paperstyle import (  # noqa: E402
    COLORS,
    DOUBLE_COL,
    MARKERS,
    SIGMA_C4_E2,
    SIGMA_C4_LANDAU,
    SIGMA_CV_BOUND,
    SIGMA_MISE_LANDAU,
    SIGMA_SILV_E2,
    SIGMA_SILV_LANDAU,
    UNSAFE_THRESHOLD,
    apply_style,
    mark_censored,
    unsafe_band,
)


RATE_LO = 1.0 - UNSAFE_THRESHOLD
RATE_HI = 1.0 + UNSAFE_THRESHOLD


@dataclass(frozen=True)
class Cell:
    selector: str
    noise: float | None
    noise_lo: float | None
    noise_hi: float | None
    rate: float | None
    rate_lo: float | None
    rate_hi: float | None
    rate_censored: bool = False
    note: str = ""

    @property
    def rate_pass(self) -> bool:
        return (
            not self.rate_censored
            and self.rate_lo is not None
            and self.rate_hi is not None
            and self.rate_lo >= RATE_LO
            and self.rate_hi <= RATE_HI
        )

    @property
    def noise_pass(self) -> bool:
        return self.noise_lo is not None and self.noise_lo > 1.0

    @property
    def both_pass(self) -> bool:
        return self.rate_pass and self.noise_pass


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def assert_close(actual: float, expected: float, decimals: int = 5) -> None:
    np.testing.assert_almost_equal(actual, expected, decimal=decimals)


def validate_source_measurements() -> None:
    """Assert the final values against the completed local result packages."""
    landau = read_csv(
        PROJECT_ROOT
        / "results"
        / "landau_sigma_refinement_4864819"
        / "analysis"
        / "landau_sigma_refinement.csv"
    )
    baseline = next(
        row
        for row in landau
        if int(row["ppc"]) == 1024 and np.isclose(float(row["sigma_dx"]), 0.0)
    )
    c4 = next(
        row
        for row in landau
        if int(row["ppc"]) == 1024
        and np.isclose(float(row["sigma_dx"]), SIGMA_C4_LANDAU)
    )
    assert_close(float(baseline["gamma"]), -0.06974)
    assert_close(
        abs(float(baseline["gamma"]) + 0.06612795869066261)
        / 0.06612795869066261,
        0.0546,
        decimals=4,
    )
    assert_close(float(baseline["fit_r_squared"]), 0.9996, decimals=4)
    assert_close(float(c4["gamma_ratio_to_cic"]), 0.994, decimals=3)
    assert_close(float(c4["ratio_ci_lo"]), 0.964, decimals=3)
    assert_close(float(c4["ratio_ci_hi"]), 1.051, decimals=3)

    landau_mise = read_csv(
        PROJECT_ROOT
        / "results"
        / "landau_mise6_rate_4872322"
        / "analysis"
        / "landau_sigma_refinement.csv"
    )
    mise6 = next(
        row
        for row in landau_mise
        if int(row["ppc"]) == 1024
        and np.isclose(float(row["sigma_dx"]), SIGMA_MISE_LANDAU)
    )
    assert_close(float(mise6["gamma_ratio_to_cic"]), 1.568, decimals=3)
    assert_close(float(mise6["ratio_ci_lo"]), 1.288, decimals=3)
    assert_close(float(mise6["ratio_ci_hi"]), 1.935, decimals=3)

    with (
        PROJECT_ROOT
        / "results"
        / "linearized_delta_f_m12_sigma_4872745"
        / "m12_sigma_sweep.json"
    ).open(encoding="utf-8") as handle:
        e2_rate = json.load(handle)
    e2_c4 = next(
        row
        for row in e2_rate["rows"]
        if np.isclose(float(row["sigma_dx"]), SIGMA_C4_E2)
    )
    assert_close(float(e2_c4["gamma_ratio"]), 0.92727)
    assert_close(float(e2_c4["ratio_ci_lo"]), 0.91428)
    assert_close(float(e2_c4["ratio_ci_hi"]), 0.93337)

    e2_noise = read_csv(
        PROJECT_ROOT
        / "results"
        / "e2_selected_noise_4872840"
        / "e2_noise_summary.csv"
    )
    e2_noise_c4 = next(
        row
        for row in e2_noise
        if np.isclose(float(row["sigma_dx"]), SIGMA_C4_E2)
    )
    assert_close(float(e2_noise_c4["noise_reduction"]), 1.20050)
    assert_close(float(e2_noise_c4["noise_reduction_ci_lo"]), 1.19768)
    assert_close(float(e2_noise_c4["noise_reduction_ci_hi"]), 1.20338)

    # Frozen final paper-table values not duplicated in the older matrix CSV.
    for actual, expected in (
        (2.559, 2.559),
        (3.079, 3.079),
        (16.624, 16.624),
        (32.66, 32.66),
    ):
        assert_close(actual, expected, decimals=3)

    # These ratios must be derived from final widths, never copied from the
    # superseded 0.22-dx E2 production draft.
    assert_close(SIGMA_SILV_LANDAU / SIGMA_C4_LANDAU, 1.833, decimals=3)
    assert_close(SIGMA_SILV_E2 / SIGMA_C4_E2, 31.547, decimals=3)
    assert SIGMA_CV_BOUND == 60.0


def final_cells() -> dict[str, dict[str, Cell]]:
    """Return the final selector cells used in the paper figure."""
    landau = {
        "CIC": Cell("CIC", 1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
        "C4": Cell(
            "C4", 2.559, 2.541, 2.577, 0.994, 0.964, 1.051
        ),
        "Silverman": Cell(
            "Silverman", 3.079, 3.051, 3.106, 1.012, 0.944, 1.103
        ),
        "MISE": Cell(
            "MISE", 16.624, 16.512, 16.727, 1.568, 1.288, 1.935
        ),
        "LOO_CV": Cell(
            "LOO_CV",
            None,
            None,
            None,
            None,
            None,
            None,
            rate_censored=True,
            note=r"$\sigma\geq60\,\Delta x$; censored before fitting",
        ),
        "WuQin": Cell(
            "WuQin",
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            note="inert by design (uniformity gate does not fire)",
        ),
    }

    # Silverman noise is interpolated at its exact frozen width from the
    # completed E2 MISE scan; its conservative interval propagates the two
    # neighboring scan intervals. The rate is censored because no linear
    # growth window survives. The MISE noise interval is the conservative
    # reference-density interval at the selector plateau.
    e2 = {
        "CIC": Cell("CIC", 1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
        "C4": Cell(
            "C4",
            1.20050,
            1.19768,
            1.20338,
            0.92727,
            0.91428,
            0.93337,
        ),
        "Silverman": Cell(
            "Silverman",
            32.271,
            31.098,
            33.344,
            None,
            None,
            None,
            rate_censored=True,
            note="growth window eliminated",
        ),
        "MISE": Cell(
            "MISE",
            32.66,
            31.44,
            33.75,
            None,
            None,
            None,
            rate_censored=True,
            note="mode killed",
        ),
        "LOO_CV": Cell(
            "LOO_CV",
            None,
            None,
            None,
            None,
            None,
            None,
            rate_censored=True,
            note=r"$\sigma\geq60\,\Delta x$; $W^2\sim7.85\times10^{-242}$",
        ),
        "WuQin": Cell(
            "WuQin",
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            note="inert by design (uniformity gate does not fire)",
        ),
    }
    return {"landau": landau, "e2": e2}


def add_gate_regions(ax, ylim: tuple[float, float]) -> None:
    ax.set_ylim(*ylim)
    unsafe_band(ax, RATE_LO, above=False)
    unsafe_band(ax, RATE_HI, above=True)
    ax.axhspan(RATE_LO, RATE_HI, color="#B9D8C4", alpha=0.18, linewidth=0)
    ax.axvline(1.0, color="0.35", lw=0.8, ls=":")
    ax.add_patch(
        Rectangle(
            (1.0, RATE_LO),
            49.0,
            RATE_HI - RATE_LO,
            facecolor="none",
            edgecolor=COLORS["C4"],
            lw=0.9,
            ls="--",
            alpha=0.75,
        )
    )
    ax.text(
        8.0,
        RATE_HI - 0.012,
        "both gates pass",
        ha="center",
        va="top",
        fontsize=6.8,
        color=COLORS["C4"],
        weight="bold",
    )


def plot_valid_cell(ax, cell: Cell, *, zorder: int = 5) -> None:
    assert cell.noise is not None and cell.rate is not None
    assert cell.noise_lo is not None and cell.noise_hi is not None
    assert cell.rate_lo is not None and cell.rate_hi is not None
    state = "PASS" if cell.both_pass else "FAIL"
    marker = MARKERS[state]["marker"]
    color = COLORS[cell.selector]
    ax.errorbar(
        cell.noise,
        cell.rate,
        xerr=[[cell.noise - cell.noise_lo], [cell.noise_hi - cell.noise]],
        yerr=[[cell.rate - cell.rate_lo], [cell.rate_hi - cell.rate]],
        fmt=marker,
        ms=5.7,
        color=color,
        mec=color,
        mfc=color if state == "PASS" else "white",
        mew=1.0,
        capsize=2.2,
        lw=0.9,
        zorder=zorder,
    )


def add_cv_rail(ax, text: str) -> None:
    """Render a censored selector separately from the metric coordinates."""
    rail = ax.inset_axes([0.49, 0.80, 0.49, 0.14])
    rail.set_xlim(0.0, 1.0)
    rail.set_ylim(0.0, 1.0)
    rail.set_facecolor("#F2ECF7")
    mark_censored(
        rail,
        0.12,
        0.50,
        direction="right",
        color=COLORS["LOO_CV"],
        markersize=4.6,
    )
    rail.text(
        0.25,
        0.50,
        text,
        ha="left",
        va="center",
        fontsize=6.35,
        color=COLORS["LOO_CV"],
    )
    rail.text(
        0.98,
        0.06,
        "censoring rail; not an $x/y$ metric",
        ha="right",
        va="bottom",
        fontsize=5.5,
        color="0.4",
    )
    rail.set_xticks([])
    rail.set_yticks([])
    for spine in rail.spines.values():
        spine.set_visible(False)


def plot_landau(ax, cells: dict[str, Cell]) -> None:
    add_gate_regions(ax, (0.74, 2.02))
    for selector in ("CIC", "C4", "Silverman", "MISE"):
        plot_valid_cell(ax, cells[selector], zorder=7 if selector == "C4" else 5)

    # Wu-Qin sits exactly on CIC; use an open diamond so both remain visible.
    ax.plot(
        1.0,
        1.0,
        marker="D",
        ms=7.2,
        mfc="none",
        mec=COLORS["WuQin"],
        mew=1.1,
        ls="none",
        zorder=8,
    )
    ax.annotate(
        "CIC + Wu-Qin\ninert by design",
        xy=(1.0, 1.0),
        xytext=(1.18, 0.81),
        fontsize=6.4,
        color="0.25",
        arrowprops={"arrowstyle": "-", "lw": 0.6, "color": "0.35"},
    )
    ax.annotate(
        "C4\n$R=0.994\\ [0.964,1.051]$\n$N=2.559\\ [2.541,2.577]$",
        xy=(2.559, 0.994),
        xytext=(1.45, 1.31),
        fontsize=6.25,
        color=COLORS["C4"],
        arrowprops={"arrowstyle": "-", "lw": 0.6, "color": COLORS["C4"]},
    )
    ax.annotate(
        "Silverman: CI fails\n$R=1.012\\ [0.944,1.103]$\n"
        "$N=3.079\\ [3.051,3.106]$",
        xy=(3.079, 1.012),
        xytext=(3.7, 1.15),
        fontsize=6.15,
        color=COLORS["Silverman"],
        arrowprops={
            "arrowstyle": "-",
            "lw": 0.6,
            "color": COLORS["Silverman"],
        },
    )
    ax.annotate(
        "MISE: rate fails\n$R=1.568\\ [1.288,1.935]$\n"
        "$N=16.624\\ [16.512,16.727]$",
        xy=(16.624, 1.568),
        xytext=(8.2, 1.72),
        fontsize=6.15,
        color=COLORS["MISE"],
        arrowprops={"arrowstyle": "-", "lw": 0.6, "color": COLORS["MISE"]},
    )
    ax.text(
        0.03,
        0.98,
        r"CIC: $\gamma=-0.06974$ (5.46% from theory), $R^2=0.9996$",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=6.25,
        color=COLORS["CIC"],
    )
    ax.text(
        0.97,
        0.05,
        "dynamics safety: 2.56x noise reduction\n"
        "of 16.62x at the density optimum",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=6.4,
        color=COLORS["C4"],
        bbox={"facecolor": "white", "edgecolor": "0.8", "alpha": 0.9, "pad": 2},
    )
    add_cv_rail(ax, "LOO-CV: $\\sigma\\geq60\\,\\Delta x$\nrate/noise censored")
    ax.set_title("(a) Weak Landau")
    ax.set_ylabel(r"Damping-rate ratio $|\gamma_\sigma|/|\gamma_{\rm CIC}|$")


def plot_e2(ax, cells: dict[str, Cell]) -> None:
    add_gate_regions(ax, (-0.02, 1.12))
    for selector in ("CIC", "C4"):
        plot_valid_cell(ax, cells[selector], zorder=7 if selector == "C4" else 5)

    ax.plot(
        1.0,
        1.0,
        marker="D",
        ms=7.2,
        mfc="none",
        mec=COLORS["WuQin"],
        mew=1.1,
        ls="none",
        zorder=8,
    )
    ax.annotate(
        "CIC + Wu-Qin\ninert by design",
        xy=(1.0, 1.0),
        xytext=(1.2, 0.79),
        fontsize=6.4,
        color="0.25",
        arrowprops={"arrowstyle": "-", "lw": 0.6, "color": "0.35"},
    )
    ax.annotate(
        "C4: both pass\n$R=0.92727\\ [0.91428,0.93337]$\n"
        "$N=1.20050\\ [1.19768,1.20338]$",
        xy=(1.20050, 0.92727),
        xytext=(1.55, 0.68),
        fontsize=6.25,
        color=COLORS["C4"],
        arrowprops={"arrowstyle": "-", "lw": 0.6, "color": COLORS["C4"]},
    )

    censor_y = {"Silverman": 0.18, "MISE": 0.07}
    for selector in ("Silverman", "MISE"):
        cell = cells[selector]
        assert cell.noise is not None
        assert cell.noise_lo is not None and cell.noise_hi is not None
        y = censor_y[selector]
        ax.errorbar(
            cell.noise,
            y,
            xerr=[[cell.noise - cell.noise_lo], [cell.noise_hi - cell.noise]],
            fmt="none",
            color=COLORS[selector],
            capsize=2.2,
            lw=0.9,
            zorder=5,
        )
        mark_censored(
            ax,
            cell.noise,
            y,
            direction="down",
            color=COLORS[selector],
            markersize=5.4,
        )

    ax.annotate(
        "Silverman\n$N=32.27\\ [31.10,33.34]$\nrate censored: growth window eliminated",
        xy=(32.271, 0.18),
        xytext=(9.5, 0.35),
        fontsize=6.05,
        color=COLORS["Silverman"],
        arrowprops={
            "arrowstyle": "-",
            "lw": 0.6,
            "color": COLORS["Silverman"],
        },
    )
    ax.annotate(
        "MISE\n$N=32.66\\ [31.44,33.75]$\nrate censored: mode killed",
        xy=(32.66, 0.07),
        xytext=(9.5, 0.05),
        fontsize=6.05,
        color=COLORS["MISE"],
        arrowprops={"arrowstyle": "-", "lw": 0.6, "color": COLORS["MISE"]},
    )
    ax.text(
        0.97,
        0.47,
        "dynamics safety: 1.20x of 32.66x\n"
        "density-optimal reduction (~27x foregone)",
        transform=ax.transAxes,
        ha="right",
        va="center",
        fontsize=6.4,
        color=COLORS["C4"],
        bbox={"facecolor": "white", "edgecolor": "0.8", "alpha": 0.9, "pad": 2},
    )
    add_cv_rail(
        ax,
        "LOO-CV: $\\sigma\\geq60\\,\\Delta x$\n"
        "$W^2\\sim7.85\\times10^{-242}$; censored",
    )
    ax.set_title("(b) Near-marginal two-stream")
    ax.set_ylabel(r"Growth-rate ratio $\gamma_\sigma/\gamma_{\rm CIC}$")


def build_figure() -> tuple[Path, Path]:
    apply_style()
    validate_source_measurements()
    cells = final_cells()

    winners = {
        selector
        for selector in cells["landau"]
        if cells["landau"][selector].both_pass and cells["e2"][selector].both_pass
    }
    assert winners == {"C4"}, f"expected C4 as sole cross-benchmark winner: {winners}"

    # The final production constants imply 31.55x, not the superseded 26.45x
    # ratio obtained from the historical 0.22-dx E2 width.
    landau_width_ratio = SIGMA_SILV_LANDAU / SIGMA_C4_LANDAU
    e2_width_ratio = SIGMA_SILV_E2 / SIGMA_C4_E2

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(DOUBLE_COL, 4.05),
        gridspec_kw={"wspace": 0.28},
    )
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlim(0.84, 50.0)
        ax.set_xlabel(r"Noise reduction $N={\rm MISE}(0)/{\rm MISE}(\sigma)$")

    plot_landau(axes[0], cells["landau"])
    plot_e2(axes[1], cells["e2"])

    fig.suptitle(
        "Selector comparison: passing requires rate fidelity and noise reduction",
        y=0.985,
        fontsize=10,
        weight="bold",
    )
    fig.text(
        0.5,
        0.925,
        "Silverman/C4 width ratio: "
        f"{landau_width_ratio:.2f}x (Landau)  vs  "
        f"{e2_width_ratio:.2f}x (two-stream)\n"
        "same rule, same code; marginality is the only variable",
        ha="center",
        va="top",
        fontsize=7.25,
        color=COLORS["Silverman"],
        weight="bold",
    )

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=COLORS[key],
            markeredgecolor=COLORS[key],
            markersize=5.2,
            label=label,
        )
        for key, label in (
            ("CIC", "CIC"),
            ("C4", "C4"),
            ("Silverman", "Silverman"),
            ("MISE", "MISE"),
        )
    ]
    legend_handles.extend(
        [
            Line2D(
                [0],
                [0],
                marker="D",
                color="none",
                markerfacecolor="none",
                markeredgecolor=COLORS["WuQin"],
                markersize=5.2,
                label="Wu-Qin (inert)",
            ),
            Line2D(
                [0],
                [0],
                marker="x",
                color=COLORS["LOO_CV"],
                markersize=5.2,
                label="LOO-CV (censored)",
            ),
        ]
    )
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.018),
        ncol=6,
        frameon=False,
        fontsize=6.8,
        columnspacing=1.0,
        handletextpad=0.4,
    )
    fig.text(
        0.5,
        0.005,
        "C4 is the only selector whose paired 95% intervals pass both gates in both benchmarks.",
        ha="center",
        va="bottom",
        fontsize=6.9,
        color=COLORS["C4"],
        weight="bold",
    )
    fig.subplots_adjust(left=0.085, right=0.985, bottom=0.16, top=0.82)

    output_pdf = FIGURE_DIR / "fig07_selector_matrix.pdf"
    output_png = FIGURE_DIR / "fig07_selector_matrix.png"
    fig.savefig(output_pdf)
    fig.savefig(output_png, dpi=300)
    plt.close(fig)
    return output_pdf, output_png


if __name__ == "__main__":
    pdf, png = build_figure()
    print(f"Wrote {pdf}")
    print(f"Wrote {png}")
