#!/usr/bin/env python3
"""Build Figure 8: mesh-phase fingerprint and grad-h drift hierarchy.

Caption
-------
A physical force is phase-independent: particles and the smooth width profile
were translated together, and only their phase relative to the mesh changed.
A force that changes sign under this translation is therefore a grid artifact.
This identifies the grid-anchored translation-symmetry floor separately from
the omitted grad-h force, which is phase-independent and was corrected by
42.8%. A discontinuous controller raises the integrated drift by 1817x,
demonstrating why a C0 width field is required.
"""

from __future__ import annotations

import csv
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np


FIGURE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = FIGURE_DIR.parents[1]
if str(FIGURE_DIR) not in sys.path:
    sys.path.insert(0, str(FIGURE_DIR))

from paperstyle import COLORS, DOUBLE_COL, apply_style  # noqa: E402


PHASE_SOURCE = (
    PROJECT_ROOT
    / "results"
    / "grid_phase_one_step_4858774"
    / "grid_phase_summary.csv"
)

EXPECTED_PHASES = np.array([0.0, 0.125, 0.25, 0.375, 0.5])
EXPECTED_FORCES = np.array(
    [4.4195e-11, 1.2229e-13, -4.2900e-11, 2.6587e-14, 4.4199e-11]
)

# Frozen full-run audit values. The first two use the same smooth C0 width
# profile; the third uses the superseded discontinuous controller.
DRIFT_GRAD_H_OFF = 1.315e-20
DRIFT_GRAD_H_ON = 7.520e-21
DRIFT_DISCONTINUOUS = 2.390e-17


def load_phase_data(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load the one-step phase audit without adding a pandas dependency."""
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    phases = np.array([float(row["phase_cells"]) for row in rows])
    forces = np.array([float(row["final_force_x"]) for row in rows])
    return phases, forces


def validate_inputs(phases: np.ndarray, forces: np.ndarray) -> None:
    """Gate the plotted mechanisms against the completed audit values."""
    np.testing.assert_allclose(phases, EXPECTED_PHASES, rtol=0.0, atol=1e-14)
    np.testing.assert_allclose(
        forces, EXPECTED_FORCES, rtol=1.1e-5, atol=0.0
    )

    # The endpoints repeat after one full cell, while the quarter-cell phase
    # reverses the force. The near-eighth-cell values locate the zero crossings.
    np.testing.assert_almost_equal(forces[0], forces[-1], decimal=13)
    assert forces[0] * forces[2] < 0.0
    suppressions = np.log10(abs(forces[0]) / np.abs(forces[[1, 3]]))
    np.testing.assert_allclose(
        suppressions, [2.5580, 3.2208], rtol=2e-4, atol=0.0
    )

    reduction = 1.0 - DRIFT_GRAD_H_ON / DRIFT_GRAD_H_OFF
    discontinuity_penalty = DRIFT_DISCONTINUOUS / DRIFT_GRAD_H_OFF
    np.testing.assert_allclose(reduction, 0.4281, rtol=2e-4, atol=0.0)
    np.testing.assert_allclose(
        discontinuity_penalty, 1817.49, rtol=2e-4, atol=0.0
    )


def build_figure(phases: np.ndarray, forces: np.ndarray) -> plt.Figure:
    """Render the signed phase sweep and integrated-drift comparison."""
    apply_style()
    fig, (ax_phase, ax_drift) = plt.subplots(
        1,
        2,
        figsize=(DOUBLE_COL, 3.25),
        gridspec_kw={"width_ratios": [1.65, 1.0], "wspace": 0.36},
    )

    # Panel (a): only the mesh phase changes.
    ax_phase.plot(
        phases,
        forces,
        color=COLORS["CIC"],
        linewidth=1.25,
        marker="o",
        markersize=4.5,
        markerfacecolor="white",
        markeredgewidth=1.0,
        zorder=3,
    )
    ax_phase.axhline(0.0, color="0.25", linewidth=0.8, zorder=1)
    ax_phase.set_yscale("symlog", linthresh=1e-13, linscale=1.0)
    ax_phase.set_xlim(-0.025, 0.525)
    ax_phase.set_ylim(-9e-11, 9e-11)
    ax_phase.set_xticks(phases)
    ax_phase.set_xticklabels(["0", "0.125", "0.25", "0.375", "0.5"])
    ax_phase.set_xlabel(r"Sub-cell mesh phase, $\delta/\Delta x$")
    ax_phase.set_ylabel(r"Total force, $\sum_i F_{i,x}$")
    ax_phase.set_title("(a) Mesh-phase fingerprint")

    ax_phase.annotate(
        "one-cell periodicity\n(endpoints agree to 3 s.f.)",
        xy=(0.5, forces[-1]),
        xytext=(0.54, 0.78),
        textcoords="axes fraction",
        arrowprops={"arrowstyle": "->", "color": "0.25", "linewidth": 0.8},
        fontsize=7.2,
        ha="left",
        va="bottom",
    )
    ax_phase.annotate(
        "half-cell shift\nreverses sign",
        xy=(0.25, forces[2]),
        xytext=(0.58, 0.12),
        textcoords="axes fraction",
        arrowprops={"arrowstyle": "->", "color": "0.25", "linewidth": 0.8},
        fontsize=7.2,
        ha="left",
        va="bottom",
    )
    ax_phase.annotate(
        "near-zero crossings:\n2.6-3.2 orders suppressed",
        xy=(0.125, forces[1]),
        xytext=(0.11, 0.70),
        textcoords="axes fraction",
        arrowprops={"arrowstyle": "->", "color": "0.35", "linewidth": 0.75},
        fontsize=7.2,
        ha="left",
        va="top",
    )
    ax_phase.annotate(
        "",
        xy=(0.375, forces[3]),
        xytext=(0.47, 0.64),
        textcoords="axes fraction",
        arrowprops={"arrowstyle": "->", "color": "0.35", "linewidth": 0.75},
    )

    # Panel (b): the smooth grad-h correction and controller discontinuity are
    # separate mechanisms and therefore shown as distinct bars.
    labels = [
        "smooth " + r"$C^0$" + "\ngrad-h off",
        "smooth " + r"$C^0$" + "\ngrad-h on",
        "discontinuous\ncontroller",
    ]
    drifts = np.array(
        [DRIFT_GRAD_H_OFF, DRIFT_GRAD_H_ON, DRIFT_DISCONTINUOUS]
    )
    bar_colors = ["#8A8A8A", COLORS["C4"], COLORS["MISE"]]
    x = np.arange(len(labels))
    bars = ax_drift.bar(
        x,
        drifts,
        width=0.62,
        color=bar_colors,
        edgecolor="0.2",
        linewidth=0.6,
        zorder=3,
    )
    ax_drift.set_yscale("log")
    ax_drift.set_ylim(4e-21, 8e-17)
    ax_drift.set_xticks(x)
    ax_drift.set_xticklabels(labels)
    ax_drift.tick_params(axis="x", labelsize=7)
    ax_drift.set_ylabel(r"Integrated drift, $|\Delta P_x|$")
    ax_drift.set_title("(b) Full-run momentum drift")

    for bar, value in zip(bars, drifts):
        ax_drift.text(
            bar.get_x() + bar.get_width() / 2.0,
            value * 1.25,
            f"{value:.2e}",
            fontsize=6.9,
            ha="center",
            va="bottom",
        )

    ax_drift.annotate(
        r"grad-h: $-42.8\%$",
        xy=(1.0, DRIFT_GRAD_H_ON),
        xytext=(0.13, 0.30),
        textcoords="axes fraction",
        arrowprops={"arrowstyle": "->", "color": COLORS["C4"], "linewidth": 0.9},
        color=COLORS["C4"],
        fontsize=7.3,
        ha="left",
    )
    ax_drift.annotate(
        "$1817\\times$\ndiscontinuity\npenalty",
        xy=(2.0, DRIFT_DISCONTINUOUS),
        xytext=(0.34, 0.69),
        textcoords="axes fraction",
        arrowprops={"arrowstyle": "->", "color": COLORS["MISE"], "linewidth": 0.9},
        color=COLORS["MISE"],
        fontsize=7.3,
        ha="center",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 0.4},
    )

    fig.subplots_adjust(bottom=0.23)
    fig.text(
        0.5,
        0.025,
        "Particles and width profile translate together: a sign-changing force "
        "under mesh phase is grid anchored, not physical.",
        ha="center",
        va="top",
        fontsize=7.5,
    )
    return fig


def main() -> None:
    phases, forces = load_phase_data(PHASE_SOURCE)
    validate_inputs(phases, forces)
    fig = build_figure(phases, forces)
    for suffix in ("pdf", "png"):
        fig.savefig(FIGURE_DIR / f"fig08_grid_phase.{suffix}")
    plt.close(fig)
    print("Figure 8 assertions passed.")
    print(f"Wrote {FIGURE_DIR / 'fig08_grid_phase.pdf'}")
    print(f"Wrote {FIGURE_DIR / 'fig08_grid_phase.png'}")


if __name__ == "__main__":
    main()
