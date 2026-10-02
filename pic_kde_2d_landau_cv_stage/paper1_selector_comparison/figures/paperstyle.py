"""Shared publication style and frozen Paper 1 selector constants.

Superseded widths that must not appear as production values in figures:
0.30, 0.35, 0.22, and Landau MISE widths 12, 16, or 20 Delta x.
"""

from __future__ import annotations

import matplotlib.pyplot as plt

SINGLE_COL = 3.4
DOUBLE_COL = 7.0
UNSAFE_THRESHOLD = 0.10

SIGMA_C4_E2 = 0.1844549
SIGMA_C4_LANDAU = 1.0
SIGMA_MISE_LANDAU = 6.0
SIGMA_SILV_LANDAU = 1.833
SIGMA_SILV_E2 = 5.819
SIGMA_CV_BOUND = 60.0

COLORS = {
    "CIC": "#14213d",
    "C4": "#2a7f62",
    "Silverman": "#e6862a",
    "MISE": "#8b1e1e",
    "LOO_CV": "#6f4a8e",
    "WuQin": "#777777",
    "deltaf": "#148c8c",
    "theory": "#111111",
}

MARKERS = {"PASS": "o", "FAIL": "o", "CENSORED": "x"}


def apply_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def mark_censored(ax, x, y, direction="right", **kwargs):
    color = kwargs.pop("color", COLORS["LOO_CV"])
    ax.scatter([x], [y], facecolors="none", edgecolors=color, marker="o", zorder=5)
    dx = 0.18 * x if x else 0.2
    if direction == "left":
        dx = -dx
    ax.annotate("", xy=(x + dx, y), xytext=(x, y), arrowprops={"arrowstyle": "->", "color": color})


def unsafe_band(ax, threshold, above=True):
    if above:
        ax.axhspan(1.0 + threshold, ax.get_ylim()[1], color="#d95f5f", alpha=0.12, zorder=0)
    else:
        ax.axhspan(ax.get_ylim()[0], 1.0 - threshold, color="#d95f5f", alpha=0.12, zorder=0)
