"""Shared Matplotlib style and frozen production constants for Paper 1.

Superseded widths that must NOT appear in any figure:

- E2 production widths: 0.30, 0.35, and 0.22 Delta x (0.22 is not a
  production value).
- Landau MISE widths: 12, 16, and 20 Delta x.

The values above may be discussed as historical audit results in prose, but
must not be plotted or labeled as final selector widths.
"""

from __future__ import annotations

from typing import Literal

import matplotlib as mpl


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
    "CIC": "#14213D",
    "C4": "#2A7F62",
    "Silverman": "#E07A1F",
    "MISE": "#8B1E1E",
    "LOO_CV": "#6A3D9A",
    "WuQin": "#737373",
    "deltaf": "#138A8A",
    "theory": "#000000",
}

LINESTYLES = {
    "theory": "--",
}

# Complete plotting keyword dictionaries keep PASS and FAIL distinct even when
# printed in greyscale. CENSORED uses a cross; mark_censored adds the bound
# circle and arrow required for a right- or left-censored selector.
MARKERS = {
    "PASS": {
        "marker": "o",
        "fillstyle": "full",
        "markeredgewidth": 0.9,
    },
    "FAIL": {
        "marker": "o",
        "fillstyle": "none",
        "markeredgewidth": 1.0,
    },
    "CENSORED": {
        "marker": "x",
        "fillstyle": "none",
        "markeredgewidth": 1.2,
    },
}


def apply_style() -> None:
    """Apply the shared publication style to subsequent Matplotlib figures."""
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.alpha": 0.25,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def mark_censored(
    ax,
    x: float,
    y: float,
    direction: Literal["right", "left", "up", "down"] = "right",
    **kwargs,
):
    """Draw a censored point as an open marker, cross, and bound arrow.

    Extra keyword arguments are forwarded to both marker artists. The arrow is
    drawn in display-point units, so it remains legible on linear and log axes.
    """
    offsets = {
        "right": (14, 0),
        "left": (-14, 0),
        "up": (0, 14),
        "down": (0, -14),
    }
    if direction not in offsets:
        raise ValueError("direction must be 'right', 'left', 'up', or 'down'")

    color = kwargs.pop("color", COLORS["LOO_CV"])
    marker_size = kwargs.pop("markersize", 5.0)
    label = kwargs.pop("label", None)
    open_marker, = ax.plot(
        [x],
        [y],
        linestyle="none",
        marker="o",
        markerfacecolor="none",
        markeredgecolor=color,
        markeredgewidth=1.0,
        markersize=marker_size,
        label=label,
        **kwargs,
    )
    cross, = ax.plot(
        [x],
        [y],
        linestyle="none",
        marker=MARKERS["CENSORED"]["marker"],
        color=color,
        markeredgewidth=MARKERS["CENSORED"]["markeredgewidth"],
        markersize=marker_size,
        label="_nolegend_",
        **kwargs,
    )
    arrow = ax.annotate(
        "",
        xy=(x, y),
        xytext=offsets[direction],
        textcoords="offset points",
        arrowprops={
            "arrowstyle": "<-",
            "color": color,
            "linewidth": 0.9,
            "shrinkA": 2.0,
            "shrinkB": 2.0,
        },
        annotation_clip=False,
    )
    return open_marker, cross, arrow


def unsafe_band(ax, threshold: float, above: bool = True, **kwargs):
    """Shade the unsafe side of a horizontal threshold.

    Call this helper after setting the final y limits. For a damage axis use
    ``unsafe_band(ax, UNSAFE_THRESHOLD, above=True)``. For a fidelity-ratio
    axis, call it at ``1 - UNSAFE_THRESHOLD`` with ``above=False`` and at
    ``1 + UNSAFE_THRESHOLD`` with ``above=True``.
    """
    lower, upper = ax.get_ylim()
    color = kwargs.pop("color", "#F3A6A6")
    alpha = kwargs.pop("alpha", 0.12)
    if above:
        start, stop = threshold, upper
    else:
        start, stop = lower, threshold
    return ax.axhspan(
        start,
        stop,
        color=color,
        alpha=alpha,
        linewidth=0.0,
        **kwargs,
    )


__all__ = [
    "COLORS",
    "DOUBLE_COL",
    "LINESTYLES",
    "MARKERS",
    "SIGMA_C4_E2",
    "SIGMA_C4_LANDAU",
    "SIGMA_CV_BOUND",
    "SIGMA_MISE_LANDAU",
    "SIGMA_SILV_E2",
    "SIGMA_SILV_LANDAU",
    "SINGLE_COL",
    "UNSAFE_THRESHOLD",
    "apply_style",
    "mark_censored",
    "unsafe_band",
]
