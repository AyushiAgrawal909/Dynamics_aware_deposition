#!/usr/bin/env python3
"""Plot raw, compensated, and matched-Gaussian filter responses."""

from __future__ import annotations

import math
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from paperstyle import COLORS, DOUBLE_COL, apply_style


FIGURE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = FIGURE_DIR.parents[1]
PAPER_FIGURE_DIR = PROJECT_ROOT.parent / "paper1_source_package" / "figs"

N_PASSES = 4
SIGMA_EQ_CELLS = math.sqrt(N_PASSES / 2.0)
LANDAU_MODE = 2.0 * math.pi / 64.0
E2_MODE = 2.0 * math.pi * 12.0 / 128.0
MEASURED_E2_ATTENUATION = 0.0028624
OUTPUT_STEM = "fig_compensator"


def compensator_coefficients(order: int, n_passes: int = N_PASSES) -> list[int]:
    """Return C(n+j-1, j), j=0,...,order."""
    return [math.comb(n_passes + j - 1, j) for j in range(order + 1)]


def compensated_response(
    kdx: np.ndarray | float,
    order: int,
    n_passes: int = N_PASSES,
) -> np.ndarray:
    """Return (1-s)^n times the order-p truncated compensator."""
    kdx_array = np.asarray(kdx, dtype=np.float64)
    s = np.sin(kdx_array / 2.0) ** 2
    polynomial = np.polynomial.polynomial.polyval(
        s, compensator_coefficients(order, n_passes)
    )
    return (1.0 - s) ** n_passes * polynomial


def raw_binomial_response(
    kdx: np.ndarray | float,
    n_passes: int = N_PASSES,
) -> np.ndarray:
    """Return the exact n-pass binomial transfer function."""
    kdx_array = np.asarray(kdx, dtype=np.float64)
    s = np.sin(kdx_array / 2.0) ** 2
    return (1.0 - s) ** n_passes


def matched_gaussian_response(kdx: np.ndarray | float) -> np.ndarray:
    """Return exp[-(k sigma_eq)^2/2] for sigma_eq=sqrt(n/2) dx."""
    kdx_array = np.asarray(kdx, dtype=np.float64)
    return np.exp(-0.5 * (SIGMA_EQ_CELLS * kdx_array) ** 2)


def verify_e2_response() -> tuple[float, float]:
    """Check the protected E2 attenuation against its measurement."""
    assert compensator_coefficients(1) == [1, 4]
    assert compensator_coefficients(2) == [1, 4, 10]
    assert compensator_coefficients(3) == [1, 4, 10, 20]

    e2_w = float(compensated_response(E2_MODE, order=3))
    e2_w_squared = e2_w**2
    e2_attenuation = 1.0 - e2_w_squared
    measured_w_squared = 1.0 - MEASURED_E2_ATTENUATION

    assert abs(e2_w_squared - measured_w_squared) / measured_w_squared < 0.003, (
        "p=3, n=4 E2 transmission differs from the protected measurement by "
        "more than 0.3%"
    )
    assert (
        abs(e2_attenuation - MEASURED_E2_ATTENUATION)
        / MEASURED_E2_ATTENUATION
        < 0.003
    ), (
        "p=3, n=4 E2 attenuation differs from 0.0028624 by more than 0.3%"
    )
    return e2_w_squared, e2_attenuation


def main() -> None:
    e2_w_squared, e2_attenuation = verify_e2_response()
    print(
        "E2 protected mode: "
        f"W^2={e2_w_squared:.9f}, 1-W^2={e2_attenuation:.9f}, "
        f"measured attenuation={MEASURED_E2_ATTENUATION:.7f}"
    )

    apply_style()
    matplotlib.rcParams["savefig.bbox"] = None

    kdx = np.linspace(0.0, math.pi, 1601)
    raw = raw_binomial_response(kdx)
    compensated = {
        order: compensated_response(kdx, order=order)
        for order in (1, 2, 3)
    }
    gaussian = matched_gaussian_response(kdx)

    fig, ax = plt.subplots(figsize=(DOUBLE_COL, 3.0))
    fig.subplots_adjust(left=0.085, right=0.985, bottom=0.19, top=0.86)

    ax.plot(
        kdx,
        raw,
        color=COLORS["CIC"],
        linewidth=1.55,
        label=r"raw binomial, $n=4$",
        zorder=4,
    )
    ax.plot(
        kdx,
        compensated[1],
        color=COLORS["C4"],
        linewidth=1.45,
        label=r"compensated $p=1$, $n=4$",
        zorder=4,
    )
    ax.plot(
        kdx,
        compensated[2],
        color=COLORS["Silverman"],
        linewidth=1.45,
        label=r"compensated $p=2$, $n=4$",
        zorder=4,
    )
    ax.plot(
        kdx,
        compensated[3],
        color=COLORS["MISE"],
        linewidth=1.55,
        label=r"compensated $p=3$, $n=4$",
        zorder=5,
    )
    ax.plot(
        kdx,
        gaussian,
        color=COLORS["WuQin"],
        linestyle="--",
        linewidth=1.25,
        label=r"matched Gaussian, $\sigma_{\rm eq}=\sqrt{2}\,\Delta x$",
        zorder=3,
    )

    ax.axvline(
        LANDAU_MODE,
        color=COLORS["C4"],
        linestyle=":",
        linewidth=1.05,
        zorder=2,
    )
    ax.axvline(
        E2_MODE,
        color=COLORS["MISE"],
        linestyle=":",
        linewidth=1.05,
        zorder=2,
    )
    ax.plot(
        E2_MODE,
        math.sqrt(1.0 - MEASURED_E2_ATTENUATION),
        marker="o",
        markersize=6.2,
        markerfacecolor="white",
        markeredgecolor=COLORS["MISE"],
        markeredgewidth=1.35,
        linestyle="none",
        label="measured E2 single-mode check",
        zorder=7,
    )
    ax.text(
        LANDAU_MODE + 0.025,
        0.035,
        r"Landau: $m=1$, $N_x=64$",
        color=COLORS["C4"],
        fontsize=7.0,
        rotation=90,
        ha="left",
        va="bottom",
    )
    ax.text(
        E2_MODE + 0.025,
        0.035,
        r"E2: $m=12$, $N_x=128$",
        color=COLORS["MISE"],
        fontsize=7.0,
        rotation=90,
        ha="left",
        va="bottom",
    )

    ax.annotate(
        "passband error near zero\n"
        r"raw: $\mathcal{O}[(k\Delta x)^2]$" "\n"
        r"$p=1$: $\mathcal{O}[(k\Delta x)^4]$" "\n"
        r"$p=2$: $\mathcal{O}[(k\Delta x)^6]$" "\n"
        r"$p=3$: $\mathcal{O}[(k\Delta x)^8]$",
        xy=(0.30, 0.999),
        xytext=(0.74, 0.965),
        fontsize=7.1,
        color="#303030",
        ha="left",
        va="top",
        arrowprops={
            "arrowstyle": "->",
            "color": "#505050",
            "linewidth": 0.75,
        },
        bbox={
            "boxstyle": "round,pad=0.18",
            "facecolor": "white",
            "edgecolor": "#B5B5B5",
            "linewidth": 0.5,
            "alpha": 0.92,
        },
        zorder=8,
    )

    shoulder_x = 1.55
    shoulder_y = float(compensated_response(shoulder_x, order=3))
    ax.annotate(
        "higher order: more low-$k$ fidelity,\n"
        "but less stop-band suppression\n"
        r"Landau $\rightarrow (p=1,\ n=16)$" "\n"
        r"E2 $\rightarrow (p=3,\ n=4)$",
        xy=(shoulder_x, shoulder_y),
        xytext=(1.44, 0.055),
        fontsize=7.2,
        color="#303030",
        ha="left",
        va="bottom",
        arrowprops={
            "arrowstyle": "->",
            "color": COLORS["MISE"],
            "linewidth": 0.8,
        },
        bbox={
            "boxstyle": "round,pad=0.18",
            "facecolor": "white",
            "edgecolor": "#B5B5B5",
            "linewidth": 0.5,
            "alpha": 0.92,
        },
        zorder=8,
    )

    ax.set_xlim(0.0, math.pi)
    ax.set_ylim(0.0, 1.05)
    ax.set_xticks(
        [0.0, math.pi / 4.0, math.pi / 2.0, 3.0 * math.pi / 4.0, math.pi]
    )
    ax.set_xticklabels(
        [r"$0$", r"$\pi/4$", r"$\pi/2$", r"$3\pi/4$", r"$\pi$"]
    )
    ax.set_yticks(np.arange(0.0, 1.01, 0.2))
    ax.set_xlabel(r"Wavenumber $k\Delta x$")
    ax.set_ylabel(r"Filter response $W(k)$")
    ax.set_title(
        "Compensator order trades passband fidelity for stop-band suppression",
        loc="left",
    )
    ax.legend(
        loc="upper right",
        frameon=True,
        framealpha=0.94,
        facecolor="white",
        edgecolor="#B5B5B5",
        fontsize=7.2,
        handlelength=2.6,
        borderpad=0.45,
    )

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
