#!/usr/bin/env python3
"""Generate Paper 1 Figure 2: final matched-transfer anchors."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from paperstyle import (
    COLORS,
    DOUBLE_COL,
    LINESTYLES,
    MARKERS,
    SIGMA_C4_LANDAU,
    SIGMA_CV_BOUND,
    SIGMA_MISE_LANDAU,
    apply_style,
    mark_censored,
)


OUTPUT_DIR = Path(__file__).resolve().parent
FORBIDDEN_LANDAU_MISE_WIDTHS = (12.0, 16.0, 20.0)


def matched_transfer_squared(
    sigma_cells: np.ndarray | float,
    mode: int,
    n_cells: int,
) -> np.ndarray:
    """Return CIC deposit-gather times matched Gaussian transfer."""
    sigma = np.asarray(sigma_cells, dtype=np.float64)
    cic_squared = np.sinc(mode / n_cells) ** 4
    gaussian_squared = np.exp(
        -(2.0 * np.pi * mode * sigma / n_cells) ** 2
    )
    return cic_squared * gaussian_squared


def main() -> None:
    apply_style()

    cic_principal = float(np.sinc(1.0 / 64.0) ** 4)
    assert f"{cic_principal:.10f}" == "0.9983947798"

    e2_m8_w2_at_cv = float(
        np.exp(-(2.0 * np.pi * 8.0 * SIGMA_CV_BOUND / 128.0) ** 2)
    )
    assert f"{e2_m8_w2_at_cv:.2e}" == "7.85e-242"

    plotted_widths = (
        SIGMA_C4_LANDAU,
        SIGMA_MISE_LANDAU,
        SIGMA_CV_BOUND,
    )
    assert not any(
        np.isclose(width, forbidden, rtol=0.0, atol=1.0e-12)
        for width in plotted_widths
        for forbidden in FORBIDDEN_LANDAU_MISE_WIDTHS
    )

    sigma = np.geomspace(0.1, 80.0, 1000)
    landau = matched_transfer_squared(sigma, mode=1, n_cells=64)
    e2 = matched_transfer_squared(sigma, mode=12, n_cells=128)

    c4_transfer = float(
        matched_transfer_squared(SIGMA_C4_LANDAU, mode=1, n_cells=64)
    )
    mise_transfer = float(
        matched_transfer_squared(SIGMA_MISE_LANDAU, mode=1, n_cells=64)
    )
    cv_transfer = float(
        matched_transfer_squared(SIGMA_CV_BOUND, mode=1, n_cells=64)
    )
    mise_erasure = 1.0 / mise_transfer

    fig, ax = plt.subplots(figsize=(DOUBLE_COL, 3.8))
    ax.plot(
        sigma,
        landau,
        color=COLORS["theory"],
        linestyle=LINESTYLES["theory"],
        linewidth=1.35,
        label=r"Landau protected mode: $(m,N_x)=(1,64)$",
        zorder=3,
    )
    ax.plot(
        sigma,
        e2,
        color=COLORS["theory"],
        linestyle=(0, (2.0, 1.5)),
        linewidth=1.05,
        alpha=0.28,
        label=r"E2 protected mode: $(m,N_x)=(12,128)$",
        zorder=2,
    )

    ax.plot(
        SIGMA_C4_LANDAU,
        c4_transfer,
        linestyle="none",
        color=COLORS["C4"],
        markerfacecolor=COLORS["C4"],
        markeredgecolor=COLORS["C4"],
        markersize=6,
        label=r"C4 dynamics-safe: $\sigma=1\,\Delta x$",
        zorder=6,
        **MARKERS["PASS"],
    )
    ax.annotate(
        "C4 dynamics-safe",
        xy=(SIGMA_C4_LANDAU, c4_transfer),
        xytext=(0.35, 0.13),
        arrowprops={
            "arrowstyle": "->",
            "color": COLORS["C4"],
            "linewidth": 0.8,
        },
        color=COLORS["C4"],
        fontsize=8,
    )

    ax.plot(
        SIGMA_MISE_LANDAU,
        mise_transfer,
        linestyle="none",
        color=COLORS["MISE"],
        markerfacecolor="none",
        markeredgecolor=COLORS["MISE"],
        markersize=6,
        label=r"Landau MISE optimum: $\sigma=6\,\Delta x$",
        zorder=6,
        **MARKERS["FAIL"],
    )
    ax.annotate(
        f"{mise_erasure:.2f}x erasure",
        xy=(SIGMA_MISE_LANDAU, mise_transfer),
        xytext=(7.8, 0.045),
        arrowprops={
            "arrowstyle": "->",
            "color": COLORS["MISE"],
            "linewidth": 0.8,
        },
        color=COLORS["MISE"],
        fontsize=8,
        ha="left",
    )

    mark_censored(
        ax,
        SIGMA_CV_BOUND,
        cv_transfer,
        direction="right",
        color=COLORS["LOO_CV"],
        markersize=6,
        label=r"LOO-CV: $\sigma\geq60\,\Delta x$",
        zorder=7,
    )
    ax.annotate(
        r"$W^2\sim7.85\times10^{-242}$ at E2 $m=8$;"
        "\nrepresentable in float64 but operationally zero"
        "\nin the FFT field solve, ~225 decades below relative roundoff",
        xy=(SIGMA_CV_BOUND, cv_transfer),
        xytext=(3.0, 2.5e-12),
        arrowprops={
            "arrowstyle": "->",
            "color": COLORS["LOO_CV"],
            "linewidth": 0.8,
        },
        color=COLORS["LOO_CV"],
        fontsize=8,
        ha="left",
        va="bottom",
    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.1, 80.0)
    ax.set_ylim(1.0e-16, 2.0)
    ax.set_xlabel(r"Gaussian width, $\sigma/\Delta x$")
    ax.set_ylabel(r"Matched principal transfer, $T_{\rm eff}^2$")
    ax.set_title("Matched transfer anchors at final selector widths")
    ax.legend(loc="lower left", frameon=False)
    fig.savefig(OUTPUT_DIR / "fig02_transfer_anchors.pdf")
    fig.savefig(OUTPUT_DIR / "fig02_transfer_anchors.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
