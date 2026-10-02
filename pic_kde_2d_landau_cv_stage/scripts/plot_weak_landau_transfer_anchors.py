"""Plot matched mode transfer at dynamics, MISE, and censored CV widths."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def transfer(sigma_dx: np.ndarray, nx: int = 64, mode: int = 1) -> np.ndarray:
    x = np.pi * mode / nx
    cic_matched = (np.sin(x) / x) ** 4
    k_dx = 2.0 * np.pi * mode / nx
    return cic_matched * np.exp(-(k_dx * sigma_dx) ** 2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    sigma = np.geomspace(0.1, 60.0, 500)
    fig, ax = plt.subplots(figsize=(7.4, 4.7))
    ax.plot(sigma, transfer(sigma), color="#213547", linewidth=2.2, label="matched CIC + Gaussian")
    ax.axvspan(16.0, 20.0, color="#d88923", alpha=0.20, label=r"measured MISE minima: $16$--$20\Delta x$")

    points = {
        0.3: ("dynamics-safe", "#17866b", "D"),
        16.0: ("MISE, PPC 16/32", "#bd6416", "o"),
        20.0: ("MISE, PPC 8", "#8f3f12", "o"),
    }
    for width, (label, color, marker) in points.items():
        value = float(transfer(np.asarray([width]))[0])
        ax.scatter([width], [value], s=58, marker=marker, color=color, zorder=4, label=label)

    cv_width = 60.0
    cv_transfer = float(transfer(np.asarray([cv_width]))[0])
    ax.scatter(
        [cv_width], [cv_transfer], s=72, facecolors="none", edgecolors="#8b2e57",
        linewidths=1.8, zorder=5, label=r"LOO-CV boundary: $\sigma_{CV}\geq60\Delta x$",
    )
    ax.annotate(
        "CV remains boundary-censored",
        xy=(cv_width, cv_transfer), xytext=(24.0, 1.5e-12),
        arrowprops={"arrowstyle": "->", "color": "#8b2e57"},
        color="#8b2e57", fontsize=9,
    )
    ax.annotate("11.8x erasure", xy=(16.0, transfer(np.asarray([16.0]))[0]), xytext=(7.0, 0.035), fontsize=9)
    ax.annotate("47.3x erasure", xy=(20.0, transfer(np.asarray([20.0]))[0]), xytext=(8.0, 0.006), fontsize=9)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.1, 65.0)
    ax.set_ylim(1.0e-16, 1.4)
    ax.set_xlabel(r"bandwidth $\sigma/\Delta x$")
    ax.set_ylabel(r"resolved-mode transfer $T_{\mathrm{eff}}^2$")
    ax.set_title("Weak Landau: density-optimal smoothing erases the physical mode")
    ax.grid(which="both", alpha=0.18)
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=240, bbox_inches="tight")
    print(args.output)


if __name__ == "__main__":
    main()
