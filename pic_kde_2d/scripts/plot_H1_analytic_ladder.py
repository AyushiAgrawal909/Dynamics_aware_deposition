#!/usr/bin/env python3
"""Plot the preregistered H1 local-over-global locality gate."""

from __future__ import annotations

import argparse
import csv
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


METRICS = (
    ("density_mise", "Total density MISE"),
    ("shoulder_density_mse", "Shoulder density MSE"),
    ("field_mse", "Field MSE"),
    ("protected_relative_l2", "Protected-signal error"),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    rows = list(csv.DictReader(args.input.open(encoding="utf-8")))
    rows.sort(key=lambda row: float(row["dynamic_range"]))

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), sharey=True)
    colors = ("#1f6f8b", "#d17a22", "#2a9d68", "#b23a48")
    x = np.arange(len(METRICS))
    for ax, row in zip(axes, rows):
        means = []
        lower = []
        upper = []
        for key, _ in METRICS:
            mean = float(row[f"{key}_local_over_global"])
            lo = float(row[f"{key}_ratio_ci_lo"])
            hi = float(row[f"{key}_ratio_ci_hi"])
            means.append(mean)
            lower.append(mean - lo)
            upper.append(hi - mean)
        ax.bar(x, means, color=colors, width=0.68, zorder=2)
        ax.errorbar(
            x,
            means,
            yerr=np.asarray([lower, upper]),
            fmt="none",
            ecolor="#202020",
            capsize=4,
            lw=1.2,
            zorder=3,
        )
        ax.axhline(1.0, color="#222222", lw=1.2, label="equal to global C4")
        ax.axhline(
            0.9,
            color="#9b2226",
            ls="--",
            lw=1.2,
            label="10% improvement gate",
        )
        ax.set_xticks(x, [label for _, label in METRICS], rotation=24, ha="right")
        ax.set_title(f"Peak/floor = {float(row['dynamic_range']):g}")
        ax.grid(axis="y", alpha=0.2, zorder=0)
    axes[0].set_ylabel("C4-capped Abramson / global C4 error")
    axes[0].legend(frameon=False, loc="upper left")
    axes[0].set_ylim(0.0, 3.25)
    fig.suptitle("H1 analytic locality gate: density and protected-signal criteria fail")
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=220)


if __name__ == "__main__":
    main()
