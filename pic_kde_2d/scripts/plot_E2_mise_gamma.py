#!/usr/bin/env python3
"""Build E2 headline plots: density selectors improve while dynamics break."""

from __future__ import annotations

import argparse
import csv
import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read_csv(path: pathlib.Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def as_float(rows: list[dict], key: str) -> np.ndarray:
    return np.asarray([float(row[key]) for row in rows], dtype=np.float64)


def interpolate_gamma(mise_sigma: np.ndarray, gamma_rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    gamma_sigma = as_float(gamma_rows, "sigma_dx")
    gamma_ratio = as_float(gamma_rows, "growth_ratio_vs_reference")
    order = np.argsort(gamma_sigma)
    gamma_sigma = gamma_sigma[order]
    gamma_ratio = gamma_ratio[order]
    interpolated = np.interp(mise_sigma, gamma_sigma, gamma_ratio, left=np.nan, right=np.nan)
    return gamma_sigma, gamma_ratio


def nearest_value(rows: list[dict], key_x: str, key_y: str, target: float) -> float:
    """Return y at the row whose x is closest to target."""
    row = min(rows, key=lambda item: abs(float(item[key_x]) - target))
    return float(row[key_y])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mise-csv", type=pathlib.Path, required=True)
    parser.add_argument("--gamma-csv", type=pathlib.Path, required=True)
    parser.add_argument("--cv-csv", type=pathlib.Path, default=None)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--safe-sigma", type=float, default=0.35)
    parser.add_argument("--degraded-sigma", type=float, default=0.45)
    parser.add_argument("--xmax", type=float, default=12.0)
    args = parser.parse_args()

    mise_rows = read_csv(args.mise_csv)
    gamma_rows = read_csv(args.gamma_csv)
    cv_rows = read_csv(args.cv_csv) if args.cv_csv is not None and args.cv_csv.exists() else []
    sigma = as_float(mise_rows, "sigma_dx")
    mise = as_float(mise_rows, "mise_mean")
    mise_lo = as_float(mise_rows, "mise_ci_lo")
    mise_hi = as_float(mise_rows, "mise_ci_hi")
    gamma_sigma, gamma_ratio = interpolate_gamma(sigma, gamma_rows)

    order = np.argsort(sigma)
    sigma = sigma[order]
    mise = mise[order]
    mise_lo = mise_lo[order]
    mise_hi = mise_hi[order]

    positive = sigma > 0.0
    sigma_plot = sigma[positive]
    mise_plot = mise[positive]
    mise_lo_plot = mise_lo[positive]
    mise_hi_plot = mise_hi[positive]
    mise_norm = mise_plot / nearest_value(mise_rows, "sigma_dx", "mise_mean", args.safe_sigma)
    mise_lo_norm = mise_lo_plot / nearest_value(mise_rows, "sigma_dx", "mise_mean", args.safe_sigma)
    mise_hi_norm = mise_hi_plot / nearest_value(mise_rows, "sigma_dx", "mise_mean", args.safe_sigma)

    gamma_damage = 1.0 - gamma_ratio
    sigma_cv = None
    if cv_rows:
        sigma_cv = float(min(cv_rows, key=lambda row: float(row["cv_mean"]))["sigma_dx"])
    sigma_mise = float(min(mise_rows, key=lambda row: float(row["mise_mean"]))["sigma_dx"])

    fig, ax_mise = plt.subplots(figsize=(8.8, 5.2))
    ax_damage = ax_mise.twinx()

    ax_mise.axvspan(args.degraded_sigma, args.xmax, color="#f2b8a8", alpha=0.25, label=r"dynamically degraded: $\gamma_\sigma/\gamma_0<0.85$")
    ax_mise.axvspan(args.safe_sigma, args.degraded_sigma, color="#f8df9b", alpha=0.22, label="boundary")
    ax_mise.axvspan(sigma_mise, args.xmax, color="#a86cc1", alpha=0.12, label="selector region")

    ax_mise.plot(sigma_plot, mise_norm, "o-", color="#1b4d89", label=r"Density MISE / MISE$(0.35\Delta x)$")
    ax_mise.fill_between(sigma_plot, mise_lo_norm, mise_hi_norm, color="#1b4d89", alpha=0.18, linewidth=0)
    ax_mise.set_yscale("log")
    ax_mise.set_xscale("log")
    ax_mise.set_xlim(0.09, args.xmax * 1.05)
    ax_mise.set_xlabel(r"$\sigma / \Delta x$")
    ax_mise.set_ylabel("Normalized density MISE (lower is better)", color="#1b4d89")
    ax_mise.tick_params(axis="y", labelcolor="#1b4d89")

    ax_damage.plot(gamma_sigma, gamma_damage, "s--", color="#a33a2b", label=r"Growth-rate damage $1-\gamma_\sigma/\gamma_0$")
    ax_damage.axhline(0.15, color="#a33a2b", linestyle=":", linewidth=1.3)
    ax_damage.set_ylabel("Growth-rate damage (higher is worse)", color="#a33a2b")
    ax_damage.tick_params(axis="y", labelcolor="#a33a2b")
    ax_damage.set_ylim(0.0, 0.55)

    ax_mise.axvline(args.safe_sigma, color="black", linestyle="-.", linewidth=1.2)
    ax_mise.text(args.safe_sigma * 0.93, max(mise_norm) * 0.55, r"$\sigma_{\rm safe}\approx0.35\Delta x$", fontsize=9, ha="right")

    ax_mise.axvline(sigma_mise, color="#5f2a7a", linestyle="--", linewidth=1.2)
    ax_mise.text(sigma_mise * 0.96, min(mise_norm) * 1.8, rf"MISE min $\geq {sigma_mise:g}\Delta x$", color="#5f2a7a", fontsize=9, ha="right")
    if sigma_cv is not None:
        ax_mise.axvline(sigma_cv, color="#6b3fa0", linestyle=":", linewidth=1.8)
        ax_mise.text(sigma_cv * 0.96, min(mise_norm) * 3.0, rf"CV min $\geq {sigma_cv:g}\Delta x$", color="#6b3fa0", fontsize=9, ha="right")

    ax_mise.set_title("E2: density selectors keep improving after dynamics are damaged")
    lines_1, labels_1 = ax_mise.get_legend_handles_labels()
    lines_2, labels_2 = ax_damage.get_legend_handles_labels()
    ax_mise.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper right")
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    print(args.output)


if __name__ == "__main__":
    main()
