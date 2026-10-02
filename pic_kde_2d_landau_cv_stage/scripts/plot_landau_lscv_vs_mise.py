#!/usr/bin/env python3
"""Plot the aligned Landau LSCV and reference-MISE objective curves."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def interpolate(rows: list[dict], x_name: str, y_name: str, x: np.ndarray) -> np.ndarray:
    source_x = np.asarray([float(row[x_name]) for row in rows], dtype=np.float64)
    source_y = np.asarray([float(row[y_name]) for row in rows], dtype=np.float64)
    order = np.argsort(source_x)
    return np.interp(x, source_x[order], source_y[order])


def normalized_excess(values: np.ndarray) -> np.ndarray:
    shifted = np.asarray(values, dtype=np.float64) - float(np.min(values))
    scale = float(np.max(shifted))
    return shifted / scale if scale > 0.0 else shifted


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lscv-summary", type=Path, required=True)
    parser.add_argument("--lscv-result", type=Path, required=True)
    parser.add_argument("--mise-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    lscv = read_csv(args.lscv_summary)
    mise = read_csv(args.mise_summary)
    result = json.loads(args.lscv_result.read_text(encoding="utf-8"))
    sigma = np.asarray([float(row["sigma_dx"]) for row in lscv], dtype=np.float64)
    lscv_mean = np.asarray([float(row["lscv_mean"]) for row in lscv], dtype=np.float64)
    lscv_lo = np.asarray([float(row["lscv_ci_lo"]) for row in lscv], dtype=np.float64)
    lscv_hi = np.asarray([float(row["lscv_ci_hi"]) for row in lscv], dtype=np.float64)
    mise_on_grid = interpolate(mise, "sigma_dx", "mise_mean", sigma)

    sigma_cv = float(result["sigma_cv_quadratic_dx"])
    sigma_mise = float(
        min(mise, key=lambda row: float(row["mise_mean"]))["sigma_dx"]
    )
    lscv_scale = float(np.ptp(lscv_mean))
    lscv_norm = normalized_excess(lscv_mean)
    lscv_lo_norm = (lscv_lo - float(np.min(lscv_mean))) / lscv_scale
    lscv_hi_norm = (lscv_hi - float(np.min(lscv_mean))) / lscv_scale
    mise_norm = normalized_excess(mise_on_grid)

    fig, ax = plt.subplots(figsize=(5.2, 3.2), dpi=180)
    ax.fill_between(
        sigma,
        lscv_lo_norm,
        lscv_hi_norm,
        color="#6b3fa0",
        alpha=0.16,
        linewidth=0.0,
        label="LSCV 95% CI",
    )
    ax.plot(sigma, lscv_norm, color="#6b3fa0", marker="o", ms=3.0, label="LSCV")
    ax.plot(sigma, mise_norm, color="#8f1d21", marker="s", ms=3.0, label="reference MISE")
    ax.axvline(sigma_cv, color="#6b3fa0", linestyle="--", linewidth=1.1)
    ax.axvline(sigma_mise, color="#8f1d21", linestyle=":", linewidth=1.1)
    ax.annotate(
        rf"$\sigma_{{CV}}={sigma_cv:.2f}\,\Delta x$",
        xy=(sigma_cv, 0.0),
        xytext=(sigma_cv + 1.0, 0.32),
        arrowprops={"arrowstyle": "->", "color": "#6b3fa0"},
        color="#6b3fa0",
    )
    ax.annotate(
        rf"$\sigma_{{MISE}}={sigma_mise:.1f}\,\Delta x$",
        xy=(sigma_mise, float(np.interp(sigma_mise, sigma, mise_norm))),
        xytext=(max(0.3, sigma_mise - 4.8), 0.58),
        arrowprops={"arrowstyle": "->", "color": "#8f1d21"},
        color="#8f1d21",
    )
    ax.set_xlim(0.0, min(20.0, float(np.max(sigma))))
    ax.set_ylim(-0.03, 1.05)
    ax.set_xlabel(r"bandwidth $\sigma/\Delta x$")
    ax.set_ylabel("normalized objective excess above minimum")
    ax.set_title("Aligned Landau density objectives have the same interior scale")
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(args.output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
