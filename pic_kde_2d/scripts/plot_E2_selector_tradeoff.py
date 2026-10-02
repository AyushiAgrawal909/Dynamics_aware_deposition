#!/usr/bin/env python3
"""Redraw the E2 selector trade-off figure for paper readability."""

from __future__ import annotations

import argparse
import csv
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


def read_csv(path: pathlib.Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def interp(rows: list[dict], x_name: str, y_name: str, x: float) -> float:
    xs = np.asarray([float(row[x_name]) for row in rows], dtype=np.float64)
    ys = np.asarray([float(row[y_name]) for row in rows], dtype=np.float64)
    order = np.argsort(xs)
    return float(np.interp(x, xs[order], ys[order]))


def clamped_fidelity(ratio: np.ndarray | float) -> np.ndarray | float:
    """Display fidelity as [0, 1.05], with negative/no-growth fits clamped."""
    return np.clip(ratio, 0.0, 1.0)


def build_plot(args: argparse.Namespace) -> None:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    noise_rows = read_csv(args.noise_csv)
    gamma_rows = read_csv(args.gamma_csv)
    table_rows = read_csv(args.headline_csv)
    selector_rows = read_csv(args.selector_csv)

    frontier_noise = np.asarray([float(row["noise_reduction"]) for row in noise_rows], dtype=np.float64)
    frontier_sigma = np.asarray([float(row["sigma_dx"]) for row in noise_rows], dtype=np.float64)
    gamma_sigma = np.asarray([float(row["sigma_dx"]) for row in gamma_rows], dtype=np.float64)
    gamma_sigma_values = np.asarray([float(row["gamma_sigma"]) for row in gamma_rows], dtype=np.float64)
    gamma_cic = float(next(row["gamma_sigma"] for row in table_rows if row["choice"] == "cic"))
    gamma_ratio = gamma_sigma_values / gamma_cic
    order = np.argsort(gamma_sigma)
    gamma_sigma = gamma_sigma[order]
    gamma_ratio = gamma_ratio[order]
    frontier_ratio = np.interp(frontier_sigma, gamma_sigma, gamma_ratio)

    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    unsafe_color = "#f0c7bd"
    ax.axhspan(0.0, args.unsafe_threshold, color=unsafe_color, alpha=0.42, zorder=0)
    ax.text(
        1.12,
        args.unsafe_threshold * 0.55,
        "dynamically unsafe",
        color="#8a2d22",
        fontsize=9,
        va="center",
    )
    ax.axhline(args.unsafe_threshold, color="#8a2d22", linestyle=":", linewidth=1.4)

    frontier_line = ax.plot(
        frontier_noise,
        clamped_fidelity(frontier_ratio),
        color="#174a7c",
        linewidth=2.2,
        label="smoothing frontier",
    )[0]
    measured_noise = np.asarray([interp(noise_rows, "sigma_dx", "noise_reduction", s) for s in gamma_sigma])
    measured_points = ax.scatter(
        measured_noise,
        clamped_fidelity(gamma_ratio),
        color="#174a7c",
        s=26,
        zorder=3,
        label="measured growth fits",
    )

    points = {row["choice"]: row for row in table_rows}
    cic = points["cic"]
    density = points["density_mise"]
    selector_row = min(selector_rows, key=lambda row: abs(float(row["eps_gamma"]) - args.eps_gamma))
    dynamics_sigma = float(selector_row["sigma_star"])
    dynamics_noise = float(selector_row["noise_reduction_at_star"])
    dynamics_ratio = float(selector_row["predicted_ratio"])

    def point_xy(row: dict) -> tuple[float, float]:
        return (
            float(row["noise_reduction"]),
            float(np.clip(float(row["gamma_ratio_to_cic"]), 0.0, 1.05)),
        )

    ax.scatter(*point_xy(cic), s=95, color="#111111", zorder=5)
    ax.annotate("CIC", point_xy(cic), xytext=(8, -16), textcoords="offset points", fontsize=9, weight="bold")

    ax.scatter(dynamics_noise, clamped_fidelity(dynamics_ratio), s=130, color="#1d6f42", zorder=6, marker="D")
    ax.annotate(
        f"dynamics selector\n$\\sigma={dynamics_sigma:.2f}\\Delta x$",
        (dynamics_noise, clamped_fidelity(dynamics_ratio)),
        xytext=(36, -2),
        textcoords="offset points",
        fontsize=9,
        color="#1d6f42",
        weight="bold",
        va="top",
        arrowprops={"arrowstyle": "->", "color": "#1d6f42", "lw": 1.0},
    )

    density_xy = point_xy(density)
    ax.scatter(*density_xy, s=135, color="#9f2f22", zorder=6, marker="X")
    ax.annotate(
        "density selectors\n(MISE = CV coincide)\n$\\sigma=12\\Delta x$",
        density_xy,
        xytext=(-190, 70),
        textcoords="offset points",
        fontsize=9,
        color="#8a2d22",
        weight="bold",
        arrowprops={"arrowstyle": "->", "color": "#8a2d22", "lw": 1.1},
    )

    killed_mask = frontier_sigma >= 1.0
    if np.any(killed_mask):
        x_mid = float(frontier_noise[killed_mask][0] * 1.25)
        ax.annotate(
            "mode killed",
            (x_mid, 0.018),
            xytext=(0, 42),
            textcoords="offset points",
            ha="center",
            fontsize=9,
            color="#8a2d22",
            arrowprops={"arrowstyle": "->", "color": "#8a2d22", "lw": 1.0},
        )

    ax.set_xscale("log")
    ax.set_xlim(max(0.9, frontier_noise.min() * 0.9), frontier_noise.max() * 1.25)
    ax.set_ylim(0.0, 1.03)
    ax.set_xlabel("noise reduction factor  MISE(0) / MISE($\\sigma$)")
    ax.set_ylabel("growth-rate fidelity  $\\gamma_\\sigma / \\gamma_{\\rm CIC}$")
    ax.set_title("E2 selector trade-off: stop at the CIC-referenced dynamics constraint", pad=10)
    ax.grid(True, which="both", axis="x", alpha=0.18)
    ax.grid(True, which="major", axis="y", alpha=0.18)
    selector_handle = Line2D(
        [0],
        [0],
        marker="D",
        color="none",
        markerfacecolor="#1d6f42",
        markeredgecolor="#1d6f42",
        markersize=8,
        label="dynamics selector",
    )
    density_handle = Line2D(
        [0],
        [0],
        marker="X",
        color="none",
        markerfacecolor="#9f2f22",
        markeredgecolor="#9f2f22",
        markersize=8,
        label="density selectors",
    )
    unsafe_handle = Patch(facecolor=unsafe_color, alpha=0.42, label=f"unsafe: fidelity < {args.unsafe_threshold:.2f}")
    ax.legend(
        handles=[frontier_line, measured_points, unsafe_handle, selector_handle, density_handle],
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        fontsize=8,
        frameon=True,
    )
    fig.tight_layout(rect=(0.0, 0.0, 0.82, 1.0))
    fig.savefig(args.output, dpi=240)
    print(f"wrote {args.output}")


def build_parser() -> argparse.ArgumentParser:
    root = pathlib.Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Plot E2 noise-reduction vs fidelity tradeoff.")
    parser.add_argument("--noise-csv", type=pathlib.Path, default=root / "results" / "part2" / "noise_model.csv")
    parser.add_argument(
        "--gamma-csv",
        type=pathlib.Path,
        default=root / "results" / "part2" / "gamma_sweep_full_schemafix_failed_checkpoint.csv",
    )
    parser.add_argument(
        "--headline-csv",
        type=pathlib.Path,
        default=root / "results" / "part2" / "headline_comparison" / "headline_table.csv",
    )
    parser.add_argument(
        "--selector-csv",
        type=pathlib.Path,
        default=root / "results" / "part2" / "selector_result_cic.csv",
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=root / "results" / "part2" / "headline_comparison" / "selector_tradeoff_noise_reduction.png",
    )
    parser.add_argument("--eps-gamma", type=float, default=0.10)
    parser.add_argument("--unsafe-threshold", type=float, default=0.90)
    return parser


def main() -> None:
    build_plot(build_parser().parse_args())


if __name__ == "__main__":
    main()
