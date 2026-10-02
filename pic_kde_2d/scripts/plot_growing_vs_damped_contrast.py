#!/usr/bin/env python3
"""Two-panel growing-vs-damped selector contrast."""

from __future__ import annotations

import argparse
import csv
import glob
import pathlib
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read_csv(path: pathlib.Path | str) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def interp(rows: list[dict], x_name: str, y_name: str, x: float) -> float:
    xs = np.asarray([float(row[x_name]) for row in rows], dtype=np.float64)
    ys = np.asarray([float(row[y_name]) for row in rows], dtype=np.float64)
    order = np.argsort(xs)
    return float(np.interp(x, xs[order], ys[order]))


def collect_landau_gamma(e1_dir: pathlib.Path) -> list[dict]:
    rows: list[dict] = []
    pattern = str(e1_dir / "shards" / "k1_sigma*" / "E1_revised_summary_*.csv")
    for path in sorted(glob.glob(pattern)):
        data = read_csv(path)[0]
        rows.append(data)
    rows.sort(key=lambda row: float(row["sigma_dx"]))
    if not rows:
        raise RuntimeError(f"no E1 k=1 gamma summaries found under {e1_dir}")
    return rows


def landau_fidelity_rows(e1_dir: pathlib.Path, landau_mise_rows: list[dict], eps: float) -> tuple[list[dict], float]:
    gamma_rows = collect_landau_gamma(e1_dir)
    gamma_ref = float(next(row["gamma"] for row in gamma_rows if abs(float(row["sigma_dx"]) - 0.25) < 1.0e-12))
    out = []
    for row in gamma_rows:
        sigma = float(row["sigma_dx"])
        rel = abs(float(row["gamma"]) / gamma_ref - 1.0)
        fidelity = max(0.0, 1.0 - rel)
        out.append(
            {
                "sigma_dx": sigma,
                "noise_reduction": interp(landau_mise_rows, "sigma_dx", "noise_reduction", sigma),
                "fidelity": fidelity,
                "gamma": float(row["gamma"]),
                "gamma_ref": gamma_ref,
                "relative_gamma_error": rel,
            }
        )
    feasible = [row["sigma_dx"] for row in out if row["relative_gamma_error"] <= eps]
    sigma_safe = max(feasible) if feasible else float("nan")
    return out, sigma_safe


def add_common_format(ax, title: str, y_label: str | None = None) -> None:
    ax.axhspan(0.0, 0.85, color="#f0c7bd", alpha=0.42, zorder=0)
    ax.axhline(0.85, color="#8a2d22", linestyle=":", linewidth=1.3)
    ax.set_xscale("log")
    ax.set_ylim(0.0, 1.03)
    ax.set_title(title)
    ax.set_xlabel("noise reduction factor")
    if y_label:
        ax.set_ylabel(y_label)
    ax.grid(True, which="both", axis="x", alpha=0.16)
    ax.grid(True, which="major", axis="y", alpha=0.16)


def build_plot(args: argparse.Namespace) -> None:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    e2_noise = read_csv(args.e2_noise_csv)
    e2_gamma = read_csv(args.e2_gamma_csv)
    e2_table = read_csv(args.e2_headline_csv)
    landau_mise = read_csv(args.landau_mise_csv)

    # Add noise reduction to Landau MISE rows, preserving the raw file.
    landau_mise0 = float(landau_mise[0]["mise_mean"])
    for row in landau_mise:
        row["noise_reduction"] = landau_mise0 / float(row["mise_mean"])

    landau_rows, landau_safe_sigma = landau_fidelity_rows(args.e1_dir, landau_mise, args.eps_gamma)

    fig, axes = plt.subplots(1, 2, figsize=(13.4, 5.2), sharey=True)
    ax = axes[0]
    e2_sigma = np.asarray([float(row["sigma_dx"]) for row in e2_noise], dtype=np.float64)
    e2_noise_reduction = np.asarray([float(row["noise_reduction"]) for row in e2_noise], dtype=np.float64)
    gamma_sigma = np.asarray([float(row["sigma_dx"]) for row in e2_gamma], dtype=np.float64)
    gamma_ratio_to_cic = np.asarray(
        [
            float(row["gamma_sigma"])
            / float(next(item["gamma_sigma"] for item in e2_table if item["choice"] == "cic"))
            for row in e2_gamma
        ],
        dtype=np.float64,
    )
    order = np.argsort(gamma_sigma)
    gamma_sigma = gamma_sigma[order]
    gamma_ratio_to_cic = gamma_ratio_to_cic[order]
    e2_fidelity = np.clip(np.interp(e2_sigma, gamma_sigma, gamma_ratio_to_cic), 0.0, 1.0)
    ax.plot(e2_noise_reduction, e2_fidelity, color="#174a7c", linewidth=2.3)
    measured_noise = np.asarray([interp(e2_noise, "sigma_dx", "noise_reduction", s) for s in gamma_sigma])
    ax.scatter(measured_noise, np.clip(gamma_ratio_to_cic, 0.0, 1.0), color="#174a7c", s=24)
    e2_points = {row["choice"]: row for row in e2_table}
    for key, label, color, marker in [
        ("cic", "CIC", "#111111", "o"),
        ("dynamics", "dynamics\ncurrent table\n$0.35\\Delta x$", "#1d6f42", "D"),
        ("density_mise", "density\n$12\\Delta x$", "#9f2f22", "X"),
    ]:
        row = e2_points[key]
        x = float(row["noise_reduction"])
        y = float(np.clip(float(row["gamma_ratio_to_cic"]), 0.0, 1.0))
        ax.scatter(x, y, color=color, marker=marker, s=95, zorder=5)
        ax.annotate(label, (x, y), xytext=(6, 7), textcoords="offset points", fontsize=9, color=color)
    add_common_format(ax, "Growing mode: two-stream", "mode fidelity")
    ax.text(1.1, 0.37, "unsafe", color="#8a2d22", fontsize=10)

    ax = axes[1]
    landau_noise = np.asarray([row["noise_reduction"] for row in landau_rows], dtype=np.float64)
    landau_fidelity = np.asarray([row["fidelity"] for row in landau_rows], dtype=np.float64)
    ax.plot(landau_noise, landau_fidelity, color="#174a7c", linewidth=2.3)
    ax.scatter(landau_noise, landau_fidelity, color="#174a7c", s=24)
    # Mark baseline, dynamics-safe point, and literal MISE minimum.
    base = landau_rows[0]
    ax.scatter(base["noise_reduction"], base["fidelity"], color="#111111", s=95, zorder=5)
    ax.annotate("small-$\\sigma$\nreference", (base["noise_reduction"], base["fidelity"]), xytext=(6, -30), textcoords="offset points", fontsize=9)
    safe_row = min(landau_rows, key=lambda row: abs(row["sigma_dx"] - landau_safe_sigma))
    ax.scatter(safe_row["noise_reduction"], safe_row["fidelity"], color="#1d6f42", marker="D", s=105, zorder=5)
    ax.annotate(
        f"dynamics safe\n$\\sigma\\approx{landau_safe_sigma:.1f}\\Delta x$",
        (safe_row["noise_reduction"], safe_row["fidelity"]),
        xytext=(-92, -58),
        textcoords="offset points",
        fontsize=9,
        color="#1d6f42",
        arrowprops={"arrowstyle": "->", "color": "#1d6f42", "lw": 0.9},
    )
    density_sigma = float(min(landau_mise, key=lambda row: float(row["mise_mean"]))["sigma_dx"])
    density_noise = interp(landau_mise, "sigma_dx", "noise_reduction", density_sigma)
    ax.axvline(density_noise, color="#9f2f22", linestyle="--", linewidth=1.4)
    ax.scatter(
        density_noise,
        1.02,
        facecolors="none",
        edgecolors="#9f2f22",
        marker="o",
        s=115,
        linewidth=1.8,
        zorder=5,
        clip_on=False,
    )
    ax.annotate(
        "density optimum;\nfidelity not measured",
        (density_noise, 1.02),
        xytext=(-190, 14),
        textcoords="offset points",
        fontsize=9,
        color="#8a2d22",
        arrowprops={"arrowstyle": "->", "color": "#8a2d22", "lw": 1.0},
    )
    add_common_format(ax, "Damped mode: Landau")
    ax.text(1.1, 0.37, "unsafe", color="#8a2d22", fontsize=10)

    fig.suptitle("Growing vs damped modes: measured fidelity ranges and open validation gaps", y=1.02)
    fig.tight_layout()
    fig.savefig(args.output, dpi=240, bbox_inches="tight")
    print(f"wrote {args.output}")
    print(f"Landau dynamics-safe sigma: {landau_safe_sigma}")


def build_parser() -> argparse.ArgumentParser:
    root = pathlib.Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Plot growing-vs-damped selector contrast.")
    parser.add_argument("--e2-noise-csv", type=pathlib.Path, default=root / "results" / "part2" / "noise_model.csv")
    parser.add_argument(
        "--e2-gamma-csv",
        type=pathlib.Path,
        default=root / "results" / "part2" / "gamma_sweep_full_schemafix_failed_checkpoint.csv",
    )
    parser.add_argument(
        "--e2-headline-csv",
        type=pathlib.Path,
        default=root / "results" / "part2" / "headline_comparison" / "headline_table.csv",
    )
    parser.add_argument("--landau-mise-csv", type=pathlib.Path, default=root / "results" / "landau_mise" / "landau_mise_summary.csv")
    parser.add_argument("--e1-dir", type=pathlib.Path, default=root / "results" / "E1_remote")
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=root / "results" / "part2" / "headline_comparison" / "growing_vs_damped_contrast.png",
    )
    parser.add_argument("--eps-gamma", type=float, default=0.06)
    return parser


def main() -> None:
    build_plot(build_parser().parse_args())


if __name__ == "__main__":
    main()
