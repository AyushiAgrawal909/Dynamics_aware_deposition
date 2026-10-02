#!/usr/bin/env python3
"""
scripts/plot_phase4_results.py
==============================

Plot Phase 4 results from CSV outputs.

This script is optional for remote execution; it is primarily for local analysis.
"""

from __future__ import annotations

import argparse
import csv
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read_csv_rows(path: pathlib.Path) -> list[dict]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def as_float(rows: list[dict], key: str) -> np.ndarray:
    return np.asarray([float(row[key]) for row in rows], dtype=np.float64)


def resolve_csv(results_dir: pathlib.Path, subdir: str, stem: str, tag: str) -> pathlib.Path:
    tagged = results_dir / subdir / f"{stem}_{tag}.csv" if tag else None
    if tagged and tagged.exists():
        return tagged
    plain = results_dir / subdir / f"{stem}.csv"
    if plain.exists():
        return plain
    matches = sorted((results_dir / subdir).glob(f"{stem}*.csv"))
    if not matches:
        raise FileNotFoundError(f"Could not find CSV for {subdir}/{stem} under {results_dir}")
    return matches[0]


def has_csv(results_dir: pathlib.Path, subdir: str, stem: str, tag: str) -> bool:
    try:
        resolve_csv(results_dir, subdir, stem, tag)
    except FileNotFoundError:
        return False
    return True


def plot_E1(results_dir: pathlib.Path, plot_dir: pathlib.Path, tag: str) -> None:
    rows = read_csv_rows(resolve_csv(results_dir, "E1", "E1_summary", tag))
    fig, ax = plt.subplots(figsize=(8, 6))
    k_modes = sorted({int(row["k_mode"]) for row in rows})
    for k_mode in k_modes:
        sub = [row for row in rows if int(row["k_mode"]) == k_mode]
        sub.sort(key=lambda row: float(row["sigma_dx"]))
        sigma = as_float(sub, "sigma_dx")
        gamma_mean = as_float(sub, "gamma_mean")
        ci_lo = as_float(sub, "gamma_CI_lo")
        ci_hi = as_float(sub, "gamma_CI_hi")
        ax.errorbar(sigma, gamma_mean, yerr=[gamma_mean - ci_lo, ci_hi - gamma_mean], marker="o", capsize=3, label=f"k_mode={k_mode}")
    ax.axhline(float(rows[0]["gamma_theory"]), linestyle="--", color="k", label="theory")
    ax.set_xlabel(r"$\sigma / dx$")
    ax.set_ylabel(r"$\gamma$")
    ax.set_title("E1: Landau damping vs sigma")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_dir / "E1_gamma_vs_sigma.png", dpi=120)
    plt.close(fig)


def plot_E2(results_dir: pathlib.Path, plot_dir: pathlib.Path, tag: str) -> None:
    rows = read_csv_rows(resolve_csv(results_dir, "E2", "E2_summary", tag))
    rows.sort(key=lambda row: float(row["sigma_dx"]))
    sigma = as_float(rows, "sigma_dx")
    gamma_mean = as_float(rows, "gamma_mean")
    gamma_std = as_float(rows, "gamma_std")
    peak_mean = as_float(rows, "peak_mean")
    peak_std = as_float(rows, "peak_std")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.errorbar(sigma, gamma_mean, yerr=gamma_std, marker="o", capsize=3)
    ax1.set_xlabel(r"$\sigma / dx$")
    ax1.set_ylabel(r"$\gamma_{\rm linear}$")
    ax1.set_title("E2: Two-stream growth")

    ax2.errorbar(sigma, peak_mean, yerr=peak_std, marker="o", capsize=3)
    ax2.set_xlabel(r"$\sigma / dx$")
    ax2.set_ylabel("Peak amplitude")
    ax2.set_yscale("log")
    ax2.set_title("E2: Two-stream saturation peak")

    fig.tight_layout()
    fig.savefig(plot_dir / "E2_two_stream.png", dpi=120)
    plt.close(fig)


def plot_E3(results_dir: pathlib.Path, plot_dir: pathlib.Path, tag: str) -> None:
    rows = read_csv_rows(resolve_csv(results_dir, "E3", "E3_summary", tag))
    cases = [row["case"] for row in rows]
    metric_key = "tracked_mode_mse_mean" if "tracked_mode_mse_mean" in rows[0] else "field_mse_mean"
    metric_std_key = "tracked_mode_mse_std" if "tracked_mode_mse_std" in rows[0] else "field_mse_std"
    field_mean = as_float(rows, metric_key)
    field_std = as_float(rows, metric_std_key)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(cases, field_mean, yerr=field_std, capsize=5)
    ax.set_ylabel("Tracked-mode MSE vs high-ppc CIC reference")
    ax.set_yscale("log")
    ax.set_title("E3: anisotropic versus isotropic sigma")
    fig.tight_layout()
    fig.savefig(plot_dir / "E3_field_mse.png", dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot Phase 4 results.")
    parser.add_argument("--results-root", type=pathlib.Path, default=pathlib.Path("results"))
    parser.add_argument("--plot-dir", type=pathlib.Path, default=pathlib.Path("plots/phase4"))
    parser.add_argument("--tag", type=str, default="")
    args = parser.parse_args()
    results_dir = args.results_root
    plot_dir = args.plot_dir
    plot_dir.mkdir(parents=True, exist_ok=True)
    if has_csv(results_dir, "E1", "E1_summary", args.tag):
        plot_E1(results_dir, plot_dir, args.tag)
    if has_csv(results_dir, "E2", "E2_summary", args.tag):
        plot_E2(results_dir, plot_dir, args.tag)
    if has_csv(results_dir, "E3", "E3_summary", args.tag):
        plot_E3(results_dir, plot_dir, args.tag)
    print(f"Plots written to {plot_dir}")
