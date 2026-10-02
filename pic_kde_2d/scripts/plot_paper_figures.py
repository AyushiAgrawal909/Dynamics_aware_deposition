#!/usr/bin/env python3
"""Generate publication figures currently supported by completed E2 data."""

from __future__ import annotations

import argparse
import csv
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read_csv(path: pathlib.Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def col(rows: list[dict], name: str) -> np.ndarray:
    return np.asarray([float(row[name]) for row in rows], dtype=np.float64)


def select_sigma(rows: list[dict], value_col: str) -> float:
    return float(min(rows, key=lambda row: float(row[value_col]))["sigma_dx"])


def figure1(args: argparse.Namespace) -> None:
    mise = read_csv(args.mise_linear)
    cv = read_csv(args.cv_linear)
    gamma = read_csv(args.gamma)
    sigma = col(mise, "sigma_dx")
    mask = sigma > 0.0
    sigma = sigma[mask]
    m = col(mise, "mise_mean")[mask]
    m_lo = col(mise, "mise_ci_lo")[mask]
    m_hi = col(mise, "mise_ci_hi")[mask]
    norm = float(next(row for row in mise if abs(float(row["sigma_dx"]) - args.safe_sigma) < 1e-12)["mise_mean"])
    gs = col(gamma, "sigma_dx")
    gr = col(gamma, "growth_ratio_vs_reference")
    sigma_mise = select_sigma(mise, "mise_mean")
    sigma_cv = select_sigma(cv, "cv_mean")

    fig, ax = plt.subplots(figsize=(8.9, 5.1))
    ax2 = ax.twinx()
    ax.axvspan(args.degraded_sigma, args.xmax, color="#f2b8a8", alpha=0.28, label=r"degraded: $\gamma_\sigma/\gamma_0<0.85$")
    ax.axvspan(args.safe_sigma, args.degraded_sigma, color="#f8df9b", alpha=0.24, label="boundary")
    ax.plot(sigma, m / norm, "o-", color="#174a7c", label="Density MISE")
    ax.fill_between(sigma, m_lo / norm, m_hi / norm, color="#174a7c", alpha=0.18)
    ax2.errorbar(
        gs,
        gr,
        yerr=[gr - col(gamma, "growth_ratio_CI_lo"), col(gamma, "growth_ratio_CI_hi") - gr],
        fmt="s--",
        color="#9f2f22",
        capsize=3,
        label=r"$\gamma_\sigma/\gamma_0$",
    )
    ax.axvline(args.safe_sigma, color="black", linestyle="-.", linewidth=1.2)
    ax.axvline(sigma_mise, color="#5f2a7a", linestyle="--", linewidth=1.4)
    ax.axvline(sigma_cv, color="#6b3fa0", linestyle=":", linewidth=2.0)
    ax.text(args.safe_sigma * 0.92, 0.7, r"$\sigma_{\rm safe}\approx0.35$", ha="right", fontsize=9)
    ax.text(sigma_cv * 0.95, 0.055, rf"$\sigma_{{MISE}},\sigma_{{CV}}\geq {sigma_cv:g}$", ha="right", color="#5f2a7a", fontsize=9)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.09, args.xmax * 1.05)
    ax2.set_ylim(0.0, 1.08)
    ax.set_xlabel(r"$\sigma/\Delta x$")
    ax.set_ylabel(r"Density MISE / MISE$(0.35\Delta x)$")
    ax2.set_ylabel(r"Growth-rate fidelity $\gamma_\sigma/\gamma_0$")
    ax.set_title("Figure 1: density selectors point into the dynamically degraded zone")
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc="upper right")
    fig.tight_layout()
    fig.savefig(args.output / "figure1_headline_selectors_degraded.png", dpi=220)
    plt.close(fig)


def figure2(args: argparse.Namespace) -> None:
    rows = read_csv(args.spectrum)
    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    colors = {0.0: "#333333", 0.35: "#1b7f5a", 0.68: "#d68b00", 3.0: "#9f2f22", 12.0: "#5f2a7a"}
    labels = {0.0: "CIC / sigma=0", 0.35: "safe 0.35", 0.68: "unsafe 0.68", 3.0: "wide 3.0", 12.0: "selector 12"}
    for sigma in [0.0, 0.35, 0.68, 3.0, 12.0]:
        sub = [row for row in rows if abs(float(row["sigma_dx"]) - sigma) < 1e-12]
        mode = col(sub, "mode")
        power = col(sub, "power")
        keep = mode > 0
        ax.plot(mode[keep], power[keep], marker="o", markersize=2.5, linewidth=1.4, color=colors[sigma], label=labels[sigma])
    ax.axvline(args.kx_mode, color="black", linestyle="-.", linewidth=1.1)
    ax.text(args.kx_mode * 1.04, ax.get_ylim()[1] * 0.35, "physical mode m=12", rotation=90, va="center", fontsize=9)
    ax.set_yscale("log")
    ax.set_xlim(1, 64)
    ax.set_xlabel("x-mode number m")
    ax.set_ylabel(r"Mean $|\hat{\rho}(m,k_y=0)|^2$")
    ax.set_title("Figure 2: smoothing erases the unstable-mode spectrum")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(args.output / "figure2_spectrum_smoothing.png", dpi=220)
    plt.close(fig)


def cold_growth(a: float, coupling: np.ndarray) -> np.ndarray:
    """Cold symmetric two-stream growth for effective coupling A."""
    coupling = np.asarray(coupling, dtype=np.float64)
    disc = np.sqrt(coupling * coupling + 8.0 * coupling * a * a)
    y_minus = (2.0 * a * a + coupling - disc) / 2.0
    return np.sqrt(np.maximum(-y_minus, 0.0))


def figure3(args: argparse.Namespace) -> None:
    gamma = read_csv(args.gamma)
    gs = col(gamma, "sigma_dx")
    gr = col(gamma, "growth_ratio_vs_reference")
    lo = col(gamma, "growth_ratio_CI_lo")
    hi = col(gamma, "growth_ratio_CI_hi")
    kdx = 2.0 * np.pi * args.kx_mode / 128.0
    k_phys = 2.0 * np.pi * args.kx_mode / 64.0
    a = k_phys * 0.8
    gamma_ref = float(next(row for row in gamma if abs(float(row["sigma_dx"]) - 0.25) < 1e-12)["gamma_linear"])
    y = -gamma_ref * gamma_ref
    coupling_ref = ((y - a * a) ** 2) / (y + a * a)
    sig = np.geomspace(0.1, 1.5, 160)
    transfer_ratio = np.exp(-(kdx * sig) ** 2) / np.exp(-(kdx * 0.25) ** 2)
    theory = cold_growth(a, coupling_ref * transfer_ratio) / gamma_ref

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    ax.axhspan(0.0, 0.85, color="#f2b8a8", alpha=0.28, label="degraded")
    ax.errorbar(gs, gr, yerr=[gr - lo, hi - gr], fmt="o", color="#9f2f22", capsize=4, label="PIC E2")
    ax.plot(sig, theory, "-", color="#1b4d89", label="cold transfer-calibrated curve")
    ax.axvline(args.safe_sigma, color="black", linestyle="-.", linewidth=1.2, label=r"$\sigma_{\rm safe}$")
    ax.set_xscale("log")
    ax.set_xlim(0.2, 0.8)
    ax.set_ylim(0.35, 1.06)
    ax.set_xlabel(r"$\sigma/\Delta x$")
    ax.set_ylabel(r"$\gamma_\sigma/\gamma_0$")
    ax.set_title("Figure 3: growth-rate fidelity collapses with smoothing")
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(args.output / "figure3_growth_fidelity_theory.png", dpi=220)
    plt.close(fig)


def figure4(args: argparse.Namespace) -> None:
    mise = read_csv(args.mise_nonlinear)
    cv = read_csv(args.cv_nonlinear)
    gamma = read_csv(args.gamma)
    sigma = col(mise, "sigma_dx")
    mask = sigma > 0.0
    sigma = sigma[mask]
    m = col(mise, "mise_mean")[mask]
    m_lo = col(mise, "mise_ci_lo")[mask]
    m_hi = col(mise, "mise_ci_hi")[mask]
    norm = float(next(row for row in mise if abs(float(row["sigma_dx"]) - args.safe_sigma) < 1e-12)["mise_mean"])
    gs = col(gamma, "sigma_dx")
    gr = col(gamma, "growth_ratio_vs_reference")
    sigma_mise = select_sigma(mise, "mise_mean")
    sigma_cv = select_sigma(cv, "cv_mean")

    fig, ax = plt.subplots(figsize=(8.9, 5.1))
    ax2 = ax.twinx()
    ax.axvspan(args.degraded_sigma, args.xmax, color="#f2b8a8", alpha=0.28, label="degraded by linear E2 dynamics")
    ax.axvspan(args.safe_sigma, args.degraded_sigma, color="#f8df9b", alpha=0.24, label="boundary")
    ax.plot(sigma, m / norm, "o-", color="#174a7c", label="Nonlinear-window MISE")
    ax.fill_between(sigma, m_lo / norm, m_hi / norm, color="#174a7c", alpha=0.18)
    ax2.plot(gs, gr, "s--", color="#9f2f22", label=r"$\gamma_\sigma/\gamma_0$")
    ax.axvline(args.safe_sigma, color="black", linestyle="-.", linewidth=1.2)
    ax.axvline(sigma_mise, color="#5f2a7a", linestyle="--", linewidth=1.4)
    ax.axvline(sigma_cv, color="#6b3fa0", linestyle=":", linewidth=2.0)
    ax.text(sigma_cv * 1.05, 0.92, rf"$\sigma_{{CV}}={sigma_cv:g}$", color="#6b3fa0", fontsize=9)
    ax.text(sigma_mise * 0.96, 0.72, rf"$\sigma_{{MISE}}\geq {sigma_mise:g}$", color="#5f2a7a", fontsize=9, ha="right")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.09, args.xmax * 1.05)
    ax2.set_ylim(0.0, 1.08)
    ax.set_xlabel(r"$\sigma/\Delta x$")
    ax.set_ylabel(r"Nonlinear MISE / MISE$(0.35\Delta x)$")
    ax2.set_ylabel(r"Linear growth fidelity $\gamma_\sigma/\gamma_0$")
    ax.set_title("Figure 4: structured-reference defense")
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc="upper right")
    fig.tight_layout()
    fig.savefig(args.output / "figure4_structured_reference.png", dpi=220)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--mise-linear", type=pathlib.Path, required=True)
    p.add_argument("--cv-linear", type=pathlib.Path, required=True)
    p.add_argument("--gamma", type=pathlib.Path, required=True)
    p.add_argument("--spectrum", type=pathlib.Path, required=True)
    p.add_argument("--mise-nonlinear", type=pathlib.Path, required=True)
    p.add_argument("--cv-nonlinear", type=pathlib.Path, required=True)
    p.add_argument("--output", type=pathlib.Path, required=True)
    p.add_argument("--safe-sigma", type=float, default=0.35)
    p.add_argument("--degraded-sigma", type=float, default=0.45)
    p.add_argument("--xmax", type=float, default=12.0)
    p.add_argument("--kx-mode", type=int, default=12)
    return p


def main() -> None:
    args = build_parser().parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    figure1(args)
    figure2(args)
    figure3(args)
    figure4(args)
    print(args.output)


if __name__ == "__main__":
    main()
