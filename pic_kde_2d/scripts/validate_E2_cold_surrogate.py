#!/usr/bin/env python3
"""Step 2.2: zero-parameter cold two-stream surrogate validation.

Uses the actual E2 parameters and the measured extended gamma sweep. The model
has no fitted parameters:

    A0 = (omega_p / (k v0))^2 * sinc(k dx / 2)^4
    A_sigma = A0 * exp(-(k sigma)^2)

where sigma is in physical units and k is the physical x-wavenumber.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read_csv(path: pathlib.Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def explicit_sinc(x: float | np.ndarray) -> float | np.ndarray:
    x_arr = np.asarray(x, dtype=np.float64)
    out = np.ones_like(x_arr)
    mask = np.abs(x_arr) > 1.0e-14
    out[mask] = np.sin(x_arr[mask]) / x_arr[mask]
    if np.ndim(x) == 0:
        return float(out)
    return out


def cold_growth_rate(k: float, v0: float, A: np.ndarray) -> np.ndarray:
    """Cold symmetric two-stream growth rate for coupling A."""
    A = np.asarray(A, dtype=np.float64)
    inside = (np.sqrt(A * A + 8.0 * A) - (A + 2.0)) / 2.0
    return k * v0 * np.sqrt(np.maximum(inside, 0.0))


def compute_prediction(sigma_dx: np.ndarray, args: argparse.Namespace) -> dict:
    dx = args.Lx / args.Nx
    k = 2.0 * math.pi * args.kx_mode / args.Lx
    mode_fraction = args.kx_mode / args.Nx
    if mode_fraction >= 0.25:
        raise AssertionError(f"alias safety failed: kx_mode/Nx={mode_fraction:.3f} >= 0.25")
    x = k * dx / 2.0
    s_cic_single = explicit_sinc(x) ** 2
    s_cic_deposit_gather = s_cic_single ** 2
    A0 = (args.omega_p / (k * args.v0)) ** 2 * s_cic_deposit_gather
    sigma_phys = sigma_dx * dx
    W2 = np.exp(-(k * sigma_phys) ** 2)
    A_sigma = A0 * W2
    gamma = cold_growth_rate(k, args.v0, A_sigma)
    sigma_min = float(np.min(sigma_dx))
    gamma_ref = cold_growth_rate(k, args.v0, np.asarray([A0 * math.exp(-(k * sigma_min * dx) ** 2)]))[0]
    ratio = gamma / gamma_ref if gamma_ref != 0.0 else np.full_like(gamma, np.nan)
    sigma_stabilize = math.sqrt(math.log(A0)) / (k * dx) if A0 > 1.0 else 0.0
    return {
        "dx": dx,
        "k": k,
        "kdx": k * dx,
        "mode_fraction": mode_fraction,
        "s_cic_single": s_cic_single,
        "s_cic_deposit_gather": s_cic_deposit_gather,
        "A0": A0,
        "sigma_stabilize_dx": sigma_stabilize,
        "gamma_pred": gamma,
        "ratio_pred": ratio,
        "A_sigma": A_sigma,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gamma-csv", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--Nx", type=int, default=128)
    parser.add_argument("--Lx", type=float, default=64.0)
    parser.add_argument("--kx-mode", type=int, default=12)
    parser.add_argument("--v0", type=float, default=0.8)
    parser.add_argument("--omega-p", type=float, default=1.0)
    parser.add_argument("--prekill-max-sigma", type=float, default=0.8)
    parser.add_argument("--measured-stabilization-sigma", type=float, default=1.0)
    parser.add_argument("--rmse-threshold", type=float, default=0.15)
    parser.add_argument("--crossing-rel-threshold", type=float, default=0.20)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    rows = read_csv(args.gamma_csv)
    sigma = np.asarray([float(row["sigma_dx"]) for row in rows], dtype=np.float64)
    ratio = np.asarray([float(row["gamma_ratio"]) for row in rows], dtype=np.float64)
    ratio_lo = np.asarray([float(row["ci_lo"]) / float(rows[0]["gamma_sigma"]) for row in rows], dtype=np.float64)
    ratio_hi = np.asarray([float(row["ci_hi"]) / float(rows[0]["gamma_sigma"]) for row in rows], dtype=np.float64)
    pred = compute_prediction(sigma, args)
    ratio_pred = np.asarray(pred["ratio_pred"], dtype=np.float64)

    prekill = sigma <= args.prekill_max_sigma
    rmse = float(np.sqrt(np.mean((ratio[prekill] - ratio_pred[prekill]) ** 2)))
    sigma_cross = float(pred["sigma_stabilize_dx"])
    crossing_rel_error = abs(sigma_cross - args.measured_stabilization_sigma) / args.measured_stabilization_sigma
    rmse_pass = bool(rmse < args.rmse_threshold)
    crossing_pass = bool(crossing_rel_error < args.crossing_rel_threshold)
    passed = bool(rmse_pass and crossing_pass)

    summary = {
        "A0": float(pred["A0"]),
        "S_CIC_single": float(pred["s_cic_single"]),
        "S_CIC_deposit_gather": float(pred["s_cic_deposit_gather"]),
        "alias_safe": bool(pred["mode_fraction"] < 0.25),
        "k": float(pred["k"]),
        "kdx": float(pred["kdx"]),
        "mode_fraction": float(pred["mode_fraction"]),
        "sigma_stabilize_pred_dx": sigma_cross,
        "sigma_stabilize_measured_dx": args.measured_stabilization_sigma,
        "stabilization_rel_error": crossing_rel_error,
        "prekill_sigma_max": args.prekill_max_sigma,
        "prekill_rmse": rmse,
        "rmse_pass": rmse_pass,
        "crossing_pass": crossing_pass,
        "passed": passed,
    }
    with (args.output / "cold_surrogate_validation.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)

    out_rows = []
    for i, sig in enumerate(sigma):
        out_rows.append(
            {
                "sigma_dx": sig,
                "gamma_ratio_measured": ratio[i],
                "gamma_ratio_pred": ratio_pred[i],
                "A_sigma": float(pred["A_sigma"][i]),
                "prekill_used_for_rmse": bool(prekill[i]),
            }
        )
    with (args.output / "cold_surrogate_curve.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(out_rows[0].keys()))
        writer.writeheader()
        writer.writerows(out_rows)

    sig_dense = np.geomspace(0.2, 3.0, 300)
    pred_dense = compute_prediction(sig_dense, args)
    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    ax.axhspan(-3.2, 0.0, color="#d8d8d8", alpha=0.28, label="stabilized / damped fit")
    ax.errorbar(
        sigma,
        ratio,
        yerr=[ratio - ratio_lo, ratio_hi - ratio],
        fmt="o",
        color="#9f2f22",
        capsize=3,
        label="PIC measurement",
    )
    ax.plot(sig_dense, pred_dense["ratio_pred"], color="#174a7c", linewidth=2.0, label="zero-parameter cold surrogate")
    ax.axvline(sigma_cross, color="#174a7c", linestyle="--", linewidth=1.2, label=rf"predicted kill $\sigma={sigma_cross:.2f}$")
    ax.axvline(args.measured_stabilization_sigma, color="black", linestyle="-.", linewidth=1.2, label="measured kill ~1.0")
    ax.axvline(args.prekill_max_sigma, color="#555555", linestyle=":", linewidth=1.1, label="RMSE cutoff")
    ax.set_xscale("log")
    ax.set_xlim(0.22, 12.5)
    ax.set_ylim(-3.1, 1.1)
    ax.set_xlabel(r"$\sigma/\Delta x$")
    ax.set_ylabel(r"$\gamma_\sigma/\gamma_0$")
    ax.set_title("Step 2.2: cold surrogate validation")
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(args.output / "cold_surrogate_validation.png", dpi=220)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
