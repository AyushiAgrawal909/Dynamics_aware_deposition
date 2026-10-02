#!/usr/bin/env python3
"""Validate delta-f variance reduction against the same matched spectral full-f method."""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import sys

import numpy as np

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pic_kde_2d.benchmarks.theory import landau_gamma_theory
from pic_kde_2d.conventions import landau_box_length, marker_grid_shape, thermal_velocity
from pic_kde_2d.core.grid2d import Grid2D
from pic_kde_2d.core.particles2d import init_landau_2d
from pic_kde_2d.core.pic_loop import run_simulation
from pic_kde_2d.core.pic_loop_delta_f import run_simulation_delta_f
from pic_kde_2d.methods.delta_f import init_delta_f_from_particles, make_delta_f_methods
from pic_kde_2d.methods.spectral_W import make_W_gaussian
from pic_kde_2d.methods.spectral_W_wrappers import make_spectral_W_gaussian_methods
from pic_kde_2d.validation.landau_fit import bootstrap_gamma_ci, fit_gamma_ensemble_fixed_window


def parse_float_list(text: str) -> list[float]:
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def write_csv(path: pathlib.Path, rows: list[dict]) -> None:
    fieldnames = sorted({key for row in rows for key in row}) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def offmode_field_floor(history, grid: Grid2D, mode: tuple[int, int], tail_points: int) -> float:
    """Estimate field noise after removing the resolved physical-mode energy."""
    amps = np.abs(np.asarray(history.mode_amplitudes_Ex[mode], dtype=np.complex128))
    field = np.asarray(history.field_energy, dtype=np.float64)
    tracked_energy = grid.Lx * grid.Ly * amps**2
    residual = np.maximum(field - tracked_energy, 0.0)
    return float(np.mean(residual[-tail_points:]))


def build_case(args, alpha: float, seed: int):
    Lx = landau_box_length(args.kx_mode)
    grid = Grid2D(Nx=args.Nx, Ny=args.Ny, Lx=Lx, Ly=args.Ly)
    nx_marker, ny_marker = marker_grid_shape(args.Nx, args.Ny, args.ppc)
    kx = 2.0 * np.pi * args.kx_mode / Lx
    common = dict(
        Nx_marker=nx_marker,
        Ny_marker=ny_marker,
        Lx=Lx,
        Ly=args.Ly,
        v_thermal=thermal_velocity(),
        kx_mode=args.kx_mode,
        ky_mode=0,
    )
    # Use the same density-only perturbation in both methods. Delta-f captures
    # this initial condition directly in weights without adding marker noise.
    full = init_landau_2d(
        alpha=alpha,
        velocity_perturbation=False,
        rng=np.random.default_rng(seed),
        **common,
    )
    markers = init_landau_2d(
        alpha=0.0,
        velocity_perturbation=False,
        rng=np.random.default_rng(seed),
        **common,
    )
    delta = init_delta_f_from_particles(
        markers,
        f0_v_thermal=thermal_velocity(),
        f0_density=1.0,
        alpha=alpha,
        kx_phys=kx,
    )
    return grid, full, delta


def run_method(args, alpha: float, seed: int, method: str):
    grid, full, delta = build_case(args, alpha, seed)
    sigma = args.sigma_dx * grid.dx
    mode = (args.kx_mode, 0)
    if method == "full_f":
        deposit, gather = make_spectral_W_gaussian_methods(grid, sigma, sigma)
        history = run_simulation(
            full, grid, args.dt, args.n_steps, deposit, gather,
            diag_stride=args.diag_stride, tracked_modes=[mode],
        )
        validity_ratio = None
    else:
        W = make_W_gaussian(grid, sigma, sigma)
        deposit, gather = make_delta_f_methods(W=W)
        history = run_simulation_delta_f(
            delta, grid, args.dt, args.n_steps, deposit, gather,
            diag_stride=args.diag_stride, tracked_modes=[mode],
            validity_threshold=args.validity_threshold,
        )
        validity_ratio = float(max(history.delta_f_ratio))
    amp = np.abs(np.asarray(history.mode_amplitudes_Ex[mode], dtype=np.complex128))
    return {
        "times": np.asarray(history.times, dtype=np.float64),
        "amplitude": amp,
        "offmode_noise_floor": offmode_field_floor(history, grid, mode, args.tail_points),
        "validity_ratio_max": validity_ratio,
        "charge_error_max": float(np.max(np.abs(np.asarray(history.total_charge) - history.total_charge[0]))),
    }


def run_validation(args: argparse.Namespace) -> dict:
    args.output.mkdir(parents=True, exist_ok=True)
    alphas = parse_float_list(args.alphas)
    rows: list[dict] = []
    amplitudes: dict[tuple[float, str], list[np.ndarray]] = {}
    times_by_key: dict[tuple[float, str], np.ndarray] = {}
    for alpha in alphas:
        for method in ["full_f", "delta_f"]:
            amplitudes[(alpha, method)] = []
            floors = []
            for seed in range(args.seed_start, args.seed_start + args.seeds):
                out = run_method(args, alpha, seed, method)
                floors.append(out["offmode_noise_floor"])
                amplitudes[(alpha, method)].append(out["amplitude"])
                times_by_key[(alpha, method)] = out["times"]
                rows.append({
                    "alpha": alpha,
                    "method": method,
                    "seed": seed,
                    "offmode_noise_floor": out["offmode_noise_floor"],
                    "validity_ratio_max": out["validity_ratio_max"],
                    "charge_error_max": out["charge_error_max"],
                })
                print(f"alpha={alpha:g} method={method} seed={seed} noise={out['offmode_noise_floor']:.6e}", flush=True)

    gamma_theory = landau_gamma_theory(0.4)
    summary: list[dict] = []
    for alpha in alphas:
        by_method = {}
        for method in ["full_f", "delta_f"]:
            subset = [row for row in rows if row["alpha"] == alpha and row["method"] == method]
            floors = np.asarray([row["offmode_noise_floor"] for row in subset], dtype=np.float64)
            fit = fit_gamma_ensemble_fixed_window(amplitudes[(alpha, method)], times_by_key[(alpha, method)], 5.0, 30.0)
            ci = bootstrap_gamma_ci(
                amplitudes[(alpha, method)], times_by_key[(alpha, method)],
                n_boot=args.nboot, fixed_window=True, t_fit_start=5.0, t_fit_stop=30.0,
            )
            by_method[method] = float(np.mean(floors))
            summary.append({
                "alpha": alpha,
                "method": method,
                "noise_floor_mean": float(np.mean(floors)),
                "noise_floor_std": float(np.std(floors)),
                "gamma": fit["gamma"],
                "gamma_CI_lo": ci["ci_lo"],
                "gamma_CI_hi": ci["ci_hi"],
                "gamma_theory": gamma_theory,
                "gamma_relative_error": abs(fit["gamma"] - gamma_theory) / abs(gamma_theory),
                "validity_ratio_max": max(
                    (row["validity_ratio_max"] or 0.0 for row in subset), default=0.0
                ),
                "charge_error_max": max(row["charge_error_max"] for row in subset),
            })
        reduction = by_method["full_f"] / max(by_method["delta_f"], 1.0e-30)
        for row in summary:
            if row["alpha"] == alpha:
                row["variance_reduction_factor"] = reduction

    write_csv(args.output / "delta_f_validation.csv", rows)
    write_csv(args.output / "delta_f_summary.csv", summary)
    delta_rows = {row["alpha"]: row for row in summary if row["method"] == "delta_f"}
    scaling = None
    if 0.001 in delta_rows and 0.01 in delta_rows:
        scaling = delta_rows[0.01]["noise_floor_mean"] / max(delta_rows[0.001]["noise_floor_mean"], 1.0e-30)
    target_alpha_row = delta_rows.get(0.01)
    charge_ok = all(row["charge_error_max"] < 1.0e-12 for row in summary if row["method"] == "delta_f")
    gamma_ok = all(row["gamma_relative_error"] < 0.20 for row in summary if row["method"] == "delta_f" and row["alpha"] in (0.001, 0.01))
    reduction_ok = bool(target_alpha_row and target_alpha_row["variance_reduction_factor"] > 30.0)
    scaling_ok = bool(scaling is not None and 30.0 < scaling < 300.0)
    validity_ok = all(row["validity_ratio_max"] < args.validity_threshold for row in summary if row["method"] == "delta_f" and row["alpha"] in (0.001, 0.01))
    result = {
        "summary_csv": str(args.output / "delta_f_summary.csv"),
        "n_seeds": args.seeds,
        "sigma_dx": args.sigma_dx,
        "delta_f_alpha_scaling_0.01_over_0.001": scaling,
        "acceptance": {
            "charge_neutrality": charge_ok,
            "gamma_within_20_percent": gamma_ok,
            "variance_reduction_at_alpha_0.01_gt_30": reduction_ok,
            "alpha_squared_scaling_30_to_300": scaling_ok,
            "validity_through_run": validity_ok,
            "passed": bool(charge_ok and gamma_ok and reduction_ok and scaling_ok and validity_ok),
        },
        "summary": summary,
    }
    with (args.output / "delta_f_result.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Phase 5 delta-f variance validation.")
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--alphas", default="0.001,0.01,0.1")
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--Nx", type=int, default=64)
    parser.add_argument("--Ny", type=int, default=16)
    parser.add_argument("--ppc", type=int, default=32)
    parser.add_argument("--kx-mode", type=int, default=1)
    parser.add_argument("--Ly", type=float, default=8.0)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--n-steps", type=int, default=400)
    parser.add_argument("--diag-stride", type=int, default=4)
    parser.add_argument("--sigma-dx", type=float, default=1.5)
    parser.add_argument("--tail-points", type=int, default=10)
    parser.add_argument("--validity-threshold", type=float, default=0.5)
    parser.add_argument("--nboot", type=int, default=1000)
    return parser


if __name__ == "__main__":
    run_validation(build_parser().parse_args())
