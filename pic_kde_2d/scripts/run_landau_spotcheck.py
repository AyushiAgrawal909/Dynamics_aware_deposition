#!/usr/bin/env python3
"""
Run one Landau damping spot check and save the tracked-mode amplitude history.
"""

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

from pic_kde_2d.benchmarks.landau_2d import fit_mode_gamma
from pic_kde_2d.benchmarks.theory import landau_gamma_theory, landau_omega_theory
from pic_kde_2d.conventions import landau_box_length, marker_grid_shape, thermal_velocity
from pic_kde_2d.core.grid2d import Grid2D
from pic_kde_2d.core.particles2d import init_landau_2d
from pic_kde_2d.core.pic_loop import run_simulation
from pic_kde_2d.methods.spectral_W_wrappers import make_spectral_W_gaussian_methods


def fit_gamma_r2(times: np.ndarray, amplitudes: np.ndarray, t_min: float, t_max: float) -> tuple[float, float, int]:
    mask = (times >= t_min) & (times <= t_max) & np.isfinite(amplitudes) & (amplitudes > 1.0e-12)
    if np.count_nonzero(mask) < 3:
        return float("nan"), float("nan"), int(np.count_nonzero(mask))
    x = times[mask]
    y = np.log(amplitudes[mask])
    slope, intercept = np.polyfit(x, y, 1)
    y_hat = slope * x + intercept
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else float("nan")
    return float(slope), float(r2), int(np.count_nonzero(mask))


def write_history(path: pathlib.Path, times: np.ndarray, mode_values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    amplitudes = np.abs(mode_values)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["time", "mode_real", "mode_imag", "amplitude", "log_amplitude"])
        writer.writeheader()
        for time, mode, amplitude in zip(times, mode_values, amplitudes):
            writer.writerow(
                {
                    "time": float(time),
                    "mode_real": float(np.real(mode)),
                    "mode_imag": float(np.imag(mode)),
                    "amplitude": float(amplitude),
                    "log_amplitude": float(np.log(max(amplitude, 1.0e-300))),
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a one-seed Landau damping spot check.")
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--Nx", type=int, default=64)
    parser.add_argument("--Ny", type=int, default=8)
    parser.add_argument("--ppc", type=int, default=64)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--n-steps", type=int, default=1200)
    parser.add_argument("--diag-stride", type=int, default=2)
    parser.add_argument("--kx-mode", type=int, default=1)
    parser.add_argument("--k-lambdaD", type=float, default=0.4)
    parser.add_argument("--Ly", type=float, default=8.0)
    parser.add_argument("--alpha", type=float, default=0.02)
    parser.add_argument("--sigma-dx", type=float, default=0.25)
    parser.add_argument("--sigma-y-dx", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-velocity-perturbation", action="store_true")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    v_thermal = thermal_velocity()
    Lx = landau_box_length(args.kx_mode, k_lambdaD=args.k_lambdaD)
    grid = Grid2D(Nx=args.Nx, Ny=args.Ny, Lx=Lx, Ly=args.Ly)
    deposit, gather = make_spectral_W_gaussian_methods(
        grid,
        sigma_x=args.sigma_dx * grid.dx,
        sigma_y=args.sigma_y_dx * grid.dy,
    )
    nx_marker, ny_marker = marker_grid_shape(args.Nx, args.Ny, args.ppc)
    particles = init_landau_2d(
        Nx_marker=nx_marker,
        Ny_marker=ny_marker,
        Lx=Lx,
        Ly=args.Ly,
        v_thermal=v_thermal,
        alpha=args.alpha,
        kx_mode=args.kx_mode,
        ky_mode=0,
        rng=np.random.default_rng(args.seed),
        velocity_perturbation=not args.no_velocity_perturbation,
    )
    history = run_simulation(
        particles,
        grid,
        args.dt,
        args.n_steps,
        deposit,
        gather,
        diag_stride=args.diag_stride,
        tracked_modes=[(args.kx_mode, 0)],
    )
    arrays = history.to_arrays()
    times = arrays["times"]
    mode_values = arrays[f"Ex_mode_{args.kx_mode}_0"]
    amplitudes = np.abs(mode_values)
    gamma_2_15, r2_2_15, n_2_15 = fit_gamma_r2(times, amplitudes, 2.0, 15.0)
    gamma_5_30, r2_5_30, n_5_30 = fit_gamma_r2(times, amplitudes, 5.0, 30.0)
    gamma_5_40, r2_5_40, n_5_40 = fit_gamma_r2(times, amplitudes, 5.0, 40.0)
    gamma_theory = landau_gamma_theory(k_lambdaD=args.k_lambdaD, v_thermal=v_thermal)
    omega_theory = landau_omega_theory(k_lambdaD=args.k_lambdaD, v_thermal=v_thermal)

    history_path = args.output / "landau_spotcheck_history.csv"
    write_history(history_path, times, mode_values)
    summary = {
        "history_csv": str(history_path),
        "Nx": args.Nx,
        "Ny": args.Ny,
        "ppc": args.ppc,
        "seed": args.seed,
        "sigma_dx": args.sigma_dx,
        "velocity_perturbation": not args.no_velocity_perturbation,
        "gamma_theory": gamma_theory,
        "omega_theory": omega_theory,
        "gamma_fit_2_15": fit_mode_gamma(times, amplitudes, 2.0, 15.0, amp_floor=1.0e-12),
        "gamma_fit_2_15_r2": r2_2_15,
        "gamma_fit_2_15_points": n_2_15,
        "gamma_fit_5_30": gamma_5_30,
        "gamma_fit_5_30_r2": r2_5_30,
        "gamma_fit_5_30_points": n_5_30,
        "gamma_fit_5_30_abs_error": abs(gamma_5_30 - gamma_theory),
        "gamma_fit_5_40": gamma_5_40,
        "gamma_fit_5_40_r2": r2_5_40,
        "gamma_fit_5_40_points": n_5_40,
        "gamma_fit_5_40_abs_error": abs(gamma_5_40 - gamma_theory),
    }
    summary_path = args.output / "landau_spotcheck_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
