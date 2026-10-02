#!/usr/bin/env python3
"""Diagnose delta-f damping bias across smoothing widths and fit windows."""

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
from pic_kde_2d.core.pic_loop_delta_f import run_simulation_delta_f
from pic_kde_2d.methods.delta_f import init_delta_f_from_particles, make_delta_f_methods
from pic_kde_2d.methods.spectral_W import make_W_gaussian
from pic_kde_2d.validation.landau_fit import (
    bootstrap_gamma_ci,
    ensemble_average_amplitude,
    fit_gamma_ensemble,
    fit_gamma_ensemble_fixed_window,
)


def parse_float_list(text: str) -> list[float]:
    return [float(value) for value in text.split(",") if value.strip()]


def run_one(args, sigma_dx: float, seed: int):
    Lx = landau_box_length(1)
    grid = Grid2D(args.Nx, args.Ny, Lx, args.Ly)
    nx_marker, ny_marker = marker_grid_shape(args.Nx, args.Ny, args.ppc)
    markers = init_landau_2d(
        nx_marker, ny_marker, Lx, args.Ly, thermal_velocity(), 0.0, 1,
        velocity_perturbation=False, rng=np.random.default_rng(seed),
    )
    delta = init_delta_f_from_particles(
        markers, thermal_velocity(), 1.0, args.alpha, 2.0 * np.pi / Lx,
    )
    W = make_W_gaussian(grid, sigma_dx * grid.dx, sigma_dx * grid.dy)
    deposit, gather = make_delta_f_methods(W=W)
    history = run_simulation_delta_f(
        delta, grid, args.dt, args.n_steps, deposit, gather,
        diag_stride=args.diag_stride, tracked_modes=[(1, 0)],
    )
    return (
        np.asarray(history.times),
        np.abs(np.asarray(history.mode_amplitudes_Ex[(1, 0)])),
        float(max(history.delta_f_ratio)),
    )


def main(args) -> None:
    args.output.mkdir(parents=True, exist_ok=True)
    gamma_theory = landau_gamma_theory(0.4)
    summaries: list[dict] = []
    trajectory_rows: list[dict] = []
    for sigma in parse_float_list(args.sigmas):
        amplitudes = []
        validity = []
        times = None
        for seed in range(args.seeds):
            times, amp, ratio = run_one(args, sigma, seed)
            amplitudes.append(amp)
            validity.append(ratio)
            print(f"sigma={sigma:g} seed={seed} validity={ratio:.6f}", flush=True)
        mean_amp = ensemble_average_amplitude(amplitudes)
        for time, amp in zip(times, mean_amp):
            trajectory_rows.append({"sigma_dx": sigma, "time": float(time), "mean_amplitude": float(amp)})
        for tmin, tmax in [(2.0, 10.0), (2.0, 15.0), (5.0, 15.0), (5.0, 20.0), (5.0, 30.0)]:
            fit = fit_gamma_ensemble_fixed_window(amplitudes, times, tmin, tmax)
            ci = bootstrap_gamma_ci(
                amplitudes, times, n_boot=args.nboot, fixed_window=True,
                t_fit_start=tmin, t_fit_stop=tmax,
            )
            summaries.append({
                "sigma_dx": sigma,
                "fit_strategy": "fixed",
                "t_min": tmin,
                "t_max": tmax,
                "gamma": fit["gamma"],
                "gamma_CI_lo": ci["ci_lo"],
                "gamma_CI_hi": ci["ci_hi"],
                "gamma_relative_error": abs(fit["gamma"] - gamma_theory) / abs(gamma_theory),
                "validity_ratio_max": max(validity),
            })
        fit = fit_gamma_ensemble(
            amplitudes, times, t_fit_start=5.0, t_fit_stop_max=30.0,
            noise_floor_start=40.0, noise_factor=8.0,
        )
        ci = bootstrap_gamma_ci(
            amplitudes, times, n_boot=args.nboot, fixed_window=False,
            t_fit_start=5.0, t_fit_stop_max=30.0,
            noise_floor_start=40.0, noise_factor=8.0,
        )
        summaries.append({
            "sigma_dx": sigma,
            "fit_strategy": fit.get("fit_strategy", "adaptive"),
            "t_min": fit.get("t_fit_start"),
            "t_max": fit.get("t_fit_stop"),
            "gamma": fit["gamma"],
            "gamma_CI_lo": ci["ci_lo"],
            "gamma_CI_hi": ci["ci_hi"],
            "gamma_relative_error": abs(fit["gamma"] - gamma_theory) / abs(gamma_theory),
            "validity_ratio_max": max(validity),
        })

    with (args.output / "delta_f_gamma_diagnostic.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    with (args.output / "delta_f_gamma_trajectories.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sigma_dx", "time", "mean_amplitude"])
        writer.writeheader()
        writer.writerows(trajectory_rows)
    with (args.output / "delta_f_gamma_diagnostic.json").open("w", encoding="utf-8") as handle:
        json.dump({"gamma_theory": gamma_theory, "rows": summaries}, handle, indent=2, sort_keys=True)
    print(json.dumps({"gamma_theory": gamma_theory, "rows": summaries}, indent=2, sort_keys=True), flush=True)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=pathlib.Path, required=True)
    p.add_argument("--sigmas", default="0.25,1.5")
    p.add_argument("--seeds", type=int, default=10)
    p.add_argument("--alpha", type=float, default=0.001)
    p.add_argument("--Nx", type=int, default=64)
    p.add_argument("--Ny", type=int, default=16)
    p.add_argument("--ppc", type=int, default=32)
    p.add_argument("--Ly", type=float, default=8.0)
    p.add_argument("--dt", type=float, default=0.05)
    p.add_argument("--n-steps", type=int, default=1200)
    p.add_argument("--diag-stride", type=int, default=4)
    p.add_argument("--nboot", type=int, default=1000)
    return p


if __name__ == "__main__":
    main(parser().parse_args())
