#!/usr/bin/env python3
"""Run one static-width full-f case with complete energy diagnostics."""

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

from pic_kde_2d.benchmarks.theory import langmuir_omega_r
from pic_kde_2d.conventions import landau_box_length, marker_grid_shape, thermal_velocity
from pic_kde_2d.core.grid2d import Grid2D
from pic_kde_2d.core.particles2d import init_landau_2d_eigenmode, init_two_stream_2d
from pic_kde_2d.core.pic_loop import run_simulation
from pic_kde_2d.methods.spectral_W_wrappers import make_spectral_W_gaussian_methods


def build(args):
    if args.benchmark.startswith("landau"):
        grid = Grid2D(Nx=64, Ny=16, Lx=landau_box_length(1, 0.4), Ly=8.0)
        nx_marker, ny_marker = marker_grid_shape(grid.Nx, grid.Ny, args.ppc)
        particles = init_landau_2d_eigenmode(
            Nx_marker=nx_marker,
            Ny_marker=ny_marker,
            Lx=grid.Lx,
            Ly=grid.Ly,
            v_thermal=thermal_velocity(),
            alpha=args.alpha,
            kx_mode=1,
            omega_r=langmuir_omega_r(0.4),
            rng=np.random.default_rng(args.seed),
        )
        mode = (1, 0)
    else:
        grid = Grid2D(Nx=128, Ny=16, Lx=64.0, Ly=16.0)
        nx_marker, ny_marker = marker_grid_shape(grid.Nx, grid.Ny, args.ppc)
        particles = init_two_stream_2d(
            Nx_marker=nx_marker,
            Ny_marker=ny_marker,
            Lx=grid.Lx,
            Ly=grid.Ly,
            v_beam=0.8,
            v_thermal=0.05,
            alpha=0.001,
            kx_mode=12,
            rng=np.random.default_rng(args.seed),
        )
        mode = (12, 0)
    return grid, particles, mode


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--benchmark", choices=("landau_weak", "landau_strong", "e2"), required=True)
    parser.add_argument("--case", choices=("cic", "c4", "large"), required=True)
    parser.add_argument("--sigma-x-dx", type=float, required=True)
    parser.add_argument("--sigma-y-dy", type=float, required=True)
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--ppc", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--n-steps", type=int, required=True)
    parser.add_argument("--diag-stride", type=int, default=2)
    args = parser.parse_args()
    grid, particles, mode = build(args)
    deposit, gather = make_spectral_W_gaussian_methods(
        grid,
        args.sigma_x_dx * grid.dx,
        args.sigma_y_dy * grid.dy,
    )
    history = run_simulation(
        particles,
        grid,
        args.dt,
        args.n_steps,
        deposit,
        gather,
        diag_stride=args.diag_stride,
        tracked_modes=[mode],
    )
    arrays = history.to_arrays()
    times = arrays["times"]
    total = arrays["total_energy"]
    field = arrays["field_energy"]
    kinetic = arrays["kinetic_energy"]
    delta_total = total - total[0]
    relative = delta_total / max(abs(float(total[0])), 1.0e-300)
    peak_index = int(np.argmax(field))
    trend = float(np.polyfit(times, relative, 1)[0])
    summary = {
        "benchmark": args.benchmark,
        "case": args.case,
        "seed": args.seed,
        "sigma_x_dx": args.sigma_x_dx,
        "sigma_y_dy": args.sigma_y_dy,
        "alpha": args.alpha,
        "ppc": args.ppc,
        "dt": args.dt,
        "t_end": float(times[-1]),
        "initial_total_energy": float(total[0]),
        "final_relative_energy_drift": float(relative[-1]),
        "max_abs_relative_energy_drift": float(np.max(np.abs(relative))),
        "relative_energy_drift_slope_per_time": trend,
        "max_abs_drift_over_peak_field_energy": float(
            np.max(np.abs(delta_total)) / max(float(np.max(field)), 1.0e-300)
        ),
        "peak_field_energy": float(field[peak_index]),
        "peak_field_time": float(times[peak_index]),
        "kinetic_change_at_peak_field": float(kinetic[peak_index] - kinetic[0]),
        "energy_exchange_residual_at_peak": float(delta_total[peak_index]),
        "controller_refresh_count": 0,
        "controller_refresh_work": 0.0,
        "static_width": True,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output / "energy_history.npz", **arrays)
    with (args.output / "energy_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary))
        writer.writeheader()
        writer.writerow(summary)
    (args.output / "run_parameters.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
