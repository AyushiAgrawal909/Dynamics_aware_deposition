#!/usr/bin/env python3
"""Run one matched-binomial full-f Landau realization."""

from __future__ import annotations

import argparse
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
from pic_kde_2d.core.particles2d import init_landau_2d_eigenmode
from pic_kde_2d.core.pic_loop import run_simulation
from pic_kde_2d.methods.spectral_W_wrappers import (
    make_spectral_W_binomial_methods,
    make_spectral_W_compensated_binomial_methods,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--passes", type=int, required=True)
    parser.add_argument(
        "--filter-family",
        choices=("binomial", "compensated_binomial"),
        default="binomial",
    )
    parser.add_argument("--compensation-order", type=int, default=1)
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--ppc", type=int, default=1024)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--Nx", type=int, default=64)
    parser.add_argument("--Ny", type=int, default=16)
    parser.add_argument("--Ly", type=float, default=8.0)
    parser.add_argument("--kx-mode", type=int, default=1)
    parser.add_argument("--k-lambdaD", type=float, default=0.4)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--n-steps", type=int, required=True)
    parser.add_argument("--diag-stride", type=int, default=2)
    args = parser.parse_args()
    if args.passes < 0:
        raise ValueError("passes must be nonnegative")

    grid = Grid2D(
        Nx=args.Nx,
        Ny=args.Ny,
        Lx=landau_box_length(args.kx_mode, args.k_lambdaD),
        Ly=args.Ly,
    )
    nx_marker, ny_marker = marker_grid_shape(grid.Nx, grid.Ny, args.ppc)
    particles = init_landau_2d_eigenmode(
        Nx_marker=nx_marker,
        Ny_marker=ny_marker,
        Lx=grid.Lx,
        Ly=grid.Ly,
        v_thermal=thermal_velocity(),
        alpha=args.alpha,
        kx_mode=args.kx_mode,
        omega_r=langmuir_omega_r(args.k_lambdaD),
        rng=np.random.default_rng(args.seed),
    )
    method_factory = (
        make_spectral_W_compensated_binomial_methods
        if args.filter_family == "compensated_binomial"
        else make_spectral_W_binomial_methods
    )
    if args.filter_family == "compensated_binomial":
        deposit, gather = method_factory(
            grid,
            args.passes,
            args.passes,
            compensation_order=args.compensation_order,
        )
    else:
        deposit, gather = method_factory(grid, args.passes, args.passes)
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
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output / "history.npz",
        **arrays,
        ppc=np.asarray([args.ppc], dtype=np.int64),
        passes=np.asarray([args.passes], dtype=np.int64),
        alpha=np.asarray([args.alpha], dtype=np.float64),
        seed=np.asarray([args.seed], dtype=np.int64),
        filter_family=np.asarray([args.filter_family]),
        compensation_order=np.asarray([args.compensation_order], dtype=np.int64),
    )
    metadata = {
        "filter": f"matched {args.filter_family} on rho and E",
        "filter_family": args.filter_family,
        "compensation_order": (
            args.compensation_order
            if args.filter_family == "compensated_binomial"
            else None
        ),
        "passes_x": args.passes,
        "passes_y": args.passes,
        "sigma_eq_dx": (
            float(np.sqrt(args.passes / 2.0))
            if args.filter_family == "binomial"
            else None
        ),
        "alpha": args.alpha,
        "k_lambdaD": args.k_lambdaD,
        "ppc": args.ppc,
        "seed": args.seed,
        "dt": args.dt,
        "n_steps": args.n_steps,
        "quiet_start_layout": "global",
    }
    (args.output / "run_parameters.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
