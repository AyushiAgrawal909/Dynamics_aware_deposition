#!/usr/bin/env python3
"""Measure matched-binomial density MISE for one paired benchmark seed."""

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
from pic_kde_2d.core.pic_loop import pic_step
from pic_kde_2d.methods.cic import deposit_cic_2d
from pic_kde_2d.methods.spectral_W import (
    apply_W_to_grid,
    make_W_binomial,
    make_W_compensated_binomial,
)
from pic_kde_2d.methods.spectral_W_wrappers import make_spectral_W_gaussian_methods


PASSES = (0, 1, 2, 4, 8, 12, 13, 16)


def particles(ppc: int, seed: int, grid: Grid2D, args):
    nx_marker, ny_marker = marker_grid_shape(grid.Nx, grid.Ny, ppc)
    if args.benchmark.startswith("landau"):
        return init_landau_2d_eigenmode(
            Nx_marker=nx_marker,
            Ny_marker=ny_marker,
            Lx=grid.Lx,
            Ly=grid.Ly,
            v_thermal=thermal_velocity(),
            alpha=args.alpha,
            kx_mode=1,
            omega_r=langmuir_omega_r(0.4),
            rng=np.random.default_rng(seed),
        )
    return init_two_stream_2d(
        Nx_marker=nx_marker,
        Ny_marker=ny_marker,
        Lx=grid.Lx,
        Ly=grid.Ly,
        v_beam=0.8,
        v_thermal=0.05,
        alpha=0.001,
        kx_mode=12,
        rng=np.random.default_rng(seed),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--benchmark", choices=("landau_weak", "landau_strong", "e2"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--ppc", type=int, required=True)
    parser.add_argument("--reference-ppc", type=int, required=True)
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument(
        "--filter-family",
        choices=("binomial", "compensated_binomial"),
        default="binomial",
    )
    parser.add_argument("--passes", type=int, nargs="+", default=list(PASSES))
    parser.add_argument("--compensation-orders", type=int, nargs="+", default=[1])
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--n-steps", type=int, required=True)
    parser.add_argument("--diag-stride", type=int, default=2)
    parser.add_argument("--eval-t-min", type=float, required=True)
    parser.add_argument("--eval-t-max", type=float, required=True)
    args = parser.parse_args()
    if len(set(args.passes)) != len(args.passes) or 0 not in args.passes:
        raise ValueError("--passes must be unique and include the CIC baseline 0")
    if any(value < 0 for value in args.passes):
        raise ValueError("filter passes must be nonnegative")
    if any(value < 1 for value in args.compensation_orders):
        raise ValueError("compensation orders must be positive")
    if args.filter_family == "binomial" and args.compensation_orders != [1]:
        raise ValueError("ordinary binomial filtering does not take compensation orders")
    if args.benchmark.startswith("landau"):
        grid = Grid2D(Nx=64, Ny=16, Lx=landau_box_length(1, 0.4), Ly=8.0)
    else:
        grid = Grid2D(Nx=128, Ny=16, Lx=64.0, Ly=16.0)
    low = particles(args.ppc, args.seed, grid, args)
    reference = particles(args.reference_ppc, args.seed, grid, args)
    deposit_live, gather_live = make_spectral_W_gaussian_methods(
        grid,
        0.25 * grid.dx,
        0.25 * grid.dy,
    )
    filter_factory = (
        make_W_compensated_binomial
        if args.filter_family == "compensated_binomial"
        else make_W_binomial
    )
    orders = (
        list(dict.fromkeys(args.compensation_orders))
        if args.filter_family == "compensated_binomial"
        else [0]
    )
    keys = [(order, passes) for order in orders for passes in args.passes]
    filters = {}
    for order, passes in keys:
        if passes == 0:
            continue
        if args.filter_family == "compensated_binomial":
            filters[(order, passes)] = filter_factory(
                grid,
                passes,
                passes,
                compensation_order=order,
            )
        else:
            filters[(order, passes)] = filter_factory(grid, passes, passes)
    errors = {key: [] for key in keys}
    times = []
    for step in range(args.n_steps + 1):
        time = step * args.dt
        if (
            step % args.diag_stride == 0
            and args.eval_t_min <= time <= args.eval_t_max
        ):
            rho_low = deposit_cic_2d(
                low.x, low.y, low.w, grid.Nx, grid.Ny, grid.dx, grid.dy
            )
            rho_ref = deposit_cic_2d(
                reference.x,
                reference.y,
                reference.w,
                grid.Nx,
                grid.Ny,
                grid.dx,
                grid.dy,
            )
            for order, passes in keys:
                candidate = (
                    rho_low
                    if passes == 0
                    else apply_W_to_grid(rho_low, filters[(order, passes)], grid)
                )
                errors[(order, passes)].append(
                    float(np.mean((candidate - rho_ref) ** 2))
                )
            times.append(time)
        if step == args.n_steps:
            break
        pic_step(low, grid, args.dt, deposit_live, gather_live)
        pic_step(reference, grid, args.dt, deposit_live, gather_live)
    if not times:
        raise RuntimeError("no noise diagnostic samples were recorded")
    rows = [
        {
            "benchmark": args.benchmark,
            "seed": args.seed,
            "passes": passes,
            "sigma_eq_dx": (
                float(np.sqrt(passes / 2.0))
                if args.filter_family == "binomial"
                else None
            ),
            "filter_family": args.filter_family,
            "compensation_order": order if order > 0 else None,
            "mise": float(np.mean(errors[(order, passes)])),
            "n_times": len(errors[(order, passes)]),
            "eval_t_start": min(times),
            "eval_t_stop": max(times),
        }
        for order, passes in keys
    ]
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "binomial_noise_seed.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output / "metadata.json").write_text(
        json.dumps(
            {
                "benchmark": args.benchmark,
                "seed": args.seed,
                "ppc": args.ppc,
                "reference_ppc": args.reference_ppc,
                "alpha": args.alpha,
                "eval_window": [args.eval_t_min, args.eval_t_max],
                "passes": list(args.passes),
                "filter_family": args.filter_family,
                "compensation_orders": orders,
                "live_method": "matched Gaussian sigma=0.25 dx",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(rows), flush=True)


if __name__ == "__main__":
    main()
