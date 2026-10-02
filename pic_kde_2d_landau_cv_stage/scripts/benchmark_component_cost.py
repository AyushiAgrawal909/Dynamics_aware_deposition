#!/usr/bin/env python3
"""Benchmark the measured component cost of CIC, C4, and linearized delta-f.

Each invocation profiles one method in a fresh process.  The Slurm wrapper runs
all methods sequentially on the same allocated node, avoiding cross-node timing
and peak-RSS comparisons.  A full step contains two force evaluations, matching
the production velocity-Verlet and explicit-midpoint delta-f loops.
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import resource
import sys
import time

import numpy as np

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pic_kde_2d.benchmarks.theory import landau_gamma_theory, landau_omega_theory
from pic_kde_2d.conventions import Q_OVER_M, landau_box_length, marker_grid_shape
from pic_kde_2d.core.grid2d import Grid2D
from pic_kde_2d.core.particles2d import init_landau_2d
from pic_kde_2d.methods.cic import deposit_cic_2d, gather_cic_2d
from pic_kde_2d.methods.delta_f import (
    background_velocity_score,
    init_delta_f_from_particles,
)
from pic_kde_2d.methods.spectral_W import apply_W_to_grid, make_W_gaussian


COMPONENTS = ("deposit", "solve", "filter", "gather", "push")


def now() -> float:
    return time.perf_counter()


def add_elapsed(totals: dict[str, float], name: str, start: float) -> None:
    totals[name] += now() - start


def make_full_particles(grid: Grid2D, ppc: int, seed: int, alpha: float):
    nx_marker, ny_marker = marker_grid_shape(grid.Nx, grid.Ny, ppc)
    return init_landau_2d(
        Nx_marker=nx_marker,
        Ny_marker=ny_marker,
        Lx=grid.Lx,
        Ly=grid.Ly,
        v_thermal=1.0,
        alpha=alpha,
        kx_mode=1,
        rng=np.random.default_rng(seed),
        omega_real=landau_omega_theory(0.4, 1.0),
        gamma=landau_gamma_theory(0.4, 1.0),
    )


def force_full(p, grid: Grid2D, W, totals: dict[str, float]):
    start = now()
    rho = deposit_cic_2d(p.x, p.y, p.w, grid.Nx, grid.Ny, grid.dx, grid.dy)
    add_elapsed(totals, "deposit", start)
    if W is not None:
        start = now()
        rho = apply_W_to_grid(rho, W, grid)
        add_elapsed(totals, "filter", start)
    start = now()
    Ex, Ey = grid.poisson_solve(rho)
    add_elapsed(totals, "solve", start)
    if W is not None:
        start = now()
        Ex = apply_W_to_grid(Ex, W, grid)
        Ey = apply_W_to_grid(Ey, W, grid)
        add_elapsed(totals, "filter", start)
    start = now()
    Ex_p, Ey_p = gather_cic_2d(
        Ex, Ey, p.x, p.y, grid.Nx, grid.Ny, grid.dx, grid.dy
    )
    add_elapsed(totals, "gather", start)
    return Ex_p, Ey_p


def step_full(p, grid: Grid2D, W, dt: float) -> dict[str, float]:
    totals = {name: 0.0 for name in COMPONENTS}
    Ex_p, Ey_p = force_full(p, grid, W, totals)
    start = now()
    p.vx += 0.5 * dt * Q_OVER_M * Ex_p
    p.vy += 0.5 * dt * Q_OVER_M * Ey_p
    p.x = (p.x + dt * p.vx) % p.Lx
    p.y = (p.y + dt * p.vy) % p.Ly
    add_elapsed(totals, "push", start)
    Ex_p, Ey_p = force_full(p, grid, W, totals)
    start = now()
    p.vx += 0.5 * dt * Q_OVER_M * Ex_p
    p.vy += 0.5 * dt * Q_OVER_M * Ey_p
    add_elapsed(totals, "push", start)
    return totals


def force_delta(p, grid: Grid2D, totals: dict[str, float]):
    start = now()
    rho = deposit_cic_2d(
        p.x, p.y, p.w_delta, grid.Nx, grid.Ny, grid.dx, grid.dy
    )
    add_elapsed(totals, "deposit", start)
    start = now()
    Ex, Ey = grid.poisson_solve(rho)
    add_elapsed(totals, "solve", start)
    start = now()
    Ex_p, Ey_p = gather_cic_2d(
        Ex, Ey, p.x, p.y, grid.Nx, grid.Ny, grid.dx, grid.dy
    )
    add_elapsed(totals, "gather", start)
    return Ex_p, Ey_p


def weight_rhs(p, Ex_p: np.ndarray, Ey_p: np.ndarray) -> np.ndarray:
    score_x, score_y = background_velocity_score(p)
    return -Q_OVER_M * (Ex_p * score_x + Ey_p * score_y) * p.w_full


def step_linearized_delta(p, grid: Grid2D, dt: float) -> dict[str, float]:
    totals = {name: 0.0 for name in COMPONENTS}
    start = now()
    x_initial = p.x.copy()
    y_initial = p.y.copy()
    w_initial = p.w_delta.copy()
    add_elapsed(totals, "push", start)
    Ex_p, Ey_p = force_delta(p, grid, totals)
    start = now()
    rhs_initial = weight_rhs(p, Ex_p, Ey_p)
    p.x = (x_initial + 0.5 * dt * p.vx) % p.Lx
    p.y = (y_initial + 0.5 * dt * p.vy) % p.Ly
    p.w_delta = w_initial + 0.5 * dt * rhs_initial
    p.w_delta -= np.mean(p.w_delta)
    add_elapsed(totals, "push", start)
    Ex_p, Ey_p = force_delta(p, grid, totals)
    start = now()
    rhs_mid = weight_rhs(p, Ex_p, Ey_p)
    p.x = (x_initial + dt * p.vx) % p.Lx
    p.y = (y_initial + dt * p.vy) % p.Ly
    p.w_delta = w_initial + dt * rhs_mid
    p.w_delta -= np.mean(p.w_delta)
    add_elapsed(totals, "push", start)
    return totals


def persistent_array_bytes(p, grid: Grid2D, W) -> int:
    arrays = [p.x, p.y, p.vx, p.vy]
    if hasattr(p, "w"):
        arrays.append(p.w)
    else:
        arrays.extend((p.w_full, p.w_delta))
    arrays.extend((grid.x, grid.y, grid.kx, grid.ky, grid.KX, grid.KY, grid.k2, grid.k2_inv))
    if W is not None:
        arrays.append(W)
    return int(sum(np.asarray(array).nbytes for array in arrays))


def run(args: argparse.Namespace) -> None:
    args.output.mkdir(parents=True, exist_ok=True)
    grid = Grid2D(Nx=args.Nx, Ny=args.Ny, Lx=landau_box_length(1, 0.4), Ly=args.Ly)
    p = make_full_particles(grid, args.ppc, args.seed, args.alpha)
    W = None
    if args.method == "c4":
        W = make_W_gaussian(grid, args.sigma_dx * grid.dx, args.sigma_dx * grid.dy)
    if args.method == "deltaf":
        equilibrium = make_full_particles(grid, args.ppc, args.seed, 0.0)
        kx = 2.0 * np.pi / grid.Lx
        p = init_delta_f_from_particles(
            equilibrium,
            f0_v_thermal=1.0,
            f0_density=1.0,
            alpha=args.alpha,
            kx_phys=kx,
            omega_real=landau_omega_theory(0.4, 1.0),
            gamma=landau_gamma_theory(0.4, 1.0),
        )

    step_fn = step_linearized_delta if args.method == "deltaf" else step_full
    for _ in range(args.warmup_steps):
        if args.method == "deltaf":
            step_fn(p, grid, args.dt)
        else:
            step_fn(p, grid, W, args.dt)

    samples = {name: [] for name in COMPONENTS}
    total_samples = []
    for _ in range(args.steps):
        start = now()
        if args.method == "deltaf":
            measured = step_fn(p, grid, args.dt)
        else:
            measured = step_fn(p, grid, W, args.dt)
        total_samples.append(now() - start)
        for name in COMPONENTS:
            samples[name].append(measured[name])

    explicit_bytes = persistent_array_bytes(p, grid, W)
    max_rss_native = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    # Linux reports KiB; macOS reports bytes. The production job runs on Linux,
    # but retaining the platform branch makes local smoke tests meaningful.
    max_rss_bytes = max_rss_native if sys.platform == "darwin" else max_rss_native * 1024
    row = {
        "method": args.method,
        "protocol": "linearized_delta_f" if args.method == "deltaf" else "full_f",
        "Nx": grid.Nx,
        "Ny": grid.Ny,
        "ppc": args.ppc,
        "n_particles": p.n_particles,
        "steps": args.steps,
        "warmup_steps": args.warmup_steps,
        "deposit_ms_per_step": 1.0e3 * float(np.mean(samples["deposit"])),
        "solve_ms_per_step": 1.0e3 * float(np.mean(samples["solve"])),
        "filter_ms_per_step": 1.0e3 * float(np.mean(samples["filter"])),
        "gather_ms_per_step": 1.0e3 * float(np.mean(samples["gather"])),
        "push_ms_per_step": 1.0e3 * float(np.mean(samples["push"])),
        "total_ms_per_step": 1.0e3 * float(np.mean(total_samples)),
        "total_ms_step_std": 1.0e3 * float(np.std(total_samples, ddof=1)),
        "persistent_array_bytes": explicit_bytes,
        "persistent_bytes_per_particle": explicit_bytes / float(p.n_particles),
        "process_peak_rss_bytes": max_rss_bytes,
    }
    component_sum = sum(row[f"{name}_ms_per_step"] for name in COMPONENTS)
    row["unattributed_ms_per_step"] = row["total_ms_per_step"] - component_sum
    with (args.output / f"cost_{args.method}.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    (args.output / f"cost_{args.method}.json").write_text(
        json.dumps(row, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(row, indent=2, sort_keys=True), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("cic", "c4", "deltaf"), required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--Nx", type=int, default=64)
    parser.add_argument("--Ny", type=int, default=16)
    parser.add_argument("--Ly", type=float, default=8.0)
    parser.add_argument("--ppc", type=int, default=1024)
    parser.add_argument("--alpha", type=float, default=0.001)
    parser.add_argument("--sigma-dx", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--steps", type=int, default=200)
    args = parser.parse_args()
    if args.steps < 100:
        raise SystemExit("cost benchmark requires at least 100 measured steps")
    run(args)


if __name__ == "__main__":
    main()
