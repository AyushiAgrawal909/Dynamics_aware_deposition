#!/usr/bin/env python3
"""Landau density-MISE scan for Figure 6."""

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

from pic_kde_2d.conventions import landau_box_length, marker_grid_shape, thermal_velocity
from pic_kde_2d.core.grid2d import Grid2D
from pic_kde_2d.core.particles2d import init_landau_2d_eigenmode
from pic_kde_2d.core.pic_loop import pic_step
from pic_kde_2d.benchmarks.theory import landau_omega_theory
from pic_kde_2d.methods.cic import deposit_cic_2d
from pic_kde_2d.methods.spectral_W import apply_W_to_grid, make_W_gaussian
from pic_kde_2d.methods.spectral_W_wrappers import make_spectral_W_gaussian_methods
from pic_kde_2d.methods.wu_qin_bandwidth import identify_cv_minimum


def parse_float_list(text: str) -> list[float]:
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def write_csv(path: pathlib.Path, rows: list[dict]) -> None:
    fields = sorted({key for row in rows for key in row.keys()}) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def bootstrap_mean_ci(values: np.ndarray, nboot: int, seed: int) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    sampled = values[rng.integers(0, len(values), size=(nboot, len(values)))]
    means = np.mean(sampled, axis=1)
    return float(np.mean(values)), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def make_particles(ppc: int, seed: int, grid: Grid2D, args: argparse.Namespace):
    nx_marker, ny_marker = marker_grid_shape(grid.Nx, grid.Ny, ppc)
    omega_r = landau_omega_theory(args.k_lambdaD, thermal_velocity())
    grid_shape = (
        (grid.Nx, grid.Ny)
        if args.quiet_start_layout == "per-cell"
        else None
    )
    return init_landau_2d_eigenmode(
        Nx_marker=nx_marker,
        Ny_marker=ny_marker,
        Lx=grid.Lx,
        Ly=grid.Ly,
        v_thermal=thermal_velocity(),
        alpha=args.alpha,
        kx_mode=args.kx_mode,
        omega_r=omega_r,
        ky_mode=0,
        rng=np.random.default_rng(seed),
        grid_shape=grid_shape,
    )


def cic_density(particles, grid: Grid2D) -> np.ndarray:
    return deposit_cic_2d(particles.x, particles.y, particles.w, grid.Nx, grid.Ny, grid.dx, grid.dy)


def run_one_seed(seed: int, sigma_grid: list[float], grid: Grid2D, args: argparse.Namespace) -> list[dict]:
    low = make_particles(args.ppc, seed, grid, args)
    ref = make_particles(args.reference_ppc, seed, grid, args)
    deposit_live, gather_live = make_spectral_W_gaussian_methods(
        grid, args.live_sigma_dx * grid.dx, args.live_sigma_dx * grid.dy
    )
    filters = {s: make_W_gaussian(grid, s * grid.dx, s * grid.dy) for s in sigma_grid if s > 0.0}
    err = {s: [] for s in sigma_grid}
    for step in range(args.n_steps + 1):
        t = step * args.dt
        if step % args.diag_stride == 0 and args.eval_t_min <= t <= args.eval_t_max:
            rho_ref = cic_density(ref, grid)
            rho_low = cic_density(low, grid)
            for sigma in sigma_grid:
                rho = rho_low if sigma == 0.0 else apply_W_to_grid(rho_low, filters[sigma], grid)
                err[sigma].append(float(np.mean((rho - rho_ref) ** 2)))
        if step == args.n_steps:
            break
        pic_step(low, grid, args.dt, deposit_live, gather_live)
        pic_step(ref, grid, args.dt, deposit_live, gather_live)
    return [{"seed": seed, "sigma_dx": s, "mise_time_avg": float(np.mean(err[s])), "n_times": len(err[s])} for s in sigma_grid]


def run_scan(args: argparse.Namespace) -> dict:
    args.output.mkdir(parents=True, exist_ok=True)
    Lx = landau_box_length(args.kx_mode, args.k_lambdaD)
    grid = Grid2D(Nx=args.Nx, Ny=args.Ny, Lx=Lx, Ly=args.Ly)
    sigma_grid = parse_float_list(args.sigma_dx)
    seed_rows: list[dict] = []
    for seed in range(args.seed_start, args.seed_start + args.seeds):
        seed_rows.extend(run_one_seed(seed, sigma_grid, grid, args))
        print(f"Landau MISE completed seed={seed}", flush=True)
    summary: list[dict] = []
    for sigma in sigma_grid:
        values = np.asarray([r["mise_time_avg"] for r in seed_rows if r["sigma_dx"] == sigma], dtype=np.float64)
        mean, lo, hi = bootstrap_mean_ci(values, args.nboot, 33300 + int(round(sigma * 1000.0)))
        summary.append(
            {
                "sigma_dx": sigma,
                "mise_mean": mean,
                "mise_ci_lo": lo,
                "mise_ci_hi": hi,
                "n_seeds": len(values),
                "n_times": int(next(r["n_times"] for r in seed_rows if r["sigma_dx"] == sigma)),
            }
        )
    write_csv(args.output / "landau_mise_seed_values.csv", seed_rows)
    write_csv(args.output / "landau_mise_summary.csv", summary)
    ordered = sorted(summary, key=lambda row: float(row["sigma_dx"]))
    selection = identify_cv_minimum(
        [float(row["sigma_dx"]) for row in ordered],
        [float(row["mise_mean"]) for row in ordered],
    )
    result = {
        "summary_csv": str(args.output / "landau_mise_summary.csv"),
        "sigma_mise_dx": selection.bandwidth,
        "sigma_mise_relation": "=" if selection.identified else ">=",
        "sigma_mise_lower_bound_dx": selection.lower_bound,
        "minimum_status": selection.status,
        "parameters": {
            "Nx": grid.Nx,
            "Ny": grid.Ny,
            "Lx": grid.Lx,
            "Ly": grid.Ly,
            "dx": grid.dx,
            "k_lambdaD": args.k_lambdaD,
            "ppc": args.ppc,
            "reference_ppc": args.reference_ppc,
            "eval_window": [args.eval_t_min, args.eval_t_max],
            "quiet_start_layout": args.quiet_start_layout,
        },
    }
    with (args.output / "landau_mise_result.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return result


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run Landau MISE scan for Figure 6.")
    p.add_argument("--output", type=pathlib.Path, default=PROJECT_ROOT / "results" / "landau_mise")
    p.add_argument("--sigma-dx", type=str, default="0.0,0.25,0.5,1.0,2.0,3.0,4.0,5.0,6.0,8.0,12.0")
    p.add_argument("--seed-start", type=int, default=0)
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--ppc", type=int, default=32)
    p.add_argument("--reference-ppc", type=int, default=512)
    p.add_argument("--Nx", type=int, default=64)
    p.add_argument("--Ny", type=int, default=16)
    p.add_argument("--Ly", type=float, default=16.0)
    p.add_argument("--kx-mode", type=int, default=1)
    p.add_argument("--k-lambdaD", type=float, default=0.4)
    p.add_argument("--alpha", type=float, default=0.02)
    p.add_argument("--live-sigma-dx", type=float, default=0.25)
    p.add_argument(
        "--quiet-start-layout",
        choices=("global", "per-cell"),
        default="global",
        help="velocity quadrature assignment; matrix runs use global to match the rate harness",
    )
    p.add_argument("--dt", type=float, default=0.05)
    p.add_argument("--n-steps", type=int, default=600)
    p.add_argument("--diag-stride", type=int, default=10)
    p.add_argument("--eval-t-min", type=float, default=5.0)
    p.add_argument("--eval-t-max", type=float, default=30.0)
    p.add_argument("--nboot", type=int, default=1000)
    return p


if __name__ == "__main__":
    run_scan(build_parser().parse_args())
