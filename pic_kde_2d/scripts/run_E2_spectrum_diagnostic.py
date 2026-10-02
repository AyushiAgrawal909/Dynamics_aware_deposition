#!/usr/bin/env python3
"""E2 spectrum diagnostic for the mechanism figure.

Runs the executed E2 reference dynamics (`sigma_live=0.25 dx`) and records the
1D ky=0 density spectrum after applying diagnostic spectral-W filters.
"""

from __future__ import annotations

import argparse
import csv
import pathlib
import sys

import numpy as np

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pic_kde_2d.conventions import marker_grid_shape
from pic_kde_2d.core.grid2d import Grid2D
from pic_kde_2d.core.particles2d import init_two_stream_2d
from pic_kde_2d.core.pic_loop import pic_step
from pic_kde_2d.methods.cic import deposit_cic_2d
from pic_kde_2d.methods.spectral_W import apply_W_to_grid, make_W_gaussian
from pic_kde_2d.methods.spectral_W_wrappers import make_spectral_W_gaussian_methods


def parse_float_list(text: str) -> list[float]:
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def write_csv(path: pathlib.Path, rows: list[dict]) -> None:
    fieldnames = sorted({key for row in rows for key in row.keys()}) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def make_particles(ppc: int, seed: int, args: argparse.Namespace):
    nx_marker, ny_marker = marker_grid_shape(args.Nx, args.Ny, ppc)
    return init_two_stream_2d(
        Nx_marker=nx_marker,
        Ny_marker=ny_marker,
        Lx=args.Lx,
        Ly=args.Ly,
        v_beam=args.v_beam,
        v_thermal=args.v_thermal,
        alpha=args.alpha,
        kx_mode=args.kx_mode,
        ky_mode=0,
        rng=np.random.default_rng(seed),
    )


def density_cic(particles, grid: Grid2D) -> np.ndarray:
    return deposit_cic_2d(
        particles.x,
        particles.y,
        particles.w,
        grid.Nx,
        grid.Ny,
        grid.dx,
        grid.dy,
    )


def run_diagnostic(args: argparse.Namespace) -> dict:
    args.output.mkdir(parents=True, exist_ok=True)
    grid = Grid2D(Nx=args.Nx, Ny=args.Ny, Lx=args.Lx, Ly=args.Ly)
    sigma_grid = parse_float_list(args.sigma_dx)
    filters = {
        sigma: make_W_gaussian(grid, sigma * grid.dx, sigma * grid.dy)
        for sigma in sigma_grid
        if sigma > 0.0
    }
    spectrum_accum = {
        sigma: np.zeros(grid.Nx, dtype=np.float64)
        for sigma in sigma_grid
    }
    n_samples = 0

    sigma_live_x = args.live_sigma_dx * grid.dx
    sigma_live_y = args.live_sigma_dx * grid.dy
    deposit_live, gather_live = make_spectral_W_gaussian_methods(grid, sigma_live_x, sigma_live_y)

    for seed in range(args.seed_start, args.seed_start + args.seeds):
        particles = make_particles(args.ppc, seed, args)
        for step in range(args.n_steps + 1):
            time = step * args.dt
            record = step % args.diag_stride == 0 and args.eval_t_min <= time <= args.eval_t_max
            if record:
                rho_cic = density_cic(particles, grid)
                rho_cic = rho_cic - np.mean(rho_cic)
                for sigma in sigma_grid:
                    rho = rho_cic if sigma == 0.0 else apply_W_to_grid(rho_cic, filters[sigma], grid)
                    rho_hat = np.fft.fft(np.mean(rho, axis=1))
                    power = np.abs(rho_hat) ** 2 / float(grid.Nx * grid.Nx)
                    spectrum_accum[sigma] += power
                n_samples += 1
            if step == args.n_steps:
                break
            pic_step(particles, grid, args.dt, deposit_live, gather_live)
        print(f"E2 spectrum completed seed={seed}", flush=True)

    rows: list[dict] = []
    modes = np.arange(grid.Nx, dtype=np.int64)
    folded_modes = np.minimum(modes, grid.Nx - modes)
    for sigma in sigma_grid:
        power_mean = spectrum_accum[sigma] / float(n_samples)
        for mode in range(grid.Nx // 2 + 1):
            rows.append(
                {
                    "sigma_dx": sigma,
                    "mode": mode,
                    "power": float(np.mean(power_mean[folded_modes == mode])),
                    "n_samples": n_samples,
                }
            )
    write_csv(args.output / "E2_spectrum_summary.csv", rows)
    result = {
        "summary_csv": str(args.output / "E2_spectrum_summary.csv"),
        "n_samples": n_samples,
        "sigma_dx": sigma_grid,
        "kx_mode": args.kx_mode,
        "eval_window": [args.eval_t_min, args.eval_t_max],
    }
    print(result, flush=True)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run E2 density-spectrum diagnostic.")
    parser.add_argument("--output", type=pathlib.Path, default=PROJECT_ROOT / "results" / "E2_spectrum")
    parser.add_argument("--sigma-dx", type=str, default="0.0,0.35,0.68,3.0,12.0")
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--ppc", type=int, default=32)
    parser.add_argument("--Nx", type=int, default=128)
    parser.add_argument("--Ny", type=int, default=16)
    parser.add_argument("--Lx", type=float, default=64.0)
    parser.add_argument("--Ly", type=float, default=16.0)
    parser.add_argument("--kx-mode", type=int, default=12)
    parser.add_argument("--v-beam", type=float, default=0.8)
    parser.add_argument("--v-thermal", type=float, default=0.05)
    parser.add_argument("--alpha", type=float, default=0.001)
    parser.add_argument("--live-sigma-dx", type=float, default=0.25)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--n-steps", type=int, default=80)
    parser.add_argument("--diag-stride", type=int, default=2)
    parser.add_argument("--eval-t-min", type=float, default=0.5)
    parser.add_argument("--eval-t-max", type=float, default=4.0)
    return parser


if __name__ == "__main__":
    run_diagnostic(build_parser().parse_args())
