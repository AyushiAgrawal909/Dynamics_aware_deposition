#!/usr/bin/env python3
"""Measure E2 density and field noise for arbitrary ``(sigma_x, sigma_y)`` pairs."""

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

from pic_kde_2d.conventions import marker_grid_shape
from pic_kde_2d.core.grid2d import Grid2D
from pic_kde_2d.core.particles2d import init_two_stream_2d
from pic_kde_2d.core.pic_loop import pic_step
from pic_kde_2d.methods.cic import deposit_cic_2d
from pic_kde_2d.methods.spectral_W import apply_W_to_grid, make_W_gaussian
from pic_kde_2d.methods.spectral_W_wrappers import (
    make_spectral_W_gaussian_methods,
)


def parse_configs(text: str) -> list[tuple[str, float, float]]:
    """Parse ``label:sigma_x:sigma_y`` entries separated by commas."""
    configs = []
    labels = set()
    for item in text.split(","):
        parts = item.strip().split(":")
        if len(parts) != 3:
            raise argparse.ArgumentTypeError(
                "configs must use label:sigma_x:sigma_y"
            )
        label, sigma_x_text, sigma_y_text = parts
        sigma_x = float(sigma_x_text)
        sigma_y = float(sigma_y_text)
        if (
            not label
            or label in labels
            or not np.isfinite(sigma_x)
            or not np.isfinite(sigma_y)
            or sigma_x < 0.0
            or sigma_y < 0.0
        ):
            raise argparse.ArgumentTypeError(
                "config labels must be unique and widths finite and nonnegative"
            )
        labels.add(label)
        configs.append((label, sigma_x, sigma_y))
    if not configs or configs[0][1:] != (0.0, 0.0):
        raise argparse.ArgumentTypeError("the first config must be the CIC baseline")
    return configs


def make_particles(ppc: int, seed: int, args: argparse.Namespace):
    """Construct the frozen globally shuffled E2 ensemble."""
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


def cic_density(particles, grid: Grid2D) -> np.ndarray:
    return deposit_cic_2d(
        particles.x,
        particles.y,
        particles.w,
        grid.Nx,
        grid.Ny,
        grid.dx,
        grid.dy,
    )


def relative_charge_error(rho: np.ndarray, particles, grid: Grid2D) -> float:
    deposited = float(np.sum(rho) * grid.dx * grid.dy)
    expected = float(particles.total_charge())
    return float(abs(deposited - expected) / max(abs(expected), np.finfo(float).tiny))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--configs", type=parse_configs, required=True)
    parser.add_argument("--ppc", type=int, default=32)
    parser.add_argument("--reference-ppc", type=int, default=1024)
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
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    grid = Grid2D(Nx=args.Nx, Ny=args.Ny, Lx=args.Lx, Ly=args.Ly)
    low = make_particles(args.ppc, args.seed, args)
    reference = make_particles(args.reference_ppc, args.seed, args)
    live_sigma_x = args.live_sigma_dx * grid.dx
    live_sigma_y = args.live_sigma_dx * grid.dy
    deposit_live, gather_live = make_spectral_W_gaussian_methods(
        grid, live_sigma_x, live_sigma_y
    )
    filters = {
        label: make_W_gaussian(
            grid,
            sigma_x * grid.dx,
            sigma_y * grid.dy,
        )
        for label, sigma_x, sigma_y in args.configs
    }
    values = {
        label: {
            "density_mse": [],
            "field_mse": [],
            "neutrality": [],
            "charge_error": [],
        }
        for label, _, _ in args.configs
    }

    for step in range(args.n_steps + 1):
        time = step * args.dt
        record = (
            step % args.diag_stride == 0
            and args.eval_t_min <= time <= args.eval_t_max
        )
        if record:
            rho_low = cic_density(low, grid)
            rho_ref = cic_density(reference, grid)
            rhs_ref = rho_ref - float(np.mean(rho_ref))
            Ex_ref, Ey_ref = grid.poisson_solve(rhs_ref)
            for label, _, _ in args.configs:
                rho_candidate = apply_W_to_grid(
                    rho_low, filters[label], grid
                )
                rhs_candidate = rho_candidate - float(np.mean(rho_candidate))
                Ex_candidate, Ey_candidate = grid.poisson_solve(rhs_candidate)
                values[label]["density_mse"].append(
                    float(np.mean((rho_candidate - rho_ref) ** 2))
                )
                values[label]["field_mse"].append(
                    float(
                        np.mean(
                            (Ex_candidate - Ex_ref) ** 2
                            + (Ey_candidate - Ey_ref) ** 2
                        )
                    )
                )
                values[label]["neutrality"].append(
                    float(
                        abs(np.sum(rhs_candidate))
                        / max(
                            np.sum(np.abs(rhs_candidate)),
                            np.finfo(float).tiny,
                        )
                    )
                )
                values[label]["charge_error"].append(
                    relative_charge_error(rho_candidate, low, grid)
                )

        if step == args.n_steps:
            break
        pic_step(low, grid, args.dt, deposit_live, gather_live)
        pic_step(reference, grid, args.dt, deposit_live, gather_live)

    rows = []
    for label, sigma_x, sigma_y in args.configs:
        row = {
            "label": label,
            "seed": args.seed,
            "sigma_x_dx": sigma_x,
            "sigma_y_dy": sigma_y,
            "density_mse": float(np.mean(values[label]["density_mse"])),
            "field_mse": float(np.mean(values[label]["field_mse"])),
            "max_neutrality_defect": float(np.max(values[label]["neutrality"])),
            "max_relative_charge_error": float(
                np.max(values[label]["charge_error"])
            ),
            "n_times": len(values[label]["density_mse"]),
        }
        rows.append(row)

    with (args.output / "joint_noise_seed.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result = {
        "seed": args.seed,
        "configs": args.configs,
        "ppc": args.ppc,
        "reference_ppc": args.reference_ppc,
        "rows": rows,
    }
    (args.output / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
