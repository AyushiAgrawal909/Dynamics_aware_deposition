#!/usr/bin/env python3
"""Measure one seed of the dense, model-admissible E2 Gaussian noise design."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np


WORKSPACE = Path(__file__).resolve().parents[2]
PROJECT = WORKSPACE / "pic_kde_2d"
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from pic_kde_2d.conventions import marker_grid_shape
from pic_kde_2d.core.grid2d import Grid2D
from pic_kde_2d.core.particles2d import init_two_stream_2d
from pic_kde_2d.core.pic_loop import pic_step
from pic_kde_2d.methods.cic import deposit_cic_2d
from pic_kde_2d.methods.spectral_W import apply_W_to_grid, make_W_gaussian
from pic_kde_2d.methods.spectral_W_wrappers import make_spectral_W_gaussian_methods

from e2_anisotropic_design import Q_BUDGET_GRID_TIME, make_design


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
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--q-budget", type=float, default=Q_BUDGET_GRID_TIME)
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
    parser.add_argument(
        "--diagnostic-grid",
        action="store_true",
        help="allow a non-production grid for an implementation smoke test",
    )
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    args.output.mkdir(parents=True)

    if (args.Nx, args.Ny, args.Lx, args.Ly) != (128, 16, 64.0, 16.0) and not args.diagnostic_grid:
        raise ValueError("non-production grids require --diagnostic-grid")
    design = make_design(args.q_budget)
    grid = Grid2D(args.Nx, args.Ny, args.Lx, args.Ly)
    low = make_particles(args.ppc, args.seed, args)
    reference = make_particles(args.reference_ppc, args.seed, args)
    deposit_live, gather_live = make_spectral_W_gaussian_methods(
        grid,
        args.live_sigma_dx * grid.dx,
        args.live_sigma_dx * grid.dy,
    )
    filters = {
        row["label"]: make_W_gaussian(
            grid,
            row["sigma_x_dx"] * grid.dx,
            row["sigma_y_dy"] * grid.dy,
        )
        for row in design
    }
    density_values = {row["label"]: [] for row in design}
    field_values = {row["label"]: [] for row in design}
    neutrality_values = {row["label"]: [] for row in design}
    charge_values = {row["label"]: [] for row in design}

    for step in range(args.n_steps + 1):
        time = step * args.dt
        if step % args.diag_stride == 0 and args.eval_t_min <= time <= args.eval_t_max:
            rho_low = cic_density(low, grid)
            rho_reference = cic_density(reference, grid)
            rhs_reference = rho_reference - float(np.mean(rho_reference))
            Ex_reference, Ey_reference = grid.poisson_solve(rhs_reference)
            for row in design:
                label = row["label"]
                rho_candidate = apply_W_to_grid(rho_low, filters[label], grid)
                rhs_candidate = rho_candidate - float(np.mean(rho_candidate))
                Ex_candidate, Ey_candidate = grid.poisson_solve(rhs_candidate)
                density_values[label].append(float(np.mean((rho_candidate - rho_reference) ** 2)))
                field_values[label].append(
                    float(
                        np.mean(
                            (Ex_candidate - Ex_reference) ** 2
                            + (Ey_candidate - Ey_reference) ** 2
                        )
                    )
                )
                neutrality_values[label].append(
                    float(
                        abs(np.sum(rhs_candidate))
                        / max(np.sum(np.abs(rhs_candidate)), np.finfo(float).tiny)
                    )
                )
                charge_values[label].append(relative_charge_error(rho_candidate, low, grid))
        if step == args.n_steps:
            break
        pic_step(low, grid, args.dt, deposit_live, gather_live)
        pic_step(reference, grid, args.dt, deposit_live, gather_live)

    rows: list[dict] = []
    for design_row in design:
        label = design_row["label"]
        if not density_values[label]:
            raise RuntimeError("the requested evaluation window contains no diagnostics")
        rows.append(
            {
                **design_row,
                "seed": args.seed,
                "density_mse": float(np.mean(density_values[label])),
                "field_mse": float(np.mean(field_values[label])),
                "max_neutrality_defect": float(np.max(neutrality_values[label])),
                "max_relative_charge_error": float(np.max(charge_values[label])),
                "n_times": len(density_values[label]),
            }
        )
    with (args.output / "anisotropic_dense_noise_seed.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result = {
        "status": "complete",
        "seed": args.seed,
        "ppc": args.ppc,
        "reference_ppc": args.reference_ppc,
        "q_budget": args.q_budget,
        "protected_mode": [12, 1],
        "budget_level": "grid-time-discrete 10% E2 rate boundary",
        "design_points": len(design),
        "design": design,
        "runtime": {
            "Nx": args.Nx,
            "Ny": args.Ny,
            "Lx": args.Lx,
            "Ly": args.Ly,
            "kx_mode": args.kx_mode,
            "v_beam": args.v_beam,
            "v_thermal": args.v_thermal,
            "alpha": args.alpha,
            "live_sigma_dx": args.live_sigma_dx,
            "dt": args.dt,
            "n_steps": args.n_steps,
            "diag_stride": args.diag_stride,
            "eval_window": [args.eval_t_min, args.eval_t_max],
        },
        "rows": rows,
    }
    (args.output / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: result[key] for key in result if key not in {"design", "rows"}}, indent=2))


if __name__ == "__main__":
    main()
