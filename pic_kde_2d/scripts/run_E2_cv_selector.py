#!/usr/bin/env python3
"""Particle-only CV bandwidth selector for the E2 two-stream setup.

This is a least-squares leave-one-out CV estimate for the same binned estimator
used in the E2 method: CIC particle binning followed by a spectral Gaussian
post-filter W. The live trajectory is fixed at the executed E2 reference method
(`sigma_live=0.25 dx`); candidate sigma values are diagnostic only.
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

from pic_kde_2d.conventions import marker_grid_shape
from pic_kde_2d.core.grid2d import Grid2D
from pic_kde_2d.core.particles2d import init_two_stream_2d
from pic_kde_2d.core.pic_loop import pic_step
from pic_kde_2d.methods.cic import deposit_cic_2d
from pic_kde_2d.methods.spectral_W import apply_W_to_grid, make_W_gaussian
from pic_kde_2d.methods.spectral_W_wrappers import make_spectral_W_gaussian_methods
from pic_kde_2d.methods.wu_qin_bandwidth import identify_cv_minimum


def parse_float_list(text: str) -> list[float]:
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def write_csv(path: pathlib.Path, rows: list[dict]) -> None:
    fieldnames = sorted({key for row in rows for key in row.keys()}) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def bootstrap_mean_ci(values: np.ndarray, nboot: int, seed: int) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=np.float64)
    sampled = values[rng.integers(0, len(values), size=(nboot, len(values)))]
    means = np.mean(sampled, axis=1)
    return float(np.mean(values)), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


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


def probability_density_cic(particles, grid: Grid2D) -> np.ndarray:
    """CIC empirical probability density, integrating to one."""
    n_particles = particles.n_particles
    weights = np.full(n_particles, 1.0 / float(n_particles), dtype=np.float64)
    return deposit_cic_2d(
        particles.x,
        particles.y,
        weights,
        grid.Nx,
        grid.Ny,
        grid.dx,
        grid.dy,
    )


def self_kernel_value(grid: Grid2D, W: np.ndarray | None) -> float:
    """Estimate the CIC+W self contribution K_h(0) for LOO correction.

    We average a small set of sub-cell offsets because a binned CIC estimator is
    weakly position-dependent at the cell scale.
    """
    offsets = [
        (0.13, 0.17),
        (0.31, 0.41),
        (0.57, 0.23),
        (0.79, 0.67),
        (0.43, 0.89),
        (0.91, 0.11),
        (0.25, 0.75),
        (0.75, 0.25),
    ]
    values: list[float] = []
    for ox, oy in offsets:
        x = np.asarray([(0.5 * grid.Nx + ox) * grid.dx], dtype=np.float64) % grid.Lx
        y = np.asarray([(0.5 * grid.Ny + oy) * grid.dy], dtype=np.float64) % grid.Ly
        rho_one = deposit_cic_2d(
            x,
            y,
            np.asarray([1.0], dtype=np.float64),
            grid.Nx,
            grid.Ny,
            grid.dx,
            grid.dy,
        )
        rho_smooth = rho_one if W is None else apply_W_to_grid(rho_one, W, grid)
        # CIC adjoint equivalent to gathering rho_smooth back to the same marker.
        values.append(float(np.sum(rho_smooth * rho_one) * grid.dx * grid.dy))
    return float(np.mean(values))


def cv_score_for_density(
    rho_prob_cic: np.ndarray,
    W: np.ndarray | None,
    k_self: float,
    n_particles: int,
    grid: Grid2D,
) -> float:
    """Least-squares leave-one-out CV score for one density snapshot."""
    rho_smooth = rho_prob_cic if W is None else apply_W_to_grid(rho_prob_cic, W, grid)
    term_integrated_square = float(np.sum(rho_smooth * rho_smooth) * grid.dx * grid.dy)
    raw_at_particles_mean = float(np.sum(rho_smooth * rho_prob_cic) * grid.dx * grid.dy)
    loo_mean = (raw_at_particles_mean - k_self / float(n_particles)) * (
        float(n_particles) / float(n_particles - 1)
    )
    return term_integrated_square - 2.0 * loo_mean


def run_one_seed(seed: int, sigma_grid: list[float], grid: Grid2D, args: argparse.Namespace) -> list[dict]:
    particles = make_particles(args.ppc, seed, args)
    n_particles = particles.n_particles
    sigma_live_x = args.live_sigma_dx * grid.dx
    sigma_live_y = args.live_sigma_dx * grid.dy
    deposit_live, gather_live = make_spectral_W_gaussian_methods(grid, sigma_live_x, sigma_live_y)

    filters = {
        sigma: make_W_gaussian(grid, sigma * grid.dx, sigma * grid.dy)
        for sigma in sigma_grid
        if sigma > 0.0
    }
    self_terms = {
        sigma: self_kernel_value(grid, None if sigma == 0.0 else filters[sigma])
        for sigma in sigma_grid
    }
    cv_values: dict[float, list[float]] = {sigma: [] for sigma in sigma_grid}

    for step in range(args.n_steps + 1):
        time = step * args.dt
        record = (
            step % args.diag_stride == 0
            and args.eval_t_min <= time <= args.eval_t_max
        )
        if record:
            rho_prob_cic = probability_density_cic(particles, grid)
            for sigma in sigma_grid:
                score = cv_score_for_density(
                    rho_prob_cic=rho_prob_cic,
                    W=None if sigma == 0.0 else filters[sigma],
                    k_self=self_terms[sigma],
                    n_particles=n_particles,
                    grid=grid,
                )
                cv_values[sigma].append(score)

        if step == args.n_steps:
            break
        pic_step(particles, grid, args.dt, deposit_live, gather_live)

    return [
        {
            "seed": seed,
            "sigma_dx": sigma,
            "cv_time_avg": float(np.mean(cv_values[sigma])),
            "n_times": len(cv_values[sigma]),
            "self_kernel": self_terms[sigma],
        }
        for sigma in sigma_grid
    ]


def classify_cv(summary: list[dict]) -> dict:
    """Classify unguarded LOO-CV without inventing a boundary optimum."""
    ordered = sorted(summary, key=lambda row: float(row["sigma_dx"]))
    selection = identify_cv_minimum(
        [float(row["sigma_dx"]) for row in ordered],
        [float(row["cv_mean"]) for row in ordered],
    )
    if not selection.identified:
        classification = f"unguarded_cv_{selection.status}"
    elif float(selection.bandwidth) <= 0.35:
        classification = "cv_selects_dynamically_safe_sigma"
    elif float(selection.bandwidth) > 0.45:
        classification = "cv_selects_dynamically_degraded_sigma"
    else:
        classification = "cv_selects_boundary_sigma"
    result = selection.to_dict()
    result.update(
        {
            "sigma_cv_dx": selection.bandwidth,
            "sigma_cv_relation": "=" if selection.identified else ">=",
            "sigma_cv_lower_bound_dx": selection.lower_bound,
            "classification": classification,
        }
    )
    return result


def run_selector(args: argparse.Namespace) -> dict:
    args.output.mkdir(parents=True, exist_ok=True)
    grid = Grid2D(Nx=args.Nx, Ny=args.Ny, Lx=args.Lx, Ly=args.Ly)
    sigma_grid = parse_float_list(args.sigma_dx)
    seeds = list(range(args.seed_start, args.seed_start + args.seeds))

    seed_rows: list[dict] = []
    for seed in seeds:
        seed_rows.extend(run_one_seed(seed, sigma_grid, grid, args))
        print(f"E2 CV completed seed={seed}", flush=True)

    summary: list[dict] = []
    for sigma in sigma_grid:
        values = np.asarray([row["cv_time_avg"] for row in seed_rows if row["sigma_dx"] == sigma])
        mean, lo, hi = bootstrap_mean_ci(values, args.nboot, 77100 + int(round(sigma * 1000)))
        summary.append(
            {
                "sigma_dx": sigma,
                "cv_mean": mean,
                "cv_ci_lo": lo,
                "cv_ci_hi": hi,
                "n_seeds": len(values),
                "n_times": int(next(row["n_times"] for row in seed_rows if row["sigma_dx"] == sigma)),
                "self_kernel": float(next(row["self_kernel"] for row in seed_rows if row["sigma_dx"] == sigma)),
            }
        )

    result = {
        "parameters": {
            "Nx": args.Nx,
            "Ny": args.Ny,
            "Lx": args.Lx,
            "Ly": args.Ly,
            "dx": grid.dx,
            "dy": grid.dy,
            "kx_mode": args.kx_mode,
            "k_dx": float(2.0 * np.pi * args.kx_mode / args.Nx),
            "v_beam": args.v_beam,
            "v_thermal": args.v_thermal,
            "alpha": args.alpha,
            "ppc": args.ppc,
            "live_sigma_dx": args.live_sigma_dx,
            "dt": args.dt,
            "n_steps": args.n_steps,
            "diag_stride": args.diag_stride,
            "eval_window": [args.eval_t_min, args.eval_t_max],
            "n_seeds": len(seeds),
        },
        "selector": (
            "unguarded least-squares leave-one-out CV on CIC plus spectral-W "
            "binned KDE; boundary minima are reported as censored"
        ),
        "summary": summary,
        "gate": classify_cv(summary),
    }
    write_csv(args.output / "E2_cv_seed_values.csv", seed_rows)
    write_csv(args.output / "E2_cv_summary.csv", summary)
    with (args.output / "E2_cv_selector_result.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run particle-only CV selector for E2.")
    parser.add_argument("--output", type=pathlib.Path, default=PROJECT_ROOT / "results" / "E2_cv_selector")
    parser.add_argument(
        "--sigma-dx",
        type=str,
        default="0.0,0.10,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.68,0.75,1.0,1.25,1.5,2.0,3.0,4.0,5.0,6.0,8.0,12.0",
    )
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
    parser.add_argument("--nboot", type=int, default=2000)
    return parser


if __name__ == "__main__":
    run_selector(build_parser().parse_args())
