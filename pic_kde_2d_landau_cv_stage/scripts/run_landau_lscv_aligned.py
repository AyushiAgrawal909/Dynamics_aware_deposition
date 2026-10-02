#!/usr/bin/env python3
"""Aligned weak-Landau LSCV scan for the final Paper 1 matrix.

This runner freezes the same PPC, global-shuffle quiet start, live method, and
time window used by the final Landau MISE cell.  It also evaluates the exact
particle-phase-averaged CIC self term instead of an offset approximation.
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

from pic_kde_2d.benchmarks.theory import landau_omega_theory
from pic_kde_2d.conventions import landau_box_length, marker_grid_shape, thermal_velocity
from pic_kde_2d.core.grid2d import Grid2D
from pic_kde_2d.core.particles2d import init_landau_2d_eigenmode
from pic_kde_2d.core.pic_loop import pic_step
from pic_kde_2d.methods.cic import deposit_cic_2d
from pic_kde_2d.methods.lscv import (
    cic_filter_self_matrix,
    cic_weight_second_moment,
    lscv_score,
    mean_cic_filter_self,
)
from pic_kde_2d.methods.spectral_W import make_W_gaussian
from pic_kde_2d.methods.spectral_W_wrappers import make_spectral_W_gaussian_methods
from pic_kde_2d.methods.wu_qin_bandwidth import identify_cv_minimum


DEFAULT_SIGMAS = (
    "0,0.25,0.5,1,2,3,4,4.5,5,5.5,6,6.2,6.4,6.5,6.6,6.61,"
    "6.7,6.8,7,7.5,8,10,12,16,20,30,60"
)


def parse_float_list(text: str) -> list[float]:
    return sorted({float(item.strip()) for item in text.split(",") if item.strip()})


def write_csv(path: pathlib.Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def make_grid(args: argparse.Namespace) -> Grid2D:
    return Grid2D(
        Nx=args.Nx,
        Ny=args.Ny,
        Lx=landau_box_length(args.kx_mode, args.k_lambdaD),
        Ly=args.Ly,
    )


def make_particles(seed: int, grid: Grid2D, args: argparse.Namespace):
    nx_marker, ny_marker = marker_grid_shape(grid.Nx, grid.Ny, args.ppc)
    return init_landau_2d_eigenmode(
        Nx_marker=nx_marker,
        Ny_marker=ny_marker,
        Lx=grid.Lx,
        Ly=grid.Ly,
        v_thermal=thermal_velocity(),
        alpha=args.alpha,
        kx_mode=args.kx_mode,
        omega_r=landau_omega_theory(args.k_lambdaD, thermal_velocity()),
        ky_mode=0,
        rng=np.random.default_rng(seed),
        grid_shape=None,
        position_jitter=args.position_sampling == "stratified-jitter",
    )


def probability_density_cic(particles, grid: Grid2D) -> np.ndarray:
    n = particles.n_particles
    return deposit_cic_2d(
        particles.x,
        particles.y,
        np.full(n, 1.0 / float(n), dtype=np.float64),
        grid.Nx,
        grid.Ny,
        grid.dx,
        grid.dy,
    )


def quadratic_minimum(sigmas: np.ndarray, scores: np.ndarray) -> float:
    """Refine a sampled interior minimum with its three-point quadratic."""
    index = int(np.argmin(scores))
    if index == 0 or index == len(sigmas) - 1:
        return float(sigmas[index])
    x = sigmas[index - 1 : index + 2]
    y = scores[index - 1 : index + 2]
    a, b, _ = np.polyfit(x, y, 2)
    if not np.isfinite(a) or a <= 0.0:
        return float(sigmas[index])
    vertex = float(-b / (2.0 * a))
    return vertex if float(x[0]) <= vertex <= float(x[-1]) else float(sigmas[index])


def scan(args: argparse.Namespace) -> None:
    args.output.mkdir(parents=True, exist_ok=True)
    grid = make_grid(args)
    sigmas = parse_float_list(args.sigma_dx)
    filters = {
        sigma: make_W_gaussian(grid, sigma * grid.dx, sigma * grid.dy)
        for sigma in sigmas
        if sigma > 0.0
    }
    self_matrices = {
        sigma: cic_filter_self_matrix(
            grid,
            None if sigma == 0.0 else filters[sigma],
        )
        for sigma in sigmas
    }
    deposit_live, gather_live = make_spectral_W_gaussian_methods(
        grid,
        args.live_sigma_dx * grid.dx,
        args.live_sigma_dx * grid.dy,
    )

    rows: list[dict] = []
    time_rows: list[dict] = []
    for seed in range(args.seed_start, args.seed_start + args.seeds):
        particles = make_particles(seed, grid, args)
        values = {sigma: [] for sigma in sigmas}
        self_values = {sigma: [] for sigma in sigmas}
        for step in range(args.n_steps + 1):
            time = step * args.dt
            if (
                step % args.diag_stride == 0
                and args.eval_t_min <= time <= args.eval_t_max
            ):
                rho_probability = probability_density_cic(particles, grid)
                moment = cic_weight_second_moment(particles.x, particles.y, grid)
                for sigma in sigmas:
                    mean_self = mean_cic_filter_self(
                        moment,
                        self_matrices[sigma],
                    )
                    score = lscv_score(
                        rho_probability,
                        None if sigma == 0.0 else filters[sigma],
                        mean_self,
                        particles.n_particles,
                        grid,
                    )
                    values[sigma].append(score)
                    self_values[sigma].append(mean_self)
                    if args.save_time_resolved:
                        time_rows.append(
                            {
                                "seed": seed,
                                "time": time,
                                "sigma_dx": sigma,
                                "lscv_score": score,
                                "mean_self": mean_self,
                            }
                        )
            if step == args.n_steps:
                break
            pic_step(particles, grid, args.dt, deposit_live, gather_live)

        for sigma in sigmas:
            rows.append(
                {
                    "seed": seed,
                    "sigma_dx": sigma,
                    "lscv_time_avg": float(np.mean(values[sigma])),
                    "mean_self_time_avg": float(np.mean(self_values[sigma])),
                    "n_times": len(values[sigma]),
                }
            )
        print(f"Landau aligned LSCV completed seed={seed}", flush=True)

    write_csv(args.output / "landau_lscv_seed_values.csv", rows)
    if args.save_time_resolved:
        write_csv(args.output / "landau_lscv_time_values.csv", time_rows)
    metadata = {
        "protocol": "final Paper 1 weak-Landau noise cell",
        "Nx": grid.Nx,
        "Ny": grid.Ny,
        "Lx": grid.Lx,
        "Ly": grid.Ly,
        "ppc": args.ppc,
        "seed_start": args.seed_start,
        "seeds": args.seeds,
        "alpha": args.alpha,
        "live_sigma_dx": args.live_sigma_dx,
        "dt": args.dt,
        "n_steps": args.n_steps,
        "diag_stride": args.diag_stride,
        "eval_window": [args.eval_t_min, args.eval_t_max],
        "quiet_start_layout": "global",
        "position_sampling": args.position_sampling,
        "self_term": "exact particle-phase-averaged CIC quadratic form",
        "sigma_grid_dx": sigmas,
        "time_resolved_scores": bool(args.save_time_resolved),
    }
    (args.output / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def bootstrap_selection(
    score_matrix: np.ndarray,
    sigmas: np.ndarray,
    nboot: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    selected = np.empty(nboot, dtype=np.float64)
    for draw in range(nboot):
        indices = rng.integers(0, score_matrix.shape[0], size=score_matrix.shape[0])
        selected[draw] = quadratic_minimum(
            sigmas,
            np.mean(score_matrix[indices], axis=0),
        )
    return selected


def reduce(args: argparse.Namespace) -> None:
    files = sorted(args.input_root.glob("seed*/landau_lscv_seed_values.csv"))
    if not files:
        files = sorted(args.input_root.glob("**/landau_lscv_seed_values.csv"))
    rows: list[dict] = []
    for path in files:
        with path.open(newline="", encoding="utf-8") as handle:
            rows.extend(csv.DictReader(handle))
    seeds = sorted({int(row["seed"]) for row in rows})
    if len(seeds) != args.expected_seeds:
        raise RuntimeError(
            f"expected {args.expected_seeds} independent seeds, found {seeds}"
        )
    sigmas = np.asarray(
        sorted({float(row["sigma_dx"]) for row in rows}),
        dtype=np.float64,
    )
    score_matrix = np.asarray(
        [
            [
                float(
                    next(
                        row["lscv_time_avg"]
                        for row in rows
                        if int(row["seed"]) == seed
                        and float(row["sigma_dx"]) == sigma
                    )
                )
                for sigma in sigmas
            ]
            for seed in seeds
        ],
        dtype=np.float64,
    )
    if np.any(np.var(score_matrix, axis=0) == 0.0):
        raise RuntimeError("zero across-seed LSCV variance; initializer audit failed")

    means = np.mean(score_matrix, axis=0)
    selection = identify_cv_minimum(sigmas, means)
    refined = quadratic_minimum(sigmas, means) if selection.identified else None
    seed_minima: list[dict] = []
    for seed_index, seed in enumerate(seeds):
        seed_selection = identify_cv_minimum(sigmas, score_matrix[seed_index])
        seed_minima.append(
            {
                "seed": seed,
                "minimum_status": seed_selection.status,
                "sigma_cv_grid_dx": seed_selection.bandwidth,
                "sigma_cv_quadratic_dx": (
                    quadratic_minimum(sigmas, score_matrix[seed_index])
                    if seed_selection.identified
                    else ""
                ),
                "lscv_minimum": seed_selection.score,
            }
        )
    interior_seed_widths = np.asarray(
        [
            float(row["sigma_cv_quadratic_dx"])
            for row in seed_minima
            if row["minimum_status"] == "interior"
        ],
        dtype=np.float64,
    )
    selected_boot = (
        bootstrap_selection(score_matrix, sigmas, args.nboot, args.bootstrap_seed)
        if selection.identified
        else np.asarray([], dtype=np.float64)
    )
    rng = np.random.default_rng(args.bootstrap_seed + 1)
    summary: list[dict] = []
    for index, sigma in enumerate(sigmas):
        draws = np.mean(
            score_matrix[
                rng.integers(
                    0,
                    score_matrix.shape[0],
                    size=(args.nboot, score_matrix.shape[0]),
                ),
                index,
            ],
            axis=1,
        )
        summary.append(
            {
                "sigma_dx": float(sigma),
                "lscv_mean": float(means[index]),
                "lscv_ci_lo": float(np.percentile(draws, 2.5)),
                "lscv_ci_hi": float(np.percentile(draws, 97.5)),
                "n_seeds": len(seeds),
            }
        )
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "landau_lscv_summary.csv", summary)
    write_csv(args.output / "landau_lscv_seed_minima.csv", seed_minima)
    result = {
        "minimum": selection.to_dict(),
        "sigma_cv_grid_dx": selection.bandwidth,
        "sigma_cv_quadratic_dx": refined,
        "sigma_cv_quadratic_ci_lo": (
            float(np.percentile(selected_boot, 2.5)) if selected_boot.size else None
        ),
        "sigma_cv_quadratic_ci_hi": (
            float(np.percentile(selected_boot, 97.5)) if selected_boot.size else None
        ),
        "per_seed_minimum": {
            "n_interior": int(interior_seed_widths.size),
            "n_total": len(seeds),
            "mean_dx": (
                float(np.mean(interior_seed_widths))
                if interior_seed_widths.size
                else None
            ),
            "std_dx": (
                float(np.std(interior_seed_widths, ddof=1))
                if interior_seed_widths.size > 1
                else None
            ),
            "median_dx": (
                float(np.median(interior_seed_widths))
                if interior_seed_widths.size
                else None
            ),
            "min_dx": (
                float(np.min(interior_seed_widths))
                if interior_seed_widths.size
                else None
            ),
            "max_dx": (
                float(np.max(interior_seed_widths))
                if interior_seed_widths.size
                else None
            ),
        },
        "n_seeds": len(seeds),
        "source_files": [str(path) for path in files],
        "protocol_checks": {
            "ppc": args.ppc,
            "quiet_start_layout": "global",
            "eval_window": [args.eval_t_min, args.eval_t_max],
            "self_term": "exact particle-phase-averaged CIC quadratic form",
            "zero_variance_guard": "passed",
        },
    }
    (args.output / "landau_lscv_result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("scan", "reduce"))
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--input-root", type=pathlib.Path)
    parser.add_argument("--sigma-dx", default=DEFAULT_SIGMAS)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--expected-seeds", type=int, default=10)
    parser.add_argument("--ppc", type=int, default=1024)
    parser.add_argument("--Nx", type=int, default=64)
    parser.add_argument("--Ny", type=int, default=16)
    parser.add_argument("--Ly", type=float, default=8.0)
    parser.add_argument("--kx-mode", type=int, default=1)
    parser.add_argument("--k-lambdaD", type=float, default=0.4)
    parser.add_argument("--alpha", type=float, default=0.01)
    parser.add_argument("--live-sigma-dx", type=float, default=0.25)
    parser.add_argument(
        "--position-sampling",
        choices=("regular", "stratified-jitter"),
        default="regular",
        help=(
            "Regular keeps the production quiet lattice. Stratified-jitter "
            "provides independent position ensembles for static-CV uncertainty."
        ),
    )
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--n-steps", type=int, default=600)
    parser.add_argument("--diag-stride", type=int, default=10)
    parser.add_argument("--eval-t-min", type=float, default=5.0)
    parser.add_argument("--eval-t-max", type=float, default=30.0)
    parser.add_argument("--nboot", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=76123)
    parser.add_argument(
        "--save-time-resolved",
        action="store_true",
        help="retain every snapshot score curve for sigma_CV(t)",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.mode == "scan":
        scan(args)
    else:
        if args.input_root is None:
            raise SystemExit("reduce mode requires --input-root")
        reduce(args)


if __name__ == "__main__":
    main()
