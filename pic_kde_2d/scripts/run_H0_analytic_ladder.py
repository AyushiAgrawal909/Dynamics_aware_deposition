#!/usr/bin/env python3
"""Run the estimator-only H0 CIC/global-C4/capped-Abramson ladder."""

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

from pic_kde_2d.benchmarks.heterogeneous import (
    analytic_importance_sampled_density,
    init_importance_sampled_landau_2d,
)
from pic_kde_2d.conventions import landau_box_length, marker_grid_shape, thermal_velocity
from pic_kde_2d.core.grid2d import Grid2D
from pic_kde_2d.methods.adaptive_sigma import abramson_c4_bandwidth
from pic_kde_2d.methods.cic import deposit_cic_2d
from pic_kde_2d.methods.spectral_W import (
    apply_W_to_grid,
    deposit_binned_adaptive_spectral_W,
    make_W_gaussian,
)


def parse_float_list(text: str) -> list[float]:
    return [float(value.strip()) for value in text.split(",") if value.strip()]


def write_csv(path: pathlib.Path, rows: list[dict]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def bootstrap_mean(values: np.ndarray, nboot: int, seed: int) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, values.size, size=(nboot, values.size))
    means = np.mean(values[indices], axis=1)
    return (
        float(np.mean(values)),
        float(np.percentile(means, 2.5)),
        float(np.percentile(means, 97.5)),
    )


def density_deciles(node_density: np.ndarray) -> np.ndarray:
    ranks = np.argsort(np.argsort(node_density, kind="stable"), kind="stable")
    return np.minimum(9, 10 * ranks // len(ranks))


def run(args: argparse.Namespace) -> dict:
    args.output.mkdir(parents=True, exist_ok=True)
    grid = Grid2D(
        Nx=args.Nx,
        Ny=args.Ny,
        Lx=landau_box_length(args.kx_mode, args.k_lambdaD),
        Ly=args.Ly,
    )
    nx_marker, ny_marker = marker_grid_shape(grid.Nx, grid.Ny, args.ppc)
    dynamic_ranges = parse_float_list(args.dynamic_ranges)
    rows = []
    decile_rows = []
    clip_rows = []

    for dynamic_range in dynamic_ranges:
        kappa = 0.5 * np.log(dynamic_range)
        for seed in range(args.seed_start, args.seed_start + args.seeds):
            particles, profile = init_importance_sampled_landau_2d(
                Nx_marker=nx_marker,
                Ny_marker=ny_marker,
                Lx=grid.Lx,
                Ly=grid.Ly,
                v_thermal=thermal_velocity(),
                alpha=args.alpha,
                kx_mode=args.kx_mode,
                marker_kappa=kappa,
                rng=np.random.default_rng(seed),
                randomized_positions=True,
            )
            rho_ref = analytic_importance_sampled_density(
                grid.x,
                args.alpha,
                args.kx_mode,
                grid.Lx,
            )[:, None] * np.ones((1, grid.Ny))
            rho_cic = deposit_cic_2d(
                particles.x,
                particles.y,
                particles.w,
                grid.Nx,
                grid.Ny,
                grid.dx,
                grid.dy,
            )
            W = make_W_gaussian(
                grid,
                args.sigma_c4_dx * grid.dx,
                args.sigma_c4_dx * grid.dy,
            )
            rho_global = apply_W_to_grid(rho_cic, W, grid)

            marker_density = profile.pdf(particles.x)
            positive_density = marker_density[marker_density > 0.0]
            density_floor = max(
                float(np.min(positive_density)) * 1.0e-6,
                np.finfo(np.float64).tiny,
            )
            safe_density = np.maximum(marker_density, density_floor)
            geometric_mean = float(np.exp(np.mean(np.log(safe_density))))
            raw_sigma_cells = args.sigma_c4_dx * np.sqrt(
                geometric_mean / safe_density
            )
            upper_clipped = raw_sigma_cells > args.sigma_c4_dx
            lower_clipped = raw_sigma_cells < args.sigma_min_dx
            clip_rows.append(
                {
                    "dynamic_range": dynamic_range,
                    "seed": seed,
                    "upper_clip_fraction": float(np.mean(upper_clipped)),
                    "lower_clip_fraction": float(np.mean(lower_clipped)),
                    "unclipped_fraction": float(
                        np.mean(~upper_clipped & ~lower_clipped)
                    ),
                    "raw_sigma_min_dx": float(np.min(raw_sigma_cells)),
                    "raw_sigma_max_dx": float(np.max(raw_sigma_cells)),
                }
            )
            sigma_cells = abramson_c4_bandwidth(
                marker_density,
                sigma_pilot=np.asarray([args.sigma_c4_dx, args.sigma_c4_dx]),
                sigma_c4_max=np.asarray([args.sigma_c4_dx, args.sigma_c4_dx]),
                sigma_min=np.asarray([args.sigma_min_dx, args.sigma_min_dx]),
            )
            rho_local = deposit_binned_adaptive_spectral_W(
                particles,
                grid,
                sigma_cells[:, 0] * grid.dx,
                sigma_cells[:, 1] * grid.dy,
                n_bins=args.width_bins,
            )

            node_marker_density = np.repeat(profile.pdf(grid.x), grid.Ny)
            deciles = density_deciles(node_marker_density)
            ref_mode = grid.mode_amplitude(rho_ref, args.kx_mode, 0)
            for method, rho in (
                ("cic", rho_cic),
                ("global_c4", rho_global),
                ("abramson_c4_capped", rho_local),
            ):
                error = (rho - rho_ref).ravel() ** 2
                mode_error = abs(
                    grid.mode_amplitude(rho, args.kx_mode, 0) - ref_mode
                ) / max(abs(ref_mode), 1.0e-30)
                rows.append(
                    {
                        "dynamic_range": dynamic_range,
                        "seed": seed,
                        "method": method,
                        "mise": float(np.mean(error)),
                        "protected_mode_relative_error": float(mode_error),
                        "sum_w2": float(np.sum(particles.w**2)),
                        "effective_marker_count": float(
                            np.sum(np.abs(particles.w)) ** 2
                            / np.sum(particles.w**2)
                        ),
                        "sigma_min_observed_dx": (
                            0.0
                            if method == "cic"
                            else args.sigma_c4_dx
                            if method == "global_c4"
                            else float(np.min(sigma_cells))
                        ),
                        "sigma_max_observed_dx": (
                            0.0
                            if method == "cic"
                            else args.sigma_c4_dx
                            if method == "global_c4"
                            else float(np.max(sigma_cells))
                        ),
                    }
                )
                for decile in range(10):
                    decile_rows.append(
                        {
                            "dynamic_range": dynamic_range,
                            "seed": seed,
                            "method": method,
                            "marker_density_decile": decile,
                            "mse": float(np.mean(error[deciles == decile])),
                        }
                    )

    summary = []
    for dynamic_range in dynamic_ranges:
        for method in ("cic", "global_c4", "abramson_c4_capped"):
            selected = [
                row
                for row in rows
                if row["dynamic_range"] == dynamic_range and row["method"] == method
            ]
            values = np.asarray([row["mise"] for row in selected])
            relative_spread = float(
                np.std(values, ddof=1)
                / max(abs(np.mean(values)), np.max(np.abs(values)), np.finfo(float).tiny)
            )
            if relative_spread < args.min_relative_seed_spread:
                raise RuntimeError(
                    f"INCOMPLETE: {method} D={dynamic_range:g} relative seed "
                    f"spread {relative_spread:.3e} is below the guard"
                )
            mean, lo, hi = bootstrap_mean(
                values,
                args.nboot,
                31000 + int(dynamic_range) + len(method),
            )
            cic_mean = np.mean(
                [
                    row["mise"]
                    for row in rows
                    if row["dynamic_range"] == dynamic_range and row["method"] == "cic"
                ]
            )
            summary.append(
                {
                    "dynamic_range": dynamic_range,
                    "method": method,
                    "mise_mean": mean,
                    "mise_ci_lo": lo,
                    "mise_ci_hi": hi,
                    "noise_reduction_vs_cic": float(cic_mean / mean),
                    "relative_seed_spread": relative_spread,
                    "n_seeds": args.seeds,
                }
            )

    clip_summary = []
    for dynamic_range in dynamic_ranges:
        selected = [
            row for row in clip_rows if row["dynamic_range"] == dynamic_range
        ]
        summary_row = {"dynamic_range": dynamic_range, "n_seeds": args.seeds}
        for metric in (
            "upper_clip_fraction",
            "lower_clip_fraction",
            "unclipped_fraction",
        ):
            values = np.asarray([row[metric] for row in selected])
            mean, lo, hi = bootstrap_mean(
                values,
                args.nboot,
                41000 + int(dynamic_range) + len(metric),
            )
            summary_row[metric] = mean
            summary_row[f"{metric}_ci_lo"] = lo
            summary_row[f"{metric}_ci_hi"] = hi
        clip_summary.append(summary_row)

    write_csv(args.output / "H0_seed_metrics.csv", rows)
    write_csv(args.output / "H0_decile_metrics.csv", decile_rows)
    write_csv(args.output / "H0_summary.csv", summary)
    write_csv(args.output / "H0_clip_seed_metrics.csv", clip_rows)
    write_csv(args.output / "H0_clip_summary.csv", clip_summary)
    result = {
        "status": "complete",
        "scope": "estimator_only_no_dynamics_or_conservation_claim",
        "dynamic_ranges": dynamic_ranges,
        "methods": ["cic", "global_c4", "abramson_c4_capped"],
        "sigma_c4_dx": args.sigma_c4_dx,
        "n_seeds": args.seeds,
        "summary_csv": str(args.output / "H0_summary.csv"),
        "clip_summary_csv": str(args.output / "H0_clip_summary.csv"),
    }
    (args.output / "H0_result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--dynamic-ranges", default="10,100,1000")
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--Nx", type=int, default=64)
    parser.add_argument("--Ny", type=int, default=16)
    parser.add_argument("--Ly", type=float, default=8.0)
    parser.add_argument("--ppc", type=int, default=32)
    parser.add_argument("--kx-mode", type=int, default=1)
    parser.add_argument("--k-lambdaD", type=float, default=0.4)
    parser.add_argument("--alpha", type=float, default=0.01)
    parser.add_argument("--sigma-c4-dx", type=float, default=1.0)
    parser.add_argument("--sigma-min-dx", type=float, default=0.1)
    parser.add_argument("--width-bins", type=int, default=32)
    parser.add_argument("--nboot", type=int, default=2000)
    parser.add_argument("--min-relative-seed-spread", type=float, default=1.0e-10)
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
