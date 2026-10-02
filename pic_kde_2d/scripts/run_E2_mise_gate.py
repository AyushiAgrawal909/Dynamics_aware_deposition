#!/usr/bin/env python3
"""E2 density MISE gate using a high-ppc reference on fixed reference dynamics.

The live trajectory is deliberately fixed at the executed E2 reference method
(`sigma_live=0.25 dx`) for every diagnostic candidate. Candidate sigma values
are used only for density reconstruction, so this script measures the density
selector rather than folding dynamic force suppression into its target.
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
    """Parse comma-separated floating point values."""
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def write_csv(path: pathlib.Path, rows: list[dict]) -> None:
    """Write rows to CSV without adding a pandas dependency."""
    fieldnames = sorted({key for row in rows for key in row.keys()}) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def make_particles(ppc: int, seed: int, args: argparse.Namespace):
    """Construct the exact E2 two-stream initialization at a requested ppc."""
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
    """Deposit unfiltered CIC density used for the high-ppc truth output."""
    return deposit_cic_2d(
        particles.x,
        particles.y,
        particles.w,
        grid.Nx,
        grid.Ny,
        grid.dx,
        grid.dy,
    )


def bootstrap_mean_ci(values: np.ndarray, nboot: int, seed: int) -> tuple[float, float, float]:
    """Bootstrap a mean over paired-seed values."""
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=np.float64)
    sampled = values[rng.integers(0, len(values), size=(nboot, len(values)))]
    means = np.mean(sampled, axis=1)
    return float(np.mean(values)), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def run_one_seed(
    seed: int,
    sigma_grid: list[float],
    grid: Grid2D,
    args: argparse.Namespace,
) -> tuple[list[dict], np.ndarray, np.ndarray]:
    """Run paired sample/reference trajectories and return density-error metrics."""
    particles_low = make_particles(args.ppc, seed, args)
    particles_ref = make_particles(args.reference_ppc, seed, args)

    sigma_live_x = args.live_sigma_dx * grid.dx
    sigma_live_y = args.live_sigma_dx * grid.dy
    deposit_live, gather_live = make_spectral_W_gaussian_methods(grid, sigma_live_x, sigma_live_y)

    filters = {
        sigma: make_W_gaussian(grid, sigma * grid.dx, sigma * grid.dy)
        for sigma in sigma_grid
        if sigma > 0.0
    }
    squared_errors: dict[float, list[float]] = {sigma: [] for sigma in sigma_grid}
    rho_reference: list[np.ndarray] = []
    reference_times: list[float] = []

    for step in range(args.n_steps + 1):
        time = step * args.dt
        record = (
            step % args.diag_stride == 0
            and args.eval_t_min <= time <= args.eval_t_max
        )
        if record:
            rho_ref = cic_density(particles_ref, grid)
            rho_low_cic = cic_density(particles_low, grid)
            rho_reference.append(rho_ref)
            reference_times.append(time)
            for sigma in sigma_grid:
                rho_candidate = (
                    rho_low_cic
                    if sigma == 0.0
                    else apply_W_to_grid(rho_low_cic, filters[sigma], grid)
                )
                squared_errors[sigma].append(float(np.mean((rho_candidate - rho_ref) ** 2)))

        if step == args.n_steps:
            break
        pic_step(particles_low, grid, args.dt, deposit_live, gather_live)
        pic_step(particles_ref, grid, args.dt, deposit_live, gather_live)

    rows = [
        {
            "seed": seed,
            "sigma_dx": sigma,
            "mise_time_avg": float(np.mean(squared_errors[sigma])),
            "n_times": len(squared_errors[sigma]),
        }
        for sigma in sigma_grid
    ]
    return rows, np.asarray(rho_reference), np.asarray(reference_times)


def classify_gate(summary: list[dict]) -> dict:
    """Classify the requested gate from the measured discrete sigma sweep."""
    ordered = sorted(summary, key=lambda row: float(row["sigma_dx"]))
    selection = identify_cv_minimum(
        [float(row["sigma_dx"]) for row in ordered],
        [float(row["mise_mean"]) for row in ordered],
    )
    if not selection.identified:
        return {
            "sigma_mise_dx": None,
            "sigma_mise_relation": ">=" if selection.status == "right_censored" else "<=",
            "sigma_mise_bound_dx": (
                selection.lower_bound
                if selection.status == "right_censored"
                else selection.search_min
            ),
            "classification": f"density_optimum_{selection.status}",
            "interpretation": selection.explanation,
        }
    sigma_opt = float(selection.bandwidth)
    classification = "needs_refinement"
    interpretation = "The discrete optimum is at the safety boundary; refine the sigma grid."
    if sigma_opt <= 0.35:
        classification = "density_optimal_is_dynamically_safe_stop"
        interpretation = "Density-optimal smoothing is already in the dynamically safe range."
    elif sigma_opt > 0.45:
        classification = "density_optimal_is_dynamically_degraded_proceed"
        interpretation = "Density-optimal smoothing is in the dynamically degraded region."
    else:
        row_035 = min(summary, key=lambda row: abs(float(row["sigma_dx"]) - 0.35))
        row_068 = min(summary, key=lambda row: abs(float(row["sigma_dx"]) - 0.68))
        intervals_overlap = not (
            row_035["mise_ci_hi"] < row_068["mise_ci_lo"]
            or row_068["mise_ci_hi"] < row_035["mise_ci_lo"]
        )
        if intervals_overlap:
            classification = "mise_does_not_discriminate_safe_from_unsafe_proceed_adjusted_claim"
            interpretation = "MISE confidence intervals overlap between safe and unsafe sigma values."
    return {
        "sigma_mise_dx": sigma_opt,
        "sigma_mise_relation": "=",
        "sigma_mise_bound_dx": None,
        "classification": classification,
        "interpretation": interpretation,
    }


def run_gate(args: argparse.Namespace) -> dict:
    """Generate rho_ref snapshots and compute the E2 density MISE sweep."""
    args.output.mkdir(parents=True, exist_ok=True)
    sigma_grid = parse_float_list(args.sigma_dx)
    grid = Grid2D(Nx=args.Nx, Ny=args.Ny, Lx=args.Lx, Ly=args.Ly)
    seeds = list(range(args.seed_start, args.seed_start + args.seeds))
    seed_rows: list[dict] = []
    references: list[np.ndarray] = []
    times: np.ndarray | None = None

    for seed in seeds:
        rows, rho_ref, seed_times = run_one_seed(seed, sigma_grid, grid, args)
        seed_rows.extend(rows)
        references.append(rho_ref)
        if times is None:
            times = seed_times
        elif not np.array_equal(times, seed_times):
            raise RuntimeError("reference diagnostic times differ across seeds")
        print(f"E2 MISE completed seed={seed}", flush=True)

    if times is None:
        raise RuntimeError("no reference snapshots were recorded")
    summary_rows: list[dict] = []
    for sigma in sigma_grid:
        values = np.asarray(
            [row["mise_time_avg"] for row in seed_rows if row["sigma_dx"] == sigma],
            dtype=np.float64,
        )
        mean, lo, hi = bootstrap_mean_ci(values, args.nboot, 56100 + int(round(sigma * 1000)))
        summary_rows.append(
            {
                "sigma_dx": sigma,
                "mise_mean": mean,
                "mise_ci_lo": lo,
                "mise_ci_hi": hi,
                "n_seeds": len(values),
                "n_times": int(next(row["n_times"] for row in seed_rows if row["sigma_dx"] == sigma)),
            }
        )

    gate = classify_gate(summary_rows)
    reference_path = args.output / "rho_ref_e2_linear_window.npz"
    np.savez_compressed(
        reference_path,
        rho_ref=np.asarray(references, dtype=np.float64),
        times=times,
        seeds=np.asarray(seeds, dtype=np.int64),
        reference_ppc=np.asarray([args.reference_ppc], dtype=np.int64),
    )
    write_csv(args.output / "E2_mise_seed_values.csv", seed_rows)
    write_csv(args.output / "E2_mise_summary.csv", summary_rows)
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
            "reference_ppc": args.reference_ppc,
            "live_sigma_dx": args.live_sigma_dx,
            "dt": args.dt,
            "n_steps": args.n_steps,
            "diag_stride": args.diag_stride,
            "eval_window": [args.eval_t_min, args.eval_t_max],
            "n_seeds": len(seeds),
        },
        "reference_definition": (
            "unsmoothed high-ppc CIC density snapshots on trajectories evolved "
            "with executed E2 reference dynamics sigma_live=0.25 dx"
        ),
        "reference_npz": str(reference_path),
        "summary": summary_rows,
        "gate": gate,
    }
    with (args.output / "E2_mise_gate_result.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return result


def build_parser() -> argparse.ArgumentParser:
    """Create command-line parser with the executed E2 configuration as defaults."""
    parser = argparse.ArgumentParser(description="Compute the diagnostic density MISE gate for E2.")
    parser.add_argument("--output", type=pathlib.Path, default=PROJECT_ROOT / "results" / "E2_mise_gate")
    parser.add_argument(
        "--sigma-dx",
        type=str,
        default="0.0,0.10,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.68,0.75,1.0,1.25,1.5,2.0,3.0,4.0,5.0,6.0,8.0,12.0",
    )
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--seeds", type=int, default=10)
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
    parser.add_argument("--nboot", type=int, default=2000)
    return parser


if __name__ == "__main__":
    run_gate(build_parser().parse_args())
