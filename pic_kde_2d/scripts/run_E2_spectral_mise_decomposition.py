#!/usr/bin/env python3
"""Decompose E2 density MSE into physical, other-low-k, and high-k bands."""

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


def parse_float_list(text: str) -> list[float]:
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def write_csv(path: pathlib.Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row.keys()}) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
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


def cic_density(particles, grid: Grid2D) -> np.ndarray:
    return deposit_cic_2d(particles.x, particles.y, particles.w, grid.Nx, grid.Ny, grid.dx, grid.dy)


def band_masks(grid: Grid2D, kx_mode: int, low_k_cut: int) -> dict[str, np.ndarray]:
    mx = np.minimum(np.arange(grid.Nx), grid.Nx - np.arange(grid.Nx))[:, None]
    my = np.arange(grid.Ny // 2 + 1)[None, :]
    physical = (mx == kx_mode) & (my == 0)
    low = (mx <= low_k_cut) & (my <= low_k_cut) & ~physical
    high = ~(physical | low)
    return {"physical": physical, "low_other": low, "high": high}


def spectral_band_errors(delta: np.ndarray, masks: dict[str, np.ndarray]) -> dict[str, float]:
    """Return band MSE contributions whose sum approximately equals real-space MSE."""
    nx, ny = delta.shape
    hat = np.fft.rfft2(delta)
    power = np.abs(hat) ** 2
    # rfft omits negative ky partners. Double non-DC/non-Nyquist ky modes.
    weights = np.ones_like(power)
    if ny > 1:
        weights[:, 1:-1] *= 2.0
    norm = float((nx * ny) ** 2)
    return {name: float(np.sum(power[mask] * weights[mask]) / norm) for name, mask in masks.items()}


def run_one_seed(seed: int, sigma_grid: list[float], grid: Grid2D, args: argparse.Namespace) -> list[dict]:
    particles_low = make_particles(args.ppc, seed, args)
    particles_ref = make_particles(args.reference_ppc, seed, args)
    deposit_live, gather_live = make_spectral_W_gaussian_methods(
        grid,
        args.live_sigma_dx * grid.dx,
        args.live_sigma_dx * grid.dy,
    )
    filters = {sigma: make_W_gaussian(grid, sigma * grid.dx, sigma * grid.dy) for sigma in sigma_grid if sigma > 0.0}
    masks = band_masks(grid, args.kx_mode, args.low_k_cut)
    rows: list[dict] = []
    for step in range(args.n_steps + 1):
        time = step * args.dt
        record = step % args.diag_stride == 0 and args.eval_t_min <= time <= args.eval_t_max
        if record:
            rho_ref = cic_density(particles_ref, grid)
            rho_low = cic_density(particles_low, grid)
            for sigma in sigma_grid:
                rho = rho_low if sigma == 0.0 else apply_W_to_grid(rho_low, filters[sigma], grid)
                delta = rho - rho_ref
                bands = spectral_band_errors(delta, masks)
                total = float(np.mean(delta * delta))
                rows.append(
                    {
                        "seed": seed,
                        "time": time,
                        "sigma_dx": sigma,
                        "mse_total": total,
                        "mse_physical": bands["physical"],
                        "mse_low_other": bands["low_other"],
                        "mse_high": bands["high"],
                        "physical_fraction": bands["physical"] / total if total > 0.0 else float("nan"),
                        "high_fraction": bands["high"] / total if total > 0.0 else float("nan"),
                    }
                )
        if step == args.n_steps:
            break
        pic_step(particles_low, grid, args.dt, deposit_live, gather_live)
        pic_step(particles_ref, grid, args.dt, deposit_live, gather_live)
    return rows


def bootstrap_mean(values: np.ndarray, nboot: int, seed: int) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    draws = values[rng.integers(0, len(values), size=(nboot, len(values)))]
    means = np.mean(draws, axis=1)
    return float(np.mean(values)), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def summarize(rows: list[dict], args: argparse.Namespace) -> list[dict]:
    sigma_values = sorted({float(row["sigma_dx"]) for row in rows})
    summary = []
    for sigma in sigma_values:
        sub = [row for row in rows if float(row["sigma_dx"]) == sigma]
        seed_ids = sorted({int(row["seed"]) for row in sub})
        seed_means = []
        for seed in seed_ids:
            srows = [row for row in sub if int(row["seed"]) == seed]
            seed_means.append(
                {
                    key: float(np.mean([row[key] for row in srows]))
                    for key in ["mse_total", "mse_physical", "mse_low_other", "mse_high", "physical_fraction", "high_fraction"]
                }
            )
        out = {"sigma_dx": sigma, "n_seeds": len(seed_ids)}
        for key in ["mse_total", "mse_physical", "mse_low_other", "mse_high", "physical_fraction", "high_fraction"]:
            vals = np.asarray([row[key] for row in seed_means], dtype=np.float64)
            mean, lo, hi = bootstrap_mean(vals, args.nboot, 88100 + int(1000 * sigma) + len(key))
            out[key] = mean
            out[f"{key}_ci_lo"] = lo
            out[f"{key}_ci_hi"] = hi
        summary.append(out)
    total0 = next(row["mse_total"] for row in summary if abs(row["sigma_dx"]) < 1.0e-12)
    high0 = next(row["mse_high"] for row in summary if abs(row["sigma_dx"]) < 1.0e-12)
    for row in summary:
        row["total_noise_reduction"] = total0 / row["mse_total"]
        row["highk_reduction"] = high0 / row["mse_high"] if row["mse_high"] > 0.0 else float("inf")
    return summary


def run(args: argparse.Namespace) -> dict:
    args.output.mkdir(parents=True, exist_ok=True)
    sigma_grid = parse_float_list(args.sigma_dx)
    grid = Grid2D(Nx=args.Nx, Ny=args.Ny, Lx=args.Lx, Ly=args.Ly)
    all_rows = []
    for seed in range(args.seed_start, args.seed_start + args.seeds):
        all_rows.extend(run_one_seed(seed, sigma_grid, grid, args))
        print(f"spectral MSE decomposition seed={seed}", flush=True)
    summary = summarize(all_rows, args)
    write_csv(args.output / "E2_spectral_mise_decomposition_samples.csv", all_rows)
    write_csv(args.output / "E2_spectral_mise_decomposition_summary.csv", summary)
    result = {
        "summary_csv": str(args.output / "E2_spectral_mise_decomposition_summary.csv"),
        "sample_csv": str(args.output / "E2_spectral_mise_decomposition_samples.csv"),
        "bands": {
            "physical": f"(mx,my)=({args.kx_mode},0)",
            "low_other": f"mx,my <= {args.low_k_cut}, excluding physical",
            "high": "all remaining resolved modes",
        },
        "n_seeds": args.seeds,
    }
    with (args.output / "E2_spectral_mise_decomposition.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=pathlib.Path, default=PROJECT_ROOT / "results" / "E2_spectral_mise_decomposition")
    parser.add_argument("--sigma-dx", default="0.0,0.25,0.30,0.35,0.45,0.68,1.0,3.0,12.0")
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--ppc", type=int, default=32)
    parser.add_argument("--reference-ppc", type=int, default=1024)
    parser.add_argument("--Nx", type=int, default=128)
    parser.add_argument("--Ny", type=int, default=16)
    parser.add_argument("--Lx", type=float, default=64.0)
    parser.add_argument("--Ly", type=float, default=16.0)
    parser.add_argument("--kx-mode", type=int, default=12)
    parser.add_argument("--low-k-cut", type=int, default=16)
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
    run(build_parser().parse_args())

