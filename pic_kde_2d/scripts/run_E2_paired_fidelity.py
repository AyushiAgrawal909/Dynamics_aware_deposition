#!/usr/bin/env python3
"""Paired common-random-number E2 growth-fidelity ratios.

This script is for the statistical gate on the dynamics selector. It runs the
same seed across all requested sigma values and bootstraps paired seed indices,
so growth-rate ratios exploit common random numbers instead of comparing two
independent noisy ensembles.
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
from pic_kde_2d.core.pic_loop import run_simulation
from pic_kde_2d.methods.spectral_W_wrappers import make_spectral_W_gaussian_methods
from pic_kde_2d.validation.landau_fit import fit_gamma_ensemble_fixed_window


def parse_float_list(text: str) -> list[float]:
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def write_csv(path: pathlib.Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row.keys()}) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run_one(sigma_dx: float, seed: int, args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray]:
    grid = Grid2D(Nx=args.Nx, Ny=args.Ny, Lx=args.Lx, Ly=args.Ly)
    deposit, gather = make_spectral_W_gaussian_methods(grid, sigma_dx * grid.dx, sigma_dx * grid.dy)
    nx_marker, ny_marker = marker_grid_shape(args.Nx, args.Ny, args.ppc)
    particles = init_two_stream_2d(
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
    history = run_simulation(
        particles,
        grid,
        args.dt,
        args.n_steps,
        deposit,
        gather,
        diag_stride=args.diag_stride,
        tracked_modes=[(args.kx_mode, 0)],
    )
    times = np.asarray(history.times, dtype=np.float64)
    amp = np.abs(np.asarray(history.mode_amplitudes_Ex[(args.kx_mode, 0)], dtype=np.complex128))
    return times, amp


def fit_gamma(amplitudes: list[np.ndarray], times: np.ndarray, args: argparse.Namespace) -> float:
    return float(
        fit_gamma_ensemble_fixed_window(
            amplitudes,
            times,
            t_fit_start=args.fit_t_min,
            t_fit_stop=args.fit_t_max,
            amp_floor=1.0e-30,
        )["gamma"]
    )


def paired_ratio_ci(
    amplitudes: dict[float, list[np.ndarray]],
    sigma: float,
    reference_sigma: float,
    times: np.ndarray,
    args: argparse.Namespace,
) -> dict:
    rng = np.random.default_rng(args.bootstrap_seed + int(round(1000.0 * sigma)))
    n = len(amplitudes[reference_sigma])
    ratios = []
    gamma_values = []
    ref_values = []
    for _ in range(args.nboot):
        idx = rng.integers(0, n, size=n)
        sample = [amplitudes[sigma][int(i)] for i in idx]
        ref = [amplitudes[reference_sigma][int(i)] for i in idx]
        gamma = fit_gamma(sample, times, args)
        gamma_ref = fit_gamma(ref, times, args)
        if np.isfinite(gamma) and np.isfinite(gamma_ref) and gamma_ref != 0.0:
            ratios.append(gamma / gamma_ref)
            gamma_values.append(gamma)
            ref_values.append(gamma_ref)
    values = np.asarray(ratios, dtype=np.float64)
    gamma_arr = np.asarray(gamma_values, dtype=np.float64)
    ref_arr = np.asarray(ref_values, dtype=np.float64)
    return {
        "ratio_mean": float(np.mean(values)),
        "ratio_sem": float(np.std(values)),
        "ratio_ci_lo": float(np.percentile(values, 2.5)),
        "ratio_ci_hi": float(np.percentile(values, 97.5)),
        "gamma_bootstrap_mean": float(np.mean(gamma_arr)),
        "gamma_ref_bootstrap_mean": float(np.mean(ref_arr)),
    }


def run(args: argparse.Namespace) -> dict:
    args.output.mkdir(parents=True, exist_ok=True)
    sigma_grid = sorted(set(parse_float_list(args.sigma_dx) + [args.reference_sigma_dx]))
    seeds = list(range(args.seed_start, args.seed_start + args.seeds))
    amplitudes: dict[float, list[np.ndarray]] = {}
    times_by_sigma: dict[float, np.ndarray] = {}
    for sigma in sigma_grid:
        amplitudes[sigma] = []
        for seed in seeds:
            times, amp = run_one(sigma, seed, args)
            amplitudes[sigma].append(amp)
            times_by_sigma[sigma] = times
            print(f"paired E2 sigma={sigma:g} seed={seed}", flush=True)

    times = times_by_sigma[args.reference_sigma_dx]
    gamma_by_sigma = {sigma: fit_gamma(amplitudes[sigma], times_by_sigma[sigma], args) for sigma in sigma_grid}
    rows = []
    for sigma in sigma_grid:
        ratio = paired_ratio_ci(amplitudes, sigma, args.reference_sigma_dx, times, args)
        central_ratio = gamma_by_sigma[sigma] / gamma_by_sigma[args.reference_sigma_dx]
        pass_central = central_ratio >= args.min_ratio
        pass_statistical = ratio["ratio_ci_lo"] >= args.min_ratio
        inconclusive = pass_central and not pass_statistical
        rows.append(
            {
                "sigma_dx": sigma,
                "reference_sigma_dx": args.reference_sigma_dx,
                "gamma": gamma_by_sigma[sigma],
                "gamma_reference": gamma_by_sigma[args.reference_sigma_dx],
                "central_ratio": central_ratio,
                "paired_ratio_mean": ratio["ratio_mean"],
                "paired_ratio_sem": ratio["ratio_sem"],
                "paired_ratio_ci_lo": ratio["ratio_ci_lo"],
                "paired_ratio_ci_hi": ratio["ratio_ci_hi"],
                "min_ratio": args.min_ratio,
                "pass_central": pass_central,
                "pass_statistical": pass_statistical,
                "inconclusive": inconclusive,
                "n_seeds": args.seeds,
            }
        )
    write_csv(args.output / "E2_paired_fidelity.csv", rows)
    result = {
        "csv": str(args.output / "E2_paired_fidelity.csv"),
        "reference_sigma_dx": args.reference_sigma_dx,
        "min_ratio": args.min_ratio,
        "n_seeds": args.seeds,
        "rows": rows,
    }
    with (args.output / "E2_paired_fidelity.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=pathlib.Path, default=PROJECT_ROOT / "results" / "part2" / "E2_paired_fidelity")
    parser.add_argument("--sigma-dx", default="0.0,0.25,0.30,0.35,0.45")
    parser.add_argument("--reference-sigma-dx", type=float, default=0.0)
    parser.add_argument("--min-ratio", type=float, default=0.90)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--seeds", type=int, default=120)
    parser.add_argument("--ppc", type=int, default=32)
    parser.add_argument("--Nx", type=int, default=128)
    parser.add_argument("--Ny", type=int, default=16)
    parser.add_argument("--Lx", type=float, default=64.0)
    parser.add_argument("--Ly", type=float, default=16.0)
    parser.add_argument("--kx-mode", type=int, default=12)
    parser.add_argument("--v-beam", type=float, default=0.8)
    parser.add_argument("--v-thermal", type=float, default=0.05)
    parser.add_argument("--alpha", type=float, default=0.001)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--n-steps", type=int, default=300)
    parser.add_argument("--diag-stride", type=int, default=2)
    parser.add_argument("--fit-t-min", type=float, default=0.5)
    parser.add_argument("--fit-t-max", type=float, default=4.0)
    parser.add_argument("--nboot", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=77231)
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())

