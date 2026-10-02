#!/usr/bin/env python3
"""Measure Landau damping fidelity at wide smoothing values."""

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

from pic_kde_2d.benchmarks.theory import landau_gamma_theory
from pic_kde_2d.conventions import landau_box_length, marker_grid_shape, thermal_velocity
from pic_kde_2d.core.grid2d import Grid2D
from pic_kde_2d.core.particles2d import init_landau_2d_eigenmode
from pic_kde_2d.core.pic_loop import run_simulation
from pic_kde_2d.methods.spectral_W_wrappers import make_spectral_W_gaussian_methods
from pic_kde_2d.benchmarks.theory import langmuir_omega_r
from pic_kde_2d.validation.landau_fit import bootstrap_gamma_ci, fit_gamma_ensemble


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
    Lx = landau_box_length(args.kx_mode, k_lambdaD=args.k_lambdaD)
    grid = Grid2D(Nx=args.Nx, Ny=args.Ny, Lx=Lx, Ly=args.Ly)
    sigma = sigma_dx * grid.dx
    deposit, gather = make_spectral_W_gaussian_methods(grid, sigma, sigma)
    nx_marker, ny_marker = marker_grid_shape(args.Nx, args.Ny, args.ppc)
    particles = init_landau_2d_eigenmode(
        Nx_marker=nx_marker,
        Ny_marker=ny_marker,
        Lx=Lx,
        Ly=args.Ly,
        v_thermal=thermal_velocity(),
        alpha=args.alpha,
        kx_mode=args.kx_mode,
        omega_r=langmuir_omega_r(args.k_lambdaD),
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


def run(args: argparse.Namespace) -> dict:
    args.output.mkdir(parents=True, exist_ok=True)
    sigma_grid = sorted(set(parse_float_list(args.sigma_dx) + [args.reference_sigma_dx]))
    rows = []
    gamma_theory = landau_gamma_theory(k_lambdaD=args.k_lambdaD)
    amplitudes: dict[float, list[np.ndarray]] = {}
    times_by_sigma: dict[float, np.ndarray] = {}
    for sigma in sigma_grid:
        amplitudes[sigma] = []
        for seed in range(args.seed_start, args.seed_start + args.seeds):
            times, amp = run_one(sigma, seed, args)
            amplitudes[sigma].append(amp)
            times_by_sigma[sigma] = times
            print(f"Landau tail sigma={sigma:g} seed={seed}", flush=True)
    gamma_ref = None
    for sigma in sigma_grid:
        fit = fit_gamma_ensemble(
            amplitudes[sigma],
            times_by_sigma[sigma],
            t_fit_start=args.fit_t_min,
            t_fit_stop_max=args.fit_t_max,
            noise_floor_start=args.noise_floor_start,
            noise_factor=args.noise_factor,
        )
        ci = bootstrap_gamma_ci(
            amplitudes[sigma],
            times_by_sigma[sigma],
            n_boot=args.nboot,
            seed=100300 + int(round(1000.0 * sigma)),
            t_fit_start=args.fit_t_min,
            t_fit_stop_max=args.fit_t_max,
            noise_floor_start=args.noise_floor_start,
            noise_factor=args.noise_factor,
        )
        if abs(sigma - args.reference_sigma_dx) < 1.0e-12:
            gamma_ref = fit["gamma"]
        rows.append(
            {
                "sigma_dx": sigma,
                "gamma": fit["gamma"],
                "gamma_ci_lo": ci["ci_lo"],
                "gamma_ci_hi": ci["ci_hi"],
                "gamma_sem": ci["gamma_sem"],
                "gamma_theory": gamma_theory,
                "relative_error_to_theory": abs(fit["gamma"] - gamma_theory) / abs(gamma_theory),
                "n_fit_points": fit["n_points"],
                "fit_t_start": fit["t_fit"][0],
                "fit_t_stop": fit["t_fit"][1],
                "noise_floor": fit["noise_floor"],
                "n_seeds": args.seeds,
            }
        )
    if gamma_ref is not None:
        for row in rows:
            row["gamma_ref"] = gamma_ref
            row["fidelity_to_ref"] = row["gamma"] / gamma_ref if gamma_ref != 0.0 else float("nan")
    write_csv(args.output / "landau_tail_fidelity.csv", rows)
    result = {
        "csv": str(args.output / "landau_tail_fidelity.csv"),
        "reference_sigma_dx": args.reference_sigma_dx,
        "n_seeds": args.seeds,
        "gamma_theory": gamma_theory,
        "rows": rows,
    }
    with (args.output / "landau_tail_fidelity.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=pathlib.Path, default=PROJECT_ROOT / "results" / "landau_tail_fidelity")
    parser.add_argument("--sigma-dx", default="0.25,3,4,6,8,12")
    parser.add_argument("--reference-sigma-dx", type=float, default=0.25)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--seeds", type=int, default=60)
    parser.add_argument("--ppc", type=int, default=64)
    parser.add_argument("--Nx", type=int, default=64)
    parser.add_argument("--Ny", type=int, default=16)
    parser.add_argument("--Ly", type=float, default=8.0)
    parser.add_argument("--kx-mode", type=int, default=1)
    parser.add_argument("--k-lambdaD", type=float, default=0.4)
    parser.add_argument("--alpha", type=float, default=0.02)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--n-steps", type=int, default=1200)
    parser.add_argument("--diag-stride", type=int, default=2)
    parser.add_argument("--fit-t-min", type=float, default=5.0)
    parser.add_argument("--fit-t-max", type=float, default=30.0)
    parser.add_argument("--noise-floor-start", type=float, default=40.0)
    parser.add_argument("--noise-factor", type=float, default=8.0)
    parser.add_argument("--nboot", type=int, default=1000)
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())

