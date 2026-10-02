#!/usr/bin/env python3
"""
Revised E1: sigma-selection rule with fixed dx, ensemble fitting, and CIC correction.
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

from pic_kde_2d.benchmarks.theory import landau_gamma_theory, langmuir_omega_r
from pic_kde_2d.conventions import landau_box_length, marker_grid_shape, thermal_velocity
from pic_kde_2d.core.grid2d import Grid2D
from pic_kde_2d.core.particles2d import init_landau_2d_eigenmode
from pic_kde_2d.core.pic_loop import run_simulation
from pic_kde_2d.methods.spectral_W_wrappers import make_spectral_W_gaussian_methods
from pic_kde_2d.validation.landau_fit import (
    bootstrap_gamma_ci,
    fit_gamma_ensemble,
    fit_gamma_ensemble_fixed_window,
)
from pic_kde_2d.validation.transfer_empirical import (
    sigma_max_cic_corrected,
    sigma_max_pure_gaussian,
)


def parse_int_list(text: str) -> list[int]:
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def parse_float_list(text: str) -> list[float]:
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def write_csv(path: pathlib.Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()}) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def round_to_multiple(value: int, multiple: int) -> int:
    return max(multiple, multiple * ((value + multiple // 2) // multiple))


def grid_for_kmode(k_mode: int, k_lambdaD: float, dx_target: float, x_multiple: int) -> tuple[int, float]:
    Lx = landau_box_length(k_mode, k_lambdaD=k_lambdaD)
    Nx = round_to_multiple(int(np.round(Lx / dx_target)), x_multiple)
    return Nx, Lx


def run_one(
    k_mode: int,
    sigma_dx: float,
    seed: int,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, int, float]:
    Nx, Lx = grid_for_kmode(k_mode, args.k_lambdaD, args.dx_target, args.x_multiple)
    grid = Grid2D(Nx=Nx, Ny=args.Ny, Lx=Lx, Ly=args.Ly)
    if abs(grid.dx - args.dx_target) / args.dx_target > args.dx_assert_rel_tol:
        raise RuntimeError(f"dx scaling drifted: k={k_mode} dx={grid.dx} target={args.dx_target}")
    sigma_x = sigma_dx * grid.dx
    sigma_y = args.sigma_y_dx * grid.dy
    deposit, gather = make_spectral_W_gaussian_methods(grid, sigma_x, sigma_y)
    Nx_marker, Ny_marker = marker_grid_shape(Nx, args.Ny, args.ppc)
    particles = init_landau_2d_eigenmode(
        Nx_marker=Nx_marker,
        Ny_marker=Ny_marker,
        Lx=Lx,
        Ly=args.Ly,
        v_thermal=thermal_velocity(),
        alpha=args.alpha,
        kx_mode=k_mode,
        omega_r=langmuir_omega_r(args.k_lambdaD, v_thermal=thermal_velocity()),
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
        tracked_modes=[(k_mode, 0)],
        verbose=False,
    )
    times = np.asarray(history.times, dtype=np.float64)
    amps = np.abs(np.asarray(history.mode_amplitudes_Ex[(k_mode, 0)], dtype=np.complex128))
    return times, amps, Nx, grid.dx


def run_E1(args: argparse.Namespace) -> dict:
    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)
    k_modes = parse_int_list(args.k_modes)
    sigma_grid = parse_float_list(args.sigma_dx)
    seeds = list(range(args.seed_start, args.seed_start + args.seeds))
    gamma_theory = landau_gamma_theory(args.k_lambdaD, v_thermal=thermal_velocity())

    summary_rows: list[dict] = []
    sigma_opt_rows: list[dict] = []

    for k_mode in k_modes:
        gamma_by_sigma: dict[float, float] = {}
        sem_by_sigma: dict[float, float] = {}
        Nx, _ = grid_for_kmode(k_mode, args.k_lambdaD, args.dx_target, args.x_multiple)
        for sigma_dx in sigma_grid:
            amplitude_list: list[np.ndarray] = []
            times_ref: np.ndarray | None = None
            dx_seen = float("nan")
            for seed in seeds:
                times, amps, _, dx_seen = run_one(k_mode, sigma_dx, seed, args)
                times_ref = times
                amplitude_list.append(amps)
                print(f"E1 revised k={k_mode} sigma={sigma_dx:.2f} seed={seed}", flush=True)
            if times_ref is None:
                raise RuntimeError("no E1 amplitudes collected")
            if args.fit_mode == "fixed":
                fit = fit_gamma_ensemble_fixed_window(
                    amplitude_list,
                    times_ref,
                    t_fit_start=args.fit_t_min,
                    t_fit_stop=args.fit_t_max,
                )
                fit.update(
                    {
                        "fit_strategy": "fixed_window",
                        "noise_floor": float("nan"),
                        "t_fit_start": args.fit_t_min,
                        "t_fit_stop": args.fit_t_max,
                    }
                )
                ci = bootstrap_gamma_ci(
                    amplitude_list,
                    times_ref,
                    n_boot=args.nboot,
                    seed=12345 + 1000 * k_mode + int(round(100.0 * sigma_dx)),
                    fixed_window=True,
                    t_fit_start=args.fit_t_min,
                    t_fit_stop=args.fit_t_max,
                )
            else:
                fit = fit_gamma_ensemble(
                    amplitude_list,
                    times_ref,
                    t_fit_start=args.fit_t_min,
                    t_fit_stop_max=args.fit_t_max,
                    noise_floor_start=args.noise_floor_start,
                    noise_factor=args.noise_factor,
                )
                ci = bootstrap_gamma_ci(
                    amplitude_list,
                    times_ref,
                    n_boot=args.nboot,
                    seed=12345 + 1000 * k_mode + int(round(100.0 * sigma_dx)),
                    t_fit_start=args.fit_t_min,
                    t_fit_stop_max=args.fit_t_max,
                    noise_floor_start=args.noise_floor_start,
                    noise_factor=args.noise_factor,
                )
            gamma = float(fit["gamma"])
            gamma_by_sigma[sigma_dx] = gamma
            sem_by_sigma[sigma_dx] = float(ci["gamma_sem"])
            summary_rows.append(
                {
                    "Nx": Nx,
                    "dx": dx_seen,
                    "gamma": gamma,
                    "gamma_CI_hi": ci["ci_hi"],
                    "gamma_CI_lo": ci["ci_lo"],
                    "gamma_sem": ci["gamma_sem"],
                    "gamma_theory": gamma_theory,
                    "fit_strategy": fit["fit_strategy"],
                    "k_mode": k_mode,
                    "n_fit_points": fit["n_points"],
                    "n_seeds": len(seeds),
                    "noise_floor": fit["noise_floor"],
                    "sigma_dx": sigma_dx,
                    "t_fit_start": fit["t_fit_start"],
                    "t_fit_stop": fit["t_fit_stop"],
                }
            )
            print(
                f"E1 revised ensemble k={k_mode} sigma={sigma_dx:.2f}: "
                f"gamma={gamma:.6f} sem={ci['gamma_sem']:.3e}",
                flush=True,
            )

        small = [gamma_by_sigma[sigma] for sigma in sigma_grid if sigma <= args.gamma_ref_sigma_max]
        gamma_ref = float(np.nanmean(small))
        sigma_opt = float("nan")
        for sigma_dx in sorted(sigma_grid):
            gamma = gamma_by_sigma[sigma_dx]
            sem = sem_by_sigma[sigma_dx]
            threshold = max(args.gamma_relative_tolerance * abs(gamma_ref), args.sem_factor * sem)
            if np.isfinite(gamma) and abs(gamma - gamma_ref) < threshold:
                sigma_opt = sigma_dx

        pred_pure = sigma_max_pure_gaussian(Nx, k_mode, args.A_min)
        pred_cic = sigma_max_cic_corrected(Nx, k_mode, args.A_min, cic_power=args.cic_power)
        rel_err_pure = abs(sigma_opt - pred_pure) / pred_pure if np.isfinite(sigma_opt) else float("nan")
        rel_err_cic = abs(sigma_opt - pred_cic) / pred_cic if pred_cic > 0.0 and np.isfinite(sigma_opt) else float("nan")
        sigma_opt_rows.append(
            {
                "Nx": Nx,
                "gamma_ref": gamma_ref,
                "gamma_ref_rel_error_to_theory": abs(gamma_ref - gamma_theory) / abs(gamma_theory),
                "k_mode": k_mode,
                "pass_gamma_ref": abs(gamma_ref - gamma_theory) / abs(gamma_theory) < args.gamma_ref_theory_tolerance,
                "pass_pure_prediction": bool(np.isfinite(rel_err_pure) and rel_err_pure < args.prediction_relative_tolerance),
                "pred_cic_corrected": pred_cic,
                "pred_pure_gaussian": pred_pure,
                "rel_err_cic": rel_err_cic,
                "rel_err_pure": rel_err_pure,
                "residual_cic": sigma_opt - pred_cic if np.isfinite(sigma_opt) else float("nan"),
                "residual_pure": sigma_opt - pred_pure if np.isfinite(sigma_opt) else float("nan"),
                "sigma_opt_dx": sigma_opt,
            }
        )

    suffix = f"_{args.tag}" if args.tag else ""
    summary_path = output_dir / f"E1_revised_summary{suffix}.csv"
    sigma_opt_path = output_dir / f"E1_revised_sigma_opt{suffix}.csv"
    write_csv(summary_path, summary_rows)
    write_csv(sigma_opt_path, sigma_opt_rows)

    gamma_ref_ok = all(bool(row["pass_gamma_ref"]) for row in sigma_opt_rows)
    pred_ok = all(bool(row["pass_pure_prediction"]) for row in sigma_opt_rows)
    result = {
        "summary_csv": str(summary_path),
        "sigma_opt_csv": str(sigma_opt_path),
        "passed": bool(gamma_ref_ok and pred_ok),
        "gamma_ref_ok": bool(gamma_ref_ok),
        "prediction_ok": bool(pred_ok),
        "n_summary_rows": len(summary_rows),
        "n_seeds": len(seeds),
    }
    json_path = output_dir / f"E1_revised_result{suffix}.json"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run revised Phase 4 E1.")
    parser.add_argument("--output", type=pathlib.Path, default=PROJECT_ROOT / "results" / "E1_revised")
    parser.add_argument("--tag", type=str, default="")
    parser.add_argument("--k-modes", type=str, default="1,2,4,8")
    parser.add_argument("--sigma-dx", type=str, default="0.25,0.5,0.75,1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5")
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--ppc", type=int, default=32)
    parser.add_argument("--Ny", type=int, default=32)
    parser.add_argument("--Ly", type=float, default=8.0)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--n-steps", type=int, default=1200)
    parser.add_argument("--diag-stride", type=int, default=2)
    parser.add_argument("--fit-mode", choices=("fixed", "adaptive"), default="fixed")
    parser.add_argument("--fit-t-min", type=float, default=5.0)
    parser.add_argument("--fit-t-max", type=float, default=30.0)
    parser.add_argument("--noise-floor-start", type=float, default=40.0)
    parser.add_argument("--noise-factor", type=float, default=8.0)
    parser.add_argument("--sigma-y-dx", type=float, default=3.0)
    parser.add_argument("--k-lambdaD", type=float, default=0.4)
    parser.add_argument("--alpha", type=float, default=0.02)
    parser.add_argument("--nboot", type=int, default=1000)
    parser.add_argument("--dx-target", type=float, default=0.2454369260617026)
    parser.add_argument("--dx-assert-rel-tol", type=float, default=0.02)
    parser.add_argument("--x-multiple", type=int, default=8)
    parser.add_argument("--gamma-ref-sigma-max", type=float, default=0.5)
    parser.add_argument("--gamma-relative-tolerance", type=float, default=0.15)
    parser.add_argument("--gamma-ref-theory-tolerance", type=float, default=0.10)
    parser.add_argument("--sem-factor", type=float, default=2.0)
    parser.add_argument("--A-min", type=float, default=0.85)
    parser.add_argument("--cic-power", type=float, default=4.0)
    parser.add_argument("--prediction-relative-tolerance", type=float, default=0.30)
    return parser


if __name__ == "__main__":
    run_E1(build_parser().parse_args())
