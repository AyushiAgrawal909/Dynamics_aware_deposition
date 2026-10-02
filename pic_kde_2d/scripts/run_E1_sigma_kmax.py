#!/usr/bin/env python3
"""
scripts/run_E1_sigma_kmax.py
============================

Experiment E1: validate the sigma_x * k_mode scaling rule for Landau damping
using the fixed-sigma spectral-W method.

This implementation is meluxina-friendly:
- no pandas dependency
- optional seed slicing for batch splitting
- CSV + JSON outputs only
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import sys
from typing import Iterable

import numpy as np

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pic_kde_2d.benchmarks.landau_2d import fit_mode_gamma
from pic_kde_2d.benchmarks.theory import landau_gamma_theory
from pic_kde_2d.conventions import landau_box_length, marker_grid_shape, thermal_velocity
from pic_kde_2d.core.grid2d import Grid2D
from pic_kde_2d.core.particles2d import init_landau_2d
from pic_kde_2d.core.pic_loop import run_simulation
from pic_kde_2d.methods.spectral_W_wrappers import make_spectral_W_gaussian_methods


def parse_int_list(text: str) -> list[int]:
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def parse_float_list(text: str) -> list[float]:
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def bootstrap_mean_ci(values: Iterable[float], nboot: int = 1000, alpha: float = 0.05, seed: int = 12345) -> tuple[float, float, float]:
    samples = np.asarray(list(values), dtype=np.float64)
    samples = samples[np.isfinite(samples)]
    if samples.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    boots = np.empty(nboot, dtype=np.float64)
    for idx in range(nboot):
        boots[idx] = np.mean(rng.choice(samples, size=samples.size, replace=True))
    lo, hi = np.quantile(boots, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(np.mean(samples)), float(lo), float(hi)


def fit_gamma_series(times: np.ndarray, amplitudes: np.ndarray, t_min: float, t_max: float) -> float:
    """Fit gamma from one amplitude series."""
    return fit_mode_gamma(times, amplitudes, t_min, t_max, amp_floor=1.0e-12)


def bootstrap_ensemble_gamma_ci(
    times: np.ndarray,
    amplitude_series: list[np.ndarray],
    t_min: float,
    t_max: float,
    nboot: int,
    seed: int,
) -> tuple[float, float, float]:
    """Fit gamma to the ensemble-mean amplitude, bootstrapping over seeds."""
    finite_series = [np.asarray(series, dtype=np.float64) for series in amplitude_series if np.all(np.isfinite(series))]
    if not finite_series:
        return float("nan"), float("nan"), float("nan")
    stack = np.vstack(finite_series)
    gamma = fit_gamma_series(times, np.mean(stack, axis=0), t_min, t_max)
    rng = np.random.default_rng(seed)
    boots = np.empty(nboot, dtype=np.float64)
    for idx in range(nboot):
        sample_idx = rng.integers(0, stack.shape[0], size=stack.shape[0])
        boots[idx] = fit_gamma_series(times, np.mean(stack[sample_idx], axis=0), t_min, t_max)
    boots = boots[np.isfinite(boots)]
    if boots.size == 0:
        return gamma, float("nan"), float("nan")
    lo, hi = np.quantile(boots, [0.025, 0.975])
    return float(gamma), float(lo), float(hi)


def write_csv(path: pathlib.Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()}) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def append_csv_row(path: pathlib.Path, row: dict, fieldnames: list[str]) -> None:
    """Append one CSV row immediately so long jobs preserve partial results."""
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)
        handle.flush()


def as_float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def as_int(value) -> int:
    return int(float(value))


def sigma_max_predicted_dx(nx: int, k_mode: int, a_min: float) -> float:
    """Matched deposit+gather Gaussian preservation bound in cell units."""
    return float(np.sqrt(-np.log(a_min)) * nx / (2.0 * np.pi * k_mode))


def build_E1_criterion_tables(
    rows: list[dict],
    k_modes: list[int],
    sigma_dx_list: list[float],
    nboot: int,
    gamma_ref_sigma_max: float,
    gamma_relative_tolerance: float,
    a_min: float,
    prediction_relative_tolerance: float,
) -> tuple[list[dict], list[dict], dict]:
    """Build E1 summaries from existing gamma samples without rerunning PIC."""
    summary_rows: list[dict] = []
    sigma_opt_rows: list[dict] = []

    for k_mode in k_modes:
        mode_rows = [row for row in rows if as_int(row["k_mode"]) == k_mode]
        nx_values = [as_int(row["Nx"]) for row in mode_rows if np.isfinite(as_float(row.get("Nx")))]
        nx = int(nx_values[0]) if nx_values else 0
        gamma_theory_values = [as_float(row.get("gamma_theory")) for row in mode_rows]
        gamma_theory_values = [value for value in gamma_theory_values if np.isfinite(value)]
        gamma_theory = float(gamma_theory_values[0]) if gamma_theory_values else float("nan")

        ref_values = [
            as_float(row["gamma_measured"])
            for row in mode_rows
            if as_float(row["sigma_dx"]) <= gamma_ref_sigma_max and np.isfinite(as_float(row["gamma_measured"]))
        ]
        gamma_ref = float(np.mean(ref_values)) if ref_values else float("nan")

        passing_sigma: list[float] = []
        for sigma_dx in sigma_dx_list:
            gamma_values = [
                as_float(row["gamma_measured"])
                for row in mode_rows
                if as_float(row["sigma_dx"]) == sigma_dx
            ]
            mean, ci_lo, ci_hi = bootstrap_mean_ci(gamma_values, nboot=nboot, alpha=0.05, seed=12345 + k_mode)
            rel_to_ref = abs(mean - gamma_ref) / max(abs(gamma_ref), 1.0e-30) if np.isfinite(mean) and np.isfinite(gamma_ref) else float("nan")
            preserves_reference = bool(np.isfinite(rel_to_ref) and rel_to_ref < gamma_relative_tolerance)
            if preserves_reference:
                passing_sigma.append(sigma_dx)
            summary_rows.append(
                {
                    "k_mode": k_mode,
                    "sigma_dx": sigma_dx,
                    "gamma_mean": mean,
                    "gamma_CI_lo": ci_lo,
                    "gamma_CI_hi": ci_hi,
                    "gamma_ref": gamma_ref,
                    "gamma_rel_error_to_ref": rel_to_ref,
                    "preserves_reference": preserves_reference,
                    "gamma_theory": gamma_theory,
                }
            )

        sigma_opt = max(passing_sigma) if passing_sigma else float("nan")
        sigma_pred = sigma_max_predicted_dx(nx=nx, k_mode=k_mode, a_min=a_min) if nx > 0 else float("nan")
        pred_rel_error = abs(sigma_opt - sigma_pred) / sigma_pred if np.isfinite(sigma_opt) and np.isfinite(sigma_pred) and sigma_pred != 0 else float("nan")
        sigma_opt_rows.append(
            {
                "k_mode": k_mode,
                "Nx": nx,
                "gamma_ref": gamma_ref,
                "sigma_opt_dx": sigma_opt,
                "sigma_max_predicted_dx": sigma_pred,
                "sigma_opt_rel_error_to_prediction": pred_rel_error,
                "prediction_pass": bool(np.isfinite(pred_rel_error) and pred_rel_error < prediction_relative_tolerance),
                "sigma_opt_times_kmode": sigma_opt * k_mode if np.isfinite(sigma_opt) else float("nan"),
            }
        )

    prediction_pass_values = [row["prediction_pass"] for row in sigma_opt_rows]
    all_prediction_pass = bool(prediction_pass_values and all(prediction_pass_values))
    products = np.asarray(
        [row["sigma_opt_times_kmode"] for row in sigma_opt_rows if np.isfinite(row["sigma_opt_times_kmode"])],
        dtype=np.float64,
    )
    cv = float(np.std(products) / max(np.mean(products), 1.0e-30)) if products.size >= 2 else float("nan")
    result = {
        "cv_sigma_opt_times_kmode": cv,
        "passed": all_prediction_pass,
        "pass_rule": "abs(sigma_opt - sigma_max_predicted) / sigma_max_predicted < prediction_relative_tolerance for all k_modes",
        "gamma_reference_rule": f"mean finite gamma where sigma_dx <= {gamma_ref_sigma_max}",
        "gamma_relative_tolerance": gamma_relative_tolerance,
        "a_min": a_min,
        "prediction_relative_tolerance": prediction_relative_tolerance,
    }
    return summary_rows, sigma_opt_rows, result


def build_E1_ensemble_criterion_tables(
    ensemble_rows: list[dict],
    k_modes: list[int],
    sigma_dx_list: list[float],
    gamma_ref_sigma_max: float,
    gamma_relative_tolerance: float,
    a_min: float,
    prediction_relative_tolerance: float,
) -> tuple[list[dict], list[dict], dict]:
    """Build E1 summaries from ensemble-mean gamma fits."""
    summary_rows: list[dict] = []
    sigma_opt_rows: list[dict] = []

    for k_mode in k_modes:
        mode_rows = [row for row in ensemble_rows if as_int(row["k_mode"]) == k_mode]
        nx_values = [as_int(row["Nx"]) for row in mode_rows if np.isfinite(as_float(row.get("Nx")))]
        nx = int(nx_values[0]) if nx_values else 0
        gamma_theory_values = [as_float(row.get("gamma_theory")) for row in mode_rows if np.isfinite(as_float(row.get("gamma_theory")))]
        gamma_theory = float(gamma_theory_values[0]) if gamma_theory_values else float("nan")
        ref_values = [
            as_float(row["gamma_ensemble"])
            for row in mode_rows
            if as_float(row["sigma_dx"]) <= gamma_ref_sigma_max and np.isfinite(as_float(row["gamma_ensemble"]))
        ]
        gamma_ref = float(np.mean(ref_values)) if ref_values else float("nan")

        passing_sigma: list[float] = []
        for sigma_dx in sigma_dx_list:
            matches = [row for row in mode_rows if as_float(row["sigma_dx"]) == sigma_dx]
            row = matches[0] if matches else {}
            gamma = as_float(row.get("gamma_ensemble"))
            rel_to_ref = abs(gamma - gamma_ref) / max(abs(gamma_ref), 1.0e-30) if np.isfinite(gamma) and np.isfinite(gamma_ref) else float("nan")
            preserves_reference = bool(np.isfinite(rel_to_ref) and rel_to_ref < gamma_relative_tolerance)
            if preserves_reference:
                passing_sigma.append(sigma_dx)
            summary_rows.append(
                {
                    "k_mode": k_mode,
                    "sigma_dx": sigma_dx,
                    "gamma_mean": gamma,
                    "gamma_CI_lo": as_float(row.get("gamma_ensemble_CI_lo")),
                    "gamma_CI_hi": as_float(row.get("gamma_ensemble_CI_hi")),
                    "gamma_ref": gamma_ref,
                    "gamma_rel_error_to_ref": rel_to_ref,
                    "preserves_reference": preserves_reference,
                    "gamma_theory": gamma_theory,
                    "n_seeds": as_int(row.get("n_seeds", 0)) if row else 0,
                }
            )

        sigma_opt = max(passing_sigma) if passing_sigma else float("nan")
        sigma_pred = sigma_max_predicted_dx(nx=nx, k_mode=k_mode, a_min=a_min) if nx > 0 else float("nan")
        pred_rel_error = abs(sigma_opt - sigma_pred) / sigma_pred if np.isfinite(sigma_opt) and np.isfinite(sigma_pred) and sigma_pred != 0 else float("nan")
        sigma_opt_rows.append(
            {
                "k_mode": k_mode,
                "Nx": nx,
                "gamma_ref": gamma_ref,
                "sigma_opt_dx": sigma_opt,
                "sigma_max_predicted_dx": sigma_pred,
                "sigma_opt_rel_error_to_prediction": pred_rel_error,
                "prediction_pass": bool(np.isfinite(pred_rel_error) and pred_rel_error < prediction_relative_tolerance),
                "sigma_opt_times_kmode": sigma_opt * k_mode if np.isfinite(sigma_opt) else float("nan"),
            }
        )

    prediction_pass_values = [row["prediction_pass"] for row in sigma_opt_rows]
    all_prediction_pass = bool(prediction_pass_values and all(prediction_pass_values))
    products = np.asarray(
        [row["sigma_opt_times_kmode"] for row in sigma_opt_rows if np.isfinite(row["sigma_opt_times_kmode"])],
        dtype=np.float64,
    )
    cv = float(np.std(products) / max(np.mean(products), 1.0e-30)) if products.size >= 2 else float("nan")
    result = {
        "cv_sigma_opt_times_kmode": cv,
        "passed": all_prediction_pass,
        "pass_rule": "ensemble gamma: abs(sigma_opt - sigma_max_predicted) / sigma_max_predicted < prediction_relative_tolerance for all k_modes",
        "gamma_reference_rule": f"ensemble mean gamma where sigma_dx <= {gamma_ref_sigma_max}",
        "gamma_relative_tolerance": gamma_relative_tolerance,
        "a_min": a_min,
        "prediction_relative_tolerance": prediction_relative_tolerance,
    }
    return summary_rows, sigma_opt_rows, result


def round_to_multiple(value: int, multiple: int) -> int:
    return max(multiple, multiple * ((value + multiple // 2) // multiple))


def compute_mode_grid(k_mode: int, k_lambdaD: float, target_dx: float, Ny: int, Ly: float, x_multiple: int) -> tuple[int, int, float, float]:
    Lx = landau_box_length(k_mode, k_lambdaD=k_lambdaD)
    Nx = round_to_multiple(int(np.round(Lx / target_dx)), x_multiple)
    return Nx, Ny, Lx, Ly


def measure_landau_mode(
    Nx: int,
    Ny: int,
    Lx: float,
    Ly: float,
    kx_mode: int,
    sigma_x: float,
    sigma_y: float,
    ppc: int,
    dt: float,
    n_steps: int,
    diag_stride: int,
    seed: int,
    fit_t_min: float,
    fit_t_max: float,
    late_floor_t_min: float,
    late_floor_quantile: float,
    v_thermal: float = 1.0,
    alpha: float = 0.02,
) -> tuple[float, np.ndarray, np.ndarray]:
    grid = Grid2D(Nx=Nx, Ny=Ny, Lx=Lx, Ly=Ly)
    deposit, gather = make_spectral_W_gaussian_methods(grid, sigma_x, sigma_y)
    Nx_marker, Ny_marker = marker_grid_shape(Nx, Ny, ppc)
    particles = init_landau_2d(
        Nx_marker=Nx_marker,
        Ny_marker=Ny_marker,
        Lx=Lx,
        Ly=Ly,
        v_thermal=v_thermal,
        alpha=alpha,
        kx_mode=kx_mode,
        ky_mode=0,
        rng=np.random.default_rng(seed),
    )
    history = run_simulation(
        particles,
        grid,
        dt,
        n_steps,
        deposit,
        gather,
        diag_stride=diag_stride,
        tracked_modes=[(kx_mode, 0)],
        verbose=False,
    )
    amps = np.abs(np.asarray(history.mode_amplitudes_Ex[(kx_mode, 0)], dtype=np.complex128))
    times = np.asarray(history.times, dtype=np.float64)
    _ = late_floor_t_min, late_floor_quantile
    return fit_gamma_series(times, amps, fit_t_min, fit_t_max), times, amps


def run_E1(args: argparse.Namespace) -> dict:
    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    k_modes = parse_int_list(args.k_modes)
    sigma_dx_list = parse_float_list(args.sigma_dx)
    seed_values = list(range(args.seed_start, args.seed_start + args.seed_count))

    run_name = f"E1_sigma_kmax{('_' + args.tag) if args.tag else ''}"
    raw_path = output_dir / f"{run_name}.csv"
    raw_fieldnames = [
        "Lx",
        "Ly",
        "Nx",
        "Ny",
        "dx",
        "gamma_measured",
        "gamma_theory",
        "k_mode",
        "ppc",
        "seed",
        "sigma_dx",
        "sigma_x",
        "sigma_y",
    ]
    if raw_path.exists():
        raw_path.unlink()

    rows: list[dict] = []
    series_by_group: dict[tuple[int, float], list[np.ndarray]] = {}
    times_by_group: dict[tuple[int, float], np.ndarray] = {}
    gamma_theory = landau_gamma_theory(k_lambdaD=args.k_lambdaD, v_thermal=thermal_velocity())
    target_dx = args.target_dx

    for k_mode in k_modes:
        Nx, Ny, Lx, Ly = compute_mode_grid(
            k_mode=k_mode,
            k_lambdaD=args.k_lambdaD,
            target_dx=target_dx,
            Ny=args.Ny,
            Ly=args.Ly,
            x_multiple=args.x_multiple,
        )
        grid = Grid2D(Nx=Nx, Ny=Ny, Lx=Lx, Ly=Ly)
        for sigma_dx in sigma_dx_list:
            sigma_x = sigma_dx * grid.dx
            sigma_y = args.sigma_y_dx * grid.dy
            group_key = (k_mode, sigma_dx)
            for seed in seed_values:
                gamma_measured, times, amps = measure_landau_mode(
                    Nx=Nx,
                    Ny=Ny,
                    Lx=Lx,
                    Ly=Ly,
                    kx_mode=k_mode,
                    sigma_x=sigma_x,
                    sigma_y=sigma_y,
                    ppc=args.ppc,
                    dt=args.dt,
                    n_steps=args.n_steps,
                    diag_stride=args.diag_stride,
                    seed=seed,
                    fit_t_min=args.fit_t_min,
                    fit_t_max=args.fit_t_max,
                    late_floor_t_min=args.late_floor_t_min,
                    late_floor_quantile=args.late_floor_quantile,
                    v_thermal=thermal_velocity(),
                    alpha=args.alpha,
                )
                times_by_group[group_key] = times
                series_by_group.setdefault(group_key, []).append(amps)
                row = {
                    "k_mode": k_mode,
                    "Lx": Lx,
                    "Ly": Ly,
                    "Nx": Nx,
                    "Ny": Ny,
                    "dx": grid.dx,
                    "ppc": args.ppc,
                    "sigma_dx": sigma_dx,
                    "sigma_x": sigma_x,
                    "sigma_y": sigma_y,
                    "seed": seed,
                    "gamma_measured": gamma_measured,
                    "gamma_theory": gamma_theory,
                }
                rows.append(row)
                append_csv_row(raw_path, row, raw_fieldnames)
                print(
                    f"E1 k_mode={k_mode:>2d} sigma={sigma_dx:>4.2f}dx seed={seed:>3d} "
                    f"Nx={Nx:>3d} dx={grid.dx: .4f} gamma={gamma_measured: .5f}",
                    flush=True,
                )

    ensemble_rows: list[dict] = []
    for k_mode in k_modes:
        Nx, Ny, Lx, Ly = compute_mode_grid(
            k_mode=k_mode,
            k_lambdaD=args.k_lambdaD,
            target_dx=target_dx,
            Ny=args.Ny,
            Ly=args.Ly,
            x_multiple=args.x_multiple,
        )
        grid = Grid2D(Nx=Nx, Ny=Ny, Lx=Lx, Ly=Ly)
        for sigma_dx in sigma_dx_list:
            group_key = (k_mode, sigma_dx)
            series = series_by_group.get(group_key, [])
            times = times_by_group.get(group_key)
            if times is None or not series:
                gamma_ensemble = gamma_ci_lo = gamma_ci_hi = float("nan")
            else:
                gamma_ensemble, gamma_ci_lo, gamma_ci_hi = bootstrap_ensemble_gamma_ci(
                    times=times,
                    amplitude_series=series,
                    t_min=args.fit_t_min,
                    t_max=args.fit_t_max,
                    nboot=args.nboot,
                    seed=54321 + 97 * k_mode + int(round(1000.0 * sigma_dx)),
                )
            ensemble_rows.append(
                {
                    "k_mode": k_mode,
                    "Lx": Lx,
                    "Ly": Ly,
                    "Nx": Nx,
                    "Ny": Ny,
                    "dx": grid.dx,
                    "ppc": args.ppc,
                    "sigma_dx": sigma_dx,
                    "sigma_x": sigma_dx * grid.dx,
                    "sigma_y": args.sigma_y_dx * grid.dy,
                    "n_seeds": len(series),
                    "gamma_ensemble": gamma_ensemble,
                    "gamma_ensemble_CI_lo": gamma_ci_lo,
                    "gamma_ensemble_CI_hi": gamma_ci_hi,
                    "gamma_theory": gamma_theory,
                }
            )

    # Rewrite the raw CSV at the end to keep column ordering consistent even if
    # a future row schema changes. The incrementally written file above is for
    # crash/cancellation resilience during long k=8 runs.
    write_csv(raw_path, rows)
    ensemble_raw_path = output_dir / f"E1_ensemble_gamma{('_' + args.tag) if args.tag else ''}.csv"
    write_csv(ensemble_raw_path, ensemble_rows)

    summary_rows, sigma_opt_rows, criterion_result = build_E1_criterion_tables(
        rows=rows,
        k_modes=k_modes,
        sigma_dx_list=sigma_dx_list,
        nboot=args.nboot,
        gamma_ref_sigma_max=args.gamma_ref_sigma_max,
        gamma_relative_tolerance=args.gamma_relative_tolerance,
        a_min=args.a_min,
        prediction_relative_tolerance=args.prediction_relative_tolerance,
    )

    summary_path = output_dir / f"E1_summary{('_' + args.tag) if args.tag else ''}.csv"
    sigma_opt_path = output_dir / f"E1_sigma_opt{('_' + args.tag) if args.tag else ''}.csv"
    write_csv(summary_path, summary_rows)
    write_csv(sigma_opt_path, sigma_opt_rows)

    ensemble_summary_rows, ensemble_sigma_opt_rows, ensemble_criterion_result = build_E1_ensemble_criterion_tables(
        ensemble_rows=ensemble_rows,
        k_modes=k_modes,
        sigma_dx_list=sigma_dx_list,
        gamma_ref_sigma_max=args.gamma_ref_sigma_max,
        gamma_relative_tolerance=args.gamma_relative_tolerance,
        a_min=args.a_min,
        prediction_relative_tolerance=args.prediction_relative_tolerance,
    )
    ensemble_summary_path = output_dir / f"E1_ensemble_summary{('_' + args.tag) if args.tag else ''}.csv"
    ensemble_sigma_opt_path = output_dir / f"E1_ensemble_sigma_opt{('_' + args.tag) if args.tag else ''}.csv"
    write_csv(ensemble_summary_path, ensemble_summary_rows)
    write_csv(ensemble_sigma_opt_path, ensemble_sigma_opt_rows)

    result = {
        "raw_csv": str(raw_path),
        "ensemble_raw_csv": str(ensemble_raw_path),
        "summary_csv": str(summary_path),
        "sigma_opt_csv": str(sigma_opt_path),
        "ensemble_summary_csv": str(ensemble_summary_path),
        "ensemble_sigma_opt_csv": str(ensemble_sigma_opt_path),
        "ensemble_result": ensemble_criterion_result,
        "n_rows": len(rows),
        "k_modes": k_modes,
        "sigma_dx": sigma_dx_list,
        "seed_start": args.seed_start,
        "seed_count": args.seed_count,
    }
    result.update(criterion_result)

    json_path = output_dir / f"E1_result{('_' + args.tag) if args.tag else ''}.json"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)

    print("\n=== E1 SUMMARY ===")
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Phase 4 experiment E1.")
    parser.add_argument("--output", type=pathlib.Path, default=PROJECT_ROOT / "results" / "E1")
    parser.add_argument("--tag", type=str, default="")
    parser.add_argument("--k-modes", type=str, default="1,2,4,8")
    parser.add_argument("--sigma-dx", type=str, default="0.25,0.5,0.75,1.0,1.5,2.0,3.0,4.0")
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--seed-count", type=int, default=30)
    parser.add_argument("--ppc", type=int, default=32)
    parser.add_argument("--Ny", type=int, default=8)
    parser.add_argument("--Ly", type=float, default=8.0)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--n-steps", type=int, default=600)
    parser.add_argument("--diag-stride", type=int, default=2)
    parser.add_argument("--fit-t-min", type=float, default=5.0)
    parser.add_argument("--fit-t-max", type=float, default=30.0)
    parser.add_argument("--late-floor-t-min", type=float, default=20.0)
    parser.add_argument("--late-floor-quantile", type=float, default=0.01)
    parser.add_argument("--sigma-y-dx", type=float, default=3.0)
    parser.add_argument("--k-lambdaD", type=float, default=0.4)
    parser.add_argument("--alpha", type=float, default=0.02)
    parser.add_argument("--nboot", type=int, default=1000)
    parser.add_argument("--target-dx", type=float, default=float(5.0 * np.pi / 64.0))
    parser.add_argument("--x-multiple", type=int, default=8)
    parser.add_argument("--gamma-ref-sigma-max", type=float, default=0.5)
    parser.add_argument("--gamma-relative-tolerance", type=float, default=0.15)
    parser.add_argument("--a-min", type=float, default=0.85)
    parser.add_argument("--prediction-relative-tolerance", type=float, default=0.30)
    return parser


if __name__ == "__main__":
    run_E1(build_parser().parse_args())
