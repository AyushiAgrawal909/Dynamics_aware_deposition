#!/usr/bin/env python3
"""Revised E2: two-stream high-k preservation measured by ensemble growth rate."""

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
from pic_kde_2d.validation.landau_fit import bootstrap_gamma_ci, fit_gamma_ensemble_fixed_window
from pic_kde_2d.validation.transfer_empirical import sigma_max_pure_gaussian


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


def bootstrap_gamma_ratio_ci(
    predicted_amplitudes: list[np.ndarray],
    reference_amplitudes: list[np.ndarray],
    times: np.ndarray,
    args: argparse.Namespace,
) -> dict:
    """Bootstrap the growth-rate ratio using paired seed resampling."""
    if len(predicted_amplitudes) != len(reference_amplitudes):
        raise ValueError("predicted and reference amplitude ensembles must have the same seeds")
    rng = np.random.default_rng(97531)
    ratios: list[float] = []
    n_series = len(reference_amplitudes)
    for _ in range(args.nboot):
        indices = rng.integers(0, n_series, size=n_series)
        pred = [predicted_amplitudes[int(index)] for index in indices]
        ref = [reference_amplitudes[int(index)] for index in indices]
        gamma_pred = fit_gamma_ensemble_fixed_window(
            pred,
            times,
            t_fit_start=args.fit_t_min,
            t_fit_stop=args.fit_t_max,
            amp_floor=1.0e-30,
        )["gamma"]
        gamma_ref = fit_gamma_ensemble_fixed_window(
            ref,
            times,
            t_fit_start=args.fit_t_min,
            t_fit_stop=args.fit_t_max,
            amp_floor=1.0e-30,
        )["gamma"]
        if np.isfinite(gamma_pred) and np.isfinite(gamma_ref) and gamma_ref != 0.0:
            ratios.append(float(gamma_pred / gamma_ref))
    if not ratios:
        return {"ratio_sem": float("nan"), "ratio_ci_lo": float("nan"), "ratio_ci_hi": float("nan")}
    values = np.asarray(ratios, dtype=np.float64)
    return {
        "ratio_sem": float(np.std(values)),
        "ratio_ci_lo": float(np.percentile(values, 2.5)),
        "ratio_ci_hi": float(np.percentile(values, 97.5)),
    }


def matched_gaussian_transfer(sigma_dx: float, mode: int, n_cells: int) -> float:
    """Matched deposit/gather W transfer for one integer Fourier mode."""
    k_cells = 2.0 * np.pi * mode / n_cells
    return float(np.exp(-(k_cells * sigma_dx) ** 2))


def run_one(sigma_dx: float, seed: int, args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray]:
    grid = Grid2D(Nx=args.Nx, Ny=args.Ny, Lx=args.Lx, Ly=args.Ly)
    sigma_x = sigma_dx * grid.dx
    sigma_y = sigma_dx * grid.dy
    deposit, gather = make_spectral_W_gaussian_methods(grid, sigma_x, sigma_y)
    Nx_marker, Ny_marker = marker_grid_shape(args.Nx, args.Ny, args.ppc)
    particles = init_two_stream_2d(
        Nx_marker=Nx_marker,
        Ny_marker=Ny_marker,
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
        verbose=False,
    )
    times = np.asarray(history.times, dtype=np.float64)
    amps = np.abs(np.asarray(history.mode_amplitudes_Ex[(args.kx_mode, 0)], dtype=np.complex128))
    return times, amps


def run_E2(args: argparse.Namespace) -> dict:
    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)
    sigma_grid = parse_float_list(args.sigma_dx)
    seeds = list(range(args.seed_start, args.seed_start + args.seeds))
    sigma_pred = sigma_max_pure_gaussian(args.Nx, args.kx_mode, args.A_min)

    summary_rows: list[dict] = []
    amplitudes_by_sigma: dict[float, list[np.ndarray]] = {}
    times_by_sigma: dict[float, np.ndarray] = {}
    for sigma_dx in sigma_grid:
        amplitude_list: list[np.ndarray] = []
        times_ref: np.ndarray | None = None
        for seed in seeds:
            times, amps = run_one(sigma_dx, seed, args)
            times_ref = times
            amplitude_list.append(amps)
            print(f"E2 revised sigma={sigma_dx:.2f} seed={seed}", flush=True)
        if times_ref is None:
            raise RuntimeError("no E2 amplitudes collected")
        amplitudes_by_sigma[sigma_dx] = amplitude_list
        times_by_sigma[sigma_dx] = times_ref
        fit = fit_gamma_ensemble_fixed_window(
            amplitude_list,
            times_ref,
            t_fit_start=args.fit_t_min,
            t_fit_stop=args.fit_t_max,
            amp_floor=1.0e-30,
        )
        ci = bootstrap_gamma_ci(
            amplitude_list,
            times_ref,
            n_boot=args.nboot,
            seed=24680 + int(round(1000.0 * sigma_dx)),
            fixed_window=True,
            t_fit_start=args.fit_t_min,
            t_fit_stop=args.fit_t_max,
            amp_floor=1.0e-30,
        )
        summary_rows.append(
            {
                "gamma_CI_hi": ci["ci_hi"],
                "gamma_CI_lo": ci["ci_lo"],
                "gamma_linear": fit["gamma"],
                "gamma_sem": ci["gamma_sem"],
                "n_fit_points": fit["n_points"],
                "n_seeds": len(seeds),
                "sigma_dx": sigma_dx,
                "sigma_predicted_dx": sigma_pred,
            }
        )
        print(f"E2 revised ensemble sigma={sigma_dx:.2f}: gamma={fit['gamma']:.6f}", flush=True)

    by_sigma = {row["sigma_dx"]: row for row in summary_rows}
    ref_sigma = min(sigma_grid, key=lambda value: abs(value - args.reference_sigma_dx))
    pred_sigma = min(sigma_grid, key=lambda value: abs(value - sigma_pred))
    gamma_ref = float(by_sigma[ref_sigma]["gamma_linear"])
    gamma_pred = float(by_sigma[pred_sigma]["gamma_linear"])
    ratio = gamma_pred / gamma_ref if gamma_ref != 0.0 else float("nan")
    lower_limit = 1.0 - args.gamma_ratio_tolerance
    upper_limit = 1.0 + args.gamma_ratio_upper_tolerance
    ratios_by_sigma: list[dict] = []
    for sigma_dx in sigma_grid:
        sigma_ratio = float(by_sigma[sigma_dx]["gamma_linear"]) / gamma_ref if gamma_ref != 0.0 else float("nan")
        sigma_ci = bootstrap_gamma_ratio_ci(
            amplitudes_by_sigma[sigma_dx],
            amplitudes_by_sigma[ref_sigma],
            times_by_sigma[sigma_dx],
            args,
        )
        point_passed = bool(np.isfinite(sigma_ratio) and lower_limit <= sigma_ratio <= upper_limit)
        supported_below = bool(np.isfinite(sigma_ci["ratio_ci_hi"]) and sigma_ci["ratio_ci_hi"] < lower_limit)
        transfer = matched_gaussian_transfer(sigma_dx, args.kx_mode, args.Nx)
        ratio_row = {
            "sigma_dx": sigma_dx,
            "growth_ratio_vs_reference": sigma_ratio,
            "growth_ratio_sem": sigma_ci["ratio_sem"],
            "growth_ratio_CI_lo": sigma_ci["ratio_ci_lo"],
            "growth_ratio_CI_hi": sigma_ci["ratio_ci_hi"],
            "matched_W_transfer": transfer,
            "equivalent_A_min": transfer,
            "point_estimate_passed": point_passed,
            "conclusively_below_acceptance": supported_below,
        }
        ratios_by_sigma.append(ratio_row)
        by_sigma[sigma_dx].update(ratio_row)
    ratio_ci = next(row for row in ratios_by_sigma if row["sigma_dx"] == pred_sigma)
    point_estimate_passed = bool(np.isfinite(ratio) and lower_limit <= ratio <= upper_limit)
    conclusively_below_acceptance = bool(
        np.isfinite(ratio_ci["growth_ratio_CI_hi"]) and ratio_ci["growth_ratio_CI_hi"] < lower_limit
    )
    passed = point_estimate_passed

    suffix = f"_{args.tag}" if args.tag else ""
    summary_path = output_dir / f"E2_revised_summary{suffix}.csv"
    write_csv(summary_path, summary_rows)
    usable = [
        row["sigma_dx"]
        for row in ratios_by_sigma
        if row["point_estimate_passed"] and not row["conclusively_below_acceptance"]
    ]
    result = {
        "summary_csv": str(summary_path),
        "n_seeds": len(seeds),
        "predicted_sigma_dx": sigma_pred,
        "nearest_predicted_sigma_dx": pred_sigma,
        "reference_sigma_dx": ref_sigma,
        "gamma_ratio_predicted_over_reference": ratio,
        "gamma_ratio_sem": ratio_ci["growth_ratio_sem"],
        "gamma_ratio_CI_lo": ratio_ci["growth_ratio_CI_lo"],
        "gamma_ratio_CI_hi": ratio_ci["growth_ratio_CI_hi"],
        "pass_range": [lower_limit, upper_limit],
        "point_estimate_passed": point_estimate_passed,
        "conclusively_below_acceptance": conclusively_below_acceptance,
        "ratios_by_sigma": ratios_by_sigma,
        "largest_scanned_usable_sigma_dx": max(usable) if usable else None,
        "passed": passed,
    }
    json_path = output_dir / f"E2_revised_result{suffix}.json"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run revised Phase 4 E2.")
    parser.add_argument("--output", type=pathlib.Path, default=PROJECT_ROOT / "results" / "E2_revised")
    parser.add_argument("--tag", type=str, default="")
    parser.add_argument("--sigma-dx", type=str, default="0.25,0.5,0.68,0.75,1.0,1.25,1.5")
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--ppc", type=int, default=32)
    parser.add_argument("--Nx", type=int, default=128)
    parser.add_argument("--Ny", type=int, default=16)
    parser.add_argument("--Lx", type=float, default=64.0)
    parser.add_argument("--Ly", type=float, default=16.0)
    parser.add_argument("--kx-mode", type=int, default=12)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--n-steps", type=int, default=300)
    parser.add_argument("--diag-stride", type=int, default=2)
    parser.add_argument("--fit-t-min", type=float, default=0.5)
    parser.add_argument("--fit-t-max", type=float, default=4.0)
    parser.add_argument("--v-beam", type=float, default=0.8)
    parser.add_argument("--v-thermal", type=float, default=0.05)
    parser.add_argument("--alpha", type=float, default=0.001)
    parser.add_argument("--A-min", type=float, default=0.85)
    parser.add_argument("--reference-sigma-dx", type=float, default=0.25)
    parser.add_argument("--gamma-ratio-tolerance", type=float, default=0.15)
    parser.add_argument("--gamma-ratio-upper-tolerance", type=float, default=0.05)
    parser.add_argument("--nboot", type=int, default=1000)
    return parser


if __name__ == "__main__":
    run_E2(build_parser().parse_args())
