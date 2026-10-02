#!/usr/bin/env python3
"""Revised E3: anisotropic sigma against high-ppc reference mode history."""

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

from pic_kde_2d.benchmarks.theory import langmuir_omega_r
from pic_kde_2d.conventions import landau_box_length, marker_grid_shape, thermal_velocity
from pic_kde_2d.core.grid2d import Grid2D
from pic_kde_2d.core.particles2d import init_landau_2d_eigenmode
from pic_kde_2d.core.pic_loop import run_simulation
from pic_kde_2d.methods.sigma_selection import sigma_max_matched
from pic_kde_2d.methods.spectral_W_wrappers import make_spectral_W_gaussian_methods
from pic_kde_2d.validation.landau_fit import fit_gamma_ensemble_fixed_window


def write_csv(path: pathlib.Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()}) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def k_magnitude_lambdaD(kx_mode: int, ky_mode: int, Lx: float, Ly: float) -> float:
    kx = 2.0 * np.pi * kx_mode / Lx
    ky = 2.0 * np.pi * ky_mode / Ly
    return float(np.sqrt(kx**2 + ky**2))


def make_particles(
    Nx: int,
    Ny: int,
    ppc: int,
    Lx: float,
    Ly: float,
    kx_mode: int,
    ky_mode: int,
    alpha: float,
    seed: int,
) -> object:
    Nx_marker, Ny_marker = marker_grid_shape(Nx, Ny, ppc)
    omega_r = langmuir_omega_r(k_magnitude_lambdaD(kx_mode, ky_mode, Lx, Ly), v_thermal=thermal_velocity())
    return init_landau_2d_eigenmode(
        Nx_marker=Nx_marker,
        Ny_marker=Ny_marker,
        Lx=Lx,
        Ly=Ly,
        v_thermal=thermal_velocity(),
        alpha=alpha,
        kx_mode=kx_mode,
        ky_mode=ky_mode,
        omega_r=omega_r,
        rng=np.random.default_rng(seed),
    )


def run_mode_history(
    particles,
    grid: Grid2D,
    dt: float,
    n_steps: int,
    diag_stride: int,
    deposit_fn,
    gather_fn,
    mode: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    history = run_simulation(
        particles,
        grid,
        dt,
        n_steps,
        deposit_fn,
        gather_fn,
        diag_stride=diag_stride,
        tracked_modes=[mode],
        verbose=False,
    )
    times = np.asarray(history.times, dtype=np.float64)
    amp = np.abs(np.asarray(history.mode_amplitudes_Ex[mode], dtype=np.complex128))
    return times, amp


def run_E3(args: argparse.Namespace) -> dict:
    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)
    Lx = landau_box_length(args.kx_mode, k_lambdaD=args.k_lambdaD_x)
    Ly = landau_box_length(args.ky_mode, k_lambdaD=args.k_lambdaD_y)
    grid = Grid2D(Nx=args.Nx, Ny=args.Ny, Lx=Lx, Ly=Ly)
    mode = (args.kx_mode, args.ky_mode)

    sigma_x_cells = sigma_max_matched(args.Nx, args.mx_preserve, args.A_min)
    sigma_y_cells = sigma_max_matched(args.Ny, args.my_preserve, args.A_min)
    sigma_x = sigma_x_cells * grid.dx
    sigma_y = sigma_y_cells * grid.dy
    cases = {
        "anisotropic": (sigma_x, sigma_y),
        "iso_small": (min(sigma_x, sigma_y), min(sigma_x, sigma_y)),
        "iso_large": (max(sigma_x, sigma_y), max(sigma_x, sigma_y)),
    }
    reference_configs = {
        case_name: {
            "method": "spectral_W_gaussian_matched",
            "sigma_x": sx,
            "sigma_x_cells": sx / grid.dx,
            "sigma_y": sy,
            "sigma_y_cells": sy / grid.dy,
            "Nx": args.Nx,
            "Ny": args.Ny,
            "Lx": Lx,
            "Ly": Ly,
            "ppc": args.ppc_ref,
            "dt": args.dt,
            "n_steps": args.n_steps,
            "diag_stride": args.diag_stride,
            "fit_t_min": args.metric_t_min,
            "fit_t_max": args.metric_t_max,
        }
        for case_name, (sx, sy) in cases.items()
    }
    for case_name, config in reference_configs.items():
        print(f"E3 reference configuration case={case_name}: {json.dumps(config, sort_keys=True)}", flush=True)

    rows: list[dict] = []
    case_amp_lists: dict[str, list[np.ndarray]] = {name: [] for name in cases}
    ref_amp_lists: dict[str, list[np.ndarray]] = {name: [] for name in cases}
    times_ref: np.ndarray | None = None

    for seed in range(args.seed_start, args.seed_start + args.seeds):
        for case_name, (sx, sy) in cases.items():
            deposit, gather = make_spectral_W_gaussian_methods(grid, sx, sy)
            ref_particles = make_particles(
                args.Nx,
                args.Ny,
                args.ppc_ref,
                Lx,
                Ly,
                args.kx_mode,
                args.ky_mode,
                args.alpha,
                seed,
            )
            ref_times, amp_ref = run_mode_history(
                ref_particles,
                grid,
                args.dt,
                args.n_steps,
                args.diag_stride,
                deposit,
                gather,
                mode,
            )
            if times_ref is None:
                times_ref = ref_times
            elif not np.allclose(ref_times, times_ref):
                raise RuntimeError("reference diagnostic times do not match")
            ref_amp_lists[case_name].append(amp_ref)

            particles = make_particles(
                args.Nx,
                args.Ny,
                args.ppc_low,
                Lx,
                Ly,
                args.kx_mode,
                args.ky_mode,
                args.alpha,
                seed,
            )
            times, amp = run_mode_history(
                particles,
                grid,
                args.dt,
                args.n_steps,
                args.diag_stride,
                deposit,
                gather,
                mode,
            )
            if not np.allclose(times, times_ref):
                raise RuntimeError("case/reference diagnostic times do not match")
            mask = (times >= args.metric_t_min) & (times <= args.metric_t_max)
            mode_error = float(np.trapezoid((amp[mask] - amp_ref[mask]) ** 2, times[mask]))
            rows.append(
                {
                    "case": case_name,
                    "mode_error": mode_error,
                    "seed": seed,
                    "sigma_x": sx,
                    "sigma_x_cells": sx / grid.dx,
                    "sigma_y": sy,
                    "sigma_y_cells": sy / grid.dy,
                    "ppc_low": args.ppc_low,
                    "ppc_ref": args.ppc_ref,
                    "reference_method": reference_configs[case_name]["method"],
                    "reference_sigma_x": sx,
                    "reference_sigma_x_cells": sx / grid.dx,
                    "reference_sigma_y": sy,
                    "reference_sigma_y_cells": sy / grid.dy,
                    "dt": args.dt,
                    "fit_t_min": args.metric_t_min,
                    "fit_t_max": args.metric_t_max,
                }
            )
            case_amp_lists[case_name].append(amp)
            print(f"E3 revised case={case_name} seed={seed} mode_error={mode_error:.6e}", flush=True)

    if times_ref is None:
        raise RuntimeError("no E3 reference amplitudes collected")

    summary_rows: list[dict] = []
    for case_name, amp_list in case_amp_lists.items():
        sub = [row for row in rows if row["case"] == case_name]
        errors = np.asarray([row["mode_error"] for row in sub], dtype=np.float64)
        ref_gamma = fit_gamma_ensemble_fixed_window(
            ref_amp_lists[case_name],
            times_ref,
            t_fit_start=args.metric_t_min,
            t_fit_stop=args.metric_t_max,
        )["gamma"]
        gamma = fit_gamma_ensemble_fixed_window(
            amp_list,
            times_ref,
            t_fit_start=args.metric_t_min,
            t_fit_stop=args.metric_t_max,
        )["gamma"]
        damping_rel_error = abs(gamma - ref_gamma) / max(abs(ref_gamma), 1.0e-30)
        summary_rows.append(
            {
                "case": case_name,
                "damping_rel_error_vs_ref": damping_rel_error,
                "gamma": gamma,
                "mode_error_mean": float(np.mean(errors)),
                "mode_error_std": float(np.std(errors)),
                "n_seeds": len(sub),
                "ref_gamma": ref_gamma,
                "sigma_x": cases[case_name][0],
                "sigma_x_cells": cases[case_name][0] / grid.dx,
                "sigma_y": cases[case_name][1],
                "sigma_y_cells": cases[case_name][1] / grid.dy,
                "reference_method": reference_configs[case_name]["method"],
                "reference_sigma_x": cases[case_name][0],
                "reference_sigma_x_cells": cases[case_name][0] / grid.dx,
                "reference_sigma_y": cases[case_name][1],
                "reference_sigma_y_cells": cases[case_name][1] / grid.dy,
                "ppc_low": args.ppc_low,
                "ppc_ref": args.ppc_ref,
                "Nx": args.Nx,
                "Ny": args.Ny,
                "dt": args.dt,
                "fit_t_min": args.metric_t_min,
                "fit_t_max": args.metric_t_max,
            }
        )

    by_case = {row["case"]: row for row in summary_rows}
    aniso_error = by_case["anisotropic"]["mode_error_mean"]
    iso_small_error = by_case["iso_small"]["mode_error_mean"]
    iso_large_error = by_case["iso_large"]["mode_error_mean"]
    worst_iso_error = max(iso_small_error, iso_large_error)
    improvement_vs_iso_small = (iso_small_error - aniso_error) / max(iso_small_error, 1.0e-30)
    improvement_vs_iso_large = (iso_large_error - aniso_error) / max(iso_large_error, 1.0e-30)
    improvement_vs_worst = (worst_iso_error - aniso_error) / max(worst_iso_error, 1.0e-30)
    no_overdamp = by_case["anisotropic"]["damping_rel_error_vs_ref"] <= args.damping_tolerance
    passed = bool(improvement_vs_worst >= args.improvement_threshold and no_overdamp)

    suffix = f"_{args.tag}" if args.tag else ""
    raw_path = output_dir / f"E3_revised_raw{suffix}.csv"
    summary_path = output_dir / f"E3_revised_summary{suffix}.csv"
    write_csv(raw_path, rows)
    write_csv(summary_path, summary_rows)
    result = {
        "raw_csv": str(raw_path),
        "summary_csv": str(summary_path),
        "improvement_vs_iso_small": float(improvement_vs_iso_small),
        "improvement_vs_iso_large": float(improvement_vs_iso_large),
        "improvement_vs_worst_isotropic": float(improvement_vs_worst),
        "no_overdamp": bool(no_overdamp),
        "passed": passed,
        "ref_gamma_by_case": {row["case"]: row["ref_gamma"] for row in summary_rows},
        "reference_configs": reference_configs,
        "n_seeds": args.seeds,
    }
    json_path = output_dir / f"E3_revised_result{suffix}.json"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run revised Phase 4 E3.")
    parser.add_argument("--output", type=pathlib.Path, default=PROJECT_ROOT / "results" / "E3_revised")
    parser.add_argument("--tag", type=str, default="")
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--ppc-low", type=int, default=16)
    parser.add_argument("--ppc-ref", type=int, default=256)
    parser.add_argument("--Nx", type=int, default=128)
    parser.add_argument("--Ny", type=int, default=32)
    parser.add_argument("--kx-mode", type=int, default=4)
    parser.add_argument("--ky-mode", type=int, default=1)
    parser.add_argument("--mx-preserve", type=int, default=4)
    parser.add_argument("--my-preserve", type=int, default=6)
    parser.add_argument("--k-lambdaD-x", type=float, default=0.4)
    parser.add_argument("--k-lambdaD-y", type=float, default=0.4)
    parser.add_argument("--A-min", type=float, default=0.85)
    parser.add_argument("--alpha", type=float, default=0.02)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--n-steps", type=int, default=400)
    parser.add_argument("--diag-stride", type=int, default=10)
    parser.add_argument("--metric-t-min", type=float, default=2.0)
    parser.add_argument("--metric-t-max", type=float, default=20.0)
    parser.add_argument("--improvement-threshold", type=float, default=0.15)
    parser.add_argument("--damping-tolerance", type=float, default=0.10)
    return parser


if __name__ == "__main__":
    run_E3(build_parser().parse_args())
