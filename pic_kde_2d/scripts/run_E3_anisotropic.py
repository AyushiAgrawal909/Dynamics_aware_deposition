#!/usr/bin/env python3
"""
scripts/run_E3_anisotropic.py
=============================

Experiment E3: anisotropic versus isotropic sigma on a 2D Langmuir mode k=(4,1).
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

from pic_kde_2d.conventions import landau_box_length, marker_grid_shape, thermal_velocity
from pic_kde_2d.core.grid2d import Grid2D
from pic_kde_2d.core.particles2d import init_landau_2d
from pic_kde_2d.core.pic_loop import pic_step
from pic_kde_2d.methods.cic_wrappers import cic_deposit_method, cic_gather_method
from pic_kde_2d.methods.sigma_selection import sigma_max_matched
from pic_kde_2d.methods.spectral_W_wrappers import make_spectral_W_gaussian_methods


def write_csv(path: pathlib.Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()}) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def run_with_snapshots(
    particles,
    grid: Grid2D,
    dt: float,
    n_steps: int,
    deposit_fn,
    gather_fn,
    diag_stride: int,
) -> list[dict]:
    snapshots: list[dict] = []
    for step in range(n_steps + 1):
        if step % diag_stride == 0:
            rho = deposit_fn(particles, grid)
            Ex, Ey = grid.poisson_solve(rho)
            snapshots.append(
                {
                    "step": step,
                    "time": step * dt,
                    "rho": rho.copy(),
                    "Ex": Ex.copy(),
                    "Ey": Ey.copy(),
                    "field_energy": grid.field_energy(Ex, Ey),
                }
            )
        if step == n_steps:
            break
        pic_step(particles, grid, dt, deposit_fn, gather_fn)
    return snapshots


def compare_case_to_reference(
    Nx: int,
    Ny: int,
    Lx: float,
    Ly: float,
    kx_mode: int,
    ky_mode: int,
    sigma_x: float,
    sigma_y: float,
    ppc_low: int,
    ppc_ref: int,
    dt: float,
    n_steps: int,
    diag_stride: int,
    seed: int,
    alpha: float,
) -> dict:
    grid = Grid2D(Nx=Nx, Ny=Ny, Lx=Lx, Ly=Ly)
    deposit, gather = make_spectral_W_gaussian_methods(grid, sigma_x, sigma_y)

    Nx_low, Ny_low = marker_grid_shape(Nx, Ny, ppc_low)
    low_particles = init_landau_2d(
        Nx_marker=Nx_low,
        Ny_marker=Ny_low,
        Lx=Lx,
        Ly=Ly,
        v_thermal=thermal_velocity(),
        alpha=alpha,
        kx_mode=kx_mode,
        ky_mode=ky_mode,
        rng=np.random.default_rng(seed),
    )

    Nx_ref, Ny_ref = marker_grid_shape(Nx, Ny, ppc_ref)
    ref_particles = init_landau_2d(
        Nx_marker=Nx_ref,
        Ny_marker=Ny_ref,
        Lx=Lx,
        Ly=Ly,
        v_thermal=thermal_velocity(),
        alpha=alpha,
        kx_mode=kx_mode,
        ky_mode=ky_mode,
        rng=np.random.default_rng(seed),
    )

    case_snaps = run_with_snapshots(low_particles, grid, dt, n_steps, deposit, gather, diag_stride)
    ref_snaps = run_with_snapshots(ref_particles, grid, dt, n_steps, cic_deposit_method, cic_gather_method, diag_stride)

    density_mse = []
    field_mse = []
    tracked_mode_mse = []
    for case_snap, ref_snap in zip(case_snaps, ref_snaps):
        density_mse.append(float(np.mean((case_snap["rho"] - ref_snap["rho"]) ** 2)))
        field_mse.append(
            float(
                0.5
                * (
                    np.mean((case_snap["Ex"] - ref_snap["Ex"]) ** 2)
                    + np.mean((case_snap["Ey"] - ref_snap["Ey"]) ** 2)
                )
            )
        )
        case_mode = grid.mode_amplitude(case_snap["Ex"], kx_mode, ky_mode)
        ref_mode = grid.mode_amplitude(ref_snap["Ex"], kx_mode, ky_mode)
        tracked_mode_mse.append(float(np.abs(case_mode - ref_mode) ** 2))

    return {
        "density_mse_mean": float(np.mean(density_mse)),
        "field_mse_mean": float(np.mean(field_mse)),
        "tracked_mode_mse_mean": float(np.mean(tracked_mode_mse)),
        "field_energy_final": float(case_snaps[-1]["field_energy"]),
        "reference_field_energy_final": float(ref_snaps[-1]["field_energy"]),
    }


def run_E3(args: argparse.Namespace) -> dict:
    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    Lx = landau_box_length(args.kx_mode, k_lambdaD=args.k_lambdaD)
    Ly = landau_box_length(args.ky_mode, k_lambdaD=args.k_lambdaD)
    grid = Grid2D(Nx=args.Nx, Ny=args.Ny, Lx=Lx, Ly=Ly)

    sigma_x_large = sigma_max_matched(args.Nx, args.mx_preserve, args.A_min) * grid.dx
    sigma_y_small = sigma_max_matched(args.Ny, args.my_preserve, args.A_min) * grid.dy

    cases = {
        "anisotropic": (sigma_x_large, sigma_y_small),
        "iso_smallsigma": (min(sigma_x_large, sigma_y_small), min(sigma_x_large, sigma_y_small)),
        "iso_largesigma": (max(sigma_x_large, sigma_y_small), max(sigma_x_large, sigma_y_small)),
    }

    rows: list[dict] = []
    seed_values = list(range(args.seed_start, args.seed_start + args.seed_count))
    for case_name, (sigma_x, sigma_y) in cases.items():
        for seed in seed_values:
            metrics = compare_case_to_reference(
                Nx=args.Nx,
                Ny=args.Ny,
                Lx=Lx,
                Ly=Ly,
                kx_mode=args.kx_mode,
                ky_mode=args.ky_mode,
                sigma_x=sigma_x,
                sigma_y=sigma_y,
                ppc_low=args.ppc_low,
                ppc_ref=args.ppc_ref,
                dt=args.dt,
                n_steps=args.n_steps,
                diag_stride=args.diag_stride,
                seed=seed,
                alpha=args.alpha,
            )
            row = {
                "case": case_name,
                "sigma_x": sigma_x,
                "sigma_y": sigma_y,
                "seed": seed,
                **metrics,
            }
            rows.append(row)
            print(
                f"E3 case={case_name:<15} seed={seed:>3d} "
                f"tracked_mode_mse={metrics['tracked_mode_mse_mean']:.5e} "
                f"field_mse={metrics['field_mse_mean']:.5e}",
                flush=True,
            )

    run_name = f"E3_anisotropic{('_' + args.tag) if args.tag else ''}"
    raw_path = output_dir / f"{run_name}.csv"
    write_csv(raw_path, rows)

    summary_rows: list[dict] = []
    for case_name in cases:
        sub = [row for row in rows if row["case"] == case_name]
        density_vals = np.asarray([row["density_mse_mean"] for row in sub], dtype=np.float64)
        field_vals = np.asarray([row["field_mse_mean"] for row in sub], dtype=np.float64)
        tracked_vals = np.asarray([row["tracked_mode_mse_mean"] for row in sub], dtype=np.float64)
        floor_vals = np.asarray([row["field_energy_final"] for row in sub], dtype=np.float64)
        summary_rows.append(
            {
                "case": case_name,
                "sigma_x": cases[case_name][0],
                "sigma_y": cases[case_name][1],
                "density_mse_mean": float(np.mean(density_vals)),
                "density_mse_std": float(np.std(density_vals)),
                "field_mse_mean": float(np.mean(field_vals)),
                "field_mse_std": float(np.std(field_vals)),
                "tracked_mode_mse_mean": float(np.mean(tracked_vals)),
                "tracked_mode_mse_std": float(np.std(tracked_vals)),
                "field_energy_final_mean": float(np.mean(floor_vals)),
                "field_energy_final_std": float(np.std(floor_vals)),
            }
        )

    summary_path = output_dir / f"E3_summary{('_' + args.tag) if args.tag else ''}.csv"
    write_csv(summary_path, summary_rows)

    by_case = {row["case"]: row for row in summary_rows}
    aniso_metric = by_case["anisotropic"]["tracked_mode_mse_mean"]
    best_iso_metric = min(by_case["iso_smallsigma"]["tracked_mode_mse_mean"], by_case["iso_largesigma"]["tracked_mode_mse_mean"])
    worst_iso_metric = max(by_case["iso_smallsigma"]["tracked_mode_mse_mean"], by_case["iso_largesigma"]["tracked_mode_mse_mean"])
    improvement_vs_best = float((best_iso_metric - aniso_metric) / max(best_iso_metric, 1.0e-30))
    improvement_vs_worst = float((worst_iso_metric - aniso_metric) / max(worst_iso_metric, 1.0e-30))

    result = {
        "raw_csv": str(raw_path),
        "summary_csv": str(summary_path),
        "n_rows": len(rows),
        "improvement_vs_best_isotropic": improvement_vs_best,
        "improvement_vs_worst_isotropic": improvement_vs_worst,
        "passed": bool(improvement_vs_best >= 0.15),
        "seed_start": args.seed_start,
        "seed_count": args.seed_count,
    }

    json_path = output_dir / f"E3_result{('_' + args.tag) if args.tag else ''}.json"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)

    print("\n=== E3 SUMMARY ===")
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Phase 4 experiment E3.")
    parser.add_argument("--output", type=pathlib.Path, default=PROJECT_ROOT / "results" / "E3")
    parser.add_argument("--tag", type=str, default="")
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--seed-count", type=int, default=20)
    parser.add_argument("--ppc-low", type=int, default=16)
    parser.add_argument("--ppc-ref", type=int, default=128)
    parser.add_argument("--Nx", type=int, default=128)
    parser.add_argument("--Ny", type=int, default=32)
    parser.add_argument("--kx-mode", type=int, default=4)
    parser.add_argument("--ky-mode", type=int, default=1)
    parser.add_argument("--mx-preserve", type=int, default=4)
    parser.add_argument("--my-preserve", type=int, default=6)
    parser.add_argument("--k-lambdaD", type=float, default=0.4)
    parser.add_argument("--A-min", type=float, default=0.85)
    parser.add_argument("--alpha", type=float, default=0.02)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--n-steps", type=int, default=200)
    parser.add_argument("--diag-stride", type=int, default=10)
    return parser


if __name__ == "__main__":
    run_E3(build_parser().parse_args())
