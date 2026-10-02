#!/usr/bin/env python3
"""
scripts/run_E2_two_stream.py
============================

Experiment E2: two-stream sigma sweep for high-k mode preservation.
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

from pic_kde_2d.benchmarks.landau_2d import fit_mode_gamma
from pic_kde_2d.conventions import marker_grid_shape
from pic_kde_2d.core.grid2d import Grid2D
from pic_kde_2d.core.particles2d import init_two_stream_2d
from pic_kde_2d.core.pic_loop import run_simulation
from pic_kde_2d.methods.spectral_W_wrappers import make_spectral_W_gaussian_methods


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


def measure_two_stream_diagnostics(
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
    v_beam: float,
    v_thermal: float,
    alpha: float,
    fit_t_min: float,
    fit_t_max: float,
) -> dict:
    grid = Grid2D(Nx=Nx, Ny=Ny, Lx=Lx, Ly=Ly)
    deposit, gather = make_spectral_W_gaussian_methods(grid, sigma_x, sigma_y)
    Nx_marker, Ny_marker = marker_grid_shape(Nx, Ny, ppc)
    particles = init_two_stream_2d(
        Nx_marker=Nx_marker,
        Ny_marker=Ny_marker,
        Lx=Lx,
        Ly=Ly,
        v_beam=v_beam,
        v_thermal=v_thermal,
        alpha=alpha,
        kx_mode=kx_mode,
        ky_mode=0,
        rng=np.random.default_rng(seed),
    )
    tracked_modes = [(mode, 0) for mode in range(kx_mode - 2, kx_mode + 3)]
    history = run_simulation(
        particles,
        grid,
        dt,
        n_steps,
        deposit,
        gather,
        diag_stride=diag_stride,
        tracked_modes=tracked_modes,
        verbose=False,
    )
    times = np.asarray(history.times, dtype=np.float64)
    main_mode = np.abs(np.asarray(history.mode_amplitudes_Ex[(kx_mode, 0)], dtype=np.complex128))
    gamma_linear = fit_mode_gamma(times, main_mode, fit_t_min, fit_t_max, amp_floor=1.0e-10)
    peak_index = int(np.argmax(main_mode))
    peak_amp = float(main_mode[peak_index])
    band_energy = 0.0
    for mode in range(kx_mode - 2, kx_mode + 3):
        amps = np.asarray(history.mode_amplitudes_Ex[(mode, 0)], dtype=np.complex128)
        band_energy += float(np.abs(amps[peak_index]) ** 2)
    return {
        "gamma_linear": gamma_linear,
        "peak_amp": peak_amp,
        "peak_time": float(times[peak_index]),
        "band_energy_10_14_at_peak": band_energy,
    }


def run_E2(args: argparse.Namespace) -> dict:
    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)
    sigma_dx_list = parse_float_list(args.sigma_dx)
    seed_values = list(range(args.seed_start, args.seed_start + args.seed_count))

    rows: list[dict] = []
    dx = args.Lx / args.Nx
    dy = args.Ly / args.Ny
    for sigma_dx in sigma_dx_list:
        sigma_x = sigma_dx * dx
        sigma_y = sigma_dx * dy
        for seed in seed_values:
            diag = measure_two_stream_diagnostics(
                Nx=args.Nx,
                Ny=args.Ny,
                Lx=args.Lx,
                Ly=args.Ly,
                kx_mode=args.kx_mode,
                sigma_x=sigma_x,
                sigma_y=sigma_y,
                ppc=args.ppc,
                dt=args.dt,
                n_steps=args.n_steps,
                diag_stride=args.diag_stride,
                seed=seed,
                v_beam=args.v_beam,
                v_thermal=args.v_thermal,
                alpha=args.alpha,
                fit_t_min=args.fit_t_min,
                fit_t_max=args.fit_t_max,
            )
            row = {
                "sigma_dx": sigma_dx,
                "sigma_x": sigma_x,
                "sigma_y": sigma_y,
                "seed": seed,
                **diag,
            }
            rows.append(row)
            print(
                f"E2 sigma={sigma_dx:>4.2f}dx seed={seed:>3d} "
                f"gamma={diag['gamma_linear']: .5f} peak={diag['peak_amp']: .5e}",
                flush=True,
            )

    run_name = f"E2_two_stream{('_' + args.tag) if args.tag else ''}"
    raw_path = output_dir / f"{run_name}.csv"
    write_csv(raw_path, rows)

    summary_rows: list[dict] = []
    for sigma_dx in sigma_dx_list:
        sub = [row for row in rows if row["sigma_dx"] == sigma_dx]
        gamma_vals = np.asarray([row["gamma_linear"] for row in sub], dtype=np.float64)
        peak_vals = np.asarray([row["peak_amp"] for row in sub], dtype=np.float64)
        band_vals = np.asarray([row["band_energy_10_14_at_peak"] for row in sub], dtype=np.float64)
        summary_rows.append(
            {
                "sigma_dx": sigma_dx,
                "gamma_mean": float(np.nanmean(gamma_vals)),
                "gamma_std": float(np.nanstd(gamma_vals)),
                "peak_mean": float(np.nanmean(peak_vals)),
                "peak_std": float(np.nanstd(peak_vals)),
                "band_energy_mean": float(np.nanmean(band_vals)),
                "band_energy_std": float(np.nanstd(band_vals)),
            }
        )

    summary_path = output_dir / f"E2_summary{('_' + args.tag) if args.tag else ''}.csv"
    write_csv(summary_path, summary_rows)

    summary_by_sigma = {row["sigma_dx"]: row for row in summary_rows}
    peak_ratio = float(summary_by_sigma[1.0]["peak_mean"] / max(summary_by_sigma[0.25]["peak_mean"], 1.0e-30)) if 1.0 in summary_by_sigma and 0.25 in summary_by_sigma else float("nan")
    gamma_ratio_sigma05 = float(summary_by_sigma[0.5]["gamma_mean"] / max(summary_by_sigma[0.25]["gamma_mean"], 1.0e-30)) if 0.5 in summary_by_sigma and 0.25 in summary_by_sigma else float("nan")
    gamma_ratio_sigma1 = float(summary_by_sigma[1.0]["gamma_mean"] / max(summary_by_sigma[0.25]["gamma_mean"], 1.0e-30)) if 1.0 in summary_by_sigma and 0.25 in summary_by_sigma else float("nan")

    result = {
        "raw_csv": str(raw_path),
        "summary_csv": str(summary_path),
        "n_rows": len(rows),
        "peak_ratio_sigma1_over_sigma025": peak_ratio,
        "gamma_ratio_sigma05_over_sigma025": gamma_ratio_sigma05,
        "gamma_ratio_sigma1_over_sigma025": gamma_ratio_sigma1,
        "passed": bool(np.isfinite(gamma_ratio_sigma05) and 0.85 < gamma_ratio_sigma05 < 1.15),
        "seed_start": args.seed_start,
        "seed_count": args.seed_count,
    }

    json_path = output_dir / f"E2_result{('_' + args.tag) if args.tag else ''}.json"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)

    print("\n=== E2 SUMMARY ===")
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Phase 4 experiment E2.")
    parser.add_argument("--output", type=pathlib.Path, default=PROJECT_ROOT / "results" / "E2")
    parser.add_argument("--tag", type=str, default="")
    parser.add_argument("--sigma-dx", type=str, default="0.25,0.5,0.75,1.0,1.25,1.5")
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--seed-count", type=int, default=30)
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
    return parser


if __name__ == "__main__":
    run_E2(build_parser().parse_args())
