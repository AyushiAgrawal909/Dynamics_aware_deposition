#!/usr/bin/env python3
"""Figure 5 data: sigma_safe versus distance from two-stream marginality."""

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
    fields = sorted({key for row in rows for key in row.keys()}) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run_one(A0: float, sigma_dx: float, seed: int, args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray]:
    grid = Grid2D(Nx=args.Nx, Ny=args.Ny, Lx=args.Lx, Ly=args.Ly)
    k_phys = 2.0 * np.pi * args.kx_mode / args.Lx
    v_beam = 1.0 / (A0 * k_phys)
    sigma_x = sigma_dx * grid.dx
    sigma_y = sigma_dx * grid.dy
    deposit, gather = make_spectral_W_gaussian_methods(grid, sigma_x, sigma_y)
    nx_marker, ny_marker = marker_grid_shape(args.Nx, args.Ny, args.ppc)
    particles = init_two_stream_2d(
        Nx_marker=nx_marker,
        Ny_marker=ny_marker,
        Lx=args.Lx,
        Ly=args.Ly,
        v_beam=v_beam,
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
    amps = np.abs(np.asarray(history.mode_amplitudes_Ex[(args.kx_mode, 0)]))
    return times, amps


def bootstrap_ratio(
    amplitudes: list[np.ndarray],
    ref_amplitudes: list[np.ndarray],
    times: np.ndarray,
    args: argparse.Namespace,
    nboot: int,
    seed: int,
) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    ratios: list[float] = []
    n = len(ref_amplitudes)
    for _ in range(nboot):
        idx = rng.integers(0, n, size=n)
        sample = [amplitudes[int(i)] for i in idx]
        ref = [ref_amplitudes[int(i)] for i in idx]
        gamma = fit_gamma_ensemble_fixed_window(sample, times, args.fit_t_min, args.fit_t_max)["gamma"]
        gamma_ref = fit_gamma_ensemble_fixed_window(ref, times, args.fit_t_min, args.fit_t_max)["gamma"]
        if np.isfinite(gamma) and np.isfinite(gamma_ref) and gamma_ref != 0.0:
            ratios.append(gamma / gamma_ref)
    vals = np.asarray(ratios, dtype=np.float64)
    return float(np.mean(vals)), float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def run_scan(args: argparse.Namespace) -> dict:
    args.output.mkdir(parents=True, exist_ok=True)
    A0_values = parse_float_list(args.A0)
    sigma_grid = parse_float_list(args.sigma_dx)
    rows: list[dict] = []
    safe_rows: list[dict] = []
    k_phys = 2.0 * np.pi * args.kx_mode / args.Lx
    kdx = 2.0 * np.pi * args.kx_mode / args.Nx

    for A0 in A0_values:
        by_sigma: dict[float, list[np.ndarray]] = {}
        times_ref: np.ndarray | None = None
        for sigma in sigma_grid:
            amps: list[np.ndarray] = []
            for seed in range(args.seed_start, args.seed_start + args.seeds):
                times, amp = run_one(A0, sigma, seed, args)
                times_ref = times
                amps.append(amp)
                print(f"regime A0={A0:g} sigma={sigma:g} seed={seed}", flush=True)
            by_sigma[sigma] = amps
        if times_ref is None:
            raise RuntimeError("no amplitudes collected")
        ref_sigma = min(sigma_grid, key=lambda x: abs(x - args.reference_sigma_dx))
        gamma_ref = fit_gamma_ensemble_fixed_window(
            by_sigma[ref_sigma], times_ref, args.fit_t_min, args.fit_t_max
        )["gamma"]
        usable: list[float] = []
        for sigma in sigma_grid:
            gamma = fit_gamma_ensemble_fixed_window(
                by_sigma[sigma], times_ref, args.fit_t_min, args.fit_t_max
            )["gamma"]
            ratio = gamma / gamma_ref if gamma_ref != 0.0 else float("nan")
            mean, lo, hi = bootstrap_ratio(
                by_sigma[sigma],
                by_sigma[ref_sigma],
                times_ref,
                args,
                args.nboot,
                92100 + int(round(A0 * 1000.0)) + int(round(sigma * 1000.0)),
            )
            passed = bool(np.isfinite(ratio) and ratio >= args.safe_ratio)
            if passed:
                usable.append(sigma)
            rows.append(
                {
                    "A0": A0,
                    "v_beam": 1.0 / (A0 * k_phys),
                    "kv0": 1.0 / A0,
                    "sigma_dx": sigma,
                    "gamma": gamma,
                    "gamma_ref": gamma_ref,
                    "growth_ratio": ratio,
                    "growth_ratio_ci_lo": lo,
                    "growth_ratio_ci_hi": hi,
                    "growth_ratio_bootstrap_mean": mean,
                    "passed_point": passed,
                    "matched_W_transfer": float(np.exp(-(kdx * sigma) ** 2)),
                }
            )
        safe_rows.append(
            {
                "A0": A0,
                "kv0": 1.0 / A0,
                "v_beam": 1.0 / (A0 * k_phys),
                "sigma_safe_dx": max(usable) if usable else float("nan"),
                "gamma_ref": gamma_ref,
                "n_seeds": args.seeds,
                "safe_ratio": args.safe_ratio,
            }
        )
    write_csv(args.output / "E2_regime_gamma_summary.csv", rows)
    write_csv(args.output / "E2_regime_sigma_safe.csv", safe_rows)
    result = {"summary": str(args.output / "E2_regime_gamma_summary.csv"), "sigma_safe": safe_rows}
    with (args.output / "E2_regime_scan_result.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run E2 regime scan for Figure 5.")
    parser.add_argument("--output", type=pathlib.Path, default=PROJECT_ROOT / "results" / "E2_regime_scan")
    parser.add_argument("--A0", type=str, default="1.05,1.20,1.50,2.0,3.0")
    parser.add_argument("--sigma-dx", type=str, default="0.25,0.35,0.45,0.55,0.68,0.75,1.0,1.25,1.5,2.0")
    parser.add_argument("--reference-sigma-dx", type=float, default=0.25)
    parser.add_argument("--safe-ratio", type=float, default=0.85)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--ppc", type=int, default=32)
    parser.add_argument("--Nx", type=int, default=128)
    parser.add_argument("--Ny", type=int, default=16)
    parser.add_argument("--Lx", type=float, default=64.0)
    parser.add_argument("--Ly", type=float, default=16.0)
    parser.add_argument("--kx-mode", type=int, default=12)
    parser.add_argument("--v-thermal", type=float, default=0.05)
    parser.add_argument("--alpha", type=float, default=0.001)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--n-steps", type=int, default=160)
    parser.add_argument("--diag-stride", type=int, default=2)
    parser.add_argument("--fit-t-min", type=float, default=0.5)
    parser.add_argument("--fit-t-max", type=float, default=4.0)
    parser.add_argument("--nboot", type=int, default=500)
    return parser


if __name__ == "__main__":
    run_scan(build_parser().parse_args())
