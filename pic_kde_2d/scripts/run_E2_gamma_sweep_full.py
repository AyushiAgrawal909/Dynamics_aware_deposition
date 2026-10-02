#!/usr/bin/env python3
"""Part 2.1: extended E2 gamma sweep over the selector bandwidth range."""

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


def parse_float_list(text: str) -> list[float]:
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def write_csv(path: pathlib.Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "sigma_dx",
        "gamma_sigma",
        "gamma_sem",
        "ci_lo",
        "ci_hi",
        "gamma_ratio",
        "damage",
        "method",
        "n_seeds",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_one_fullf(sigma_dx: float, seed: int, args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray]:
    """Run one full-f E2 realization using the frozen E2 parameters."""
    grid = Grid2D(Nx=args.Nx, Ny=args.Ny, Lx=args.Lx, Ly=args.Ly)
    deposit, gather = make_spectral_W_gaussian_methods(
        grid,
        sigma_dx * grid.dx,
        sigma_dx * grid.dy,
    )
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
    amps = np.abs(np.asarray(history.mode_amplitudes_Ex[(args.kx_mode, 0)], dtype=np.complex128))
    return times, amps


def bootstrap_ratio_ci(
    amplitudes: list[np.ndarray],
    reference_amplitudes: list[np.ndarray],
    times: np.ndarray,
    args: argparse.Namespace,
    seed: int,
) -> tuple[float, float, float]:
    """Bootstrap gamma ratio against the sigma=0.25 reference using paired resampling."""
    rng = np.random.default_rng(seed)
    ratios: list[float] = []
    n = len(reference_amplitudes)
    for _ in range(args.nboot):
        idx = rng.integers(0, n, size=n)
        sample = [amplitudes[int(i)] for i in idx]
        ref = [reference_amplitudes[int(i)] for i in idx]
        gamma = fit_gamma_ensemble_fixed_window(
            sample,
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
        if np.isfinite(gamma) and np.isfinite(gamma_ref) and gamma_ref != 0.0:
            ratios.append(float(gamma / gamma_ref))
    vals = np.asarray(ratios, dtype=np.float64)
    return float(np.std(vals)), float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def checkpoint(rows: list[dict]) -> dict:
    """Check monotone damage and large-sigma collapse."""
    ordered = sorted(rows, key=lambda row: row["sigma_dx"])
    monotone_ok = True
    violations: list[dict] = []
    for prev, curr in zip(ordered, ordered[1:]):
        allowed_drop = 2.0 * max(float(prev["gamma_sem"]), float(curr["gamma_sem"]))
        if curr["gamma_ratio"] > prev["gamma_ratio"] + allowed_drop:
            monotone_ok = False
            violations.append(
                {
                    "prev_sigma": prev["sigma_dx"],
                    "curr_sigma": curr["sigma_dx"],
                    "prev_ratio": prev["gamma_ratio"],
                    "curr_ratio": curr["gamma_ratio"],
                    "allowed_noise": allowed_drop,
                }
            )
    sigma12 = min(ordered, key=lambda row: abs(row["sigma_dx"] - 12.0))
    large_sigma_ok = bool(sigma12["gamma_ratio"] < 0.2)
    passed = bool(monotone_ok and large_sigma_ok)
    return {
        "passed": passed,
        "monotone_ok": monotone_ok,
        "large_sigma_ok": large_sigma_ok,
        "gamma_ratio_sigma12": sigma12["gamma_ratio"],
        "violations": violations,
    }


def run_sweep(args: argparse.Namespace) -> dict:
    args.output.mkdir(parents=True, exist_ok=True)
    part2 = args.output / "results" / "part2"
    if args.output.name == "part2":
        part2 = args.output
    part2.mkdir(parents=True, exist_ok=True)
    sigma_grid = parse_float_list(args.sigma_dx)
    seeds = list(range(args.seed_start, args.seed_start + args.seeds))

    amplitudes_by_sigma: dict[float, list[np.ndarray]] = {}
    times_by_sigma: dict[float, np.ndarray] = {}
    for sigma in sigma_grid:
        amps: list[np.ndarray] = []
        times_ref: np.ndarray | None = None
        for seed in seeds:
            times, amp = run_one_fullf(sigma, seed, args)
            times_ref = times
            amps.append(amp)
            print(f"gamma sweep fullf sigma={sigma:g} seed={seed}", flush=True)
        if times_ref is None:
            raise RuntimeError("no amplitudes collected")
        amplitudes_by_sigma[sigma] = amps
        times_by_sigma[sigma] = times_ref

    ref_sigma = min(sigma_grid, key=lambda value: abs(value - args.reference_sigma_dx))
    ref_amps = amplitudes_by_sigma[ref_sigma]
    ref_times = times_by_sigma[ref_sigma]
    gamma0 = fit_gamma_ensemble_fixed_window(
        ref_amps,
        ref_times,
        t_fit_start=args.fit_t_min,
        t_fit_stop=args.fit_t_max,
        amp_floor=1.0e-30,
    )["gamma"]

    rows: list[dict] = []
    for sigma in sigma_grid:
        times = times_by_sigma[sigma]
        fit = fit_gamma_ensemble_fixed_window(
            amplitudes_by_sigma[sigma],
            times,
            t_fit_start=args.fit_t_min,
            t_fit_stop=args.fit_t_max,
            amp_floor=1.0e-30,
        )
        ci = bootstrap_gamma_ci(
            amplitudes_by_sigma[sigma],
            times,
            n_boot=args.nboot,
            seed=51500 + int(round(sigma * 1000.0)),
            fixed_window=True,
            t_fit_start=args.fit_t_min,
            t_fit_stop=args.fit_t_max,
            amp_floor=1.0e-30,
        )
        ratio_sem, _, _ = bootstrap_ratio_ci(
            amplitudes_by_sigma[sigma],
            ref_amps,
            times,
            args,
            61600 + int(round(sigma * 1000.0)),
        )
        ratio = fit["gamma"] / gamma0 if gamma0 != 0.0 else float("nan")
        rows.append(
            {
                "sigma_dx": sigma,
                "gamma_sigma": fit["gamma"],
                "gamma_sem": ci["gamma_sem"],
                "ci_lo": ci["ci_lo"],
                "ci_hi": ci["ci_hi"],
                "gamma_ratio": ratio,
                "damage": 1.0 - ratio,
                "method": "fullf",
                "n_seeds": len(seeds),
            }
        )
        print(f"gamma sweep sigma={sigma:g}: gamma={fit['gamma']:.6e}, ratio={ratio:.6f}", flush=True)

    csv_path = part2 / "gamma_sweep_full.csv"
    write_csv(csv_path, rows)
    check = checkpoint(rows)
    if check["passed"]:
        print("EXTENDED-GAMMA CHECKPOINT PASS", flush=True)
    else:
        print("EXTENDED-GAMMA CHECKPOINT FAILED", flush=True)
        print(json.dumps(check, indent=2, sort_keys=True), flush=True)
        raise SystemExit(2)
    result = {
        "gamma0": gamma0,
        "gamma0_sigma_dx": ref_sigma,
        "fit_window": [args.fit_t_min, args.fit_t_max],
        "csv": str(csv_path),
        "checkpoint": check,
    }
    with (part2 / "gamma_sweep_full_result.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run extended E2 gamma sweep for Part 2.1.")
    parser.add_argument("--output", type=pathlib.Path, default=PROJECT_ROOT)
    parser.add_argument("--sigma-dx", type=str, default="0.25,0.35,0.45,0.55,0.68,1.0,2.0,3.0,12.0")
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
    parser.add_argument("--reference-sigma-dx", type=float, default=0.25)
    parser.add_argument("--nboot", type=int, default=1000)
    return parser


if __name__ == "__main__":
    run_sweep(build_parser().parse_args())
