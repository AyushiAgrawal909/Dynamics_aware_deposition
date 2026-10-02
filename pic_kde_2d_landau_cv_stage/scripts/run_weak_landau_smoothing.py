#!/usr/bin/env python3
"""Weak-Landau smoothing burial scan with paired seeds and density selectors."""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import sys

import numpy as np

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if not (PROJECT_ROOT / "pic_kde_2d").is_dir():
    PROJECT_ROOT = pathlib.Path("/home/users/u103576/hexapic-main/pic_kde_2d")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pic_kde_2d.benchmarks.theory import landau_gamma_theory, langmuir_omega_r
from pic_kde_2d.conventions import landau_box_length, marker_grid_shape, thermal_velocity
from pic_kde_2d.core.grid2d import Grid2D
from pic_kde_2d.core.particles2d import init_landau_2d_eigenmode
from pic_kde_2d.core.pic_loop import pic_step, run_simulation
from pic_kde_2d.methods.cic import deposit_cic_2d
from pic_kde_2d.methods.lscv import (
    cic_filter_self_matrix,
    cic_weight_second_moment,
    lscv_score,
    mean_cic_filter_self,
)
from pic_kde_2d.methods.spectral_W import apply_W_to_grid, make_W_gaussian
from pic_kde_2d.methods.spectral_W_wrappers import make_spectral_W_gaussian_methods


DEFAULT_SIGMAS = "0,0.25,0.5,1,2,3,4,5,6,8,10,12,16,20,24,30"


def parse_sigmas(text: str) -> list[float]:
    return sorted({float(item.strip()) for item in text.split(",") if item.strip()})


def write_csv(path: pathlib.Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    names = sorted({key for row in rows for key in row}) if rows else []
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=names)
        writer.writeheader()
        writer.writerows(rows)


def bootstrap_mean(values: np.ndarray, nboot: int, seed: int) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    samples = values[rng.integers(0, len(values), size=(nboot, len(values)))]
    means = samples.mean(axis=1)
    return float(values.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def make_grid(args: argparse.Namespace) -> Grid2D:
    return Grid2D(
        Nx=args.Nx,
        Ny=args.Ny,
        Lx=landau_box_length(args.kx_mode, args.k_lambdaD),
        Ly=args.Ly,
    )


def make_particles(ppc: int, seed: int, grid: Grid2D, args: argparse.Namespace):
    nx_marker, ny_marker = marker_grid_shape(grid.Nx, grid.Ny, ppc)
    return init_landau_2d_eigenmode(
        Nx_marker=nx_marker,
        Ny_marker=ny_marker,
        Lx=grid.Lx,
        Ly=grid.Ly,
        v_thermal=thermal_velocity(),
        alpha=args.alpha,
        kx_mode=args.kx_mode,
        omega_r=langmuir_omega_r(args.k_lambdaD),
        ky_mode=0,
        rng=np.random.default_rng(seed),
    )


def floor_aware_fit(
    amplitudes: list[np.ndarray],
    times: np.ndarray,
    t_min: float,
    t_max: float,
    floor_start: float,
    noise_factor: float,
) -> dict:
    stacked = np.stack(amplitudes)
    average = (
        np.abs(np.mean(stacked, axis=0))
        if np.iscomplexobj(stacked)
        else np.mean(stacked, axis=0)
    )
    late = times >= floor_start
    floor = float(np.median(average[late])) if late.sum() >= 3 else float(average[-1])
    mask = (
        (times >= t_min)
        & (times <= t_max)
        & np.isfinite(average)
        & (average > noise_factor * floor)
        & (average > 0.0)
    )
    if mask.sum() < 5:
        return {
            "gamma": float("nan"),
            "n_points": int(mask.sum()),
            "noise_floor": floor,
            "fit_t_start": float("nan"),
            "fit_t_stop": float("nan"),
            "r_squared": float("nan"),
        }
    x = times[mask]
    y = np.log(average[mask])
    coeff = np.polyfit(x, y, 1)
    predicted = np.polyval(coeff, x)
    ss_res = float(np.sum((y - predicted) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return {
        "gamma": float(coeff[0]),
        "n_points": int(mask.sum()),
        "noise_floor": floor,
        "fit_t_start": float(x[0]),
        "fit_t_stop": float(x[-1]),
        "r_squared": float(1.0 - ss_res / ss_tot) if ss_tot > 0.0 else float("nan"),
    }


def fixed_window_fit(amplitude: np.ndarray, times: np.ndarray, t_min: float, t_max: float) -> dict:
    mask = (times >= t_min) & (times <= t_max) & np.isfinite(amplitude) & (amplitude > 0.0)
    if mask.sum() < 5:
        return {"gamma": float("nan"), "r_squared": float("nan")}
    x = times[mask]
    y = np.log(amplitude[mask])
    coeff = np.polyfit(x, y, 1)
    predicted = np.polyval(coeff, x)
    ss_res = float(np.sum((y - predicted) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return {
        "gamma": float(coeff[0]),
        "r_squared": float(1.0 - ss_res / ss_tot) if ss_tot > 0.0 else float("nan"),
    }


def bootstrap_floor_fit(amplitudes: list[np.ndarray], times: np.ndarray, args: argparse.Namespace, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(args.nboot):
        indices = rng.integers(0, len(amplitudes), size=len(amplitudes))
        fit = floor_aware_fit(
            [amplitudes[int(index)] for index in indices],
            times,
            args.fit_t_min,
            args.fit_t_max,
            args.noise_floor_start,
            args.noise_factor,
        )
        if np.isfinite(fit["gamma"]):
            values.append(fit["gamma"])
    if not values:
        return {"gamma_sem": float("nan"), "gamma_ci_lo": float("nan"), "gamma_ci_hi": float("nan"), "bootstrap_valid_fraction": 0.0}
    array = np.asarray(values)
    return {
        "gamma_sem": float(array.std()),
        "gamma_ci_lo": float(np.percentile(array, 2.5)),
        "gamma_ci_hi": float(np.percentile(array, 97.5)),
        "bootstrap_valid_fraction": len(values) / float(args.nboot),
    }


def run_gamma(args: argparse.Namespace) -> None:
    args.output.mkdir(parents=True, exist_ok=True)
    grid = make_grid(args)
    sigma = float(args.sigma_dx_value)
    deposit, gather = make_spectral_W_gaussian_methods(
        grid, sigma * grid.dx, sigma * grid.dy
    )
    mode_histories = []
    times = None
    seed_rows = []
    for seed in range(args.seed_start, args.seed_start + args.seeds):
        particles = make_particles(args.ppc, seed, grid, args)
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
        mode_history = np.asarray(
            history.mode_amplitudes_Ex[(args.kx_mode, 0)], dtype=np.complex128
        )
        amplitude = np.abs(mode_history)
        mode_histories.append(mode_history)
        adaptive = floor_aware_fit(
            [amplitude], times, args.fit_t_min, args.fit_t_max,
            args.noise_floor_start, args.noise_factor
        )
        fixed = fixed_window_fit(amplitude, times, args.fit_t_min, args.fit_t_max)
        seed_rows.append(
            {
                "seed": seed,
                "sigma_dx": sigma,
                "gamma_floor_aware": adaptive["gamma"],
                "gamma_fixed_window": fixed["gamma"],
                "fixed_window_r_squared": fixed["r_squared"],
                "adaptive_n_points": adaptive["n_points"],
                "noise_floor": adaptive["noise_floor"],
                "fit_start_snr": float(amplitude[np.argmin(np.abs(times - args.fit_t_min))] / max(adaptive["noise_floor"], 1.0e-300)),
            }
        )
        print("gamma sigma={} seed={}".format(sigma, seed), flush=True)

    assert times is not None
    ensemble = floor_aware_fit(
        mode_histories, times, args.fit_t_min, args.fit_t_max,
        args.noise_floor_start, args.noise_factor
    )
    coherent_average = np.abs(np.mean(np.stack(mode_histories), axis=0))
    fixed_ensemble = fixed_window_fit(coherent_average, times, args.fit_t_min, args.fit_t_max)
    bootstrap = bootstrap_floor_fit(
        mode_histories, times, args, 88000 + int(round(sigma * 1000.0))
    )
    seed_fixed = np.asarray([row["gamma_fixed_window"] for row in seed_rows], dtype=np.float64)
    floor_safe = max(ensemble["noise_floor"], 1.0e-300)
    initial_snr = float(coherent_average[0] / floor_safe)
    fit_start_snr = float(
        coherent_average[np.argmin(np.abs(times - args.fit_t_min))] / floor_safe
    )
    summary = {
        "sigma_dx": sigma,
        "gamma_floor_aware": ensemble["gamma"],
        "gamma_fixed_window": fixed_ensemble["gamma"],
        "gamma_seed_mean": float(np.nanmean(seed_fixed)),
        "gamma_seed_std": float(np.nanstd(seed_fixed, ddof=1)),
        "gamma_seed_min": float(np.nanmin(seed_fixed)),
        "gamma_seed_max": float(np.nanmax(seed_fixed)),
        "adaptive_fit_valid": bool(np.isfinite(ensemble["gamma"])),
        "adaptive_n_points": ensemble["n_points"],
        "adaptive_r_squared": ensemble["r_squared"],
        "fixed_window_r_squared": fixed_ensemble["r_squared"],
        "noise_floor": ensemble["noise_floor"],
        "fit_start_snr": fit_start_snr,
        "initial_snr": initial_snr,
        "resolved_efoldings_initial": float(max(0.0, np.log(initial_snr))),
        "resolved_efoldings_fit_start": float(max(0.0, np.log(fit_start_snr))),
        "initial_amplitude": float(coherent_average[0]),
        "n_seeds": args.seeds,
    }
    summary.update(bootstrap)
    tag = str(sigma).replace(".", "p")
    write_csv(args.output / ("gamma_seed_sigma_{}.csv".format(tag)), seed_rows)
    write_csv(args.output / ("gamma_summary_sigma_{}.csv".format(tag)), [summary])
    np.savez_compressed(
        args.output / ("amplitude_sigma_{}.npz".format(tag)),
        times=times,
        mode_histories=np.stack(mode_histories),
        amplitudes=np.abs(np.stack(mode_histories)),
        amplitude_coherent_mean=coherent_average,
    )


def probability_density_cic(particles, grid: Grid2D) -> np.ndarray:
    weights = np.full(particles.n_particles, 1.0 / particles.n_particles)
    return deposit_cic_2d(
        particles.x, particles.y, weights,
        grid.Nx, grid.Ny, grid.dx, grid.dy
    )


def charge_density_cic(particles, grid: Grid2D) -> np.ndarray:
    return deposit_cic_2d(
        particles.x, particles.y, particles.w,
        grid.Nx, grid.Ny, grid.dx, grid.dy
    )


def self_kernel_value(grid: Grid2D, W: np.ndarray | None) -> float:
    offsets = [(0.13, 0.17), (0.31, 0.41), (0.57, 0.23), (0.79, 0.67),
               (0.43, 0.89), (0.91, 0.11), (0.25, 0.75), (0.75, 0.25)]
    values = []
    for ox, oy in offsets:
        x = np.asarray([(0.5 * grid.Nx + ox) * grid.dx]) % grid.Lx
        y = np.asarray([(0.5 * grid.Ny + oy) * grid.dy]) % grid.Ly
        rho_one = deposit_cic_2d(
            x, y, np.asarray([1.0]), grid.Nx, grid.Ny, grid.dx, grid.dy
        )
        smoothed = rho_one if W is None else apply_W_to_grid(rho_one, W, grid)
        values.append(float(np.sum(smoothed * rho_one) * grid.dx * grid.dy))
    return float(np.mean(values))


def cv_score(rho_prob: np.ndarray, W: np.ndarray | None, self_term: float, n: int, grid: Grid2D) -> float:
    smooth = rho_prob if W is None else apply_W_to_grid(rho_prob, W, grid)
    integrated_square = float(np.sum(smooth * smooth) * grid.dx * grid.dy)
    raw_at_particles = float(np.sum(smooth * rho_prob) * grid.dx * grid.dy)
    loo_mean = (raw_at_particles - self_term / n) * (n / float(n - 1))
    return integrated_square - 2.0 * loo_mean


def run_selector(args: argparse.Namespace) -> None:
    args.output.mkdir(parents=True, exist_ok=True)
    grid = make_grid(args)
    sigmas = parse_sigmas(args.sigma_dx)
    filters = {
        sigma: make_W_gaussian(grid, sigma * grid.dx, sigma * grid.dy)
        for sigma in sigmas if sigma > 0.0
    }
    self_matrices = {
        sigma: cic_filter_self_matrix(
            grid,
            None if sigma == 0.0 else filters[sigma],
        )
        for sigma in sigmas
    }
    deposit_live, gather_live = make_spectral_W_gaussian_methods(
        grid, args.live_sigma_dx * grid.dx, args.live_sigma_dx * grid.dy
    )
    rows = []
    for seed in range(args.seed_start, args.seed_start + args.seeds):
        low = make_particles(args.ppc, seed, grid, args)
        reference = make_particles(args.reference_ppc, seed, grid, args)
        cv_values = {sigma: [] for sigma in sigmas}
        self_values = {sigma: [] for sigma in sigmas}
        mise_values = {sigma: [] for sigma in sigmas}
        for step in range(args.selector_n_steps + 1):
            time = step * args.dt
            if (
                step % args.selector_diag_stride == 0
                and args.selector_t_min <= time <= args.selector_t_max
            ):
                rho_prob = probability_density_cic(low, grid)
                rho_low = charge_density_cic(low, grid)
                rho_ref = charge_density_cic(reference, grid)
                moment = cic_weight_second_moment(low.x, low.y, grid)
                for sigma in sigmas:
                    W = None if sigma == 0.0 else filters[sigma]
                    rho_sigma = rho_low if W is None else apply_W_to_grid(rho_low, W, grid)
                    mean_self = mean_cic_filter_self(
                        moment,
                        self_matrices[sigma],
                    )
                    cv_values[sigma].append(
                        lscv_score(
                            rho_prob,
                            W,
                            mean_self,
                            low.n_particles,
                            grid,
                        )
                    )
                    self_values[sigma].append(mean_self)
                    mise_values[sigma].append(float(np.mean((rho_sigma - rho_ref) ** 2)))
            if step == args.selector_n_steps:
                break
            pic_step(low, grid, args.dt, deposit_live, gather_live)
            pic_step(reference, grid, args.dt, deposit_live, gather_live)
        for sigma in sigmas:
            rows.append(
                {
                    "seed": seed,
                    "sigma_dx": sigma,
                    "cv_time_avg": float(np.mean(cv_values[sigma])),
                    "mise_time_avg": float(np.mean(mise_values[sigma])),
                    "n_times": len(cv_values[sigma]),
                    "self_kernel": float(np.mean(self_values[sigma])),
                }
            )
        print("selector seed={}".format(seed), flush=True)

    summary = []
    for sigma in sigmas:
        selected = [row for row in rows if row["sigma_dx"] == sigma]
        cv_mean, cv_lo, cv_hi = bootstrap_mean(
            np.asarray([row["cv_time_avg"] for row in selected]), args.nboot,
            91000 + int(round(sigma * 1000.0))
        )
        mise_mean, mise_lo, mise_hi = bootstrap_mean(
            np.asarray([row["mise_time_avg"] for row in selected]), args.nboot,
            92000 + int(round(sigma * 1000.0))
        )
        summary.append(
            {
                "sigma_dx": sigma,
                "cv_mean": cv_mean,
                "cv_ci_lo": cv_lo,
                "cv_ci_hi": cv_hi,
                "mise_mean": mise_mean,
                "mise_ci_lo": mise_lo,
                "mise_ci_hi": mise_hi,
                "n_seeds": len(selected),
                "n_times": selected[0]["n_times"],
            }
        )
    sigma_cv = float(min(summary, key=lambda row: row["cv_mean"])["sigma_dx"])
    sigma_mise = float(min(summary, key=lambda row: row["mise_mean"])["sigma_dx"])
    upper = max(sigmas)
    result = {
        "selector": "least-squares leave-one-out CV for the CIC plus spectral-W estimator",
        "sigma_cv_dx": sigma_cv,
        "sigma_cv_right_censored": sigma_cv == upper,
        "sigma_mise_dx": sigma_mise,
        "sigma_mise_right_censored": sigma_mise == upper,
        "sigma_search_max_dx": upper,
        "parameters": parameter_block(args, grid),
    }
    write_csv(args.output / "selector_seed_values.csv", rows)
    write_csv(args.output / "selector_summary.csv", summary)
    (args.output / "selector_result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


def parameter_block(args: argparse.Namespace, grid: Grid2D) -> dict:
    return {
        "Nx": grid.Nx, "Ny": grid.Ny, "Lx": grid.Lx, "Ly": grid.Ly,
        "dx": grid.dx, "dy": grid.dy, "kx_mode": args.kx_mode,
        "k_lambdaD": args.k_lambdaD, "alpha": args.alpha, "ppc": args.ppc,
        "reference_ppc": args.reference_ppc, "seed_start": args.seed_start,
        "seeds": args.seeds, "dt": args.dt, "n_steps": args.n_steps,
        "diag_stride": args.diag_stride,
        "fit_window": [args.fit_t_min, args.fit_t_max],
        "noise_floor_start": args.noise_floor_start,
        "noise_factor": args.noise_factor,
        "sigma_grid_dx": parse_sigmas(args.sigma_dx),
    }


def run_merge(args: argparse.Namespace) -> None:
    summaries = []
    for path in args.output.glob("gamma_summary_sigma_*.csv"):
        with path.open(newline="", encoding="utf-8") as stream:
            row = next(csv.DictReader(stream))
        converted = {}
        for key, value in row.items():
            if key == "adaptive_fit_valid":
                converted[key] = value.lower() == "true"
            else:
                try:
                    converted[key] = float(value)
                except ValueError:
                    converted[key] = value
        summaries.append(converted)
    summaries.sort(key=lambda row: row["sigma_dx"])
    if not summaries:
        raise RuntimeError("no gamma summaries found")
    gamma_reference = summaries[0]["gamma_fixed_window"]
    for row in summaries:
        row["gamma_ratio_to_sigma0"] = row["gamma_fixed_window"] / gamma_reference
        row["gamma_damage"] = 1.0 - row["gamma_ratio_to_sigma0"]
    write_csv(args.output / "gamma_sweep_summary.csv", summaries)

    selector = json.loads((args.output / "selector_result.json").read_text(encoding="utf-8"))
    selected = {
        "cv": selector["sigma_cv_dx"],
        "mise": selector["sigma_mise_dx"],
    }
    by_sigma = {row["sigma_dx"]: row for row in summaries}
    result = {
        "parameters": parameter_block(args, make_grid(args)),
        "gamma_theory": landau_gamma_theory(args.k_lambdaD),
        "gamma_reference_sigma_dx": summaries[0]["sigma_dx"],
        "density_selectors": selector,
        "selected_dynamics": {
            name: by_sigma.get(sigma, {"status": "selected sigma absent from gamma grid"})
            for name, sigma in selected.items()
        },
    }
    (args.output / "weak_landau_result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["gamma", "selector", "merge"])
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--sigma-dx", default=DEFAULT_SIGMAS)
    parser.add_argument("--sigma-dx-value", type=float)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--ppc", type=int, default=8)
    parser.add_argument("--reference-ppc", type=int, default=512)
    parser.add_argument("--Nx", type=int, default=64)
    parser.add_argument("--Ny", type=int, default=16)
    parser.add_argument("--Ly", type=float, default=8.0)
    parser.add_argument("--kx-mode", type=int, default=1)
    parser.add_argument("--k-lambdaD", type=float, default=0.4)
    parser.add_argument("--alpha", type=float, default=0.01)
    parser.add_argument("--live-sigma-dx", type=float, default=0.25)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--n-steps", type=int, default=1800)
    parser.add_argument("--diag-stride", type=int, default=2)
    parser.add_argument("--fit-t-min", type=float, default=2.0)
    parser.add_argument("--fit-t-max", type=float, default=15.0)
    parser.add_argument("--noise-floor-start", type=float, default=70.0)
    parser.add_argument("--noise-factor", type=float, default=3.0)
    parser.add_argument("--selector-n-steps", type=int, default=600)
    parser.add_argument("--selector-diag-stride", type=int, default=10)
    parser.add_argument("--selector-t-min", type=float, default=5.0)
    parser.add_argument("--selector-t-max", type=float, default=30.0)
    parser.add_argument("--nboot", type=int, default=1000)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.mode == "gamma":
        if args.sigma_dx_value is None:
            raise SystemExit("gamma mode requires --sigma-dx-value")
        run_gamma(args)
    elif args.mode == "selector":
        run_selector(args)
    else:
        run_merge(args)


if __name__ == "__main__":
    main()
