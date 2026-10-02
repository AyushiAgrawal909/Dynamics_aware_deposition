#!/usr/bin/env python3
"""Step 2.5/2.6 head-to-head E2 comparison.

Compares CIC, density-selected smoothing, and dynamics-constrained smoothing
on the same E2 setup.  Dynamic metrics are measured from live simulations.
Density MISE is pulled from the already measured ppc-1024 reference scan.
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pic_kde_2d.conventions import marker_grid_shape  # noqa: E402
from pic_kde_2d.core.grid2d import Grid2D  # noqa: E402
from pic_kde_2d.core.particles2d import init_two_stream_2d  # noqa: E402
from pic_kde_2d.core.pic_loop import pic_step  # noqa: E402
from pic_kde_2d.methods.spectral_W_wrappers import make_spectral_W_gaussian_methods  # noqa: E402
from pic_kde_2d.validation.landau_fit import bootstrap_gamma_ci, fit_gamma_ensemble_fixed_window  # noqa: E402
from pic_kde_2d.validation.self_force_2d import test_self_force  # noqa: E402


def read_csv(path: pathlib.Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: pathlib.Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def bootstrap_mean_ci(values: np.ndarray, nboot: int, seed: int) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=np.float64)
    draws = values[rng.integers(0, len(values), size=(nboot, len(values)))]
    means = np.mean(draws, axis=1)
    return float(np.mean(values)), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def interp_from_rows(rows: list[dict], x_name: str, y_name: str, x: float) -> float:
    xs = np.asarray([float(row[x_name]) for row in rows], dtype=np.float64)
    ys = np.asarray([float(row[y_name]) for row in rows], dtype=np.float64)
    order = np.argsort(xs)
    return float(np.interp(x, xs[order], ys[order]))


def lookup_or_interp(rows: list[dict], x_name: str, y_name: str, x: float) -> float:
    for row in rows:
        if abs(float(row[x_name]) - x) < 1.0e-12:
            return float(row[y_name])
    return interp_from_rows(rows, x_name, y_name, x)


def make_particles(seed: int, args: argparse.Namespace):
    nx_marker, ny_marker = marker_grid_shape(args.Nx, args.Ny, args.ppc)
    return init_two_stream_2d(
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


def momentum(particles) -> tuple[float, float]:
    mass = np.abs(particles.w)
    return float(np.sum(mass * particles.vx)), float(np.sum(mass * particles.vy))


def run_one_dynamic(sigma_dx: float, seed: int, args: argparse.Namespace) -> dict:
    grid = Grid2D(Nx=args.Nx, Ny=args.Ny, Lx=args.Lx, Ly=args.Ly)
    particles = make_particles(seed, args)
    sigma_x = sigma_dx * grid.dx
    sigma_y = sigma_dx * grid.dy
    deposit, gather = make_spectral_W_gaussian_methods(grid, sigma_x, sigma_y)

    times: list[float] = []
    amplitudes: list[float] = []
    total_energy: list[float] = []
    total_charge: list[float] = []
    px_values: list[float] = []
    py_values: list[float] = []
    mass_total = float(np.sum(np.abs(particles.w)))
    p_norm = max(mass_total * abs(args.v_beam), 1.0e-30)

    for step in range(args.n_steps + 1):
        if step % args.diag_stride == 0:
            rho = deposit(particles, grid)
            Ex, Ey = grid.poisson_solve(rho)
            times.append(step * args.dt)
            amplitudes.append(abs(grid.mode_amplitude(Ex, args.kx_mode, 0)))
            e_total = grid.field_energy(Ex, Ey) + particles.kinetic_energy()
            total_energy.append(e_total)
            total_charge.append(particles.total_charge())
            px, py = momentum(particles)
            px_values.append(px)
            py_values.append(py)
        if step == args.n_steps:
            break
        pic_step(particles, grid, args.dt, deposit, gather)

    energy = np.asarray(total_energy, dtype=np.float64)
    px = np.asarray(px_values, dtype=np.float64)
    py = np.asarray(py_values, dtype=np.float64)
    momentum_drift = np.sqrt((px - px[0]) ** 2 + (py - py[0]) ** 2) / p_norm
    energy_drift = np.abs(energy - energy[0]) / max(abs(float(energy[0])), 1.0e-30)
    charge = np.asarray(total_charge, dtype=np.float64)
    return {
        "sigma_dx": sigma_dx,
        "seed": seed,
        "times": np.asarray(times, dtype=np.float64),
        "amplitude": np.asarray(amplitudes, dtype=np.float64),
        "max_energy_drift": float(np.max(energy_drift)),
        "final_energy_drift": float(energy_drift[-1]),
        "max_momentum_drift": float(np.max(momentum_drift)),
        "final_momentum_drift": float(momentum_drift[-1]),
        "max_charge_error": float(np.max(np.abs(charge - charge[0]))),
    }


def infer_choices(args: argparse.Namespace) -> tuple[list[dict], dict]:
    noise_rows = read_csv(args.noise_csv)
    selector_rows = read_csv(args.selector_csv)
    cv_rows = read_csv(args.cv_csv)
    density_mise_sigma = float(min(noise_rows, key=lambda row: float(row["MISE"]))["sigma_dx"])
    density_cv_sigma = float(min(cv_rows, key=lambda row: float(row["cv_mean"]))["sigma_dx"])
    selector_row = min(selector_rows, key=lambda row: abs(float(row["eps_gamma"]) - args.eps_gamma))
    dynamics_sigma = float(selector_row["sigma_star"])
    choices = [
        {"choice": "cic", "sigma_dx": 0.0, "source": "baseline"},
        {"choice": "density_mise", "sigma_dx": density_mise_sigma, "source": "MISE minimum"},
        {"choice": "density_cv", "sigma_dx": density_cv_sigma, "source": "CV minimum"},
        {"choice": "dynamics", "sigma_dx": dynamics_sigma, "source": f"eps_gamma={args.eps_gamma}"},
    ]
    metadata = {
        "sigma_density_mise": density_mise_sigma,
        "sigma_density_cv": density_cv_sigma,
        "sigma_dynamics": dynamics_sigma,
    }
    return choices, metadata


def build_headline_table(args: argparse.Namespace) -> dict:
    args.output.mkdir(parents=True, exist_ok=True)
    grid = Grid2D(Nx=args.Nx, Ny=args.Ny, Lx=args.Lx, Ly=args.Ly)
    choices, metadata = infer_choices(args)
    unique_sigmas = sorted({float(choice["sigma_dx"]) for choice in choices})
    seeds = list(range(args.seed_start, args.seed_start + args.seeds))
    dynamic_by_sigma: dict[float, list[dict]] = {sigma: [] for sigma in unique_sigmas}

    for sigma in unique_sigmas:
        for seed in seeds:
            result = run_one_dynamic(sigma, seed, args)
            dynamic_by_sigma[sigma].append(result)
            print(f"headline dynamic sigma={sigma:.3f} seed={seed}", flush=True)

    seed_metric_rows = []
    for sigma, rows in dynamic_by_sigma.items():
        for row in rows:
            seed_metric_rows.append(
                {
                    "sigma_dx": sigma,
                    "seed": row["seed"],
                    "max_energy_drift": row["max_energy_drift"],
                    "final_energy_drift": row["final_energy_drift"],
                    "max_momentum_drift": row["max_momentum_drift"],
                    "final_momentum_drift": row["final_momentum_drift"],
                    "max_charge_error": row["max_charge_error"],
                }
            )
    write_csv(args.output / "headline_seed_metrics.csv", seed_metric_rows)

    gamma_by_sigma = {}
    for sigma, rows in dynamic_by_sigma.items():
        times = rows[0]["times"]
        amp_list = [row["amplitude"] for row in rows]
        fit = fit_gamma_ensemble_fixed_window(
            amp_list,
            times,
            t_fit_start=args.fit_t_min,
            t_fit_stop=args.fit_t_max,
            amp_floor=1.0e-30,
        )
        ci = bootstrap_gamma_ci(
            amp_list,
            times,
            n_boot=args.nboot,
            seed=88000 + int(round(1000.0 * sigma)),
            fixed_window=True,
            t_fit_start=args.fit_t_min,
            t_fit_stop=args.fit_t_max,
            amp_floor=1.0e-30,
        )
        gamma_by_sigma[sigma] = {
            "gamma": fit["gamma"],
            "gamma_ci_lo": ci["ci_lo"],
            "gamma_ci_hi": ci["ci_hi"],
            "gamma_sem": ci["gamma_sem"],
            "n_fit_points": fit["n_points"],
        }

    gamma0_cic = gamma_by_sigma[0.0]["gamma"]
    gamma_rows = read_csv(args.gamma_csv)
    gamma0_selector_ref = lookup_or_interp(gamma_rows, "sigma_dx", "gamma_sigma", 0.25)
    noise_rows = read_csv(args.noise_csv)
    mise_rows = read_csv(args.mise_csv)

    table_rows = []
    for choice in choices:
        sigma = float(choice["sigma_dx"])
        dyn_rows = dynamic_by_sigma[sigma]
        energy_values = np.asarray([row["max_energy_drift"] for row in dyn_rows], dtype=np.float64)
        momentum_values = np.asarray([row["max_momentum_drift"] for row in dyn_rows], dtype=np.float64)
        charge_values = np.asarray([row["max_charge_error"] for row in dyn_rows], dtype=np.float64)
        energy_mean, energy_lo, energy_hi = bootstrap_mean_ci(
            energy_values, args.nboot, 91000 + int(round(1000.0 * sigma))
        )
        momentum_mean, momentum_lo, momentum_hi = bootstrap_mean_ci(
            momentum_values, args.nboot, 92000 + int(round(1000.0 * sigma))
        )
        charge_mean, charge_lo, charge_hi = bootstrap_mean_ci(
            charge_values, args.nboot, 93000 + int(round(1000.0 * sigma))
        )
        deposit, gather = make_spectral_W_gaussian_methods(grid, sigma * grid.dx, sigma * grid.dy)
        self_force = test_self_force(deposit, gather, grid)
        gamma = gamma_by_sigma[sigma]["gamma"]
        gamma_ratio_cic = gamma / gamma0_cic if gamma0_cic != 0.0 else float("nan")
        gamma_ratio_ci_lo_cic = gamma_by_sigma[sigma]["gamma_ci_lo"] / gamma0_cic if gamma0_cic != 0.0 else float("nan")
        gamma_ratio_ci_hi_cic = gamma_by_sigma[sigma]["gamma_ci_hi"] / gamma0_cic if gamma0_cic != 0.0 else float("nan")
        gamma_ratio_selector = gamma / gamma0_selector_ref if gamma0_selector_ref != 0.0 else float("nan")
        gamma_ratio_ci_lo_selector = (
            gamma_by_sigma[sigma]["gamma_ci_lo"] / gamma0_selector_ref
            if gamma0_selector_ref != 0.0
            else float("nan")
        )
        gamma_ratio_ci_hi_selector = (
            gamma_by_sigma[sigma]["gamma_ci_hi"] / gamma0_selector_ref
            if gamma0_selector_ref != 0.0
            else float("nan")
        )
        mise = lookup_or_interp(mise_rows, "sigma_dx", "mise_mean", sigma)
        mise_lo = lookup_or_interp(mise_rows, "sigma_dx", "mise_ci_lo", sigma)
        mise_hi = lookup_or_interp(mise_rows, "sigma_dx", "mise_ci_hi", sigma)
        noise_reduction = lookup_or_interp(noise_rows, "sigma_dx", "noise_reduction", sigma)
        table_rows.append(
            {
                "choice": choice["choice"],
                "source": choice["source"],
                "sigma_dx": sigma,
                "density_MISE": mise,
                "density_MISE_ci_lo": mise_lo,
                "density_MISE_ci_hi": mise_hi,
                "noise_reduction": noise_reduction,
                "gamma_sigma": gamma,
                "gamma_ci_lo": gamma_by_sigma[sigma]["gamma_ci_lo"],
                "gamma_ci_hi": gamma_by_sigma[sigma]["gamma_ci_hi"],
                "gamma_ratio_to_cic": gamma_ratio_cic,
                "gamma_ratio_ci_lo_to_cic": gamma_ratio_ci_lo_cic,
                "gamma_ratio_ci_hi_to_cic": gamma_ratio_ci_hi_cic,
                "gamma_ratio_to_selector_ref": gamma_ratio_selector,
                "gamma_ratio_ci_lo_to_selector_ref": gamma_ratio_ci_lo_selector,
                "gamma_ratio_ci_hi_to_selector_ref": gamma_ratio_ci_hi_selector,
                "self_force": self_force["magnitude"],
                "momentum_drift": momentum_mean,
                "momentum_drift_ci_lo": momentum_lo,
                "momentum_drift_ci_hi": momentum_hi,
                "energy_drift": energy_mean,
                "energy_drift_ci_lo": energy_lo,
                "energy_drift_ci_hi": energy_hi,
                "charge_error": charge_mean,
                "charge_error_ci_lo": charge_lo,
                "charge_error_ci_hi": charge_hi,
                "n_dynamic_seeds": len(dyn_rows),
            }
        )

    sigma_star = metadata["sigma_dynamics"]
    sigma_density = metadata["sigma_density_mise"]
    mise_star = lookup_or_interp(mise_rows, "sigma_dx", "mise_mean", sigma_star)
    mise_density = lookup_or_interp(mise_rows, "sigma_dx", "mise_mean", sigma_density)
    density_cost = mise_star / mise_density if mise_density != 0.0 else float("nan")
    for row in table_rows:
        row["density_cost_dynamics_vs_mise_optimal"] = density_cost
        row["note"] = (
            f"Preserving gamma within eps={args.eps_gamma} costs {density_cost:.2f}x "
            "density MISE vs the MISE-optimal density selector."
        )

    write_csv(args.output / "headline_table.csv", table_rows)
    make_frontier_plot(args, table_rows, gamma0_cic)
    result = {
        "choices": choices,
        "gamma0_cic": gamma0_cic,
        "gamma0_selector_ref_sigma_0p25": gamma0_selector_ref,
        "density_cost_dynamics_vs_mise_optimal": density_cost,
        "checkpoint": checkpoint(table_rows, args),
    }
    with (args.output / "headline_result.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return result


def checkpoint(table_rows: list[dict], args: argparse.Namespace) -> dict:
    dynamics = next(row for row in table_rows if row["choice"] == "dynamics")
    density = next(row for row in table_rows if row["choice"] == "density_mise")
    dynamics_pass = bool(
        dynamics["gamma_ratio_to_cic"] >= 1.0 - args.eps_gamma and dynamics["noise_reduction"] > 1.0
    )
    density_collapse = bool(density["gamma_ratio_to_cic"] < args.collapse_ratio)
    return {
        "dynamics_choice_pass": dynamics_pass,
        "density_choice_collapse": density_collapse,
        "collapse_ratio_threshold": args.collapse_ratio,
    }


def make_frontier_plot(args: argparse.Namespace, table_rows: list[dict], gamma0: float) -> None:
    noise_rows = read_csv(args.noise_csv)
    gamma_rows = read_csv(args.gamma_csv)
    measured_sigma = [0.0] + [float(row["sigma_dx"]) for row in gamma_rows]
    measured_ratio = [1.0] + [
        float(row["gamma_sigma"]) / gamma0 if gamma0 != 0.0 else float("nan")
        for row in gamma_rows
    ]
    order = np.argsort(measured_sigma)
    measured_sigma = np.asarray(measured_sigma, dtype=np.float64)[order]
    measured_ratio = np.asarray(measured_ratio, dtype=np.float64)[order]

    frontier_sigma = np.asarray([float(row["sigma_dx"]) for row in noise_rows], dtype=np.float64)
    frontier_mise = np.asarray([float(row["MISE"]) for row in noise_rows], dtype=np.float64)
    interp_ratio = np.interp(frontier_sigma, measured_sigma, measured_ratio)

    fig, ax = plt.subplots(figsize=(7.6, 5.2))
    ax.plot(frontier_mise, interp_ratio, "--", color="#777777", linewidth=1.4, label="interpolated frontier")
    measured_mise = np.asarray(
        [lookup_or_interp(noise_rows, "sigma_dx", "MISE", float(s)) for s in measured_sigma],
        dtype=np.float64,
    )
    ax.plot(measured_mise, measured_ratio, "-", color="#174a7c", linewidth=2.0, label="measured gamma points")
    ax.scatter(measured_mise, measured_ratio, color="#174a7c", s=28)

    colors = {
        "cic": "#111111",
        "density_mise": "#9f2f22",
        "density_cv": "#c96b1b",
        "dynamics": "#1d6f42",
    }
    for row in table_rows:
        ax.scatter(
            row["density_MISE"],
            row["gamma_ratio_to_cic"],
            color=colors.get(row["choice"], "black"),
            s=80,
            zorder=5,
            label=row["choice"],
        )
        ax.annotate(
            row["choice"],
            (row["density_MISE"], row["gamma_ratio_to_cic"]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )
    ax.axhline(1.0 - args.eps_gamma, color="#1d6f42", linestyle=":", linewidth=1.2, label="gamma constraint")
    ax.set_xscale("log")
    ax.set_xlabel("density MISE")
    ax.set_ylabel(r"$\gamma_\sigma/\gamma_{\rm CIC}$")
    ax.set_title("E2 trade-off frontier: density error vs growth fidelity")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(args.output / "tradeoff_frontier.png", dpi=220)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run E2 head-to-head comparison.")
    root = PROJECT_ROOT
    parser.add_argument("--output", type=pathlib.Path, default=root / "results" / "part2" / "headline_comparison")
    parser.add_argument("--noise-csv", type=pathlib.Path, default=root / "results" / "part2" / "noise_model.csv")
    parser.add_argument("--selector-csv", type=pathlib.Path, default=root / "results" / "part2" / "selector_result_cic.csv")
    parser.add_argument("--mise-csv", type=pathlib.Path, default=root / "results" / "E2_mise_cv" / "E2_mise_summary.csv")
    parser.add_argument("--cv-csv", type=pathlib.Path, default=root / "results" / "E2_mise_cv" / "E2_cv_summary.csv")
    parser.add_argument(
        "--gamma-csv",
        type=pathlib.Path,
        default=root / "results" / "part2" / "gamma_sweep_full_schemafix_failed_checkpoint.csv",
    )
    parser.add_argument("--eps-gamma", type=float, default=0.10)
    parser.add_argument("--collapse-ratio", type=float, default=0.2)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--ppc", type=int, default=32)
    parser.add_argument("--Nx", type=int, default=128)
    parser.add_argument("--Ny", type=int, default=16)
    parser.add_argument("--Lx", type=float, default=64.0)
    parser.add_argument("--Ly", type=float, default=16.0)
    parser.add_argument("--kx-mode", type=int, default=12)
    parser.add_argument("--v-beam", type=float, default=0.8)
    parser.add_argument("--v-thermal", type=float, default=0.05)
    parser.add_argument("--alpha", type=float, default=0.001)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--n-steps", type=int, default=300)
    parser.add_argument("--diag-stride", type=int, default=2)
    parser.add_argument("--fit-t-min", type=float, default=0.5)
    parser.add_argument("--fit-t-max", type=float, default=4.0)
    parser.add_argument("--nboot", type=int, default=1000)
    return parser


def main() -> None:
    build_headline_table(build_parser().parse_args())


if __name__ == "__main__":
    main()
