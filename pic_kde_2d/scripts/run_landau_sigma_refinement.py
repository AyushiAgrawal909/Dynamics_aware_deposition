#!/usr/bin/env python3
"""Run and analyze the paired weak-Landau sigma refinement."""

from __future__ import annotations

import argparse
import csv
import json
import math
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
from pic_kde_2d.validation.landau_censored_fit import fit_censored_mode, paired_bootstrap_ratio


DEFAULT_SIGMAS = "0,0.3,0.5,1,2,3,5,8,12"


def parse_floats(text: str) -> list[float]:
    return sorted({float(item.strip()) for item in text.split(",") if item.strip()})


def parse_ints(text: str) -> list[int]:
    return sorted({int(item.strip()) for item in text.split(",") if item.strip()})


def sigma_tag(value: float) -> str:
    return f"{value:.8g}".replace("-", "m").replace(".", "p")


def write_csv(path: pathlib.Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("cannot write an empty table")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def frozen_grid(args: argparse.Namespace) -> Grid2D:
    return Grid2D(
        Nx=args.Nx,
        Ny=args.Ny,
        Lx=landau_box_length(args.kx_mode, args.k_lambdaD),
        Ly=args.Ly,
    )


def simulate(args: argparse.Namespace) -> None:
    grid = frozen_grid(args)
    sigma = float(args.sigma_dx)
    deposit, gather = make_spectral_W_gaussian_methods(
        grid,
        sigma * grid.dx,
        sigma * grid.dy,
    )
    nx_marker, ny_marker = marker_grid_shape(grid.Nx, grid.Ny, args.ppc)
    particles = init_landau_2d_eigenmode(
        Nx_marker=nx_marker,
        Ny_marker=ny_marker,
        Lx=grid.Lx,
        Ly=grid.Ly,
        v_thermal=thermal_velocity(),
        alpha=args.alpha,
        kx_mode=args.kx_mode,
        omega_r=langmuir_omega_r(args.k_lambdaD),
        ky_mode=0,
        rng=np.random.default_rng(args.seed),
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
    mode = np.asarray(history.mode_amplitudes_Ex[(args.kx_mode, 0)], dtype=np.complex128)
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output / "mode_history.npz",
        times=times,
        mode=mode,
        ppc=np.asarray([args.ppc], dtype=np.int64),
        sigma_dx=np.asarray([sigma], dtype=np.float64),
        seed=np.asarray([args.seed], dtype=np.int64),
    )
    metadata = {
        "Nx": grid.Nx,
        "Ny": grid.Ny,
        "Lx": grid.Lx,
        "Ly": grid.Ly,
        "dx": grid.dx,
        "dy": grid.dy,
        "ppc": args.ppc,
        "sigma_dx": sigma,
        "seed": args.seed,
        "alpha": args.alpha,
        "dt": args.dt,
        "n_steps": args.n_steps,
        "diag_stride": args.diag_stride,
        "kx_mode": args.kx_mode,
        "k_lambdaD": args.k_lambdaD,
        "quiet_start_layout": "global",
        "initialization": "density-and-current moment-matched traveling Landau perturbation",
        "physics_chain": "CIC -> Gaussian W on rho -> spectral Poisson -> same W on E -> CIC gather",
    }
    (args.output / "run_parameters.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, sort_keys=True), flush=True)


def load_histories(
    input_root: pathlib.Path,
    ppc: int,
    sigma: float,
    seeds: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    histories = []
    times = None
    tag = sigma_tag(sigma)
    for seed in seeds:
        archive = input_root / f"ppc{ppc}" / f"sigma{tag}" / f"seed{seed}" / "mode_history.npz"
        if not archive.exists():
            raise FileNotFoundError(archive)
        with np.load(archive) as data:
            this_times = np.asarray(data["times"], dtype=np.float64)
            history = np.asarray(data["mode"], dtype=np.complex128)
            if int(data["ppc"][0]) != ppc or not np.isclose(float(data["sigma_dx"][0]), sigma):
                raise ValueError(f"metadata mismatch in {archive}")
            if int(data["seed"][0]) != seed:
                raise ValueError(f"seed mismatch in {archive}")
        if times is None:
            times = this_times
        else:
            np.testing.assert_allclose(times, this_times, rtol=0.0, atol=0.0)
        histories.append(history)
    assert times is not None
    return times, np.stack(histories)


def _fit_kwargs(args: argparse.Namespace) -> dict:
    return {
        "fit_t_start": args.fit_t_start,
        "fit_t_stop_max": args.fit_t_stop_max,
        "noise_floor_start": args.noise_floor_start,
        "noise_floor_stop": args.noise_floor_stop,
        "noise_factor": args.noise_factor,
        "envelope_period": args.envelope_period,
        "persistence_points": args.persistence_points,
        "min_points": args.min_points,
        "min_duration": args.min_duration,
        "min_model_efoldings": args.min_model_efoldings,
        "min_r_squared": args.min_r_squared,
    }


def analyze(args: argparse.Namespace) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    args.output.mkdir(parents=True, exist_ok=True)
    ppcs = parse_ints(args.ppcs)
    sigmas = parse_floats(args.sigma_grid)
    if 0.0 not in sigmas:
        raise ValueError("sigma grid must include the CIC baseline sigma=0")
    seeds = list(range(args.seed_start, args.seed_start + args.seeds))
    fit_kwargs = _fit_kwargs(args)
    gamma_theory = landau_gamma_theory(args.k_lambdaD)
    rows: list[dict] = []
    summaries = []

    for ppc in ppcs:
        times0, baseline = load_histories(args.input, ppc, 0.0, seeds)
        baseline_fit = fit_censored_mode(baseline, times0, **fit_kwargs)
        baseline_relative_error = abs(float(baseline_fit["gamma"]) - gamma_theory) / abs(gamma_theory)
        baseline_rate_usable = bool(
            baseline_fit["quality_pass"]
            and np.isfinite(baseline_relative_error)
            and baseline_relative_error <= args.baseline_theory_tolerance
        )
        ppc_rows = []
        for sigma in sigmas:
            times, histories = load_histories(args.input, ppc, sigma, seeds)
            np.testing.assert_allclose(times0, times, rtol=0.0, atol=0.0)
            fit = fit_censored_mode(histories, times, **fit_kwargs)
            if sigma == 0.0:
                bootstrap = {
                    "ratio_sem": 0.0,
                    "ratio_ci_lo": 1.0,
                    "ratio_ci_hi": 1.0,
                    "gamma_ci_lo": fit["gamma"],
                    "gamma_ci_hi": fit["gamma"],
                    "bootstrap_valid_fraction": 1.0,
                    "bootstrap_quality_fraction": float(fit["quality_pass"]),
                }
            else:
                bootstrap = paired_bootstrap_ratio(
                    baseline,
                    histories,
                    times,
                    n_boot=args.nboot,
                    seed=77000 + 1000 * ppc + int(round(100.0 * sigma)),
                    **fit_kwargs,
                )
            gamma0 = float(baseline_fit["gamma"])
            gamma = float(fit["gamma"])
            ratio = gamma / gamma0 if np.isfinite(gamma) and gamma0 != 0.0 else float("nan")
            matched_W2 = math.exp(-((2.0 * math.pi * args.kx_mode * sigma / args.Nx) ** 2))
            coupling_loss = 1.0 - matched_W2
            row = {
                "ppc": ppc,
                "sigma_dx": sigma,
                "gamma": gamma,
                "gamma_theory": gamma_theory,
                "gamma_ratio_to_cic": ratio,
                "gamma_damage": abs(ratio - 1.0) if np.isfinite(ratio) else float("nan"),
                "matched_W2": matched_W2,
                "coupling_loss": coupling_loss,
                "R_coefficient_point": (
                    abs(ratio - 1.0) / coupling_loss
                    if np.isfinite(ratio) and coupling_loss > 0.0
                    else float("nan")
                ),
                "ratio_sem": bootstrap["ratio_sem"],
                "ratio_ci_lo": bootstrap["ratio_ci_lo"],
                "ratio_ci_hi": bootstrap["ratio_ci_hi"],
                "gamma_ci_lo": bootstrap["gamma_ci_lo"],
                "gamma_ci_hi": bootstrap["gamma_ci_hi"],
                "bootstrap_valid_fraction": bootstrap["bootstrap_valid_fraction"],
                "bootstrap_quality_fraction": bootstrap["bootstrap_quality_fraction"],
                "quality_pass": fit["quality_pass"],
                "status": fit["status"],
                "floor_censored": fit["floor_censored"],
                "envelope_period": fit["envelope_period"],
                "envelope_points": fit["envelope_points"],
                "fit_t_start": fit["fit_t_start"],
                "fit_t_stop": fit["fit_t_stop"],
                "fit_duration": fit["fit_duration"],
                "fit_r_squared": fit["r_squared"],
                "model_efoldings": fit["model_efoldings"],
                "noise_floor": fit["noise_floor"],
                "floor_t_start": fit["floor_t_start"],
                "floor_t_stop": fit["floor_t_stop"],
                "fit_start_snr": fit["fit_start_snr"],
                "initial_snr": fit["initial_snr"],
                "n_points": fit["n_points"],
                "n_seeds": len(seeds),
            }
            rows.append(row)
            ppc_rows.append(row)

        central = [
            row for row in ppc_rows
            if row["quality_pass"] and np.isfinite(row["gamma_damage"]) and row["gamma_damage"] <= args.eps_gamma
        ] if baseline_rate_usable else []
        supported = [
            row for row in central
            if row["ratio_ci_lo"] >= 1.0 - args.eps_gamma
            and row["ratio_ci_hi"] <= 1.0 + args.eps_gamma
            and row["bootstrap_quality_fraction"] >= args.min_bootstrap_quality_fraction
        ]
        coefficient_rows = [
            row for row in ppc_rows
            if baseline_rate_usable
            and row["quality_pass"]
            and row["sigma_dx"] > 0.0
            and row["sigma_dx"] <= args.R_linear_sigma_max
            and np.isfinite(row["gamma_damage"])
        ]
        losses = np.asarray([row["coupling_loss"] for row in coefficient_rows], dtype=np.float64)
        damages = np.asarray([row["gamma_damage"] for row in coefficient_rows], dtype=np.float64)
        signed_shifts = np.asarray(
            [row["gamma_ratio_to_cic"] - 1.0 for row in coefficient_rows], dtype=np.float64
        )
        denominator = float(np.dot(losses, losses))
        R_coefficient = float(np.dot(losses, damages) / denominator) if denominator > 0.0 else float("nan")
        R_coefficient_signed = (
            float(np.dot(losses, signed_shifts) / denominator) if denominator > 0.0 else float("nan")
        )
        summaries.append(
            {
                "ppc": ppc,
                "gamma_cic": baseline_fit["gamma"],
                "gamma_cic_relative_error_to_theory": baseline_relative_error,
                "sigma_R_central_dx": max((row["sigma_dx"] for row in central), default=float("nan")),
                "sigma_R_ci_supported_dx": max((row["sigma_dx"] for row in supported), default=float("nan")),
                "eps_gamma": args.eps_gamma,
                "baseline_quality_pass": baseline_fit["quality_pass"],
                "baseline_rate_usable": baseline_rate_usable,
                "R_coefficient_abs": R_coefficient,
                "R_coefficient_signed": R_coefficient_signed,
                "R_coefficient_n_points": len(coefficient_rows),
                "R_coefficient_sigma_max_dx": args.R_linear_sigma_max,
            }
        )

    write_csv(args.output / "landau_sigma_refinement.csv", rows)
    write_csv(args.output / "landau_sigma_R.csv", summaries)
    finite_supported = [item for item in summaries if np.isfinite(item["sigma_R_ci_supported_dx"])]
    relaxation_test = None
    if len(finite_supported) == len(ppcs):
        by_ppc = {item["ppc"]: item for item in finite_supported}
        relaxation_test = bool(
            by_ppc[max(ppcs)]["sigma_R_ci_supported_dx"]
            >= by_ppc[min(ppcs)]["sigma_R_ci_supported_dx"]
        )
    result = {
        "parameters": {
            "ppcs": ppcs,
            "sigma_grid_dx": sigmas,
            "seeds": seeds,
            "eps_gamma": args.eps_gamma,
            "fit": fit_kwargs,
            "gamma_theory": gamma_theory,
            "baseline": "CIC, sigma=0",
            "baseline_theory_tolerance": args.baseline_theory_tolerance,
            "R_linear_sigma_max_dx": args.R_linear_sigma_max,
        },
        "sigma_R": summaries,
        "sigma_R_non_decreasing_with_ppc": relaxation_test,
        "interpretation": (
            "null: at least one PPC lacks a CI-supported sigma_R"
            if relaxation_test is None
            else "PASS" if relaxation_test else "FAIL"
        ),
    }
    (args.output / "landau_sigma_refinement.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    figure, (axis_rate, axis_snr) = plt.subplots(1, 2, figsize=(11.2, 4.6))
    colors = ["#146c94", "#d35400"]
    for color, ppc in zip(colors, ppcs):
        selected = [row for row in rows if row["ppc"] == ppc]
        x = np.asarray([row["sigma_dx"] for row in selected], dtype=np.float64)
        ratio = np.asarray([row["gamma_ratio_to_cic"] for row in selected], dtype=np.float64)
        lo = np.asarray([row["ratio_ci_lo"] for row in selected], dtype=np.float64)
        hi = np.asarray([row["ratio_ci_hi"] for row in selected], dtype=np.float64)
        yerr = np.vstack((np.maximum(0.0, ratio - lo), np.maximum(0.0, hi - ratio)))
        axis_rate.errorbar(x, ratio, yerr=yerr, color=color, marker="o", capsize=3, label=f"PPC {ppc}")
        for row in selected:
            if not row["quality_pass"] and np.isfinite(row["gamma_ratio_to_cic"]):
                axis_rate.scatter(row["sigma_dx"], row["gamma_ratio_to_cic"], facecolors="white", edgecolors=color, s=50, zorder=5)
        axis_snr.plot(x, [row["model_efoldings"] for row in selected], "o-", color=color, label=f"PPC {ppc}")
    axis_rate.axhspan(1.0 - args.eps_gamma, 1.0 + args.eps_gamma, color="#65a765", alpha=0.14, label="rate tolerance")
    axis_rate.axhline(1.0, color="black", linewidth=0.9)
    axis_snr.axhline(args.min_model_efoldings, color="#8e3b2d", linestyle="--", label="fit e-folding gate")
    for axis in (axis_rate, axis_snr):
        axis.set_xscale("symlog", linthresh=0.3)
        axis.set_xlabel(r"$\sigma/\Delta x$")
        axis.grid(alpha=0.18)
        axis.legend(frameon=False)
    axis_rate.set_ylabel(r"$\gamma_\sigma/\gamma_{\rm CIC}$")
    axis_rate.set_title("Rate constraint R")
    axis_snr.set_ylabel("resolved model e-foldings in censored fit")
    axis_snr.set_title("Measurability companion")
    figure.suptitle("Weak-Landau sigma refinement")
    figure.tight_layout()
    figure.savefig(args.output / "landau_sigma_refinement.png", dpi=230)
    plt.close(figure)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


def common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--Nx", type=int, default=64)
    parser.add_argument("--Ny", type=int, default=16)
    parser.add_argument("--Ly", type=float, default=8.0)
    parser.add_argument("--kx-mode", type=int, default=1)
    parser.add_argument("--k-lambdaD", type=float, default=0.4)
    parser.add_argument("--alpha", type=float, default=0.01)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--n-steps", type=int, default=1800)
    parser.add_argument("--diag-stride", type=int, default=2)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    simulate_parser = subparsers.add_parser("simulate")
    common_arguments(simulate_parser)
    simulate_parser.add_argument("--output", type=pathlib.Path, required=True)
    simulate_parser.add_argument("--ppc", type=int, required=True)
    simulate_parser.add_argument("--sigma-dx", type=float, required=True)
    simulate_parser.add_argument("--seed", type=int, required=True)

    analyze_parser = subparsers.add_parser("analyze")
    common_arguments(analyze_parser)
    analyze_parser.add_argument("--input", type=pathlib.Path, required=True)
    analyze_parser.add_argument("--output", type=pathlib.Path, required=True)
    analyze_parser.add_argument("--ppcs", default="32,1024")
    analyze_parser.add_argument("--sigma-grid", default=DEFAULT_SIGMAS)
    analyze_parser.add_argument("--seed-start", type=int, default=0)
    analyze_parser.add_argument("--seeds", type=int, default=10)
    analyze_parser.add_argument("--fit-t-start", type=float, default=5.0)
    analyze_parser.add_argument("--fit-t-stop-max", type=float, default=30.0)
    analyze_parser.add_argument("--noise-floor-start", type=float, default=70.0)
    analyze_parser.add_argument("--noise-floor-stop", type=float, default=90.0)
    analyze_parser.add_argument("--noise-factor", type=float, default=5.0)
    analyze_parser.add_argument(
        "--envelope-period",
        type=float,
        default=2.0 * math.pi / langmuir_omega_r(0.4),
    )
    analyze_parser.add_argument("--persistence-points", type=int, default=5)
    analyze_parser.add_argument("--min-points", type=int, default=30)
    analyze_parser.add_argument("--min-duration", type=float, default=5.0)
    analyze_parser.add_argument("--min-model-efoldings", type=float, default=0.5)
    analyze_parser.add_argument("--min-r-squared", type=float, default=0.75)
    analyze_parser.add_argument("--nboot", type=int, default=1000)
    analyze_parser.add_argument("--eps-gamma", type=float, default=0.10)
    analyze_parser.add_argument("--baseline-theory-tolerance", type=float, default=0.15)
    analyze_parser.add_argument("--min-bootstrap-quality-fraction", type=float, default=0.80)
    analyze_parser.add_argument(
        "--R-linear-sigma-max",
        dest="R_linear_sigma_max",
        type=float,
        default=2.0,
        help="largest sigma/dx included in the through-origin linear-response C_R fit",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "simulate":
        simulate(args)
    else:
        analyze(args)


if __name__ == "__main__":
    main()
