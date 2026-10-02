#!/usr/bin/env python3
"""Run H1's CIC/global-C4/C4-capped-Abramson locality gate."""

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

from pic_kde_2d.benchmarks.heterogeneous import (
    PeriodicFloorVonMisesProfile,
    init_profiled_plasma_2d,
    ion_background_on_grid,
    periodic_displacement,
)
from pic_kde_2d.conventions import marker_grid_shape
from pic_kde_2d.core.grid2d import Grid2D
from pic_kde_2d.core.immobile_ions import (
    ImmobileIonBackground,
    neutrality_diagnostics,
    with_immobile_ions,
)
from pic_kde_2d.core.particles2d import Particles2D
from pic_kde_2d.core.pic_loop import pic_step
from pic_kde_2d.methods.adaptive_sigma import abramson_c4_bandwidth
from pic_kde_2d.methods.cic import deposit_cic_2d, gather_cic_2d
from pic_kde_2d.methods.spectral_W_wrappers import (
    make_binned_adaptive_spectral_W_methods,
    make_spectral_W_gaussian_methods,
)


METHODS = ("cic", "global_c4", "abramson_c4_capped")


def copy_particles(particles: Particles2D) -> Particles2D:
    return Particles2D(
        particles.x.copy(),
        particles.y.copy(),
        particles.vx.copy(),
        particles.vy.copy(),
        particles.w.copy(),
        particles.Lx,
        particles.Ly,
    )


def cic_methods():
    def deposit(particles, grid):
        return deposit_cic_2d(
            particles.x,
            particles.y,
            particles.w,
            grid.Nx,
            grid.Ny,
            grid.dx,
            grid.dy,
        )

    def gather(Ex, Ey, particles, grid):
        return gather_cic_2d(
            Ex,
            Ey,
            particles.x,
            particles.y,
            grid.Nx,
            grid.Ny,
            grid.dx,
            grid.dy,
        )

    return deposit, gather


def build_particles(
    grid: Grid2D,
    profile: PeriodicFloorVonMisesProfile,
    ppc: int,
    seed: int,
    args: argparse.Namespace,
    *,
    alpha: float | None = None,
    randomized_positions: bool = True,
) -> Particles2D:
    nx_marker, ny_marker = marker_grid_shape(grid.Nx, grid.Ny, ppc)
    return init_profiled_plasma_2d(
        nx_marker,
        ny_marker,
        grid.Lx,
        grid.Ly,
        args.v_thermal,
        profile,
        np.random.default_rng(seed),
        alpha=args.alpha if alpha is None else alpha,
        carrier_mode=args.carrier_mode,
        shoulder_width=args.shoulder_width,
        randomized_positions=randomized_positions,
    )


def make_method(
    name: str,
    particles: Particles2D,
    grid: Grid2D,
    profile: PeriodicFloorVonMisesProfile,
    args: argparse.Namespace,
) -> tuple[callable, callable, dict]:
    if name == "cic":
        deposit, gather = cic_methods()
        return deposit, gather, {
            "sigma_min_dx": 0.0,
            "sigma_max_dx": 0.0,
            "upper_clip_fraction": 0.0,
            "lower_clip_fraction": 0.0,
        }
    if name == "global_c4":
        deposit, gather = make_spectral_W_gaussian_methods(
            grid,
            args.sigma_c4_dx * grid.dx,
            args.sigma_c4_dx * grid.dy,
        )
        return deposit, gather, {
            "sigma_min_dx": args.sigma_c4_dx,
            "sigma_max_dx": args.sigma_c4_dx,
            "upper_clip_fraction": 1.0,
            "lower_clip_fraction": 0.0,
        }
    if name != "abramson_c4_capped":
        raise ValueError(f"unknown method {name}")

    pilot_density = profile.density(particles.x)
    safe_density = np.maximum(pilot_density, np.finfo(np.float64).tiny)
    geometric_mean = float(np.exp(np.mean(np.log(safe_density))))
    raw_sigma = args.sigma_c4_dx * np.sqrt(geometric_mean / safe_density)
    sigma_cells = abramson_c4_bandwidth(
        pilot_density,
        sigma_pilot=np.asarray([args.sigma_c4_dx, args.sigma_c4_dx]),
        sigma_c4_max=np.asarray([args.sigma_c4_dx, args.sigma_c4_dx]),
        sigma_min=np.asarray([args.sigma_min_dx, args.sigma_min_dx]),
    )
    deposit, gather = make_binned_adaptive_spectral_W_methods(
        grid,
        sigma_cells[:, 0] * grid.dx,
        sigma_cells[:, 1] * grid.dy,
        particles.w,
        n_bins=args.width_bins,
    )
    return deposit, gather, {
        "sigma_min_dx": float(np.min(sigma_cells)),
        "sigma_max_dx": float(np.max(sigma_cells)),
        "upper_clip_fraction": float(np.mean(raw_sigma >= args.sigma_c4_dx)),
        "lower_clip_fraction": float(np.mean(raw_sigma <= args.sigma_min_dx)),
    }


def run_trace(
    particles: Particles2D,
    grid: Grid2D,
    background: ImmobileIonBackground,
    electron_deposit,
    gather,
    args: argparse.Namespace,
) -> dict:
    total_deposit = with_immobile_ions(electron_deposit, background)
    electron_density = []
    Ex_history = []
    Ey_history = []
    times = []
    n_steps = int(round(args.t_end / args.dt))
    for step in range(n_steps + 1):
        if step % args.diag_stride == 0:
            electron = electron_deposit(particles, grid)
            total = background.add(electron)
            Ex, Ey = grid.poisson_solve(total)
            electron_density.append(electron)
            Ex_history.append(Ex)
            Ey_history.append(Ey)
            times.append(step * args.dt)
        if step < n_steps:
            pic_step(particles, grid, args.dt, total_deposit, gather)
    return {
        "times": np.asarray(times),
        "electron_density": np.asarray(electron_density),
        "Ex": np.asarray(Ex_history),
        "Ey": np.asarray(Ey_history),
    }


def equilibrium_gate(
    grid: Grid2D,
    profile: PeriodicFloorVonMisesProfile,
    background: ImmobileIonBackground,
    args: argparse.Namespace,
) -> dict:
    electron_deposit, _ = cic_methods()
    equilibrium = build_particles(
        grid,
        profile,
        args.equilibrium_ppc,
        0,
        args,
        alpha=0.0,
        randomized_positions=False,
    )
    perturbed = build_particles(
        grid,
        profile,
        args.equilibrium_ppc,
        0,
        args,
        alpha=args.alpha,
        randomized_positions=False,
    )
    rho_equilibrium = background.add(electron_deposit(equilibrium, grid))
    rho_perturbed = background.add(electron_deposit(perturbed, grid))
    Ex_eq, Ey_eq = grid.poisson_solve(rho_equilibrium)
    Ex_pert, Ey_pert = grid.poisson_solve(rho_perturbed)
    residual_rms = float(np.sqrt(np.mean(Ex_eq**2 + Ey_eq**2)))
    signal_rms = float(
        np.sqrt(np.mean((Ex_pert - Ex_eq) ** 2 + (Ey_pert - Ey_eq) ** 2))
    )
    neutrality = neutrality_diagnostics(rho_equilibrium, grid)
    residual_ratio = residual_rms / max(signal_rms, 1.0e-300)
    passed = bool(
        neutrality["relative_integrated_charge"] <= args.neutrality_tolerance
        and residual_ratio <= args.equilibrium_residual_fraction
    )
    return {
        **neutrality,
        "residual_field_rms": residual_rms,
        "perturbation_field_rms": signal_rms,
        "residual_to_perturbation": residual_ratio,
        "neutrality_tolerance": args.neutrality_tolerance,
        "residual_fraction_tolerance": args.equilibrium_residual_fraction,
        "passed": passed,
    }


def spatial_masks(
    grid: Grid2D,
    profile: PeriodicFloorVonMisesProfile,
    args: argparse.Namespace,
) -> dict:
    density_x = profile.density(grid.x)
    density_2d = density_x[:, None] * np.ones((1, grid.Ny))
    low = np.quantile(density_x, 0.2)
    high = np.quantile(density_x, 0.8)
    displacement = periodic_displacement(grid.x, profile.shoulder_x, grid.Lx)
    shoulder_x = np.abs(displacement) <= args.shoulder_width
    return {
        "floor": density_2d <= low,
        "peak": density_2d >= high,
        "shoulder": shoulder_x[:, None] * np.ones((1, grid.Ny), dtype=bool),
    }


def protected_basis(
    grid: Grid2D,
    profile: PeriodicFloorVonMisesProfile,
    args: argparse.Namespace,
) -> np.ndarray:
    displacement = periodic_displacement(grid.x, profile.shoulder_x, grid.Lx)
    envelope = np.exp(-0.5 * (displacement / args.shoulder_width) ** 2)
    k = 2.0 * np.pi * args.carrier_mode / grid.Lx
    basis_x = envelope * np.cos(k * displacement)
    basis_x -= np.mean(basis_x)
    return basis_x[:, None] * np.ones((1, grid.Ny))


def trace_metrics(
    trace: dict,
    reference: dict,
    background: ImmobileIonBackground,
    masks: dict,
    basis: np.ndarray,
) -> dict:
    np.testing.assert_allclose(trace["times"], reference["times"], rtol=0.0, atol=0.0)
    density_error = (trace["electron_density"] - reference["electron_density"]) ** 2
    field_error = (
        (trace["Ex"] - reference["Ex"]) ** 2
        + (trace["Ey"] - reference["Ey"]) ** 2
    )
    total = trace["electron_density"] + background.rho[None, :, :]
    total_reference = reference["electron_density"] + background.rho[None, :, :]
    denominator = float(np.sum(basis**2))
    amplitude = np.sum(total * basis[None, :, :], axis=(1, 2)) / denominator
    amplitude_reference = (
        np.sum(total_reference * basis[None, :, :], axis=(1, 2)) / denominator
    )
    protected_error = float(
        np.sum((amplitude - amplitude_reference) ** 2)
        / max(np.sum(amplitude_reference**2), 1.0e-300)
    )
    integrated_charge = (
        np.sum(total, axis=(1, 2)) * background.grid.dx * background.grid.dy
    )
    result = {
        "density_mise": float(np.mean(density_error)),
        "field_mse": float(np.mean(field_error)),
        "protected_relative_l2": protected_error,
        "initial_total_charge": float(
            np.sum(total[0]) * background.grid.dx * background.grid.dy
        ),
        "max_relative_total_charge": float(
            np.max(np.abs(integrated_charge)) / background.total_charge
        ),
    }
    for name, mask in masks.items():
        result[f"{name}_density_mse"] = float(np.mean(density_error[:, mask]))
    return result


def simulate(args: argparse.Namespace) -> dict:
    args.output.mkdir(parents=True, exist_ok=True)
    grid = Grid2D(args.Nx, args.Ny, args.Lx, args.Ly)
    profile = PeriodicFloorVonMisesProfile(
        grid.Lx,
        args.dynamic_range,
        args.profile_kappa,
    )
    background = ImmobileIonBackground(
        ion_background_on_grid(profile, grid.x, grid.Ny),
        grid,
    )
    gate = equilibrium_gate(grid, profile, background, args)
    if not gate["passed"]:
        raise RuntimeError(f"H1 equilibrium gate failed: {gate}")

    reference_particles = build_particles(
        grid,
        profile,
        args.reference_ppc,
        args.seed,
        args,
    )
    reference_deposit, reference_gather = cic_methods()
    reference = run_trace(
        reference_particles,
        grid,
        background,
        reference_deposit,
        reference_gather,
        args,
    )
    reference_profile_change = float(
        np.linalg.norm(
            reference["electron_density"][-1] - reference["electron_density"][0]
        )
        / max(np.linalg.norm(reference["electron_density"][0]), 1.0e-300)
    )
    if reference_profile_change > args.max_reference_profile_change:
        raise RuntimeError(
            "H1 comparison-window gate failed: reference profile changed by "
            f"{reference_profile_change:.3%}"
        )
    base_particles = build_particles(
        grid,
        profile,
        args.ppc,
        args.seed,
        args,
    )
    masks = spatial_masks(grid, profile, args)
    basis = protected_basis(grid, profile, args)
    rows = []
    for method in METHODS:
        particles = copy_particles(base_particles)
        deposit, gather, width_info = make_method(
            method,
            particles,
            grid,
            profile,
            args,
        )
        trace = run_trace(particles, grid, background, deposit, gather, args)
        rows.append(
            {
                "dynamic_range": args.dynamic_range,
                "seed": args.seed,
                "method": method,
                **trace_metrics(trace, reference, background, masks, basis),
                **width_info,
            }
        )

    result = {
        "status": "complete",
        "dynamic_range": args.dynamic_range,
        "seed": args.seed,
        "rows": rows,
        "equilibrium_gate": gate,
        "comparison_window_gate": {
            "reference_profile_relative_change": reference_profile_change,
            "maximum_allowed": args.max_reference_profile_change,
            "passed": True,
        },
        "profile": {
            "kappa": profile.kappa,
            "peak_to_floor": profile.peak_to_floor,
            "shoulder_x": profile.shoulder_x,
        },
        "protocol": {
            "ppc": args.ppc,
            "reference_ppc": args.reference_ppc,
            "equilibrium_ppc": args.equilibrium_ppc,
            "alpha": args.alpha,
            "carrier_mode": args.carrier_mode,
            "t_end": args.t_end,
            "dt": args.dt,
            "width_policy": "fixed_initial_pilot",
        },
    }
    (args.output / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return result


def bootstrap_mean(values: np.ndarray, nboot: int, seed: int) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, values.size, size=(nboot, values.size))
    means = np.mean(values[indices], axis=1)
    return (
        float(np.mean(values)),
        float(np.percentile(means, 2.5)),
        float(np.percentile(means, 97.5)),
    )


def bootstrap_paired_ratio(
    numerator: np.ndarray,
    denominator: np.ndarray,
    nboot: int,
    seed: int,
) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, numerator.size, size=(nboot, numerator.size))
    ratios = np.mean(numerator[indices], axis=1) / np.mean(
        denominator[indices],
        axis=1,
    )
    return (
        float(np.mean(numerator) / np.mean(denominator)),
        float(np.percentile(ratios, 2.5)),
        float(np.percentile(ratios, 97.5)),
    )


def write_csv(path: pathlib.Path, rows: list[dict]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def reduce(args: argparse.Namespace) -> dict:
    args.output.mkdir(parents=True, exist_ok=True)
    seed_rows = []
    gate_rows = []
    for path in sorted(args.input.glob("D*/seed*/result.json")):
        result = json.loads(path.read_text(encoding="utf-8"))
        seed_rows.extend(result["rows"])
        gate_rows.append(
            {
                "dynamic_range": result["dynamic_range"],
                "seed": result["seed"],
                **result["equilibrium_gate"],
                "reference_profile_relative_change": result[
                    "comparison_window_gate"
                ]["reference_profile_relative_change"],
                "comparison_window_passed": result["comparison_window_gate"][
                    "passed"
                ],
            }
        )
    expected = len(args.dynamic_ranges) * args.seeds * len(METHODS)
    if len(seed_rows) != expected:
        raise RuntimeError(f"incomplete H1 ensemble: found {len(seed_rows)}, expected {expected}")
    if not all(
        row["passed"] and row["comparison_window_passed"]
        for row in gate_rows
    ):
        raise RuntimeError("one or more H1 equilibrium gates failed")
    if any(
        row["max_relative_total_charge"] > args.neutrality_tolerance
        for row in seed_rows
    ):
        raise RuntimeError("one or more H1 trajectories violated charge neutrality")

    metrics = (
        "density_mise",
        "floor_density_mse",
        "peak_density_mse",
        "shoulder_density_mse",
        "field_mse",
        "protected_relative_l2",
    )
    summary = []
    ratios = []
    for dynamic_range in args.dynamic_ranges:
        for method in METHODS:
            selected = [
                row
                for row in seed_rows
                if row["dynamic_range"] == dynamic_range and row["method"] == method
            ]
            summary_row = {
                "dynamic_range": dynamic_range,
                "method": method,
                "n_seeds": len(selected),
            }
            for metric in metrics:
                values = np.asarray([row[metric] for row in selected])
                mean, lo, hi = bootstrap_mean(
                    values,
                    args.nboot,
                    int(51000 + dynamic_range + len(method) + len(metric)),
                )
                summary_row[f"{metric}_mean"] = mean
                summary_row[f"{metric}_ci_lo"] = lo
                summary_row[f"{metric}_ci_hi"] = hi
            summary.append(summary_row)

        global_rows = sorted(
            (
                row
                for row in seed_rows
                if row["dynamic_range"] == dynamic_range
                and row["method"] == "global_c4"
            ),
            key=lambda row: row["seed"],
        )
        local_rows = sorted(
            (
                row
                for row in seed_rows
                if row["dynamic_range"] == dynamic_range
                and row["method"] == "abramson_c4_capped"
            ),
            key=lambda row: row["seed"],
        )
        ratio_row = {"dynamic_range": dynamic_range}
        for metric in metrics:
            local_values = np.asarray([row[metric] for row in local_rows])
            global_values = np.asarray([row[metric] for row in global_rows])
            value, lo, hi = bootstrap_paired_ratio(
                local_values,
                global_values,
                args.nboot,
                int(61000 + dynamic_range + len(metric)),
            )
            ratio_row[f"{metric}_local_over_global"] = value
            ratio_row[f"{metric}_ratio_ci_lo"] = lo
            ratio_row[f"{metric}_ratio_ci_hi"] = hi
        ratios.append(ratio_row)

    ratio_by_D = {row["dynamic_range"]: row for row in ratios}
    high = ratio_by_D[100.0]
    low = ratio_by_D[10.0]
    locality_passed = bool(
        high["density_mise_ratio_ci_hi"] < 0.90
        and high["shoulder_density_mse_ratio_ci_hi"] < 0.90
        and high["protected_relative_l2_ratio_ci_hi"] <= 1.10
        and low["density_mise_ratio_ci_hi"] <= 1.10
    )
    result = {
        "status": "complete",
        "locality_gate_passed": locality_passed,
        "decision": (
            "BUILD_LEARNED_RUNG"
            if locality_passed
            else "STOP_LEARNING_ANALYTIC_H1_NEGATIVE"
        ),
        "pre_registered_gate": {
            "D100_density_ratio_ci_hi_lt": 0.90,
            "D100_shoulder_ratio_ci_hi_lt": 0.90,
            "D100_protected_ratio_ci_hi_le": 1.10,
            "D10_density_ratio_ci_hi_le": 1.10,
        },
        "ratios": ratios,
    }
    write_csv(args.output / "H1_seed_metrics.csv", seed_rows)
    write_csv(args.output / "H1_summary.csv", summary)
    write_csv(args.output / "H1_local_over_global.csv", ratios)
    write_csv(args.output / "H1_equilibrium_gates.csv", gate_rows)
    (args.output / "H1_result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return result


def add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--Nx", type=int, default=64)
    parser.add_argument("--Ny", type=int, default=16)
    parser.add_argument("--Lx", type=float, default=16.0)
    parser.add_argument("--Ly", type=float, default=8.0)
    parser.add_argument("--ppc", type=int, default=32)
    parser.add_argument("--reference-ppc", type=int, default=256)
    parser.add_argument("--equilibrium-ppc", type=int, default=1024)
    parser.add_argument("--v-thermal", type=float, default=0.2)
    parser.add_argument("--alpha", type=float, default=0.01)
    parser.add_argument("--carrier-mode", type=int, default=4)
    parser.add_argument("--shoulder-width", type=float, default=1.6)
    parser.add_argument("--profile-kappa", type=float, default=4.0)
    parser.add_argument("--sigma-c4-dx", type=float, default=1.0)
    parser.add_argument("--sigma-min-dx", type=float, default=0.1)
    parser.add_argument("--width-bins", type=int, default=24)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--t-end", type=float, default=1.0)
    parser.add_argument("--diag-stride", type=int, default=2)
    parser.add_argument("--neutrality-tolerance", type=float, default=1.0e-10)
    parser.add_argument("--equilibrium-residual-fraction", type=float, default=0.10)
    parser.add_argument("--max-reference-profile-change", type=float, default=0.05)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    simulate_parser = subparsers.add_parser("simulate")
    simulate_parser.add_argument("--output", type=pathlib.Path, required=True)
    simulate_parser.add_argument("--dynamic-range", type=float, required=True)
    simulate_parser.add_argument("--seed", type=int, required=True)
    add_common(simulate_parser)

    reduce_parser = subparsers.add_parser("reduce")
    reduce_parser.add_argument("--input", type=pathlib.Path, required=True)
    reduce_parser.add_argument("--output", type=pathlib.Path, required=True)
    reduce_parser.add_argument("--dynamic-ranges", default="10,100")
    reduce_parser.add_argument("--seeds", type=int, default=10)
    reduce_parser.add_argument("--nboot", type=int, default=2000)
    reduce_parser.add_argument("--neutrality-tolerance", type=float, default=1.0e-10)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "simulate":
        simulate(args)
    else:
        args.dynamic_ranges = [
            float(value) for value in args.dynamic_ranges.split(",") if value
        ]
        reduce(args)


if __name__ == "__main__":
    main()
