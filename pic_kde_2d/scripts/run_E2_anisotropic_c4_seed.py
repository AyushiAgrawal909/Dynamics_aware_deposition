#!/usr/bin/env python3
"""Run one paired seed of the E2 anisotropic C4 noise and transverse audit."""

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
from pic_kde_2d.core.pic_loop import pic_step
from pic_kde_2d.methods.cic import deposit_cic_2d
from pic_kde_2d.methods.spectral_W import apply_W_to_grid, make_W_gaussian
from pic_kde_2d.methods.spectral_W_wrappers import make_spectral_W_gaussian_methods


def parse_float_list(text: str) -> list[float]:
    """Parse a comma-separated list of finite nonnegative widths."""
    values = [float(item.strip()) for item in text.split(",") if item.strip()]
    if not values or any(not np.isfinite(value) or value < 0.0 for value in values):
        raise ValueError("sigma-y-dy must contain finite nonnegative values")
    return values


def write_csv(path: pathlib.Path, rows: list[dict]) -> None:
    """Write dictionaries with a stable union of keys."""
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def make_particles(ppc: int, seed: int, args: argparse.Namespace):
    """Construct the frozen global-shuffle E2 particle ensemble."""
    nx_marker, ny_marker = marker_grid_shape(args.Nx, args.Ny, ppc)
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


def cic_density(particles, grid: Grid2D) -> np.ndarray:
    """Deposit the unfiltered electron density."""
    return deposit_cic_2d(
        particles.x,
        particles.y,
        particles.w,
        grid.Nx,
        grid.Ny,
        grid.dx,
        grid.dy,
    )


def rfft_parseval_components(error: np.ndarray) -> tuple[float, float]:
    """Return mean-square error in ky=0 and ky!=0 spectral sectors."""
    spectrum = np.fft.rfft2(error)
    weights = np.full(spectrum.shape[1], 2.0, dtype=np.float64)
    weights[0] = 1.0
    if error.shape[1] % 2 == 0:
        weights[-1] = 1.0
    power = np.abs(spectrum) ** 2 * weights[np.newaxis, :]
    normalization = float(np.prod(error.shape) ** 2)
    return (
        float(np.sum(power[:, 0]) / normalization),
        float(np.sum(power[:, 1:]) / normalization),
    )


def background_neutrality(rho: np.ndarray) -> float:
    """Relative k=0 defect after applying the uniform ion background."""
    rhs = rho - float(np.mean(rho))
    return float(abs(np.sum(rhs)) / max(np.sum(np.abs(rhs)), np.finfo(float).tiny))


def relative_charge_error(rho: np.ndarray, particles, grid: Grid2D) -> float:
    """Check that filtering preserves the assembled particle charge."""
    deposited = float(np.sum(rho) * grid.dx * grid.dy)
    expected = float(particles.total_charge())
    return float(abs(deposited - expected) / max(abs(expected), np.finfo(float).tiny))


def field_amplitude_spectrum(
    rho: np.ndarray, grid: Grid2D
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return field-amplitude spectrum and real-space electric field."""
    Ex, Ey = grid.poisson_solve(rho)
    norm = float(grid.Nx * grid.Ny)
    Ex_hat = np.fft.rfft2(Ex) / norm
    Ey_hat = np.fft.rfft2(Ey) / norm
    amplitude = np.sqrt(np.abs(Ex_hat) ** 2 + np.abs(Ey_hat) ** 2)
    return amplitude.astype(np.float64), Ex, Ey


def measure_density_noise(
    grid: Grid2D,
    sigma_y_values: list[float],
    args: argparse.Namespace,
) -> list[dict]:
    """Measure anisotropic density error on the frozen matrix trajectory."""
    low = make_particles(args.ppc, args.seed, args)
    reference = make_particles(args.reference_ppc, args.seed, args)
    live_sigma_x = args.live_sigma_dx * grid.dx
    live_sigma_y = args.live_sigma_dx * grid.dy
    deposit_live, gather_live = make_spectral_W_gaussian_methods(
        grid, live_sigma_x, live_sigma_y
    )
    filters = {
        sigma_y: make_W_gaussian(
            grid,
            args.sigma_x_dx * grid.dx,
            sigma_y * grid.dy,
        )
        for sigma_y in sigma_y_values
    }
    rows_by_width = {
        sigma_y: {
            "mise": [],
            "ky0_error_mse": [],
            "ky_nonzero_error_mse": [],
            "x_profile_mse": [],
            "neutrality": [],
            "charge_error": [],
        }
        for sigma_y in sigma_y_values
    }
    cic_values = {
        "mise": [],
        "ky0_error_mse": [],
        "ky_nonzero_error_mse": [],
        "x_profile_mse": [],
        "neutrality": [],
        "charge_error": [],
    }

    for step in range(args.n_steps + 1):
        time = step * args.dt
        record = (
            step % args.diag_stride == 0
            and args.eval_t_min <= time <= args.eval_t_max
        )
        if record:
            rho_low = cic_density(low, grid)
            rho_ref = cic_density(reference, grid)
            for values, rho_candidate in [
                (cic_values, rho_low),
                *[
                    (
                        rows_by_width[sigma_y],
                        apply_W_to_grid(rho_low, filters[sigma_y], grid),
                    )
                    for sigma_y in sigma_y_values
                ],
            ]:
                error = rho_candidate - rho_ref
                ky0_mse, ky_nonzero_mse = rfft_parseval_components(error)
                profile_error = np.mean(rho_candidate, axis=1) - np.mean(
                    rho_ref, axis=1
                )
                values["mise"].append(float(np.mean(error**2)))
                values["ky0_error_mse"].append(ky0_mse)
                values["ky_nonzero_error_mse"].append(ky_nonzero_mse)
                values["x_profile_mse"].append(float(np.mean(profile_error**2)))
                values["neutrality"].append(background_neutrality(rho_candidate))
                values["charge_error"].append(
                    relative_charge_error(rho_candidate, low, grid)
                )

        if step == args.n_steps:
            break
        pic_step(low, grid, args.dt, deposit_live, gather_live)
        pic_step(reference, grid, args.dt, deposit_live, gather_live)

    def summarize(values: dict, sigma_x_dx: float, sigma_y_dy: float) -> dict:
        return {
            "seed": args.seed,
            "sigma_x_dx": sigma_x_dx,
            "sigma_y_dy": sigma_y_dy,
            "is_cic": sigma_x_dx == 0.0 and sigma_y_dy == 0.0,
            "mise_time_avg": float(np.mean(values["mise"])),
            "ky0_error_mse_time_avg": float(np.mean(values["ky0_error_mse"])),
            "ky_nonzero_error_mse_time_avg": float(
                np.mean(values["ky_nonzero_error_mse"])
            ),
            "x_profile_mse_time_avg": float(np.mean(values["x_profile_mse"])),
            "max_neutrality_defect": float(np.max(values["neutrality"])),
            "max_relative_charge_error": float(np.max(values["charge_error"])),
            "n_times": len(values["mise"]),
        }

    rows = [summarize(cic_values, 0.0, 0.0)]
    rows.extend(
        summarize(rows_by_width[sigma_y], args.sigma_x_dx, sigma_y)
        for sigma_y in sigma_y_values
    )
    return rows


def run_live_transverse_audit(
    grid: Grid2D,
    sigma_y_values: list[float],
    args: argparse.Namespace,
) -> dict[str, np.ndarray]:
    """Run the actual anisotropic method and retain every resolved field mode."""
    times = []
    spectra = []
    profiles = []
    neutrality = []
    charge_error = []
    ky_power_fraction = []

    for sigma_y in sigma_y_values:
        particles = make_particles(args.ppc, args.seed, args)
        deposit, gather = make_spectral_W_gaussian_methods(
            grid,
            args.sigma_x_dx * grid.dx,
            sigma_y * grid.dy,
        )
        width_times = []
        width_spectra = []
        width_profiles = []
        width_neutrality = []
        width_charge_error = []
        width_ky_fraction = []

        for step in range(args.n_steps + 1):
            if step % args.diag_stride == 0:
                rho = deposit(particles, grid)
                rhs = rho - float(np.mean(rho))
                amplitude, _, _ = field_amplitude_spectrum(rhs, grid)
                total_power = float(np.sum(amplitude[:, 1:] ** 2))
                all_power = float(np.sum(amplitude**2))
                width_times.append(step * args.dt)
                width_spectra.append(amplitude)
                width_profiles.append(np.mean(rhs, axis=1))
                width_neutrality.append(background_neutrality(rho))
                width_charge_error.append(relative_charge_error(rho, particles, grid))
                width_ky_fraction.append(
                    total_power / max(all_power, np.finfo(float).tiny)
                )
            if step == args.n_steps:
                break
            pic_step(particles, grid, args.dt, deposit, gather)

        times.append(np.asarray(width_times, dtype=np.float64))
        spectra.append(np.asarray(width_spectra, dtype=np.float64))
        profiles.append(np.asarray(width_profiles, dtype=np.float64))
        neutrality.append(np.asarray(width_neutrality, dtype=np.float64))
        charge_error.append(np.asarray(width_charge_error, dtype=np.float64))
        ky_power_fraction.append(np.asarray(width_ky_fraction, dtype=np.float64))

    for other in times[1:]:
        np.testing.assert_allclose(times[0], other, rtol=0.0, atol=0.0)
    return {
        "times": times[0],
        "sigma_y_dy": np.asarray(sigma_y_values, dtype=np.float64),
        "field_amplitude": np.stack(spectra),
        "x_profiles": np.stack(profiles),
        "neutrality_defect": np.stack(neutrality),
        "relative_charge_error": np.stack(charge_error),
        "ky_power_fraction": np.stack(ky_power_fraction),
    }


def run(args: argparse.Namespace) -> dict:
    """Execute one paired noise shard and one live transverse audit."""
    args.output.mkdir(parents=True, exist_ok=True)
    sigma_y_values = parse_float_list(args.sigma_y_dy)
    if not any(np.isclose(value, args.sigma_x_dx) for value in sigma_y_values):
        raise ValueError("sigma-y-dy must include the isotropic sigma_y=sigma_x case")
    grid = Grid2D(Nx=args.Nx, Ny=args.Ny, Lx=args.Lx, Ly=args.Ly)
    noise_rows = measure_density_noise(grid, sigma_y_values, args)
    write_csv(args.output / "anisotropic_noise_seed.csv", noise_rows)
    audit = run_live_transverse_audit(grid, sigma_y_values, args)
    np.savez_compressed(args.output / "anisotropic_live_audit.npz", **audit)

    result = {
        "seed": args.seed,
        "sigma_x_dx": args.sigma_x_dx,
        "sigma_y_dy": sigma_y_values,
        "ppc": args.ppc,
        "reference_ppc": args.reference_ppc,
        "live_density_reference_sigma_dx": args.live_sigma_dx,
        "grid": {
            "Nx": grid.Nx,
            "Ny": grid.Ny,
            "Lx": grid.Lx,
            "Ly": grid.Ly,
            "dx": grid.dx,
            "dy": grid.dy,
        },
        "e2": {
            "kx_mode": args.kx_mode,
            "v_beam": args.v_beam,
            "v_thermal": args.v_thermal,
            "alpha": args.alpha,
            "dt": args.dt,
            "n_steps": args.n_steps,
            "eval_window": [args.eval_t_min, args.eval_t_max],
        },
        "noise_rows": noise_rows,
        "audit_scope": (
            "actual anisotropic matched deposit/gather dynamics; all resolved "
            "ky modes retained"
        ),
    }
    (args.output / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sigma-x-dx", type=float, default=0.18445488487247555)
    parser.add_argument(
        "--sigma-y-dy",
        default="0.18445488487247555,1,2,4",
    )
    parser.add_argument("--ppc", type=int, default=32)
    parser.add_argument("--reference-ppc", type=int, default=1024)
    parser.add_argument("--Nx", type=int, default=128)
    parser.add_argument("--Ny", type=int, default=16)
    parser.add_argument("--Lx", type=float, default=64.0)
    parser.add_argument("--Ly", type=float, default=16.0)
    parser.add_argument("--kx-mode", type=int, default=12)
    parser.add_argument("--v-beam", type=float, default=0.8)
    parser.add_argument("--v-thermal", type=float, default=0.05)
    parser.add_argument("--alpha", type=float, default=0.001)
    parser.add_argument("--live-sigma-dx", type=float, default=0.25)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--n-steps", type=int, default=80)
    parser.add_argument("--diag-stride", type=int, default=2)
    parser.add_argument("--eval-t-min", type=float, default=0.5)
    parser.add_argument("--eval-t-max", type=float, default=4.0)
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
