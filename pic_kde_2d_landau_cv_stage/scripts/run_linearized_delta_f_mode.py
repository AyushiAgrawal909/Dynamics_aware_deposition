#!/usr/bin/env python3
"""Run one linearized delta-f mode on a frozen Landau or E2 equilibrium."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pic_kde_2d.benchmarks.theory import landau_gamma_theory, langmuir_omega_r
from pic_kde_2d.conventions import landau_box_length, marker_grid_shape, thermal_velocity
from pic_kde_2d.core.grid2d import Grid2D
from pic_kde_2d.core.particles2d import init_landau_2d, init_two_stream_2d
from pic_kde_2d.core.pic_loop_delta_f import run_simulation_linearized_delta_f
from pic_kde_2d.methods.delta_f import init_delta_f_from_particles, make_delta_f_methods
from pic_kde_2d.methods.spectral_W import (
    make_W_binomial,
    make_W_compensated_binomial,
    make_W_gaussian,
)


def resolved_sigma_cells(args: argparse.Namespace) -> tuple[float, float]:
    """Return directional widths while preserving the isotropic CLI."""
    sigma_x_dx = args.sigma_dx if args.sigma_x_dx is None else args.sigma_x_dx
    sigma_y_dy = args.sigma_dx if args.sigma_y_dy is None else args.sigma_y_dy
    if sigma_x_dx < 0.0 or sigma_y_dy < 0.0:
        raise ValueError("Gaussian widths must be nonnegative")
    return float(sigma_x_dx), float(sigma_y_dy)


def project_density_to_mode(
    rho: np.ndarray,
    grid: Grid2D,
    mode_x: int,
    mode_y: int = 0,
) -> np.ndarray:
    """Keep only the real conjugate Fourier pair for ``(mode_x, mode_y)``."""
    rho_hat = np.fft.fft2(rho)
    projected = np.zeros_like(rho_hat)
    positive = (mode_x % grid.Nx, mode_y % grid.Ny)
    negative = ((-mode_x) % grid.Nx, (-mode_y) % grid.Ny)
    projected[positive] = rho_hat[positive]
    projected[negative] = rho_hat[negative]
    return np.fft.ifft2(projected).real


def build_markers(args: argparse.Namespace) -> tuple[Grid2D, object]:
    rng = np.random.default_rng(args.seed)
    if args.benchmark == "landau":
        grid = Grid2D(
            Nx=args.Nx,
            Ny=args.Ny,
            Lx=landau_box_length(args.mode, args.k_lambdaD),
            Ly=args.Ly,
        )
        nx_marker, ny_marker = marker_grid_shape(grid.Nx, grid.Ny, args.ppc)
        base = init_landau_2d(
            Nx_marker=nx_marker,
            Ny_marker=ny_marker,
            Lx=grid.Lx,
            Ly=grid.Ly,
            v_thermal=thermal_velocity(),
            alpha=0.0,
            kx_mode=args.mode,
            ky_mode=args.ky_mode,
            velocity_perturbation=False,
            rng=rng,
        )
        kx = 2.0 * np.pi * args.mode / grid.Lx
        ky = 2.0 * np.pi * args.ky_mode / grid.Ly
        particles = init_delta_f_from_particles(
            base,
            f0_v_thermal=thermal_velocity(),
            f0_density=1.0,
            alpha=args.alpha,
            kx_phys=kx,
            ky_phys=ky,
            omega_real=langmuir_omega_r(args.k_lambdaD),
            gamma=landau_gamma_theory(args.k_lambdaD),
        )
        return grid, particles

    grid = Grid2D(Nx=args.Nx, Ny=args.Ny, Lx=args.Lx, Ly=args.Ly)
    nx_marker, ny_marker = marker_grid_shape(grid.Nx, grid.Ny, args.ppc)
    base = init_two_stream_2d(
        Nx_marker=nx_marker,
        Ny_marker=ny_marker,
        Lx=grid.Lx,
        Ly=grid.Ly,
        v_beam=args.v_beam,
        v_thermal=args.v_thermal,
        alpha=0.0,
        kx_mode=args.mode,
        ky_mode=args.ky_mode,
        rng=rng,
    )
    kx = 2.0 * np.pi * args.mode / grid.Lx
    ky = 2.0 * np.pi * args.ky_mode / grid.Ly
    particles = init_delta_f_from_particles(
        base,
        f0_v_thermal=args.v_thermal,
        f0_density=1.0,
        alpha=args.alpha,
        kx_phys=kx,
        ky_phys=ky,
        background_kind="two_stream",
        beam_velocity=args.v_beam,
    )
    return grid, particles


def run(args: argparse.Namespace) -> dict:
    grid, particles = build_markers(args)
    if args.filter_kind in ("binomial", "compensated_binomial"):
        passes_x = args.binomial_passes_x
        passes_y = passes_x if args.binomial_passes_y is None else args.binomial_passes_y
        filter_factory = (
            make_W_compensated_binomial
            if args.filter_kind == "compensated_binomial"
            else make_W_binomial
        )
        if args.filter_kind == "compensated_binomial":
            W = filter_factory(
                grid,
                passes_x,
                passes_y,
                compensation_order=args.compensation_order,
            )
        else:
            W = filter_factory(grid, passes_x, passes_y)
        if args.filter_kind == "binomial":
            sigma_x_dx = float(np.sqrt(passes_x / 2.0))
            sigma_y_dy = float(np.sqrt(passes_y / 2.0))
        else:
            sigma_x_dx = None
            sigma_y_dy = None
    else:
        passes_x = None
        passes_y = None
        sigma_x_dx, sigma_y_dy = resolved_sigma_cells(args)
        if sigma_x_dx > 0.0 or sigma_y_dy > 0.0:
            W = make_W_gaussian(
                grid,
                sigma_x_dx * grid.dx,
                sigma_y_dy * grid.dy,
            )
        else:
            W = None
    deposit_unprojected, gather = make_delta_f_methods(W=W)

    if args.project_target_mode:
        def deposit(particles_here, grid_here):
            return project_density_to_mode(
                deposit_unprojected(particles_here, grid_here),
                grid_here,
                args.mode,
                args.ky_mode,
            )
    else:
        deposit = deposit_unprojected
    n_steps = int(round(args.t_end / args.dt))
    history = run_simulation_linearized_delta_f(
        particles,
        grid,
        args.dt,
        n_steps,
        deposit,
        gather,
        diag_stride=args.diag_stride,
        tracked_modes=[(args.mode, args.ky_mode)],
        validity_threshold=args.validity_threshold,
    )
    times = np.asarray(history.times, dtype=np.float64)
    mode = np.asarray(
        history.mode_amplitudes_Ex[(args.mode, args.ky_mode)],
        dtype=np.complex128,
    )
    validity = np.asarray(history.delta_f_ratio, dtype=np.float64)
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output / "mode_history.npz",
        times=times,
        mode=mode,
        delta_f_ratio=validity,
        seed=np.asarray([args.seed], dtype=np.int64),
    )
    result = {
        "benchmark": args.benchmark,
        "mode": args.mode,
        "kx_mode": args.mode,
        "ky_mode": args.ky_mode,
        "seed": args.seed,
        "ppc": args.ppc,
        "alpha": args.alpha,
        "filter_kind": args.filter_kind,
        "sigma_dx": (
            sigma_x_dx
            if args.filter_kind == "gaussian" and sigma_x_dx == sigma_y_dy
            else None
        ),
        "sigma_x_dx": sigma_x_dx,
        "sigma_y_dy": sigma_y_dy,
        "binomial_passes_x": passes_x,
        "binomial_passes_y": passes_y,
        "compensation_order": (
            args.compensation_order
            if args.filter_kind == "compensated_binomial"
            else None
        ),
        "dt": args.dt,
        "t_end": args.t_end,
        "max_delta_f_ratio": float(np.max(validity)),
        "validity_threshold": args.validity_threshold,
        "valid_through_run": bool(np.max(validity) < args.validity_threshold),
        "background": particles.background_kind,
        "marker_dynamics": "free_streaming_frozen_equilibrium",
        "target_mode_projection": args.project_target_mode,
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
    parser.add_argument("--benchmark", choices=("landau", "two_stream"), required=True)
    parser.add_argument("--mode", type=int, required=True)
    parser.add_argument(
        "--ky-mode",
        type=int,
        default=0,
        help="nonnegative transverse Fourier index paired with --mode",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ppc", type=int, default=32)
    parser.add_argument("--alpha", type=float, default=1.0e-4)
    parser.add_argument("--sigma-dx", type=float, default=0.0)
    parser.add_argument(
        "--filter-kind",
        choices=("gaussian", "binomial", "compensated_binomial"),
        default="gaussian",
    )
    parser.add_argument("--binomial-passes-x", type=int, default=0)
    parser.add_argument("--binomial-passes-y", type=int, default=None)
    parser.add_argument("--compensation-order", type=int, default=1)
    parser.add_argument(
        "--sigma-x-dx",
        type=float,
        default=None,
        help="x width in x-cell units; overrides --sigma-dx",
    )
    parser.add_argument(
        "--sigma-y-dy",
        type=float,
        default=None,
        help="y width in y-cell units; overrides --sigma-dx",
    )
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--t-end", type=float, default=30.0)
    parser.add_argument("--diag-stride", type=int, default=2)
    parser.add_argument("--validity-threshold", type=float, default=0.5)
    parser.add_argument(
        "--project-target-mode",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="isolate the target Fourier pair in the linear field solve",
    )
    parser.add_argument("--Nx", type=int, default=64)
    parser.add_argument("--Ny", type=int, default=16)
    parser.add_argument("--Lx", type=float, default=64.0)
    parser.add_argument("--Ly", type=float, default=8.0)
    parser.add_argument("--k-lambdaD", type=float, default=0.4)
    parser.add_argument("--v-beam", type=float, default=0.8)
    parser.add_argument("--v-thermal", type=float, default=0.05)
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
