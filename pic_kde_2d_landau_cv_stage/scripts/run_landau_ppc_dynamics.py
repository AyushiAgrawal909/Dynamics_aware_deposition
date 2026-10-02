#!/usr/bin/env python3
"""Run one weak-Landau PPC/selector/seed dynamics shard."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pic_kde_2d.benchmarks.theory import landau_omega_theory
from pic_kde_2d.conventions import landau_box_length, marker_grid_shape
from pic_kde_2d.core.grid2d import Grid2D
from pic_kde_2d.core.particles2d import init_landau_2d_eigenmode
from pic_kde_2d.core.pic_loop import run_simulation
from pic_kde_2d.methods.spectral_W_wrappers import make_spectral_W_gaussian_methods


def selected_width(selector_root: pathlib.Path, ppc: int, choice: str) -> tuple[float, str]:
    if choice == "cic":
        return 0.0, "fixed CIC"
    if choice == "c4":
        return 1.0, "final CI-supported C4"
    result = json.loads((selector_root / f"ppc{ppc}" / "selector_result.json").read_text())
    key = "mise" if choice == "mise" else "lscv"
    selected = result[key]
    if selected["status"] == "interior":
        return float(selected["quadratic_width_dx"]), f"direct {key} interior minimum"
    if selected["status"] == "right_censored":
        return float(result["sigma_grid_dx"][-1]), f"direct {key} right-censored bound"
    return float(result["sigma_grid_dx"][0]), f"direct {key} left-censored bound"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--selector-root", type=pathlib.Path, required=True)
    parser.add_argument("--ppc", type=int, required=True)
    parser.add_argument("--choice", choices=("cic", "c4", "mise", "lscv"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--n-steps", type=int, default=1800)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--diag-stride", type=int, default=2)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    grid = Grid2D(64, 16, landau_box_length(1, 0.4), 8.0)
    nx_marker, ny_marker = marker_grid_shape(grid.Nx, grid.Ny, args.ppc)
    particles = init_landau_2d_eigenmode(
        Nx_marker=nx_marker,
        Ny_marker=ny_marker,
        Lx=grid.Lx,
        Ly=grid.Ly,
        v_thermal=1.0,
        alpha=0.01,
        kx_mode=1,
        omega_r=landau_omega_theory(0.4, 1.0),
        rng=np.random.default_rng(args.seed),
        grid_shape=None,
        position_jitter=False,
    )
    sigma, source = selected_width(args.selector_root, args.ppc, args.choice)
    deposit, gather = make_spectral_W_gaussian_methods(
        grid, sigma * grid.dx, sigma * grid.dy
    )
    history = run_simulation(
        particles,
        grid,
        args.dt,
        args.n_steps,
        deposit,
        gather,
        diag_stride=args.diag_stride,
        tracked_modes=[(1, 0)],
    )
    arrays = history.to_arrays()
    np.savez_compressed(
        args.output / "history.npz",
        times=arrays["times"],
        mode=arrays["Ex_mode_1_0"],
        field_energy=arrays["field_energy"],
        kinetic_energy=arrays["kinetic_energy"],
        total_energy=arrays["total_energy"],
    )
    metadata = {
        "ppc": args.ppc,
        "choice": args.choice,
        "seed": args.seed,
        "sigma_dx": sigma,
        "width_source": source,
        "alpha": 0.01,
        "quiet_start": "regular positions, globally shuffled velocity quadrature",
        "dt": args.dt,
        "n_steps": args.n_steps,
        "diag_stride": args.diag_stride,
    }
    (args.output / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
