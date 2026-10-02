#!/usr/bin/env python3
"""Generate evolving weak-Landau density/LSCV snapshots for one PPC seed."""

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
from pic_kde_2d.core.pic_loop import pic_step
from pic_kde_2d.methods.cic import deposit_cic_2d
from pic_kde_2d.methods.lscv import (
    cic_filter_self_matrix,
    cic_weight_second_moment,
    lscv_score,
    mean_cic_filter_self,
)
from pic_kde_2d.methods.spectral_W import make_W_gaussian
from pic_kde_2d.methods.spectral_W_wrappers import make_spectral_W_gaussian_methods


DEFAULT_SIGMAS = "0,0.25,0.5,0.75,1,1.25,1.5,2,2.5,3,4,5,6,7,8,10,12,16,20,30,45,60"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--ppc", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--sigma-dx", default=DEFAULT_SIGMAS)
    parser.add_argument("--n-steps", type=int, default=600)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--diag-stride", type=int, default=10)
    parser.add_argument("--eval-t-min", type=float, default=5.0)
    parser.add_argument("--eval-t-max", type=float, default=30.0)
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
    sigmas = np.asarray(sorted({float(item) for item in args.sigma_dx.split(",") if item.strip()}))
    filters = {
        float(sigma): make_W_gaussian(grid, sigma * grid.dx, sigma * grid.dy)
        for sigma in sigmas
        if sigma > 0.0
    }
    self_matrices = {
        float(sigma): cic_filter_self_matrix(
            grid, None if sigma == 0.0 else filters[float(sigma)]
        )
        for sigma in sigmas
    }
    deposit, gather = make_spectral_W_gaussian_methods(grid, 0.25 * grid.dx, 0.25 * grid.dy)
    densities, scores, times = [], [], []
    n = particles.n_particles
    probability_weights = np.full(n, 1.0 / n)
    for step in range(args.n_steps + 1):
        time_value = step * args.dt
        if (
            step % args.diag_stride == 0
            and args.eval_t_min <= time_value <= args.eval_t_max
        ):
            rho = deposit_cic_2d(
                particles.x,
                particles.y,
                probability_weights,
                grid.Nx,
                grid.Ny,
                grid.dx,
                grid.dy,
            )
            moment = cic_weight_second_moment(particles.x, particles.y, grid)
            snapshot_scores = []
            for sigma in sigmas:
                W = None if sigma == 0.0 else filters[float(sigma)]
                mean_self = mean_cic_filter_self(moment, self_matrices[float(sigma)])
                snapshot_scores.append(lscv_score(rho, W, mean_self, n, grid))
            densities.append(rho)
            scores.append(snapshot_scores)
            times.append(time_value)
        if step == args.n_steps:
            break
        pic_step(particles, grid, args.dt, deposit, gather)
    np.savez_compressed(
        args.output / "selector_snapshots.npz",
        times=np.asarray(times),
        sigmas=sigmas,
        probability_density=np.stack(densities),
        lscv_scores=np.asarray(scores),
    )
    metadata = {
        "ppc": args.ppc,
        "seed": args.seed,
        "n_particles": n,
        "eval_window": [args.eval_t_min, args.eval_t_max],
        "live_sigma_dx": 0.25,
        "initializer": "regular positions, globally shuffled velocity quadrature",
        "n_snapshots": len(times),
    }
    (args.output / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
