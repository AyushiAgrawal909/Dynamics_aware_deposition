#!/usr/bin/env python3
"""Direct static MISE and exact LSCV bandwidth measurement at one PPC."""

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

from pic_kde_2d.benchmarks.theory import landau_omega_theory
from pic_kde_2d.conventions import landau_box_length, marker_grid_shape
from pic_kde_2d.core.grid2d import Grid2D
from pic_kde_2d.core.particles2d import init_landau_2d_eigenmode
from pic_kde_2d.methods.cic import deposit_cic_2d
from pic_kde_2d.methods.lscv import (
    cic_filter_self_matrix,
    cic_weight_second_moment,
    lscv_score,
    mean_cic_filter_self,
)
from pic_kde_2d.methods.spectral_W import apply_W_to_grid, make_W_gaussian
from pic_kde_2d.methods.wu_qin_bandwidth import identify_cv_minimum


DEFAULT_SIGMAS = "0,0.25,0.5,0.75,1,1.25,1.5,2,2.5,3,4,5,6,7,8,10,12,16,20,30,45,60"


def parse_sigmas(text: str) -> np.ndarray:
    return np.asarray(sorted({float(item) for item in text.split(",") if item.strip()}))


def quadratic_minimum(sigmas: np.ndarray, values: np.ndarray) -> float:
    index = int(np.argmin(values))
    if index == 0 or index == len(sigmas) - 1:
        return float(sigmas[index])
    x = sigmas[index - 1 : index + 2]
    y = values[index - 1 : index + 2]
    a, b, _ = np.polyfit(x, y, 2)
    if not np.isfinite(a) or a <= 0.0:
        return float(sigmas[index])
    vertex = float(-b / (2.0 * a))
    return vertex if x[0] <= vertex <= x[-1] else float(sigmas[index])


def analytic_cic_reference(grid: Grid2D, alpha: float, samples: int) -> np.ndarray:
    """Project the continuum Lagrangian density onto the same CIC grid."""
    q = (np.arange(samples, dtype=np.float64) + 0.5) * (grid.Lx / samples)
    kx = 2.0 * np.pi / grid.Lx
    x = (q + alpha * np.sin(kx * q) / kx) % grid.Lx
    i = np.floor(x / grid.dx).astype(np.int64)
    fraction = x / grid.dx - i
    density_x = np.bincount(i % grid.Nx, weights=1.0 - fraction, minlength=grid.Nx)
    density_x += np.bincount((i + 1) % grid.Nx, weights=fraction, minlength=grid.Nx)
    density_x /= samples * grid.dx
    reference = np.repeat((density_x / grid.Ly)[:, None], grid.Ny, axis=1)
    integral = float(np.sum(reference) * grid.dx * grid.dy)
    if abs(integral - 1.0) > 5.0e-13:
        raise RuntimeError(f"analytic reference normalization failed: {integral}")
    return reference


def make_particles(grid: Grid2D, ppc: int, alpha: float, seed: int):
    nx_marker, ny_marker = marker_grid_shape(grid.Nx, grid.Ny, ppc)
    return init_landau_2d_eigenmode(
        Nx_marker=nx_marker,
        Ny_marker=ny_marker,
        Lx=grid.Lx,
        Ly=grid.Ly,
        v_thermal=1.0,
        alpha=alpha,
        kx_mode=1,
        omega_r=landau_omega_theory(0.4, 1.0),
        rng=np.random.default_rng(seed),
        position_jitter=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--ppc", type=int, required=True)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=0.01)
    parser.add_argument("--Nx", type=int, default=64)
    parser.add_argument("--Ny", type=int, default=16)
    parser.add_argument("--Ly", type=float, default=8.0)
    parser.add_argument("--sigma-dx", default=DEFAULT_SIGMAS)
    parser.add_argument("--reference-samples", type=int, default=1048576)
    parser.add_argument("--nboot", type=int, default=4000)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    grid = Grid2D(args.Nx, args.Ny, landau_box_length(1, 0.4), args.Ly)
    sigmas = parse_sigmas(args.sigma_dx)
    reference = analytic_cic_reference(grid, args.alpha, args.reference_samples)
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
    mise = np.empty((args.seeds, len(sigmas)), dtype=np.float64)
    cv = np.empty_like(mise)
    seed_rows = []
    for seed in range(args.seeds):
        particles = make_particles(grid, args.ppc, args.alpha, seed)
        n = particles.n_particles
        rho = deposit_cic_2d(
            particles.x,
            particles.y,
            np.full(n, 1.0 / n),
            grid.Nx,
            grid.Ny,
            grid.dx,
            grid.dy,
        )
        moment = cic_weight_second_moment(particles.x, particles.y, grid)
        for index, sigma in enumerate(sigmas):
            W = None if sigma == 0.0 else filters[float(sigma)]
            smooth = rho if W is None else apply_W_to_grid(rho, W, grid)
            mise[seed, index] = float(np.sum((smooth - reference) ** 2) * grid.dx * grid.dy)
            mean_self = mean_cic_filter_self(moment, self_matrices[float(sigma)])
            cv[seed, index] = lscv_score(rho, W, mean_self, n, grid)
            seed_rows.append(
                {
                    "ppc": args.ppc,
                    "seed": seed,
                    "sigma_dx": float(sigma),
                    "mise": mise[seed, index],
                    "lscv": cv[seed, index],
                }
            )
        print(f"PPC selector completed ppc={args.ppc} seed={seed}", flush=True)
    if np.any(np.var(mise, axis=0) == 0.0) or np.any(np.var(cv, axis=0) == 0.0):
        raise RuntimeError("zero across-seed selector variance")
    means_mise = np.mean(mise, axis=0)
    means_cv = np.mean(cv, axis=0)
    selected_mise = identify_cv_minimum(sigmas, means_mise)
    selected_cv = identify_cv_minimum(sigmas, means_cv)
    rng = np.random.default_rng(77000 + args.ppc)
    boot_mise, boot_cv = [], []
    for _ in range(args.nboot):
        indices = rng.integers(0, args.seeds, size=args.seeds)
        draw_mise = np.mean(mise[indices], axis=0)
        draw_cv = np.mean(cv[indices], axis=0)
        if identify_cv_minimum(sigmas, draw_mise).identified:
            boot_mise.append(quadratic_minimum(sigmas, draw_mise))
        if identify_cv_minimum(sigmas, draw_cv).identified:
            boot_cv.append(quadratic_minimum(sigmas, draw_cv))

    curve_rows = []
    for index, sigma in enumerate(sigmas):
        curve_rows.append(
            {
                "ppc": args.ppc,
                "sigma_dx": float(sigma),
                "mise_mean": float(means_mise[index]),
                "mise_sem": float(np.std(mise[:, index], ddof=1) / np.sqrt(args.seeds)),
                "lscv_mean": float(means_cv[index]),
                "lscv_sem": float(np.std(cv[:, index], ddof=1) / np.sqrt(args.seeds)),
            }
        )
    with (args.output / "selector_seed_values.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(seed_rows[0]))
        writer.writeheader()
        writer.writerows(seed_rows)
    with (args.output / "selector_curves.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(curve_rows[0]))
        writer.writeheader()
        writer.writerows(curve_rows)

    def selection_record(selection, matrix, bootstrap):
        return {
            "status": selection.status,
            "grid_width_dx": selection.bandwidth,
            "quadratic_width_dx": quadratic_minimum(sigmas, np.mean(matrix, axis=0)) if selection.identified else None,
            "ci_lo": float(np.percentile(bootstrap, 2.5)) if bootstrap else None,
            "ci_hi": float(np.percentile(bootstrap, 97.5)) if bootstrap else None,
            "bootstrap_interior_fraction": len(bootstrap) / float(args.nboot),
        }

    result = {
        "ppc": args.ppc,
        "n_particles": args.ppc * grid.Nx * grid.Ny,
        "sampling": "independent stratified-jitter samples at t=0",
        "dynamics_initializer": "separate production global-shuffle quiet start",
        "reference": "analytic Lagrangian density projected through CIC",
        "mise": selection_record(selected_mise, mise, boot_mise),
        "lscv": selection_record(selected_cv, cv, boot_cv),
        "sigma_grid_dx": sigmas.tolist(),
    }
    (args.output / "selector_result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
