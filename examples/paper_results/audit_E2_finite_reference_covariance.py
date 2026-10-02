#!/usr/bin/env python3
"""Decompose the finite-reference error in the existing E2 interior replay.

The saved dense-noise products retain scalar MSEs rather than grid arrays.
This diagnostic repeats the already verified deterministic 10-seed replay,
asserts the published scalar errors, and retains only the additional
seed-variance/covariance decomposition.  It is not a new parameter campaign.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np


HERE = Path(__file__).resolve().parent
DOWNLOADS = HERE.parents[1]
PROJECT = DOWNLOADS / "pic_kde_2d"
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from pic_kde_2d.conventions import marker_grid_shape  # noqa: E402
from pic_kde_2d.core.grid2d import Grid2D  # noqa: E402
from pic_kde_2d.core.particles2d import init_two_stream_2d  # noqa: E402
from pic_kde_2d.core.pic_loop import pic_step  # noqa: E402
from pic_kde_2d.methods.cic import deposit_cic_2d  # noqa: E402
from pic_kde_2d.methods.spectral_W import apply_W_to_grid, make_W_gaussian  # noqa: E402
from pic_kde_2d.methods.spectral_W_wrappers import make_spectral_W_gaussian_methods  # noqa: E402


INTERIOR_SIGMA_Y = 0.1738592384458977


def make_particles(ppc: int, seed: int, grid: Grid2D):
    nx_marker, ny_marker = marker_grid_shape(grid.Nx, grid.Ny, ppc)
    return init_two_stream_2d(
        nx_marker, ny_marker, grid.Lx, grid.Ly, 0.8, 0.05, 0.001, 12, 0,
        np.random.default_rng(seed),
    )


def density(particles, grid: Grid2D) -> np.ndarray:
    return deposit_cic_2d(
        particles.x, particles.y, particles.w,
        grid.Nx, grid.Ny, grid.dx, grid.dy,
    )


def replay_seed(seed: int) -> dict[str, np.ndarray]:
    grid = Grid2D(128, 16, 64.0, 16.0)
    low = make_particles(32, seed, grid)
    reference = make_particles(1024, seed, grid)
    deposit_live, gather_live = make_spectral_W_gaussian_methods(
        grid, 0.25 * grid.dx, 0.25 * grid.dy
    )
    W = make_W_gaussian(grid, 0.0, INTERIOR_SIGMA_Y * grid.dy)
    values = {name: [] for name in ("rho_cic", "rho_interior", "rho_reference", "ex_cic", "ey_cic", "ex_interior", "ey_interior", "ex_reference", "ey_reference")}
    for step in range(81):
        time = step * 0.05
        if step % 2 == 0 and 0.5 <= time <= 4.0:
            rho_cic = density(low, grid)
            rho_interior = apply_W_to_grid(rho_cic, W, grid)
            rho_reference = density(reference, grid)
            ex_cic, ey_cic = grid.poisson_solve(rho_cic - np.mean(rho_cic))
            ex_interior, ey_interior = grid.poisson_solve(rho_interior - np.mean(rho_interior))
            ex_reference, ey_reference = grid.poisson_solve(rho_reference - np.mean(rho_reference))
            for name, value in (
                ("rho_cic", rho_cic), ("rho_interior", rho_interior), ("rho_reference", rho_reference),
                ("ex_cic", ex_cic), ("ey_cic", ey_cic),
                ("ex_interior", ex_interior), ("ey_interior", ey_interior),
                ("ex_reference", ex_reference), ("ey_reference", ey_reference),
            ):
                values[name].append(value)
        if step < 80:
            pic_step(low, grid, 0.05, deposit_live, gather_live)
            pic_step(reference, grid, 0.05, deposit_live, gather_live)
    return {key: np.stack(value) for key, value in values.items()}


def decompose(x: np.ndarray, r: np.ndarray) -> dict[str, float]:
    mx, mr = np.mean(x, axis=0), np.mean(r, axis=0)
    out = {
        "pair_mse": float(np.mean((x - r) ** 2)),
        "squared_mean_difference": float(np.mean((mx - mr) ** 2)),
        "arm_seed_variance": float(np.mean((x - mx) ** 2)),
        "reference_seed_variance": float(np.mean((r - mr) ** 2)),
        "seed_covariance": float(np.mean((x - mx) * (r - mr))),
    }
    out["decomposition_reconstruction"] = (
        out["squared_mean_difference"] + out["arm_seed_variance"]
        + out["reference_seed_variance"] - 2.0 * out["seed_covariance"]
    )
    np.testing.assert_allclose(
        out["decomposition_reconstruction"], out["pair_mse"], rtol=2e-13, atol=1e-18
    )
    out["reference_variance_fraction_of_pair_mse"] = out["reference_seed_variance"] / out["pair_mse"]
    out["minus_two_covariance_fraction_of_pair_mse"] = -2.0 * out["seed_covariance"] / out["pair_mse"]
    return out


def saved_means(path: Path) -> dict[tuple[str, str], float]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return {
        (row["label"], row["metric"]): float(row["mean_error"])
        for row in rows if row["label"] in {"cic", "R095_interior_vb080"}
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--saved-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    seeds = [replay_seed(seed) for seed in range(100, 110)]
    arrays = {
        key: np.stack([row[key] for row in seeds])
        for key in seeds[0]
    }
    expected = saved_means(args.saved_summary)
    rows = []
    for arm in ("cic", "interior"):
        density_row = decompose(arrays[f"rho_{arm}"], arrays["rho_reference"])
        field_x = decompose(arrays[f"ex_{arm}"], arrays["ex_reference"])
        field_y = decompose(arrays[f"ey_{arm}"], arrays["ey_reference"])
        field_row = {key: field_x[key] + field_y[key] for key in (
            "pair_mse", "squared_mean_difference", "arm_seed_variance",
            "reference_seed_variance", "seed_covariance", "decomposition_reconstruction",
        )}
        field_row["reference_variance_fraction_of_pair_mse"] = field_row["reference_seed_variance"] / field_row["pair_mse"]
        field_row["minus_two_covariance_fraction_of_pair_mse"] = -2.0 * field_row["seed_covariance"] / field_row["pair_mse"]
        label = "cic" if arm == "cic" else "R095_interior_vb080"
        np.testing.assert_allclose(density_row["pair_mse"], expected[(label, "density_mse")], rtol=0.0, atol=5e-15)
        np.testing.assert_allclose(field_row["pair_mse"], expected[(label, "field_mse")], rtol=0.0, atol=5e-15)
        for metric, result in (("density_mse", density_row), ("pre_return_grid_field_mse", field_row)):
            rows.append({
                "analysis_role": "FINITE_REFERENCE_DECOMPOSITION_OF_VERIFIED_REPLAY",
                "arm": arm, "metric": metric, **result,
                "low_ppc": 32, "reference_ppc": 1024, "n_seeds": 10,
                "window": "t=[0.5,4.0], cadence=0.1, 36 snapshots",
                "reference_limit": "finite same-seed PPC-1024 trajectory; not continuum truth",
            })
    args.output.mkdir(parents=True)
    table = args.output / "e2_finite_reference_decomposition.csv"
    with table.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    (args.output / "e2_finite_reference_decomposition.json").write_text(
        json.dumps({
            "status": "PASS_REPLAY_MATCHED_SAVED_SCALARS",
            "new_measurement": False,
            "rows": rows,
            "interpretation": "Reported reductions are finite-reference grid-error comparisons; covariance is quantified but no continuum-error claim is made.",
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
