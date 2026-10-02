#!/usr/bin/env python3
"""Replay the existing E2 noise trajectories at exact requested widths.

The original dense-noise run retained scalar errors, not density/field arrays.
This diagnostic deterministically replays its frozen trajectories and asserts
the saved q=0.50 and boundary candidates before reporting the exact v_b=0.80
R=0.95 interior width.  It is an evaluation-only replay, not bandwidth
selection and not evidence for another drift value.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


WORKSPACE = Path(__file__).resolve().parents[2]
PROJECT = WORKSPACE / "pic_kde_2d"
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from pic_kde_2d.conventions import marker_grid_shape  # noqa: E402
from pic_kde_2d.core.grid2d import Grid2D  # noqa: E402
from pic_kde_2d.core.particles2d import init_two_stream_2d  # noqa: E402
from pic_kde_2d.core.pic_loop import pic_step  # noqa: E402
from pic_kde_2d.methods.cic import deposit_cic_2d  # noqa: E402
from pic_kde_2d.methods.spectral_W import apply_W_to_grid, make_W_gaussian  # noqa: E402
from pic_kde_2d.methods.spectral_W_wrappers import (  # noqa: E402
    make_spectral_W_gaussian_methods,
)


WIDTHS = (
    ("cic", 0.0),
    ("saved_q050_anchor", 0.1716392815470568),
    ("R095_interior_vb080", 0.1738592384458977),
    ("saved_boundary_anchor", 0.24273459979982182),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def make_particles(ppc: int, seed: int, grid: Grid2D):
    nx_marker, ny_marker = marker_grid_shape(grid.Nx, grid.Ny, ppc)
    return init_two_stream_2d(
        Nx_marker=nx_marker,
        Ny_marker=ny_marker,
        Lx=grid.Lx,
        Ly=grid.Ly,
        v_beam=0.8,
        v_thermal=0.05,
        alpha=0.001,
        kx_mode=12,
        ky_mode=0,
        rng=np.random.default_rng(seed),
    )


def density(particles, grid: Grid2D) -> np.ndarray:
    return deposit_cic_2d(
        particles.x, particles.y, particles.w,
        grid.Nx, grid.Ny, grid.dx, grid.dy,
    )


def measure_seed(seed: int) -> list[dict]:
    grid = Grid2D(128, 16, 64.0, 16.0)
    low = make_particles(32, seed, grid)
    reference = make_particles(1024, seed, grid)
    deposit_live, gather_live = make_spectral_W_gaussian_methods(
        grid, 0.25 * grid.dx, 0.25 * grid.dy
    )
    filters = {
        label: make_W_gaussian(grid, 0.0, sigma_y * grid.dy)
        for label, sigma_y in WIDTHS
    }
    density_values = {label: [] for label, _ in WIDTHS}
    field_values = {label: [] for label, _ in WIDTHS}
    for step in range(81):
        time = step * 0.05
        if step % 2 == 0 and 0.5 <= time <= 4.0:
            rho_low = density(low, grid)
            rho_reference = density(reference, grid)
            reference_rhs = rho_reference - float(np.mean(rho_reference))
            ex_reference, ey_reference = grid.poisson_solve(reference_rhs)
            for label, _ in WIDTHS:
                rho_candidate = apply_W_to_grid(rho_low, filters[label], grid)
                candidate_rhs = rho_candidate - float(np.mean(rho_candidate))
                ex_candidate, ey_candidate = grid.poisson_solve(candidate_rhs)
                density_values[label].append(
                    float(np.mean((rho_candidate - rho_reference) ** 2))
                )
                field_values[label].append(
                    float(np.mean(
                        (ex_candidate - ex_reference) ** 2
                        + (ey_candidate - ey_reference) ** 2
                    ))
                )
        if step < 80:
            pic_step(low, grid, 0.05, deposit_live, gather_live)
            pic_step(reference, grid, 0.05, deposit_live, gather_live)
    return [
        {
            "analysis_role": "DETERMINISTIC_REPLAY_OF_EXISTING_NOISE_PROTOCOL",
            "seed": seed,
            "label": label,
            "sigma_x_dx": 0.0,
            "sigma_y_dy": sigma_y,
            "density_mse": float(np.mean(density_values[label])),
            "field_mse": float(np.mean(field_values[label])),
            "n_times": len(density_values[label]),
        }
        for label, sigma_y in WIDTHS
    ]


def load_saved_rows(root: Path) -> dict[tuple[int, str], dict]:
    rows: dict[tuple[int, str], dict] = {}
    for seed in range(100, 110):
        path = root / f"seed{seed}" / "result" / "anisotropic_dense_noise_seed.csv"
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row["label"] in {"q050_x000", "q100_x000"}:
                    rows[(seed, row["label"])] = row
    if len(rows) != 20:
        raise AssertionError("saved anchor rows are incomplete")
    return rows


def reduce_rows(rows: list[dict], nboot: int) -> list[dict]:
    labels = [label for label, _ in WIDTHS]
    matrix_density = np.asarray([
        [next(row["density_mse"] for row in rows if row["seed"] == seed and row["label"] == label)
         for label in labels]
        for seed in range(100, 110)
    ])
    matrix_field = np.asarray([
        [next(row["field_mse"] for row in rows if row["seed"] == seed and row["label"] == label)
         for label in labels]
        for seed in range(100, 110)
    ])
    rng = np.random.default_rng(20260911)
    indices = rng.integers(0, 10, size=(nboot, 10))
    output = []
    for metric, matrix in (("density_mse", matrix_density), ("field_mse", matrix_field)):
        means = np.mean(matrix, axis=0)
        reductions = means[0] / means
        boot_means = np.mean(matrix[indices], axis=1)
        boot_reductions = boot_means[:, [0]] / boot_means
        for index, (label, sigma_y) in enumerate(WIDTHS):
            output.append(
                {
                    "analysis_role": "FIXED_WIDTH_PAIRED_EVALUATION_NOT_SELECTION",
                    "v_beam": 0.8,
                    "label": label,
                    "sigma_x_dx": 0.0,
                    "sigma_y_dy": sigma_y,
                    "metric": metric,
                    "mean_error": float(means[index]),
                    "noise_power_reduction_vs_cic": float(reductions[index]),
                    "paired_bootstrap_ci_lo": float(np.quantile(boot_reductions[:, index], 0.025)),
                    "paired_bootstrap_ci_hi": float(np.quantile(boot_reductions[:, index], 0.975)),
                    "n_seeds": 10,
                    "bootstrap_draws": nboot,
                    "bootstrap_seed": 20260911,
                    "scope": "full-f E2 short-window grid error, t in [0.5,4.0]",
                }
            )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--saved-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--nboot", type=int, default=5000)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    saved = load_saved_rows(args.saved_root)
    rows = []
    for seed in range(100, 110):
        rows.extend(measure_seed(seed))

    # Exact replay anchors: these are saved candidates, not interpolations.
    for seed in range(100, 110):
        for replay_label, saved_label in (
            ("saved_q050_anchor", "q050_x000"),
            ("saved_boundary_anchor", "q100_x000"),
        ):
            replay = next(
                row for row in rows if row["seed"] == seed and row["label"] == replay_label
            )
            reference = saved[(seed, saved_label)]
            np.testing.assert_allclose(
                replay["density_mse"], float(reference["density_mse"]),
                rtol=0.0, atol=5.0e-15,
            )
            np.testing.assert_allclose(
                replay["field_mse"], float(reference["field_mse"]),
                rtol=0.0, atol=5.0e-15,
            )
    summaries = reduce_rows(rows, args.nboot)
    args.output.mkdir(parents=True)
    with (args.output / "per_seed_noise_replay.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (args.output / "interior_noise_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)

    interior = [row for row in summaries if row["label"] == "R095_interior_vb080"]
    result = {
        "status": "PASS_EXACT_WIDTH_REPLAY_ANCHORED_TO_SAVED_DATA",
        "scope": (
            "v_b=0.80 full-f short-window grid-error protocol only; not a noise "
            "measurement for another drift and not the projected delta-f rate run"
        ),
        "metric_definitions": {
            "density_mse": "mean grid squared error versus paired PPC-1024 CIC density",
            "field_mse": (
                "mean squared Ex+Ey grid-field error versus paired PPC-1024 CIC field; "
                "pre-return-filter and pre-gather"
            ),
            "reference": "same-seed PPC-1024 trajectory",
            "evaluation_window": [0.5, 4.0],
            "ppc": 32,
            "reference_ppc": 1024,
        },
        "replay_anchors": {
            "q050_x000_sigma_y_dy": 0.1716392815470568,
            "q100_x000_boundary_sigma_y_dy": 0.24273459979982182,
            "maximum_allowed_absolute_metric_difference": 5.0e-15,
        },
        "interior_width": 0.1738592384458977,
        "interior_results": interior,
        "boundary_result_retained_separately": [
            row for row in summaries if row["label"] == "saved_boundary_anchor"
        ],
        "input_sha256": {
            str(Path(__file__).resolve()): sha256(Path(__file__).resolve()),
            str(PROJECT / "pic_kde_2d/core/particles2d.py"): sha256(
                PROJECT / "pic_kde_2d/core/particles2d.py"
            ),
            str(PROJECT / "pic_kde_2d/core/pic_loop.py"): sha256(
                PROJECT / "pic_kde_2d/core/pic_loop.py"
            ),
            str(PROJECT / "pic_kde_2d/core/grid2d.py"): sha256(
                PROJECT / "pic_kde_2d/core/grid2d.py"
            ),
            str(PROJECT / "pic_kde_2d/methods/spectral_W.py"): sha256(
                PROJECT / "pic_kde_2d/methods/spectral_W.py"
            ),
        },
        "tables": {
            "per_seed": "per_seed_noise_replay.csv",
            "summary": "interior_noise_summary.csv",
        },
    }
    (args.output / "interior_noise_replay.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
