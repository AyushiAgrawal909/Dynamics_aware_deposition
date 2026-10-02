#!/usr/bin/env python3
"""Verify every frozen E2 drift-scan Gaussian on the implemented FFT grid."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys

import numpy as np


HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[1]
CAMPAIGN_ROOT = WORKSPACE / "pic_kde_2d_landau_cv_stage"
sys.path.insert(0, str(CAMPAIGN_ROOT))

from pic_kde_2d.core.grid2d import Grid2D  # noqa: E402
from pic_kde_2d.methods.spectral_W import (  # noqa: E402
    apply_W_to_grid,
    make_W_gaussian,
)


def mode_value(values: np.ndarray, mode_x: int, mode_y: int) -> complex:
    transformed = np.fft.fft2(values)
    return complex(transformed[mode_x, mode_y])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config["status"] != "FROZEN_BEFORE_SIMULATION_RESULTS":
        raise AssertionError("campaign is not frozen")
    grid_spec = config["grid"]
    grid = Grid2D(
        Nx=int(grid_spec["Nx"]),
        Ny=int(grid_spec["Ny"]),
        Lx=float(grid_spec["Lx"]),
        Ly=float(grid_spec["Ly"]),
    )
    mode_x, mode_y = map(int, config["mode"])
    phase = (
        2.0 * math.pi * mode_x * grid.x[:, None] / grid.Lx
        + 2.0 * math.pi * mode_y * grid.y[None, :] / grid.Ly
    )
    source = np.cos(phase)
    source_mode = mode_value(source, mode_x, mode_y)
    rows = []
    for scan in config["scan_rows"]:
        for arm, sigma_key in (
            ("fixed_gaussian", "fixed_sigma_y_dy"),
            ("constrained_gaussian", "constrained_sigma_y_dy"),
        ):
            sigma = float(scan[sigma_key])
            W = make_W_gaussian(grid, 0.0, sigma * grid.dy)
            implemented = float(W[mode_x, mode_y])
            once = apply_W_to_grid(source, W, grid)
            twice = apply_W_to_grid(once, W, grid)
            measured_once = abs(mode_value(once, mode_x, mode_y) / source_mode)
            measured_twice = abs(mode_value(twice, mode_x, mode_y) / source_mode)
            analytic_once = math.exp(
                -0.5 * (2.0 * math.pi * mode_y * sigma / grid.Ny) ** 2
            )
            analytic_twice = analytic_once**2
            for actual, expected, name in (
                (implemented, analytic_once, "implemented W"),
                (measured_once, analytic_once, "measured one application"),
                (measured_twice, analytic_twice, "measured two applications"),
            ):
                if abs(actual - expected) > 2.0e-14:
                    raise AssertionError(f"{name} mismatch at drift {scan['v_beam']}")
            rows.append(
                {
                    "v_beam": scan["v_beam"],
                    "arm": arm,
                    "sigma_x_dx": 0.0,
                    "sigma_y_dy": sigma,
                    "analytic_W_one_application": analytic_once,
                    "implemented_W_array": implemented,
                    "measured_W_one_application": measured_once,
                    "analytic_particle_force_W2": analytic_twice,
                    "measured_two_applications_W2": measured_twice,
                    "absolute_error_W": abs(measured_once - analytic_once),
                    "absolute_error_W2": abs(measured_twice - analytic_twice),
                }
            )
    fixed = [row for row in rows if row["arm"] == "fixed_gaussian"]
    if len({row["sigma_y_dy"] for row in fixed}) != 1:
        raise AssertionError("fixed Gaussian width changed across drift")
    if len({row["analytic_particle_force_W2"] for row in fixed}) != 1:
        raise AssertionError("fixed Gaussian transfer changed across drift")
    args.output.mkdir(parents=True)
    with (args.output / "implemented_transfer.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "status": "PASS",
        "mode": [mode_x, mode_y],
        "rows": len(rows),
        "fixed_sigma_y_dy": fixed[0]["sigma_y_dy"],
        "fixed_W_one_application": fixed[0]["analytic_W_one_application"],
        "fixed_particle_force_W2": fixed[0]["analytic_particle_force_W2"],
        "maximum_absolute_error_W": max(row["absolute_error_W"] for row in rows),
        "maximum_absolute_error_W2": max(row["absolute_error_W2"] for row in rows),
        "scope": "implemented spectral W; two applications represent matched deposit and gather filtering",
    }
    (args.output / "transfer_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
