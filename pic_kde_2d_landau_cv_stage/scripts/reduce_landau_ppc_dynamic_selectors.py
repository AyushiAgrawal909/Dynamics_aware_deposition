#!/usr/bin/env python3
"""Measure dynamic-window MISE and LSCV widths from paired density snapshots."""

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

from pic_kde_2d.conventions import landau_box_length
from pic_kde_2d.core.grid2d import Grid2D
from pic_kde_2d.methods.spectral_W import apply_W_to_grid, make_W_gaussian
from pic_kde_2d.methods.wu_qin_bandwidth import identify_cv_minimum


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--ppc", type=int, required=True)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--nboot", type=int, default=4000)
    args = parser.parse_args()
    densities, cv_scores = [], []
    times = sigmas = None
    for seed in range(args.seeds):
        path = args.input / f"seed{seed}" / "selector_snapshots.npz"
        with np.load(path) as data:
            current_times = np.asarray(data["times"], dtype=np.float64)
            current_sigmas = np.asarray(data["sigmas"], dtype=np.float64)
            densities.append(np.asarray(data["probability_density"], dtype=np.float64))
            cv_scores.append(np.asarray(data["lscv_scores"], dtype=np.float64))
        if times is None:
            times, sigmas = current_times, current_sigmas
        else:
            np.testing.assert_array_equal(times, current_times)
            np.testing.assert_array_equal(sigmas, current_sigmas)
    density = np.stack(densities)
    cv_scores = np.stack(cv_scores)
    # Charge-neutrality and symmetry make some individual grid entries exactly
    # seed-independent. Guard the reported ensemble statistic, not every cell.
    density_signature = np.mean(density * density, axis=(1, 2, 3))
    if np.var(density_signature) == 0.0:
        raise RuntimeError("zero across-seed density-statistic variance")
    grid = Grid2D(64, 16, landau_box_length(1, 0.4), 8.0)
    filters = {
        float(sigma): make_W_gaussian(grid, sigma * grid.dx, sigma * grid.dy)
        for sigma in sigmas
        if sigma > 0.0
    }
    sum_density = np.sum(density, axis=0)
    per_seed_mise = np.empty((args.seeds, len(sigmas)), dtype=np.float64)
    for seed in range(args.seeds):
        reference = (sum_density - density[seed]) / float(args.seeds - 1)
        for sigma_index, sigma in enumerate(sigmas):
            smooth = density[seed]
            if sigma > 0.0:
                smooth = np.stack(
                    [apply_W_to_grid(snapshot, filters[float(sigma)], grid) for snapshot in density[seed]]
                )
            per_seed_mise[seed, sigma_index] = float(
                np.mean(np.sum((smooth - reference) ** 2, axis=(1, 2)) * grid.dx * grid.dy)
            )
    per_seed_cv = np.mean(cv_scores, axis=1)
    mise_variance = np.var(per_seed_mise, axis=0)
    cv_variance = np.var(per_seed_cv, axis=0)
    if np.all(mise_variance == 0.0) or np.all(cv_variance == 0.0):
        raise RuntimeError("zero across-seed selector-curve variance")
    mean_mise = np.mean(per_seed_mise, axis=0)
    mean_cv = np.mean(per_seed_cv, axis=0)
    selected_mise = identify_cv_minimum(sigmas, mean_mise)
    selected_cv = identify_cv_minimum(sigmas, mean_cv)
    rng = np.random.default_rng(88100 + args.ppc)
    boot_mise, boot_cv = [], []
    for _ in range(args.nboot):
        indices = rng.integers(0, args.seeds, size=args.seeds)
        draw_mise = np.mean(per_seed_mise[indices], axis=0)
        draw_cv = np.mean(per_seed_cv[indices], axis=0)
        if identify_cv_minimum(sigmas, draw_mise).identified:
            boot_mise.append(quadratic_minimum(sigmas, draw_mise))
        if identify_cv_minimum(sigmas, draw_cv).identified:
            boot_cv.append(quadratic_minimum(sigmas, draw_cv))

    def record(selection, values, bootstrap):
        return {
            "status": selection.status,
            "grid_width_dx": selection.bandwidth,
            "quadratic_width_dx": quadratic_minimum(sigmas, values) if selection.identified else None,
            "ci_lo": float(np.percentile(bootstrap, 2.5)) if bootstrap else None,
            "ci_hi": float(np.percentile(bootstrap, 97.5)) if bootstrap else None,
            "bootstrap_interior_fraction": len(bootstrap) / float(args.nboot),
        }

    rows = []
    for index, sigma in enumerate(sigmas):
        rows.append(
            {
                "ppc": args.ppc,
                "sigma_dx": float(sigma),
                "mise_mean": float(mean_mise[index]),
                "mise_sem": float(np.std(per_seed_mise[:, index], ddof=1) / np.sqrt(args.seeds)),
                "lscv_mean": float(mean_cv[index]),
                "lscv_sem": float(np.std(per_seed_cv[:, index], ddof=1) / np.sqrt(args.seeds)),
            }
        )
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "selector_curves.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result = {
        "ppc": args.ppc,
        "sampling": "evolving production global-shuffle ensemble",
        "eval_window": [float(times[0]), float(times[-1])],
        "reference": "paired leave-one-seed-out ensemble density",
        "reference_noise_note": "reference variance is reduced by the other nine seeds but is nonzero",
        "mise": record(selected_mise, mean_mise, boot_mise),
        "lscv": record(selected_cv, mean_cv, boot_cv),
        "sigma_grid_dx": sigmas.tolist(),
    }
    (args.output / "selector_result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
