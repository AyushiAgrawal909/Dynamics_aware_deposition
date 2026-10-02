#!/usr/bin/env python3
"""Reduce all PPC selector snapshots against the common PPC-4096 reference."""

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


PPCS = (8, 32, 128, 512, 1024, 4096)


def quadratic_minimum(sigmas, values):
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


def load_ppc(root, ppc, seeds):
    density, cv = [], []
    times = sigmas = None
    for seed in range(seeds):
        with np.load(root / f"ppc{ppc}" / f"seed{seed}" / "selector_snapshots.npz") as data:
            current_times = np.asarray(data["times"], dtype=np.float64)
            current_sigmas = np.asarray(data["sigmas"], dtype=np.float64)
            density.append(np.asarray(data["probability_density"], dtype=np.float64))
            cv.append(np.asarray(data["lscv_scores"], dtype=np.float64))
        if times is None:
            times, sigmas = current_times, current_sigmas
        else:
            np.testing.assert_array_equal(times, current_times)
            np.testing.assert_array_equal(sigmas, current_sigmas)
    return times, sigmas, np.stack(density), np.stack(cv)


def selection_record(selection, sigmas, values, bootstrap, nboot):
    return {
        "status": selection.status,
        "grid_width_dx": selection.bandwidth,
        "quadratic_width_dx": quadratic_minimum(sigmas, values) if selection.identified else None,
        "ci_lo": float(np.percentile(bootstrap, 2.5)) if bootstrap else None,
        "ci_hi": float(np.percentile(bootstrap, 97.5)) if bootstrap else None,
        "bootstrap_interior_fraction": len(bootstrap) / float(nboot),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--nboot", type=int, default=4000)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    loaded = {ppc: load_ppc(args.input, ppc, args.seeds) for ppc in PPCS}
    ref_times, ref_sigmas, ref_density, _ = loaded[4096]
    common_reference = np.mean(ref_density, axis=0)
    grid = Grid2D(64, 16, landau_box_length(1, 0.4), 8.0)
    filters = {
        float(sigma): make_W_gaussian(grid, sigma * grid.dx, sigma * grid.dy)
        for sigma in ref_sigmas
        if sigma > 0.0
    }
    aggregate = []
    for ppc in PPCS:
        times, sigmas, density, cv_scores = loaded[ppc]
        np.testing.assert_array_equal(ref_times, times)
        np.testing.assert_array_equal(ref_sigmas, sigmas)
        per_seed_mise = np.empty((args.seeds, len(sigmas)), dtype=np.float64)
        for seed in range(args.seeds):
            reference = common_reference
            if ppc == 4096:
                reference = (np.sum(ref_density, axis=0) - ref_density[seed]) / float(args.seeds - 1)
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
        mean_mise = np.mean(per_seed_mise, axis=0)
        mean_cv = np.mean(per_seed_cv, axis=0)
        selected_mise = identify_cv_minimum(sigmas, mean_mise)
        selected_cv = identify_cv_minimum(sigmas, mean_cv)
        rng = np.random.default_rng(94100 + ppc)
        boot_mise, boot_cv = [], []
        for _ in range(args.nboot):
            indices = rng.integers(0, args.seeds, size=args.seeds)
            draw_mise = np.mean(per_seed_mise[indices], axis=0)
            draw_cv = np.mean(per_seed_cv[indices], axis=0)
            if identify_cv_minimum(sigmas, draw_mise).identified:
                boot_mise.append(quadratic_minimum(sigmas, draw_mise))
            if identify_cv_minimum(sigmas, draw_cv).identified:
                boot_cv.append(quadratic_minimum(sigmas, draw_cv))
        out = args.output / f"ppc{ppc}"
        out.mkdir(parents=True, exist_ok=True)
        curve = []
        for index, sigma in enumerate(sigmas):
            curve.append(
                {
                    "ppc": ppc,
                    "sigma_dx": float(sigma),
                    "mise_mean": float(mean_mise[index]),
                    "mise_sem": float(np.std(per_seed_mise[:, index], ddof=1) / np.sqrt(args.seeds)),
                    "lscv_mean": float(mean_cv[index]),
                    "lscv_sem": float(np.std(per_seed_cv[:, index], ddof=1) / np.sqrt(args.seeds)),
                }
            )
        with (out / "selector_curves.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(curve[0]))
            writer.writeheader()
            writer.writerows(curve)
        result = {
            "ppc": ppc,
            "sampling": "evolving production global-shuffle ensemble",
            "eval_window": [float(times[0]), float(times[-1])],
            "reference": (
                "paired leave-one-out PPC-4096 ensemble"
                if ppc == 4096
                else "common independent PPC-4096 ensemble mean"
            ),
            "mise": selection_record(selected_mise, sigmas, mean_mise, boot_mise, args.nboot),
            "lscv": selection_record(selected_cv, sigmas, mean_cv, boot_cv, args.nboot),
            "sigma_grid_dx": sigmas.tolist(),
        }
        (out / "selector_result.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        aggregate.append(result)
        print(json.dumps(result, sort_keys=True), flush=True)
    (args.output / "selector_results_all_ppc.json").write_text(
        json.dumps({"rows": aggregate}, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
