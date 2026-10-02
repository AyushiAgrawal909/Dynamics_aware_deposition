#!/usr/bin/env python3
"""Summarize the alpha-dependent linearized-delta-f validity breakdown."""

from __future__ import annotations

import argparse
import csv
import json
import pathlib

import numpy as np


def bootstrap_mean_ci(
    values: np.ndarray,
    *,
    nboot: int,
    seed: int,
) -> tuple[float, float, float]:
    """Return a percentile bootstrap CI for a mean over independent seeds."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or values.size < 2:
        raise ValueError("bootstrap requires at least two independent seeds")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, values.size, size=(nboot, values.size))
    means = np.mean(values[indices], axis=1)
    return (
        float(np.mean(values)),
        float(np.percentile(means, 2.5)),
        float(np.percentile(means, 97.5)),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--nboot", type=int, default=100_000)
    parser.add_argument("--bootstrap-seed", type=int, default=9_102_026)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    rows = []
    for alpha_path in sorted(args.input.glob("alpha_*")):
        seed_rows = []
        for history_path in sorted(alpha_path.glob("seed*/mode_history.npz")):
            with np.load(history_path) as data:
                times = np.asarray(data["times"], dtype=np.float64)
                ratios = np.asarray(data["delta_f_ratio"], dtype=np.float64)
                seed = int(np.asarray(data["seed"]).reshape(-1)[0])
            result = json.loads(
                (history_path.parent / "result.json").read_text(encoding="utf-8")
            )
            crossings = np.flatnonzero(ratios >= args.threshold)
            seed_rows.append(
                {
                    "alpha": float(result["alpha"]),
                    "seed": seed,
                    "max_delta_f_ratio": float(np.max(ratios)),
                    "first_crossing_time": (
                        float(times[crossings[0]]) if crossings.size else float("nan")
                    ),
                    "valid_through_run": bool(np.max(ratios) < args.threshold),
                }
            )
        if len(seed_rows) != args.seeds:
            raise RuntimeError(f"{alpha_path}: expected {args.seeds} seeds")
        maxima = np.asarray([row["max_delta_f_ratio"] for row in seed_rows])
        crossing_times = np.asarray([row["first_crossing_time"] for row in seed_rows])
        finite_crossings = crossing_times[np.isfinite(crossing_times)]
        mean_maximum, ci_lo, ci_hi = bootstrap_mean_ci(
            maxima,
            nboot=args.nboot,
            seed=args.bootstrap_seed + len(rows),
        )
        empirical_lo, empirical_hi = np.percentile(maxima, [2.5, 97.5])
        rows.append(
            {
                "alpha": seed_rows[0]["alpha"],
                "max_ratio_mean": mean_maximum,
                "max_ratio_ci_lo": ci_lo,
                "max_ratio_ci_hi": ci_hi,
                "max_ratio_empirical_percentile_lo": float(empirical_lo),
                "max_ratio_empirical_percentile_hi": float(empirical_hi),
                "breakdown_seed_fraction": float(np.mean(maxima >= args.threshold)),
                "median_first_crossing_time": (
                    float(np.median(finite_crossings))
                    if finite_crossings.size
                    else float("nan")
                ),
                "n_seeds": args.seeds,
            }
        )

    with (args.output / "delta_f_breakdown.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result = {
        "validity_threshold": args.threshold,
        "uncertainty": {
            "unit": "independent seed",
            "estimator": "percentile bootstrap of the seed-mean maximum ratio",
            "nboot": args.nboot,
            "bootstrap_seed": args.bootstrap_seed,
        },
        "rows": rows,
        "breakdown_demonstrated": any(
            row["breakdown_seed_fraction"] == 1.0 for row in rows
        ),
    }
    (args.output / "delta_f_breakdown.json").write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
