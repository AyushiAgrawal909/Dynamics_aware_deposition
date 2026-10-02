#!/usr/bin/env python3
"""Reduce paired binomial MISE shards to noise-reduction confidence intervals."""

from __future__ import annotations

import argparse
import csv
import json
import pathlib

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--nboot", type=int, default=4000)
    parser.add_argument("--compensation-order", type=int, default=None)
    args = parser.parse_args()
    rows = []
    for path in sorted(args.input.glob("seed*/binomial_noise_seed.csv")):
        with path.open(newline="", encoding="utf-8") as handle:
            rows.extend(csv.DictReader(handle))
    if args.compensation_order is not None:
        rows = [
            row
            for row in rows
            if row.get("compensation_order", "")
            and int(row["compensation_order"]) == args.compensation_order
        ]
    if not rows:
        raise RuntimeError("no noise rows matched the requested filter configuration")
    seeds = sorted({int(row["seed"]) for row in rows})
    if seeds != list(range(args.seeds)):
        raise RuntimeError(f"expected seeds 0..{args.seeds - 1}, found {seeds}")
    passes = sorted({int(row["passes"]) for row in rows})
    families = {row.get("filter_family", "binomial") for row in rows}
    if len(families) != 1:
        raise RuntimeError(f"mixed filter families in noise shards: {sorted(families)}")
    filter_family = families.pop()
    orders = {
        int(row["compensation_order"])
        for row in rows
        if row.get("compensation_order", "")
    }
    if filter_family == "compensated_binomial" and len(orders) != 1:
        raise RuntimeError(f"expected one compensation order, found {sorted(orders)}")
    compensation_order = next(iter(orders)) if orders else None
    matrix = np.asarray(
        [
            [
                float(next(row["mise"] for row in rows if int(row["seed"]) == seed and int(row["passes"]) == n))
                for n in passes
            ]
            for seed in seeds
        ]
    )
    if np.any(np.var(matrix, axis=0) == 0.0):
        raise RuntimeError("zero across-seed MISE variance")
    rng = np.random.default_rng(85231)
    indices = rng.integers(0, args.seeds, size=(args.nboot, args.seeds))
    boot_means = np.mean(matrix[indices], axis=1)
    boot_ratios = boot_means[:, [0]] / boot_means
    means = np.mean(matrix, axis=0)
    ratios = means[0] / means
    summary = []
    for index, n in enumerate(passes):
        summary.append(
            {
                "passes": n,
                "sigma_eq_dx": (
                    float(np.sqrt(n / 2.0))
                    if filter_family == "binomial"
                    else None
                ),
                "filter_family": filter_family,
                "compensation_order": compensation_order,
                "mise_mean": float(means[index]),
                "mise_ci_lo": float(np.percentile(boot_means[:, index], 2.5)),
                "mise_ci_hi": float(np.percentile(boot_means[:, index], 97.5)),
                "noise_reduction": float(ratios[index]),
                "noise_reduction_ci_lo": float(np.percentile(boot_ratios[:, index], 2.5)),
                "noise_reduction_ci_hi": float(np.percentile(boot_ratios[:, index], 97.5)),
                "n_seeds": args.seeds,
            }
        )
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "binomial_noise_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    result = {
        "filter_family": filter_family,
        "compensation_order": compensation_order,
        "passes": passes,
        "rows": summary,
    }
    (args.output / "binomial_noise_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
