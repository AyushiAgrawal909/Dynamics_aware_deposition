#!/usr/bin/env python3
"""Reduce paired per-seed MISE shards into selector widths and ratio CIs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


MIN_RELATIVE_SEED_SPREAD = 1.0e-10


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def paired_ratio_ci(
    numerator: np.ndarray, denominator: np.ndarray, nboot: int, seed: int
) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    n = numerator.size
    indices = rng.integers(0, n, size=(nboot, n))
    ratios = np.mean(numerator[indices], axis=1) / np.mean(denominator[indices], axis=1)
    central = float(np.mean(numerator) / np.mean(denominator))
    return central, float(np.percentile(ratios, 2.5)), float(np.percentile(ratios, 97.5))


def relative_seed_spread(values: np.ndarray) -> float:
    """Return sample standard deviation relative to the statistic scale."""
    values = np.asarray(values, dtype=np.float64)
    if values.size < 2:
        raise ValueError("at least two seeds are required for a variance check")
    scale = max(
        float(np.abs(np.mean(values))),
        float(np.max(np.abs(values))),
        np.finfo(np.float64).tiny,
    )
    return float(np.std(values, ddof=1) / scale)


def find_seed_spread_failures(
    by_sigma: dict[float, np.ndarray],
    minimum_relative_spread: float = MIN_RELATIVE_SEED_SPREAD,
) -> list[dict]:
    """Identify deterministic or suspiciously near-deterministic ensembles."""
    failures = []
    for sigma, values in sorted(by_sigma.items()):
        variance = float(np.var(values, ddof=1))
        relative_spread = relative_seed_spread(values)
        if variance == 0.0 or relative_spread < minimum_relative_spread:
            failures.append(
                {
                    "sigma_dx": sigma,
                    "variance": variance,
                    "relative_spread": relative_spread,
                }
            )
    return failures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--benchmark", choices=("landau", "e2"), required=True)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--nboot", type=int, default=10000)
    parser.add_argument(
        "--min-relative-seed-spread",
        type=float,
        default=MIN_RELATIVE_SEED_SPREAD,
        help="reject smaller across-seed relative sample standard deviations",
    )
    args = parser.parse_args()
    filename = "landau_mise_seed_values.csv" if args.benchmark == "landau" else "E2_mise_seed_values.csv"
    rows: list[dict] = []
    for seed in range(args.seeds):
        path = args.input / f"seed{seed}" / "mise" / filename
        if not path.exists():
            raise FileNotFoundError(path)
        selected = read_csv(path)
        for row in selected:
            if int(row["seed"]) != seed:
                raise ValueError(f"seed mismatch in {path}")
        rows.extend(selected)

    sigmas = sorted({float(row["sigma_dx"]) for row in rows})
    by_sigma = {
        sigma: np.asarray(
            [float(row["mise_time_avg"]) for row in rows if float(row["sigma_dx"]) == sigma],
            dtype=np.float64,
        )
        for sigma in sigmas
    }
    if any(values.size != args.seeds for values in by_sigma.values()):
        raise RuntimeError("each sigma must contain exactly one value per paired seed")
    spread_failures = find_seed_spread_failures(by_sigma, args.min_relative_seed_spread)
    if spread_failures:
        args.output.mkdir(parents=True, exist_ok=True)
        summary_path = args.output / f"{args.benchmark}_noise_summary.csv"
        if summary_path.exists():
            summary_path.unlink()
        result = {
            "benchmark": args.benchmark,
            "status": "incomplete",
            "reason": "insufficient_across_seed_variance",
            "minimum_relative_seed_spread": args.min_relative_seed_spread,
            "failures": spread_failures,
            "n_seeds": args.seeds,
        }
        (args.output / f"{args.benchmark}_noise_result.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        raise RuntimeError(
            "INCOMPLETE: deterministic or suspiciously low across-seed variance "
            f"at {len(spread_failures)} sigma value(s)"
        )
    baseline = by_sigma[0.0]
    summary = []
    for sigma in sigmas:
        values = by_sigma[sigma]
        reduction, lo, hi = paired_ratio_ci(baseline, values, args.nboot, 91000 + int(1000 * sigma))
        _, mise_lo, mise_hi = paired_ratio_ci(values, np.ones_like(values), args.nboot, 92000 + int(1000 * sigma))
        summary.append(
            {
                "sigma_dx": sigma,
                "mise_mean": float(np.mean(values)),
                "mise_ci_lo": mise_lo,
                "mise_ci_hi": mise_hi,
                "noise_reduction": reduction,
                "noise_reduction_ci_lo": lo,
                "noise_reduction_ci_hi": hi,
                "n_seeds": args.seeds,
            }
        )
    best = min(summary, key=lambda row: row["mise_mean"])
    at_boundary = np.isclose(best["sigma_dx"], max(sigmas))
    result = {
        "benchmark": args.benchmark,
        "sigma_mise_dx": best["sigma_dx"],
        "sigma_mise_relation": ">=" if at_boundary else "=",
        "right_censored": bool(at_boundary),
        "search_max_dx": max(sigmas),
        "n_seeds": args.seeds,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / f"{args.benchmark}_noise_summary.csv", summary)
    (args.output / f"{args.benchmark}_noise_result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
