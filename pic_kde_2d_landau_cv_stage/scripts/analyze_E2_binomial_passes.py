#!/usr/bin/env python3
"""Analyze the projected m=12 matched-binomial pass budget."""

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
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from analyze_E2_mode_growth import select_growth_window
from analyze_linearized_delta_f_m12_sigma import (
    estimate_growing_floor,
    fit_rate,
)


PASSES = (0, 1, 2, 4, 8, 12, 13, 16)


def load_group(
    root: pathlib.Path,
    passes: int,
    seeds: int,
    filter_family: str,
    compensation_order: int,
) -> dict:
    histories = []
    times = None
    for seed in range(seeds):
        path = root / f"passes{passes}" / f"seed{seed}"
        metadata = json.loads((path / "result.json").read_text(encoding="utf-8"))
        if (
            metadata["filter_kind"] != filter_family
            or metadata["binomial_passes_x"] != passes
            or (
                filter_family == "compensated_binomial"
                and int(metadata.get("compensation_order", 1))
                != compensation_order
            )
        ):
            raise ValueError(f"filter metadata mismatch in {path}")
        with np.load(path / "mode_history.npz") as data:
            this_times = np.asarray(data["times"], dtype=np.float64)
            histories.append(np.abs(np.asarray(data["mode"], dtype=np.complex128)))
        if times is None:
            times = this_times
        else:
            np.testing.assert_array_equal(times, this_times)
    assert times is not None
    return {"times": times, "amplitudes": np.stack(histories)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--nboot", type=int, default=2000)
    parser.add_argument("--eps-gamma", type=float, default=0.10)
    parser.add_argument(
        "--filter-family",
        choices=("binomial", "compensated_binomial"),
        default="binomial",
    )
    parser.add_argument("--passes", type=int, nargs="+", default=list(PASSES))
    parser.add_argument("--compensation-order", type=int, default=1)
    args = parser.parse_args()
    passes_values = tuple(args.passes)
    if len(set(passes_values)) != len(passes_values) or 0 not in passes_values:
        raise ValueError("--passes must be unique and include the CIC baseline 0")
    args.output.mkdir(parents=True, exist_ok=True)
    groups = {
        passes: load_group(
            args.input,
            passes,
            args.seeds,
            args.filter_family,
            args.compensation_order,
        )
        for passes in passes_values
    }
    times = groups[0]["times"]
    baseline_mean = np.mean(groups[0]["amplitudes"], axis=0)
    window = select_growth_window(
        times,
        baseline_mean,
        t_min=0.3,
        t_start_max=min(30.0, float(times[-1]) - 5.0),
        t_max=float(times[-1]),
        min_duration=5.0,
        min_points=30,
        target_r2=0.98,
        saturation_fraction=0.95,
    )
    start = float(window["t_start"])
    stop = float(window["t_stop"])
    central = {
        passes: fit_rate(groups[passes]["amplitudes"], times, start, stop)
        for passes in passes_values
    }
    rng = np.random.default_rng(84120)
    ratio_samples = {passes: [] for passes in passes_values}
    rate_samples = {passes: [] for passes in passes_values}
    for _ in range(args.nboot):
        indices = rng.integers(0, args.seeds, size=args.seeds)
        rates = {
            passes: fit_rate(groups[passes]["amplitudes"][indices], times, start, stop)
            for passes in passes_values
        }
        if rates[0] <= 0.0 or not all(np.isfinite(list(rates.values()))):
            continue
        for passes in passes_values:
            rate_samples[passes].append(rates[passes])
            ratio_samples[passes].append(rates[passes] / rates[0])
    rows = []
    for passes in passes_values:
        ratio = central[passes] / central[0]
        ratios = np.asarray(ratio_samples[passes])
        rates = np.asarray(rate_samples[passes])
        floor = estimate_growing_floor(times, groups[passes]["amplitudes"], start)
        lo = float(np.percentile(ratios, 2.5))
        hi = float(np.percentile(ratios, 97.5))
        ci_pass = lo >= 1.0 - args.eps_gamma and hi <= 1.0 + args.eps_gamma
        rows.append(
            {
                "passes": passes,
                "sigma_eq_dx": (
                    float(np.sqrt(passes / 2.0))
                    if args.filter_family == "binomial"
                    else None
                ),
                "filter_family": args.filter_family,
                "compensation_order": (
                    args.compensation_order
                    if args.filter_family == "compensated_binomial"
                    else None
                ),
                "gamma": central[passes],
                "gamma_ratio_to_cic": ratio,
                "ratio_sem": float(np.std(ratios)),
                "ratio_ci_lo": lo,
                "ratio_ci_hi": hi,
                "gamma_ci_lo": float(np.percentile(rates, 2.5)),
                "gamma_ci_hi": float(np.percentile(rates, 97.5)),
                "fit_t_start": start,
                "fit_t_stop": stop,
                "fit_r_squared": float(window["r2"]),
                "fit_efoldings": float(window["efoldings"]),
                **floor,
                "ci_supported_rate_gate": ci_pass,
                "gate_status": "PASS" if ci_pass else "FAIL",
            }
        )
    passing = [row["passes"] for row in rows if row["gate_status"] == "PASS"]
    result = {
        "benchmark": "E2 projected linearized delta-f m=12",
        "filter_family": args.filter_family,
        "compensation_order": (
            args.compensation_order
            if args.filter_family == "compensated_binomial"
            else None
        ),
        "passes": list(passes_values),
        "largest_measured_passing_n": max(passing) if passing else None,
        "analytic_n_max": 0,
        "analytic_ceiling_sigma_dx": 0.1844549,
        "rows": rows,
    }
    with (args.output / "E2_binomial_passes.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output / "E2_binomial_passes.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
