#!/usr/bin/env python3
"""Reduce repeated same-node timings into the Paper 1 cost table."""

from __future__ import annotations

import argparse
import csv
import json
import pathlib

import numpy as np


METHODS = ("cic", "c4", "deltaf")
TIMINGS = ("deposit", "solve", "filter", "gather", "push", "total")


def bootstrap_mean(values: np.ndarray, rng, nboot=10000):
    draws = np.mean(values[rng.integers(0, len(values), size=(nboot, len(values)))], axis=1)
    return float(np.mean(values)), float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--expected-repeats", type=int, default=5)
    args = parser.parse_args()
    files = sorted(args.input.glob("rep*/cost_*.csv"))
    if not files:
        files = sorted(args.input.glob("cost_*.csv"))
    records = []
    for path in files:
        with path.open(newline="", encoding="utf-8") as handle:
            row = next(csv.DictReader(handle))
        row["repeat"] = path.parent.name if path.parent != args.input else "rep0"
        records.append(row)
    repeats = sorted({row["repeat"] for row in records})
    if len(repeats) != args.expected_repeats:
        raise RuntimeError(f"expected {args.expected_repeats} repeats, found {repeats}")
    for repeat in repeats:
        found = {row["method"] for row in records if row["repeat"] == repeat}
        if found != set(METHODS):
            raise RuntimeError(f"{repeat}: incomplete methods {found}")
    rng = np.random.default_rng(99127)
    summary = []
    by_method = {}
    for method in METHODS:
        selected = [row for row in records if row["method"] == method]
        by_method[method] = selected
        output = {
            "method": method,
            "protocol": selected[0]["protocol"],
            "n_particles": int(selected[0]["n_particles"]),
            "steps_per_repeat": int(selected[0]["steps"]),
            "n_repeats": len(selected),
            "persistent_array_bytes": int(selected[0]["persistent_array_bytes"]),
            "persistent_bytes_per_particle": float(selected[0]["persistent_bytes_per_particle"]),
            "peak_rss_bytes_mean": float(np.mean([float(row["process_peak_rss_bytes"]) for row in selected])),
            "peak_rss_bytes_max": int(max(float(row["process_peak_rss_bytes"]) for row in selected)),
        }
        for timing in TIMINGS:
            values = np.asarray([float(row[f"{timing}_ms_per_step"]) for row in selected])
            mean, low, high = bootstrap_mean(values, rng)
            output[f"{timing}_ms_per_step"] = mean
            output[f"{timing}_ci_lo"] = low
            output[f"{timing}_ci_hi"] = high
        summary.append(output)
    cic_by_repeat = {
        row["repeat"]: float(row["total_ms_per_step"]) for row in by_method["cic"]
    }
    for output in summary:
        selected = by_method[output["method"]]
        ratios = np.asarray(
            [float(row["total_ms_per_step"]) / cic_by_repeat[row["repeat"]] for row in selected]
        )
        mean, low, high = bootstrap_mean(ratios, rng)
        output["overhead_vs_cic"] = mean
        output["overhead_ci_lo"] = low
        output["overhead_ci_hi"] = high
        output["extra_memory_vs_cic_bytes"] = (
            output["persistent_array_bytes"] - summary[0]["persistent_array_bytes"]
        )
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "computational_cost.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    result = {
        "rows": summary,
        "same_node_repeated_protocol": True,
        "framing_gate": {
            "c4_overhead_measured_not_assumed": True,
            "essentially_zero_threshold": 1.05,
            "c4_essentially_zero": summary[1]["overhead_ci_hi"] <= 1.05,
        },
    }
    (args.output / "computational_cost.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
