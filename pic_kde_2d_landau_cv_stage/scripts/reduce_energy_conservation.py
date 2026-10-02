#!/usr/bin/env python3
"""Reduce static-width energy histories and E2 field/kinetic partitions."""

from __future__ import annotations

import argparse
import csv
import json
import pathlib

import numpy as np


def bootstrap_ci(values, rng, nboot):
    values = np.asarray(values, dtype=np.float64)
    means = np.mean(values[rng.integers(0, len(values), size=(nboot, len(values)))], axis=1)
    return float(np.mean(values)), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--nboot", type=int, default=4000)
    args = parser.parse_args()
    raw = []
    for path in args.input.glob("*/*/seed*/energy_summary.csv"):
        with path.open(newline="", encoding="utf-8") as handle:
            raw.append(next(csv.DictReader(handle)))
    groups = sorted({(row["benchmark"], row["case"]) for row in raw})
    expected = {
        (benchmark, case)
        for benchmark in ("landau_weak", "landau_strong", "e2")
        for case in ("cic", "c4", "large")
    }
    if set(groups) != expected:
        raise RuntimeError(f"incomplete energy matrix: found {groups}")
    rng = np.random.default_rng(86211)
    summary = []
    for benchmark, case in groups:
        selected = [row for row in raw if row["benchmark"] == benchmark and row["case"] == case]
        seeds = sorted(int(row["seed"]) for row in selected)
        if seeds != list(range(args.seeds)):
            raise RuntimeError(f"{benchmark}/{case}: incomplete seeds {seeds}")
        drift = bootstrap_ci(
            [float(row["max_abs_relative_energy_drift"]) for row in selected],
            rng,
            args.nboot,
        )
        final = bootstrap_ci(
            [float(row["final_relative_energy_drift"]) for row in selected],
            rng,
            args.nboot,
        )
        slope = bootstrap_ci(
            [float(row["relative_energy_drift_slope_per_time"]) for row in selected],
            rng,
            args.nboot,
        )
        field_scaled = bootstrap_ci(
            [float(row["max_abs_drift_over_peak_field_energy"]) for row in selected],
            rng,
            args.nboot,
        )
        summary.append(
            {
                "benchmark": benchmark,
                "case": case,
                "sigma_x_dx": float(selected[0]["sigma_x_dx"]),
                "sigma_y_dy": float(selected[0]["sigma_y_dy"]),
                "max_abs_relative_drift_mean": drift[0],
                "max_abs_relative_drift_ci_lo": drift[1],
                "max_abs_relative_drift_ci_hi": drift[2],
                "final_relative_drift_mean": final[0],
                "final_relative_drift_ci_lo": final[1],
                "final_relative_drift_ci_hi": final[2],
                "drift_slope_mean_per_time": slope[0],
                "drift_slope_ci_lo": slope[1],
                "drift_slope_ci_hi": slope[2],
                "max_drift_over_peak_field_mean": field_scaled[0],
                "max_drift_over_peak_field_ci_lo": field_scaled[1],
                "max_drift_over_peak_field_ci_hi": field_scaled[2],
                "controller_refresh_work": 0.0,
                "n_seeds": args.seeds,
            }
        )

    # Preserve the full E2 ensemble partition for the growth/saturation figure.
    e2_traces = {}
    for case in ("cic", "c4", "large"):
        archives = sorted((args.input / "e2" / case).glob("seed*/energy_history.npz"))
        fields, kinetics = [], []
        times = None
        for archive in archives:
            with np.load(archive) as data:
                this_times = np.asarray(data["times"], dtype=np.float64)
                fields.append(np.asarray(data["field_energy"], dtype=np.float64))
                kinetics.append(np.asarray(data["kinetic_energy"], dtype=np.float64))
            if times is None:
                times = this_times
            else:
                np.testing.assert_array_equal(times, this_times)
        e2_traces[f"field_{case}"] = np.mean(fields, axis=0)
        e2_traces[f"kinetic_change_{case}"] = np.mean(kinetics, axis=0) - np.mean(kinetics, axis=0)[0]
    e2_traces["times"] = times

    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "energy_conservation_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    np.savez_compressed(args.output / "E2_field_kinetic_partition.npz", **e2_traces)
    result = {
        "static_width_refresh_work": 0.0,
        "rows": summary,
        "e2_partition_npz": str(args.output / "E2_field_kinetic_partition.npz"),
    }
    (args.output / "energy_conservation_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
