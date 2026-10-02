#!/usr/bin/env python3
"""Expand comparison cells into an actionable component-level build plan."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def present(row: dict, *keys: str) -> bool:
    return all(row.get(key, "").strip() for key in keys)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cells",
        type=Path,
        default=Path("paper1_selector_comparison/results/comparison_cells.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("paper1_selector_comparison/results/coverage_manifest.csv"),
    )
    args = parser.parse_args()
    with args.cells.open(newline="", encoding="utf-8") as handle:
        cells = list(csv.DictReader(handle))

    rows = []
    for cell in cells:
        selector = cell["selector_id"]
        benchmark = cell["benchmark_id"]
        if selector == "delta_f":
            selection_status = "not-applicable"
        elif cell["sigma_dx"].strip():
            selection_status = "right-censored" if cell["sigma_relation"] == ">=" else "frozen"
        else:
            selection_status = "missing"

        rate_complete = present(
            cell,
            "gamma_fit",
            "gamma_ci_lo",
            "gamma_ci_hi",
            "rate_ratio_to_cic",
            "rate_ratio_ci_lo",
            "rate_ratio_ci_hi",
            "fit_censored",
            "fit_quality_pass",
            "resolved_efoldings",
            "noise_floor",
            "floor_t_start",
            "floor_t_stop",
            "floor_scope",
        )
        noise_complete = present(
            cell,
            "mise",
            "mise_cic",
            "noise_reduction",
            "noise_reduction_ci_lo",
            "noise_reduction_ci_hi",
        )

        if selector == "delta_f" and benchmark == "two_stream_e2":
            next_action = "implement and validate a bi-Maxwellian delta-f background before comparison"
        elif selector == "delta_f":
            next_action = "rerun validated Landau delta-f at PPC 1024 through the common matrix schema"
        elif selection_status == "missing":
            next_action = "run paired noise scan to select width; then evaluate that exact width in the rate harness"
        elif not rate_complete and not noise_complete:
            next_action = "run exact frozen width in both guarded-rate and paired-noise harnesses"
        elif not rate_complete:
            next_action = "complete guarded rate, paired CI, e-folding, and floor outputs at the exact width"
        elif not noise_complete:
            next_action = "complete paired MISE and noise-reduction interval at the benchmark noise PPC"
        else:
            next_action = "none; cell has complete raw metrics"

        rows.append(
            {
                "cell_id": f"{selector}__{benchmark}",
                "selector_id": selector,
                "benchmark_id": benchmark,
                "selection_status": selection_status,
                "rate_status": "complete" if rate_complete else "missing",
                "noise_status": "complete" if noise_complete else "missing",
                "next_action": next_action,
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    missing = sum(row["next_action"] != "none; cell has complete raw metrics" for row in rows)
    print(f"Wrote {args.output}; {missing}/{len(rows)} cells still require work")


if __name__ == "__main__":
    main()
