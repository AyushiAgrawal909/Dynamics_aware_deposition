#!/usr/bin/env python3
"""Create the preregistered comparison cells and missing-build manifest."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


CELL_FIELDS = [
    "selector_id",
    "selector_label",
    "benchmark_id",
    "benchmark_label",
    "benchmark_marginality",
    "eligible_selector",
    "sigma_relation",
    "sigma_dx",
    "sigma_over_c4",
    "matched_W2_relation",
    "matched_W2_log10",
    "rate_ppc",
    "noise_ppc",
    "gamma_reference_kind",
    "gamma_reference",
    "gamma_fit",
    "gamma_ci_lo",
    "gamma_ci_hi",
    "gamma_relative_error_to_theory",
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
    "mise",
    "mise_cic",
    "noise_reduction",
    "noise_reduction_ci_lo",
    "noise_reduction_ci_hi",
    "rate_gate",
    "noise_gate",
    "gate_status",
    "provenance",
    "notes",
]


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def build(config: dict) -> tuple[list[dict], list[dict]]:
    cells: list[dict] = []
    builds: list[dict] = []
    c4 = next(selector for selector in config["selectors"] if selector["id"] == "c4")
    for selector in config["selectors"]:
        for benchmark in config["benchmarks"]:
            benchmark_id = benchmark["id"]
            sigma = selector["sigma_dx"].get(benchmark_id)
            relation = selector.get("sigma_relation", {}).get(benchmark_id, "=")
            c4_sigma = float(c4["sigma_dx"][benchmark_id])
            sigma_over_c4 = float(sigma) / c4_sigma if sigma is not None else ""
            if sigma is None:
                matched_W2_log10 = ""
                matched_W2_relation = ""
            else:
                mode = float(benchmark["rate_protocol"]["mode"])
                nx = float(benchmark["physics"]["Nx"])
                exponent = (2.0 * math.pi * mode * float(sigma) / nx) ** 2
                matched_W2_log10 = -exponent / math.log(10.0)
                matched_W2_relation = "<=" if relation == ">=" else "="
            status = "REFERENCE_PENDING" if not selector["eligible"] else "PENDING"
            row = {field: "" for field in CELL_FIELDS}
            row.update(
                {
                    "selector_id": selector["id"],
                    "selector_label": selector["label"],
                    "benchmark_id": benchmark_id,
                    "benchmark_label": benchmark["label"],
                    "benchmark_marginality": benchmark["marginality_label"],
                    "eligible_selector": selector["eligible"],
                    "sigma_relation": relation,
                    "sigma_dx": "" if sigma is None else sigma,
                    "sigma_over_c4": sigma_over_c4,
                    "matched_W2_relation": matched_W2_relation,
                    "matched_W2_log10": matched_W2_log10,
                    "rate_ppc": benchmark["rate_protocol"]["ppc"],
                    "noise_ppc": benchmark["noise_protocol"]["ppc"],
                    "gamma_reference_kind": benchmark["rate_protocol"]["reference_kind"],
                    "gamma_reference": benchmark["rate_protocol"]["reference_gamma"],
                    "fit_censored": "",
                    "fit_quality_pass": "",
                    "rate_gate": "PENDING",
                    "noise_gate": "PENDING",
                    "gate_status": status,
                    "notes": "Width must be selected first." if sigma is None else "",
                }
            )
            cells.append(row)

            if selector["id"] == "cic":
                task = "aggregate validated baseline rate, floor, and MISE"
            elif selector["id"] == "delta_f":
                task = "run formulation-matched delta-f variance reference"
            elif sigma is None:
                task = "select width at noise PPC, then run exact width in rate harness"
            else:
                task = "run exact preregistered width in rate and noise harnesses"
            builds.append(
                {
                    "cell_id": f"{selector['id']}__{benchmark_id}",
                    "selector_id": selector["id"],
                    "benchmark_id": benchmark_id,
                    "rate_ppc": benchmark["rate_protocol"]["ppc"],
                    "noise_ppc": benchmark["noise_protocol"]["ppc"],
                    "sigma_relation": relation,
                    "sigma_dx": "" if sigma is None else sigma,
                    "task": task,
                    "status": "pending",
                }
            )
    return cells, builds


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("paper1_selector_comparison/config/comparison_matrix.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("paper1_selector_comparison/results"),
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    cells, builds = build(config)
    write_csv(args.output / "comparison_cells.csv", cells, CELL_FIELDS)
    write_csv(args.output / "build_manifest.csv", builds, list(builds[0]))
    print(f"Wrote {len(cells)} cells and {len(builds)} build tasks to {args.output}")


if __name__ == "__main__":
    main()
