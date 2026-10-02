#!/usr/bin/env python3
"""Merge p=2,3 compensated-binomial rate/noise rows across benchmarks."""

from __future__ import annotations

import argparse
import csv
import json
import pathlib

import numpy as np


ORDERS = (2, 3)
PASSES = (0, 1, 2, 4)


def load(path: pathlib.Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def response(mode: int, cells: int, passes: int, order: int) -> float:
    s = float(np.sin(np.pi * mode / cells) ** 2)
    coefficient = 1.0
    compensation = 1.0
    for index in range(1, order + 1):
        coefficient *= (passes + index - 1) / index
        compensation += coefficient * s**index
    return float((1.0 - s) ** passes * compensation)


def merge_one(
    benchmark: str,
    order: int,
    rate: dict,
    noise: dict,
    mode: int,
    cells: int,
) -> list[dict]:
    for payload in (rate, noise):
        if payload.get("filter_family") != "compensated_binomial":
            raise ValueError(f"{benchmark} p={order}: wrong filter family")
        if int(payload.get("compensation_order", -1)) != order:
            raise ValueError(f"{benchmark} p={order}: wrong compensation order")
        if tuple(payload["passes"]) != PASSES:
            raise ValueError(f"{benchmark} p={order}: wrong pass grid")
    rate_rows = {int(row["passes"]): row for row in rate["rows"]}
    noise_rows = {int(row["passes"]): row for row in noise["rows"]}
    rows = []
    for passes in PASSES:
        rate_row = rate_rows[passes]
        noise_row = noise_rows[passes]
        transfer = response(mode, cells, passes, order)
        rows.append(
            {
                "benchmark": benchmark,
                "compensation_order": order,
                "passes": passes,
                "principal_W": transfer,
                "matched_principal_W2": transfer**2,
                "protected_attenuation": 1.0 - transfer**2,
                "gamma": rate_row["gamma"],
                "gamma_ratio_to_cic": rate_row["gamma_ratio_to_cic"],
                "ratio_ci_lo": rate_row["ratio_ci_lo"],
                "ratio_ci_hi": rate_row["ratio_ci_hi"],
                "rate_gate": rate_row["gate_status"],
                "noise_reduction": noise_row["noise_reduction"],
                "noise_reduction_ci_lo": noise_row["noise_reduction_ci_lo"],
                "noise_reduction_ci_hi": noise_row["noise_reduction_ci_hi"],
                "useful_gate": bool(
                    rate_row["gate_status"] == "PASS"
                    and noise_row["noise_reduction_ci_lo"] > 1.0
                ),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()

    rows = []
    for order in ORDERS:
        base = args.input / f"p{order}"
        rows += merge_one(
            "weak_landau",
            order,
            load(base / "landau_rate" / "landau_binomial_passes.json"),
            load(base / "landau_noise" / "binomial_noise_summary.json"),
            mode=1,
            cells=64,
        )
        rows += merge_one(
            "near_marginal_two_stream_m12",
            order,
            load(base / "e2_rate" / "E2_binomial_passes.json"),
            load(base / "e2_noise" / "binomial_noise_summary.json"),
            mode=12,
            cells=128,
        )

    summary = {}
    for benchmark in ("weak_landau", "near_marginal_two_stream_m12"):
        candidates = [
            row
            for row in rows
            if row["benchmark"] == benchmark and row["useful_gate"]
        ]
        best = max(candidates, key=lambda row: row["noise_reduction"], default=None)
        summary[benchmark] = {
            "best_setting": (
                {
                    "compensation_order": best["compensation_order"],
                    "passes": best["passes"],
                    "gamma_ratio_to_cic": best["gamma_ratio_to_cic"],
                    "ratio_ci_lo": best["ratio_ci_lo"],
                    "ratio_ci_hi": best["ratio_ci_hi"],
                    "noise_reduction": best["noise_reduction"],
                    "noise_reduction_ci_lo": best["noise_reduction_ci_lo"],
                    "noise_reduction_ci_hi": best["noise_reduction_ci_hi"],
                }
                if best is not None
                else None
            )
        }

    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "higher_order_compensation.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result = {
        "constraint": "paired 95% CI for gamma/gamma_CIC contained in [0.90, 1.10]",
        "orders": list(ORDERS),
        "passes": list(PASSES),
        "rows": rows,
        "summary": summary,
    }
    (args.output / "higher_order_compensation.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
