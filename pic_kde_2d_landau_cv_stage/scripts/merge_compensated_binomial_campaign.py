#!/usr/bin/env python3
"""Merge compensated-binomial rate and noise gates for Paper 1."""

from __future__ import annotations

import argparse
import csv
import json
import pathlib

import numpy as np


EXPECTED_PASSES = (0, 1, 2, 4, 8, 16)


def load(path: pathlib.Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def principal_response(mode: int, cells: int, passes: int) -> float:
    phase = np.pi * mode / cells
    return float(
        np.cos(phase) ** (2 * passes)
        * (1.0 + passes * np.sin(phase) ** 2)
    )


def merge_benchmark(
    benchmark: str,
    rate: dict,
    noise: dict,
    mode: int,
    cells: int,
) -> list[dict]:
    for payload in (rate, noise):
        if payload.get("filter_family") != "compensated_binomial":
            raise ValueError(f"{benchmark}: expected compensated-binomial payload")
        if tuple(payload["passes"]) != EXPECTED_PASSES:
            raise ValueError(f"{benchmark}: unexpected pass grid {payload['passes']}")
    rate_by_pass = {int(row["passes"]): row for row in rate["rows"]}
    noise_by_pass = {int(row["passes"]): row for row in noise["rows"]}
    rows = []
    for passes in EXPECTED_PASSES:
        rate_row = rate_by_pass[passes]
        noise_row = noise_by_pass[passes]
        response = principal_response(mode, cells, passes)
        rows.append(
            {
                "benchmark": benchmark,
                "filter_family": "compensated_binomial",
                "passes": passes,
                "principal_W": response,
                "matched_principal_W2": response**2,
                "gamma": rate_row["gamma"],
                "gamma_ratio_to_cic": rate_row["gamma_ratio_to_cic"],
                "ratio_ci_lo": rate_row["ratio_ci_lo"],
                "ratio_ci_hi": rate_row["ratio_ci_hi"],
                "fit_r_squared": rate_row["fit_r_squared"],
                "fit_efoldings": rate_row.get(
                    "fit_efoldings",
                    rate_row.get("model_efoldings"),
                ),
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
    parser.add_argument("--landau-rate", type=pathlib.Path, required=True)
    parser.add_argument("--landau-noise", type=pathlib.Path, required=True)
    parser.add_argument("--e2-rate", type=pathlib.Path, required=True)
    parser.add_argument("--e2-noise", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()

    rows = merge_benchmark(
        "weak_landau",
        load(args.landau_rate),
        load(args.landau_noise),
        mode=1,
        cells=64,
    )
    rows += merge_benchmark(
        "near_marginal_two_stream_m12",
        load(args.e2_rate),
        load(args.e2_noise),
        mode=12,
        cells=128,
    )
    summary = {}
    for benchmark in ("weak_landau", "near_marginal_two_stream_m12"):
        selected = [row for row in rows if row["benchmark"] == benchmark]
        passing = [row for row in selected if row["rate_gate"] == "PASS"]
        useful = [row for row in selected if row["useful_gate"]]
        summary[benchmark] = {
            "largest_rate_safe_passes": max(
                (row["passes"] for row in passing),
                default=None,
            ),
            "largest_useful_passes": max(
                (row["passes"] for row in useful),
                default=None,
            ),
            "best_ci_supported_noise_reduction": max(
                (row["noise_reduction"] for row in useful),
                default=None,
            ),
        }

    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "compensated_binomial_rows.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result = {
        "constraint": "paired 95% CI for gamma/gamma_CIC contained in [0.90, 1.10]",
        "filter_chain": "CIC -> W_comp -> Poisson -> W_comp -> CIC^T",
        "passes": list(EXPECTED_PASSES),
        "rows": rows,
        "summary": summary,
    }
    (args.output / "compensated_binomial_rows.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
