#!/usr/bin/env python3
"""Apply the preregistered CI-supported gates to comparison cells."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


def read_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def value(row: dict, key: str) -> float | None:
    text = row.get(key, "").strip()
    try:
        result = float(text)
    except ValueError:
        return None
    return result if math.isfinite(result) else None


def flag(row: dict, key: str) -> bool | None:
    text = row.get(key, "").strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    return None


def evaluate(rows: list[dict], config: dict) -> dict:
    gates = config["gates"]
    tolerance = float(gates["rate_ratio_tolerance"])
    theory_tolerance = float(gates["baseline_theory_relative_tolerance"])
    baseline_ok: dict[str, bool] = {}
    for row in rows:
        if row["selector_id"] == "cic":
            error = value(row, "gamma_relative_error_to_theory")
            baseline_ok[row["benchmark_id"]] = bool(
                error is not None
                and error <= theory_tolerance
                and flag(row, "fit_quality_pass") is True
                and flag(row, "fit_censored") is False
            )

    for row in rows:
        if row["selector_id"] == "delta_f":
            complete = all(
                value(row, key) is not None
                for key in ("gamma_fit", "gamma_relative_error_to_theory", "noise_reduction")
            )
            row["gate_status"] = "REFERENCE" if complete else "REFERENCE_PENDING"
            continue

        if row["selector_id"] == "cic":
            complete = all(
                value(row, key) is not None
                for key in (
                    "gamma_fit",
                    "resolved_efoldings",
                    "noise_floor",
                    "floor_t_start",
                    "floor_t_stop",
                    "noise_reduction",
                )
            )
            complete = complete and row.get("floor_scope") == "per-cell"
            row["rate_gate"] = "PASS" if baseline_ok.get(row["benchmark_id"], False) else "FAIL"
            row["noise_gate"] = "REFERENCE"
            row["gate_status"] = "BASELINE" if complete and baseline_ok.get(row["benchmark_id"], False) else "PENDING"
            continue

        required_rate = [
            value(row, "gamma_fit"),
            value(row, "rate_ratio_ci_lo"),
            value(row, "rate_ratio_ci_hi"),
            value(row, "resolved_efoldings"),
            value(row, "noise_floor"),
            value(row, "floor_t_start"),
            value(row, "floor_t_stop"),
        ]
        required_noise = [
            value(row, "noise_reduction"),
            value(row, "noise_reduction_ci_lo"),
            value(row, "noise_reduction_ci_hi"),
        ]
        if any(item is None for item in required_rate + required_noise) or row.get("floor_scope") != "per-cell":
            row["gate_status"] = "PENDING"
            continue
        if flag(row, "fit_censored") is True:
            row["rate_gate"] = "CENSORED"
            row["gate_status"] = "CENSORED"
            continue

        rate_pass = bool(
            baseline_ok.get(row["benchmark_id"], False)
            and flag(row, "fit_quality_pass") is True
            and required_rate[1] >= 1.0 - tolerance
            and required_rate[2] <= 1.0 + tolerance
        )
        noise_pass = bool(
            required_noise[0] > gates["noise_reduction_minimum"]
            and required_noise[1] > gates["noise_reduction_minimum"]
        )
        row["rate_gate"] = "PASS" if rate_pass else "FAIL"
        row["noise_gate"] = "PASS" if noise_pass else "FAIL"
        if rate_pass and noise_pass:
            row["gate_status"] = "PASS"
        elif not rate_pass and not noise_pass:
            row["gate_status"] = "FAIL_RATE_AND_NOISE"
        elif not rate_pass:
            row["gate_status"] = "FAIL_RATE"
        else:
            row["gate_status"] = "SAFE_WINDOW_CLOSED" if row["selector_id"] == "c4" else "FAIL_NOISE"

    eligible = [item for item in config["selectors"] if item["eligible"]]
    benchmarks = [item["id"] for item in config["benchmarks"]]
    winners = []
    for selector in eligible:
        statuses = [
            row["gate_status"]
            for row in rows
            if row["selector_id"] == selector["id"] and row["benchmark_id"] in benchmarks
        ]
        if len(statuses) == len(benchmarks) and all(status == "PASS" for status in statuses):
            winners.append(selector["id"])
    complete = all(row["gate_status"] not in {"PENDING", "REFERENCE_PENDING"} for row in rows)
    return {
        "complete": complete,
        "eligible_winners": winners,
        "preregistered_claim": config["claim"],
        "claim_supported": complete and winners == ["c4"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("paper1_selector_comparison/config/comparison_matrix.json"),
    )
    parser.add_argument(
        "--cells",
        type=Path,
        default=Path("paper1_selector_comparison/results/comparison_cells.csv"),
    )
    parser.add_argument(
        "--verdict",
        type=Path,
        default=Path("paper1_selector_comparison/results/comparison_verdict.json"),
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    rows = read_rows(args.cells)
    verdict = evaluate(rows, config)
    write_rows(args.cells, rows)
    args.verdict.write_text(json.dumps(verdict, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(verdict, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
