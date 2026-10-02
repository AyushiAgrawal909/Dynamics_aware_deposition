#!/usr/bin/env python3
"""
scripts/recompute_E1_criterion.py
=================================

Recompute E1 sigma-selection summaries from an existing raw E1 CSV.

This intentionally does not run PIC. It only changes the decision rule:
use the small-sigma runs as the per-mode reference, then compare the measured
sigma break point against the analytical matched-W prediction.
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import sys

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_E1_sigma_kmax import as_float, as_int, build_E1_criterion_tables, write_csv


def read_csv_rows(path: pathlib.Path) -> list[dict]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def infer_values(rows: list[dict], key: str, cast) -> list:
    values = {cast(row[key]) for row in rows}
    return sorted(values)


def main() -> None:
    parser = argparse.ArgumentParser(description="Recompute E1 criterion from existing raw CSV.")
    parser.add_argument("--raw-csv", type=pathlib.Path, required=True)
    parser.add_argument("--output-dir", type=pathlib.Path, default=None)
    parser.add_argument("--tag", type=str, default="refcriterion")
    parser.add_argument("--nboot", type=int, default=1000)
    parser.add_argument("--gamma-ref-sigma-max", type=float, default=0.5)
    parser.add_argument("--gamma-relative-tolerance", type=float, default=0.15)
    parser.add_argument("--a-min", type=float, default=0.85)
    parser.add_argument("--prediction-relative-tolerance", type=float, default=0.30)
    args = parser.parse_args()

    rows = read_csv_rows(args.raw_csv)
    output_dir = args.output_dir or args.raw_csv.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    k_modes = infer_values(rows, "k_mode", as_int)
    sigma_dx_list = infer_values(rows, "sigma_dx", as_float)
    summary_rows, sigma_opt_rows, result = build_E1_criterion_tables(
        rows=rows,
        k_modes=k_modes,
        sigma_dx_list=sigma_dx_list,
        nboot=args.nboot,
        gamma_ref_sigma_max=args.gamma_ref_sigma_max,
        gamma_relative_tolerance=args.gamma_relative_tolerance,
        a_min=args.a_min,
        prediction_relative_tolerance=args.prediction_relative_tolerance,
    )

    suffix = f"_{args.tag}" if args.tag else ""
    summary_path = output_dir / f"E1_summary{suffix}.csv"
    sigma_opt_path = output_dir / f"E1_sigma_opt{suffix}.csv"
    result_path = output_dir / f"E1_result{suffix}.json"
    write_csv(summary_path, summary_rows)
    write_csv(sigma_opt_path, sigma_opt_rows)

    result.update(
        {
            "raw_csv": str(args.raw_csv),
            "summary_csv": str(summary_path),
            "sigma_opt_csv": str(sigma_opt_path),
            "n_rows": len(rows),
            "k_modes": k_modes,
            "sigma_dx": sigma_dx_list,
        }
    )
    with result_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)

    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
