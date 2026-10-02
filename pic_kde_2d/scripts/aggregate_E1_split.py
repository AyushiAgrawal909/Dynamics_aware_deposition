#!/usr/bin/env python3
"""
Aggregate split E1 shard outputs.

Each shard may run one or more (k_mode, sigma_dx) combinations. This script
combines raw per-seed gamma rows and ensemble-gamma rows, then recomputes the
standard and ensemble E1 criteria in one output directory.
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

from run_E1_sigma_kmax import (
    as_float,
    as_int,
    build_E1_criterion_tables,
    build_E1_ensemble_criterion_tables,
    write_csv,
)


def read_csv_rows(path: pathlib.Path) -> list[dict]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def infer_values(rows: list[dict], key: str, cast) -> list:
    return sorted({cast(row[key]) for row in rows if row.get(key) not in (None, "")})


def collect_rows(root: pathlib.Path, pattern: str) -> list[dict]:
    rows: list[dict] = []
    for path in sorted(root.rglob(pattern)):
        rows.extend(read_csv_rows(path))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate split E1 shard outputs.")
    parser.add_argument("--input-root", type=pathlib.Path, required=True)
    parser.add_argument("--output-dir", type=pathlib.Path, default=None)
    parser.add_argument("--tag", type=str, default="split")
    parser.add_argument("--nboot", type=int, default=1000)
    parser.add_argument("--gamma-ref-sigma-max", type=float, default=0.5)
    parser.add_argument("--gamma-relative-tolerance", type=float, default=0.15)
    parser.add_argument("--a-min", type=float, default=0.85)
    parser.add_argument("--prediction-relative-tolerance", type=float, default=0.30)
    args = parser.parse_args()

    output_dir = args.output_dir or (args.input_root / "E1_aggregate")
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{args.tag}" if args.tag else ""

    raw_rows = collect_rows(args.input_root, "E1_sigma_kmax*.csv")
    # Avoid re-ingesting aggregate outputs when input-root == output-dir parent.
    raw_rows = [row for row in raw_rows if "gamma_measured" in row]
    ensemble_rows = collect_rows(args.input_root, "E1_ensemble_gamma*.csv")
    ensemble_rows = [row for row in ensemble_rows if "gamma_ensemble" in row]

    if not raw_rows:
        raise SystemExit(f"No raw E1 rows found under {args.input_root}")
    if not ensemble_rows:
        raise SystemExit(f"No ensemble E1 rows found under {args.input_root}")

    raw_path = output_dir / f"E1_sigma_kmax{suffix}.csv"
    ensemble_raw_path = output_dir / f"E1_ensemble_gamma{suffix}.csv"
    write_csv(raw_path, raw_rows)
    write_csv(ensemble_raw_path, ensemble_rows)

    k_modes = infer_values(raw_rows, "k_mode", as_int)
    sigma_dx_list = infer_values(raw_rows, "sigma_dx", as_float)

    summary_rows, sigma_opt_rows, result = build_E1_criterion_tables(
        rows=raw_rows,
        k_modes=k_modes,
        sigma_dx_list=sigma_dx_list,
        nboot=args.nboot,
        gamma_ref_sigma_max=args.gamma_ref_sigma_max,
        gamma_relative_tolerance=args.gamma_relative_tolerance,
        a_min=args.a_min,
        prediction_relative_tolerance=args.prediction_relative_tolerance,
    )
    ensemble_summary_rows, ensemble_sigma_opt_rows, ensemble_result = build_E1_ensemble_criterion_tables(
        ensemble_rows=ensemble_rows,
        k_modes=k_modes,
        sigma_dx_list=sigma_dx_list,
        gamma_ref_sigma_max=args.gamma_ref_sigma_max,
        gamma_relative_tolerance=args.gamma_relative_tolerance,
        a_min=args.a_min,
        prediction_relative_tolerance=args.prediction_relative_tolerance,
    )

    summary_path = output_dir / f"E1_summary{suffix}.csv"
    sigma_opt_path = output_dir / f"E1_sigma_opt{suffix}.csv"
    ensemble_summary_path = output_dir / f"E1_ensemble_summary{suffix}.csv"
    ensemble_sigma_opt_path = output_dir / f"E1_ensemble_sigma_opt{suffix}.csv"
    result_path = output_dir / f"E1_result{suffix}.json"

    write_csv(summary_path, summary_rows)
    write_csv(sigma_opt_path, sigma_opt_rows)
    write_csv(ensemble_summary_path, ensemble_summary_rows)
    write_csv(ensemble_sigma_opt_path, ensemble_sigma_opt_rows)

    result.update(
        {
            "raw_csv": str(raw_path),
            "summary_csv": str(summary_path),
            "sigma_opt_csv": str(sigma_opt_path),
            "ensemble_raw_csv": str(ensemble_raw_path),
            "ensemble_summary_csv": str(ensemble_summary_path),
            "ensemble_sigma_opt_csv": str(ensemble_sigma_opt_path),
            "ensemble_result": ensemble_result,
            "n_raw_rows": len(raw_rows),
            "n_ensemble_rows": len(ensemble_rows),
            "k_modes": k_modes,
            "sigma_dx": sigma_dx_list,
        }
    )
    with result_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)

    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
