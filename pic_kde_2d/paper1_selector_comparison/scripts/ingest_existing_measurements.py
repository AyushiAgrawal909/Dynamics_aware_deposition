#!/usr/bin/env python3
"""Populate comparison cells from validated Landau and isolated-E2 outputs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def read_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def as_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes"}


def update_landau(cell: dict, source: dict, theory: float, provenance: Path) -> None:
    gamma = float(source["gamma"])
    cell.update(
        {
            "gamma_fit": gamma,
            "gamma_ci_lo": source["gamma_ci_lo"],
            "gamma_ci_hi": source["gamma_ci_hi"],
            "gamma_relative_error_to_theory": abs(gamma - theory) / abs(theory),
            "rate_ratio_to_cic": source["gamma_ratio_to_cic"],
            "rate_ratio_ci_lo": source["ratio_ci_lo"],
            "rate_ratio_ci_hi": source["ratio_ci_hi"],
            # Adaptive floor truncation is a valid fit. Only a trajectory with
            # no pre-floor window is unresolved/censored for the matrix gate.
            "fit_censored": source["status"] == "censored-before-fit",
            "fit_quality_pass": source["quality_pass"],
            "resolved_efoldings": source["model_efoldings"],
            "noise_floor": source["noise_floor"],
            "floor_t_start": source["floor_t_start"],
            "floor_t_stop": source["floor_t_stop"],
            "floor_scope": "per-cell",
            "provenance": str(provenance),
        }
    )


def update_e2(cell: dict, source: dict, theory: float, provenance: Path) -> None:
    gamma = float(source["gamma_sigma"])
    clean = source["status"] in {"clean", "clean-baseline"}
    cell.update(
        {
            "gamma_fit": gamma,
            "gamma_ci_lo": source["gamma_ci_lo"],
            "gamma_ci_hi": source["gamma_ci_hi"],
            "gamma_relative_error_to_theory": abs(gamma - theory) / abs(theory),
            "rate_ratio_to_cic": source["gamma_ratio"],
            "rate_ratio_ci_lo": source["ratio_ci_lo"],
            "rate_ratio_ci_hi": source["ratio_ci_hi"],
            "fit_censored": not clean,
            "fit_quality_pass": clean,
            "resolved_efoldings": source["fit_efoldings"],
            "provenance": str(provenance),
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cells",
        type=Path,
        default=Path("paper1_selector_comparison/results/comparison_cells.csv"),
    )
    parser.add_argument(
        "--landau",
        type=Path,
        default=Path("results/landau_sigma_refinement_4864819/analysis/landau_sigma_refinement.csv"),
    )
    parser.add_argument(
        "--e2",
        type=Path,
        default=Path("results/E2_m8_sigma_sweep_4863559/E2_m8_sigma_sweep.csv"),
    )
    parser.add_argument(
        "--e2-mise",
        type=Path,
        default=Path("results/E2_mise_cv/E2_mise_summary.csv"),
    )
    args = parser.parse_args()
    cells = read_rows(args.cells)

    landau = read_rows(args.landau)
    landau_by_sigma = {
        float(row["sigma_dx"]): row for row in landau if int(row["ppc"]) == 1024
    }
    e2 = read_rows(args.e2) if args.e2.exists() else []
    e2_by_sigma = {float(row["sigma_dx"]): row for row in e2}
    e2_mise = read_rows(args.e2_mise)
    e2_mise_by_sigma = {float(row["sigma_dx"]): row for row in e2_mise}
    e2_mise_cic = float(e2_mise_by_sigma[0.0]["mise_mean"])

    for cell in cells:
        selector = cell["selector_id"]
        benchmark = cell["benchmark_id"]
        sigma_text = cell["sigma_dx"].strip()
        sigma = float(sigma_text) if sigma_text else None
        theory = float(cell["gamma_reference"])
        if benchmark == "weak_landau" and sigma in landau_by_sigma:
            update_landau(cell, landau_by_sigma[sigma], theory, args.landau)
        if benchmark == "two_stream_e2" and sigma in e2_by_sigma:
            update_e2(cell, e2_by_sigma[sigma], theory, args.e2)
        if benchmark == "two_stream_e2" and sigma in e2_mise_by_sigma:
            mise_row = e2_mise_by_sigma[sigma]
            mise = float(mise_row["mise_mean"])
            cell["mise"] = mise
            cell["mise_cic"] = e2_mise_cic
            cell["noise_reduction"] = e2_mise_cic / mise
            if selector == "cic":
                cell["noise_reduction_ci_lo"] = 1.0
                cell["noise_reduction_ci_hi"] = 1.0
        if selector == "cic" and cell["noise_reduction"]:
            cell["noise_gate"] = "REFERENCE"

    write_rows(args.cells, cells)
    print(f"Updated validated measurements in {args.cells}")


if __name__ == "__main__":
    main()
