#!/usr/bin/env python3
"""Build a publication-safe E2 headline table with censored density selectors."""

from __future__ import annotations

import argparse
import csv
import pathlib


def read_csv(path: pathlib.Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: pathlib.Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}g}"


def build(args: argparse.Namespace) -> None:
    headline = read_csv(args.headline_csv)
    selector = read_csv(args.selector_csv)
    rows_by_choice = {row["choice"]: row for row in headline}
    cic = rows_by_choice["cic"]
    density = rows_by_choice["density_mise"]
    cv = rows_by_choice["density_cv"]
    selector_row = min(selector, key=lambda row: abs(float(row["eps_gamma"]) - args.eps_gamma))
    constrained_sigma = float(selector_row["sigma_star"])
    constrained_ratio = float(selector_row["predicted_ratio"])
    constrained_noise = float(selector_row["noise_reduction_at_star"])

    output_rows = [
        {
            "choice": "baseline",
            "selection_rule": "CIC unsmoothed reference",
            "sigma_over_dx": "0",
            "density_MISE": fmt(float(cic["density_MISE"])),
            "noise_reduction": "1",
            "R_gamma_to_CIC": "1",
            "admissible": "reference",
            "note": "Defines gamma baseline.",
        },
        {
            "choice": "MISE oracle",
            "selection_rule": "minimum reference-density MISE",
            "sigma_over_dx": f">= {fmt(float(density['sigma_dx']), 3)}",
            "density_MISE": fmt(float(density["density_MISE"])),
            "noise_reduction": fmt(float(density["noise_reduction"])),
            "R_gamma_to_CIC": fmt(float(density["gamma_ratio_to_cic"])),
            "admissible": "no",
            "note": "Right-censored by search range; selected point reverses fitted growth.",
        },
        {
            "choice": "LOO-CV",
            "selection_rule": "least-squares leave-one-out CV",
            "sigma_over_dx": f">= {fmt(float(cv['sigma_dx']), 3)}",
            "density_MISE": fmt(float(cv["density_MISE"])),
            "noise_reduction": fmt(float(cv["noise_reduction"])),
            "R_gamma_to_CIC": fmt(float(cv["gamma_ratio_to_cic"])),
            "admissible": "no",
            "note": "Operational density selector is also right-censored at the largest scanned sigma.",
        },
        {
            "choice": "constrained",
            "selection_rule": f"max noise reduction subject to CI-supported R_gamma >= {1.0 - args.eps_gamma:.2f}",
            "sigma_over_dx": fmt(constrained_sigma, 3),
            "density_MISE": "pending paired run",
            "noise_reduction": fmt(constrained_noise),
            "R_gamma_to_CIC": fmt(constrained_ratio),
            "admissible": "pending precision",
            "note": "Central estimate only until paired common-random-number CI returns.",
        },
    ]
    write_csv(args.output, output_rows)
    print(f"wrote {args.output}")


def build_parser() -> argparse.ArgumentParser:
    root = pathlib.Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headline-csv", type=pathlib.Path, default=root / "results" / "part2" / "headline_comparison" / "headline_table.csv")
    parser.add_argument("--selector-csv", type=pathlib.Path, default=root / "results" / "part2" / "selector_result_cic.csv")
    parser.add_argument("--eps-gamma", type=float, default=0.10)
    parser.add_argument("--output", type=pathlib.Path, default=root / "results" / "part2" / "headline_comparison" / "headline_table_censored.csv")
    return parser


if __name__ == "__main__":
    build(build_parser().parse_args())

