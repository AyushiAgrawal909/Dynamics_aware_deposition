#!/usr/bin/env python3
"""Render the selector-by-benchmark matrix without promoting pending cells."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle


STATUS_COLORS = {
    "BASELINE": "#fff2cc",
    "PASS": "#d9ead3",
    "FAIL_RATE": "#f4cccc",
    "FAIL_NOISE": "#fce5cd",
    "SAFE_WINDOW_CLOSED": "#f9cb9c",
    "FAIL_RATE_AND_NOISE": "#e6b8af",
    "CENSORED": "#ead1dc",
    "REFERENCE": "#cfe2f3",
    "REFERENCE_PENDING": "#d9eaf7",
    "PENDING": "#eeeeee",
}


def number(row: dict, key: str) -> float | None:
    value = row.get(key, "").strip()
    try:
        return float(value) if value else None
    except ValueError:
        return None


def cell_text(row: dict) -> str:
    relation = row.get("sigma_relation", "=")
    sigma = row.get("sigma_dx", "")
    sigma_text = "sigma pending" if not sigma else f"sigma {relation} {float(sigma):g} dx"
    sigma_ratio = number(row, "sigma_over_c4")
    transfer_log10 = number(row, "matched_W2_log10")
    gamma_error = number(row, "gamma_relative_error_to_theory")
    efoldings = number(row, "resolved_efoldings")
    floor = number(row, "noise_floor")
    reduction = number(row, "noise_reduction")
    status = row.get("gate_status", "PENDING") or "PENDING"
    lines = [sigma_text]
    if sigma_ratio is not None and row["selector_id"] != "c4":
        lines.append(f"width / C4 {sigma_ratio:.2f}x")
    if relation == ">=":
        lines.append("selector right-censored")
    if transfer_log10 is not None and transfer_log10 < -3.0:
        transfer_relation = row.get("matched_W2_relation", "=")
        lines.append(f"mode W2 {transfer_relation} 10^{transfer_log10:.1f}")
    lines.append("rate pending" if gamma_error is None else f"rate error {100.0 * gamma_error:.1f}%")
    if efoldings is None or floor is None:
        lines.append("N_e / floor pending")
    else:
        lines.append(f"N_e {efoldings:.2f}; floor {floor:.2e}")
    lines.append("noise reduction pending" if reduction is None else f"noise reduction {reduction:.2f}x")
    lines.append(status.replace("_", " "))
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("paper1_selector_comparison/results/comparison_cells.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("paper1_selector_comparison/results/selector_benchmark_matrix.png"),
    )
    args = parser.parse_args()
    with args.input.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    selectors = list(dict.fromkeys(row["selector_id"] for row in rows))
    benchmarks = list(dict.fromkeys(row["benchmark_id"] for row in rows))
    selector_labels = {row["selector_id"]: row["selector_label"] for row in rows}
    benchmark_labels = {row["benchmark_id"]: row["benchmark_label"] for row in rows}
    benchmark_marginality = {row["benchmark_id"]: row["benchmark_marginality"] for row in rows}
    by_key = {(row["selector_id"], row["benchmark_id"]): row for row in rows}

    figure, axis = plt.subplots(figsize=(11.5, 12.0))
    axis.set_xlim(0, len(benchmarks))
    axis.set_ylim(0, len(selectors))
    axis.invert_yaxis()
    axis.set_xticks([index + 0.5 for index in range(len(benchmarks))])
    axis.set_xticklabels(
        [f"{benchmark_labels[item]}\n{benchmark_marginality[item]}" for item in benchmarks],
        fontsize=11,
        fontweight="bold",
    )
    axis.xaxis.tick_top()
    axis.set_yticks([index + 0.5 for index in range(len(selectors))])
    axis.set_yticklabels([selector_labels[item] for item in selectors], fontsize=11)
    axis.tick_params(length=0)

    for row_index, selector in enumerate(selectors):
        for column_index, benchmark in enumerate(benchmarks):
            row = by_key[(selector, benchmark)]
            status = row.get("gate_status", "PENDING") or "PENDING"
            color = STATUS_COLORS.get(status, STATUS_COLORS["PENDING"])
            axis.add_patch(Rectangle((column_index, row_index), 1, 1, facecolor=color, edgecolor="white", linewidth=3))
            axis.text(
                column_index + 0.5,
                row_index + 0.5,
                cell_text(row),
                ha="center",
                va="center",
                fontsize=9,
                linespacing=1.35,
            )

    axis.set_title("Preregistered selector comparison\n95% paired-inference gates", pad=52, fontsize=16, fontweight="bold")
    axis.text(
        0.5,
        -0.08,
        "A selector wins only with a CI-supported rate pass and CI-supported noise reduction > 1.",
        transform=axis.transAxes,
        ha="center",
        va="top",
        fontsize=10,
    )
    legend = [
        Patch(facecolor=STATUS_COLORS["PASS"], label="pass"),
        Patch(facecolor=STATUS_COLORS["BASELINE"], label="CIC baseline"),
        Patch(facecolor=STATUS_COLORS["FAIL_RATE"], label="rate failure"),
        Patch(facecolor=STATUS_COLORS["SAFE_WINDOW_CLOSED"], label="safe window closed"),
        Patch(facecolor=STATUS_COLORS["CENSORED"], label="censored"),
        Patch(facecolor=STATUS_COLORS["REFERENCE"], label="delta-f reference"),
        Patch(facecolor=STATUS_COLORS["PENDING"], label="pending build"),
    ]
    axis.legend(handles=legend, loc="lower center", bbox_to_anchor=(0.5, -0.16), ncol=4, frameon=False)
    for spine in axis.spines.values():
        spine.set_visible(False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(args.output, dpi=240, bbox_inches="tight")
    plt.close(figure)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
