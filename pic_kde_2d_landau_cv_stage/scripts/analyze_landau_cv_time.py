#!/usr/bin/env python3
"""Analyze dynamic LSCV bandwidth growth as a Landau mode damps away."""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from paper1_selector_comparison.figures.paperstyle import COLORS, SINGLE_COL, apply_style
from pic_kde_2d.methods.wu_qin_bandwidth import identify_cv_minimum


def quadratic_minimum(sigmas: np.ndarray, scores: np.ndarray) -> float:
    index = int(np.argmin(scores))
    if index == 0 or index == len(sigmas) - 1:
        return float(sigmas[index])
    x = sigmas[index - 1 : index + 2]
    y = scores[index - 1 : index + 2]
    a, b, _ = np.polyfit(x, y, 2)
    if not np.isfinite(a) or a <= 0.0:
        return float(sigmas[index])
    vertex = float(-b / (2.0 * a))
    return vertex if x[0] <= vertex <= x[-1] else float(sigmas[index])


def load_mode_envelope(trace_root: pathlib.Path) -> tuple[np.ndarray, np.ndarray]:
    records = []
    for path in trace_root.glob("seed*/mode_history.npz"):
        with np.load(path) as data:
            records.append(
                (
                    int(np.asarray(data["seed"]).ravel()[0]),
                    np.asarray(data["times"], dtype=np.float64),
                    np.asarray(data["mode"], dtype=np.complex128),
                )
            )
    records.sort(key=lambda item: item[0])
    if len(records) != 10:
        raise RuntimeError(f"expected 10 reference traces, found {len(records)}")
    times = records[0][1]
    amplitude = np.abs(np.mean(np.stack([item[2] for item in records]), axis=0))
    return times, amplitude / amplitude[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=pathlib.Path, required=True)
    parser.add_argument("--trace-root", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--expected-seeds", type=int, default=10)
    parser.add_argument("--nboot", type=int, default=2000)
    parser.add_argument("--static-sigma", type=float, default=7.24)
    parser.add_argument("--dynamic-average-sigma", type=float, default=11.50)
    args = parser.parse_args()
    rows = []
    for path in args.input.glob("seed*/landau_lscv_time_values.csv"):
        with path.open(newline="", encoding="utf-8") as handle:
            rows.extend(csv.DictReader(handle))
    seeds = sorted({int(row["seed"]) for row in rows})
    if len(seeds) != args.expected_seeds:
        raise RuntimeError(f"expected {args.expected_seeds} seeds, found {seeds}")
    times = np.asarray(sorted({float(row["time"]) for row in rows}))
    sigmas = np.asarray(sorted({float(row["sigma_dx"]) for row in rows}))
    lookup = {
        (int(row["seed"]), float(row["time"]), float(row["sigma_dx"])): float(row["lscv_score"])
        for row in rows
    }
    rng = np.random.default_rng(72991)
    summary = []
    for time_value in times:
        matrix = np.asarray(
            [
                [lookup[(seed, float(time_value), float(sigma))] for sigma in sigmas]
                for seed in seeds
            ]
        )
        mean_scores = np.mean(matrix, axis=0)
        selection = identify_cv_minimum(sigmas, mean_scores)
        selected = quadratic_minimum(sigmas, mean_scores) if selection.identified else float("nan")
        bootstrap = []
        deterministic = bool(np.all(np.var(matrix, axis=0) == 0.0))
        if not deterministic:
            for _ in range(args.nboot):
                indices = rng.integers(0, len(seeds), size=len(seeds))
                draw_scores = np.mean(matrix[indices], axis=0)
                draw_selection = identify_cv_minimum(sigmas, draw_scores)
                if draw_selection.identified:
                    bootstrap.append(quadratic_minimum(sigmas, draw_scores))
        summary.append(
            {
                "time": float(time_value),
                "sigma_cv_dx": selected,
                "sigma_cv_ci_lo": float(np.percentile(bootstrap, 2.5)) if bootstrap else float("nan"),
                "sigma_cv_ci_hi": float(np.percentile(bootstrap, 97.5)) if bootstrap else float("nan"),
                "minimum_status": selection.status,
                "bootstrap_interior_fraction": len(bootstrap) / float(args.nboot),
                "uncertainty_status": (
                    "deterministic-quiet-start; no ensemble CI"
                    if deterministic
                    else "independent-seed bootstrap"
                ),
            }
        )
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "landau_cv_time.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    result = {
        "static_sigma_cv_dx": args.static_sigma,
        "dynamic_window_average_sigma_cv_dx": args.dynamic_average_sigma,
        "time_rows": summary,
        "interpretation": (
            "As damping removes curvature, the density-optimal bandwidth grows; "
            "boundary values are censored rather than reported as optima."
        ),
    }
    (args.output / "landau_cv_time.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    apply_style()
    figure, axis = plt.subplots(figsize=(SINGLE_COL, 2.8))
    values = np.asarray([row["sigma_cv_dx"] for row in summary])
    low = np.asarray([row["sigma_cv_ci_lo"] for row in summary])
    high = np.asarray([row["sigma_cv_ci_hi"] for row in summary])
    valid = np.isfinite(values)
    axis.plot(times[valid], values[valid], color=COLORS["LOO_CV"], label=r"dynamic $\sigma_{CV}(t)$")
    axis.fill_between(times[valid], low[valid], high[valid], color=COLORS["LOO_CV"], alpha=0.18)
    axis.axhline(args.static_sigma, color=COLORS["WuQin"], linestyle="--", label="static CV")
    axis.axhline(args.dynamic_average_sigma, color=COLORS["MISE"], linestyle=":", label="dynamic-window average")
    axis.set_xlabel("time")
    axis.set_ylabel(r"$\sigma_{CV}/\Delta x$")
    axis.set_title("Density-optimal bandwidth grows as the mode disappears")
    mode_times, envelope = load_mode_envelope(args.trace_root)
    twin = axis.twinx()
    twin.plot(mode_times, envelope, color=COLORS["CIC"], alpha=0.45, label="damping envelope")
    twin.set_yscale("log")
    twin.set_ylabel("normalized mode amplitude")
    lines, labels = axis.get_legend_handles_labels()
    extra, extra_labels = twin.get_legend_handles_labels()
    axis.legend(lines + extra, labels + extra_labels, fontsize=7, loc="best")
    figure.tight_layout()
    figure.savefig(args.output / "fig_cv_time_sensitivity.pdf")
    figure.savefig(args.output / "fig_cv_time_sensitivity.png")
    plt.close(figure)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
