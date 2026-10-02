#!/usr/bin/env python3
"""Reduce the dense E2 anisotropic noise design with a fixed held-out split."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np


REPRO = Path(__file__).resolve().parent
if str(REPRO) not in sys.path:
    sys.path.insert(0, str(REPRO))

from e2_anisotropic_design import Q_BUDGET_GRID_TIME, make_design


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"refusing to write an empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def ratio_interval(
    baseline: np.ndarray,
    candidate: np.ndarray,
    nboot: int,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    central = float(np.mean(baseline) / np.mean(candidate))
    indices = rng.integers(0, baseline.size, size=(nboot, baseline.size))
    values = np.mean(baseline[indices], axis=1) / np.mean(candidate[indices], axis=1)
    return central, float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed-start", type=int, default=100)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--train-seeds", type=int, default=5)
    parser.add_argument("--nboot", type=int, default=5000)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    if args.seeds != 10 or args.train_seeds != 5:
        raise ValueError("the preregistered design requires ten seeds split five/five")

    expected_design = make_design(Q_BUDGET_GRID_TIME)
    labels = [row["label"] for row in expected_design]
    by_seed: dict[int, dict[str, dict]] = {}
    for seed in range(args.seed_start, args.seed_start + args.seeds):
        seed_root = args.input / f"seed{seed}" / "result"
        metadata = json.loads((seed_root / "result.json").read_text(encoding="utf-8"))
        if int(metadata["seed"]) != seed or metadata["status"] != "complete":
            raise AssertionError(f"incomplete or mismatched seed {seed}")
        if int(metadata["design_points"]) != len(expected_design):
            raise AssertionError("candidate count changed")
        if abs(float(metadata["q_budget"]) - Q_BUDGET_GRID_TIME) > 1.0e-15:
            raise AssertionError("admissibility budget changed")
        with (seed_root / "anisotropic_dense_noise_seed.csv").open(
            newline="", encoding="utf-8"
        ) as handle:
            rows = list(csv.DictReader(handle))
        if [row["label"] for row in rows] != labels:
            raise AssertionError(f"candidate design changed for seed {seed}")
        by_seed[seed] = {row["label"]: row for row in rows}

    seeds = np.asarray(sorted(by_seed), dtype=np.int64)
    train = seeds[: args.train_seeds]
    test = seeds[args.train_seeds :]
    if set(train).intersection(set(test)):
        raise AssertionError("training and test seed sets overlap")
    density = np.asarray(
        [[float(by_seed[seed][label]["density_mse"]) for label in labels] for seed in seeds],
        dtype=np.float64,
    )
    field = np.asarray(
        [[float(by_seed[seed][label]["field_mse"]) for label in labels] for seed in seeds],
        dtype=np.float64,
    )
    if np.any(density <= 0.0) or np.any(field <= 0.0):
        raise AssertionError("non-positive noise metric")
    train_indices = np.arange(args.train_seeds)
    test_indices = np.arange(args.train_seeds, args.seeds)
    density_selected = int(np.argmin(np.mean(density[train_indices], axis=0)))
    field_selected = int(np.argmin(np.mean(field[train_indices], axis=0)))
    density_test_best = int(np.argmin(np.mean(density[test_indices], axis=0)))
    field_test_best = int(np.argmin(np.mean(field[test_indices], axis=0)))

    rng = np.random.default_rng(271828)
    candidate_rows: list[dict] = []
    for index, design_row in enumerate(expected_design):
        density_ratio = ratio_interval(
            density[test_indices, 0], density[test_indices, index], args.nboot, rng
        )
        field_ratio = ratio_interval(
            field[test_indices, 0], field[test_indices, index], args.nboot, rng
        )
        candidate_rows.append(
            {
                **design_row,
                "train_density_mse": float(np.mean(density[train_indices, index])),
                "test_density_mse": float(np.mean(density[test_indices, index])),
                "test_density_reduction": density_ratio[0],
                "test_density_ci_lo": density_ratio[1],
                "test_density_ci_hi": density_ratio[2],
                "train_field_mse": float(np.mean(field[train_indices, index])),
                "test_field_mse": float(np.mean(field[test_indices, index])),
                "test_field_reduction": field_ratio[0],
                "test_field_ci_lo": field_ratio[1],
                "test_field_ci_hi": field_ratio[2],
                "selected_by_train_density": index == density_selected,
                "selected_by_train_field": index == field_selected,
                "best_on_test_density": index == density_test_best,
                "best_on_test_field": index == field_test_best,
            }
        )

    nested: dict[str, dict] = {}
    for metric_name, matrix in (("density", density), ("field", field)):
        selected_indices = np.empty(args.nboot, dtype=np.int64)
        reductions = np.empty(args.nboot, dtype=np.float64)
        for draw in range(args.nboot):
            train_draw = rng.integers(0, args.train_seeds, size=args.train_seeds)
            test_draw = rng.integers(args.train_seeds, args.seeds, size=args.seeds - args.train_seeds)
            selected_index = int(np.argmin(np.mean(matrix[train_draw], axis=0)))
            selected_indices[draw] = selected_index
            reductions[draw] = float(
                np.mean(matrix[test_draw, 0]) / np.mean(matrix[test_draw, selected_index])
            )
        values, counts = np.unique(selected_indices, return_counts=True)
        frequencies = sorted(
            (
                {"label": labels[int(index)], "fraction": float(count / args.nboot)}
                for index, count in zip(values, counts)
            ),
            key=lambda row: row["fraction"],
            reverse=True,
        )
        nested[metric_name] = {
            "selection_frequencies": frequencies,
            "heldout_reduction_ci_lo": float(np.percentile(reductions, 2.5)),
            "heldout_reduction_ci_hi": float(np.percentile(reductions, 97.5)),
        }

    args.output.mkdir(parents=True)
    write_csv(args.output / "anisotropic_dense_candidates.csv", candidate_rows)
    summary_rows: list[dict] = []
    for metric_name, matrix, selected_index, test_best_index in (
        ("density", density, density_selected, density_test_best),
        ("field", field, field_selected, field_test_best),
    ):
        row = candidate_rows[selected_index]
        ordered = np.argsort(np.mean(matrix[test_indices], axis=0))
        competitor_index = int(
            next(index for index in ordered if int(index) != selected_index)
            if int(ordered[0]) == selected_index
            else ordered[0]
        )
        comparison_draws = rng.integers(
            args.train_seeds, args.seeds, size=(args.nboot, args.seeds - args.train_seeds)
        )
        advantage = (
            np.mean(matrix[comparison_draws, competitor_index], axis=1)
            / np.mean(matrix[comparison_draws, selected_index], axis=1)
            - 1.0
        )
        summary_rows.append(
            {
                "metric": metric_name,
                "train_selected_label": labels[selected_index],
                "train_selected_sigma_x_dx": row["sigma_x_dx"],
                "train_selected_sigma_y_dy": row["sigma_y_dy"],
                "train_selected_q_fraction": row["q_fraction"],
                "train_selected_x_allocation": row["x_allocation"],
                "test_best_label": labels[test_best_index],
                "test_selection_matches_train": selected_index == test_best_index,
                "test_reduction": row[f"test_{metric_name}_reduction"],
                "test_ci_lo": row[f"test_{metric_name}_ci_lo"],
                "test_ci_hi": row[f"test_{metric_name}_ci_hi"],
                "nested_ci_lo": nested[metric_name]["heldout_reduction_ci_lo"],
                "nested_ci_hi": nested[metric_name]["heldout_reduction_ci_hi"],
                "nearest_test_competitor": labels[competitor_index],
                "relative_error_advantage": float(
                    np.mean(matrix[test_indices, competitor_index])
                    / np.mean(matrix[test_indices, selected_index])
                    - 1.0
                ),
                "advantage_ci_lo": float(np.percentile(advantage, 2.5)),
                "advantage_ci_hi": float(np.percentile(advantage, 97.5)),
                "bootstrap_fraction_advantage_positive": float(np.mean(advantage > 0.0)),
            }
        )
    write_csv(args.output / "anisotropic_heldout_summary.csv", summary_rows)

    result = {
        "status": "PASS",
        "protected_mode": [12, 1],
        "model_admissibility_budget": Q_BUDGET_GRID_TIME,
        "design": {
            "points": len(expected_design),
            "q_fractions": [0.0, 0.25, 0.5, 0.75, 1.0],
            "boundary_x_allocation_step": 0.05,
            "filter_family": "diagonal anisotropic Gaussian",
        },
        "fixed_split": {"train": train.tolist(), "test": test.tolist()},
        "metric_scope": {
            "density": "full-grid density squared-error power",
            "field": "pre-return-filter, pre-gather grid-field squared-error power",
        },
        "summary": summary_rows,
        "nested_bootstrap": nested,
        "optimality_statement": (
            "only discrete-design optimality among these 85 model-admissible diagonal Gaussian points; "
            "not continuous, filter-family-global, or particle-force optimality"
        ),
        "max_neutrality_defect": max(
            float(by_seed[seed][label]["max_neutrality_defect"])
            for seed in seeds
            for label in labels
        ),
        "max_relative_charge_error": max(
            float(by_seed[seed][label]["max_relative_charge_error"])
            for seed in seeds
            for label in labels
        ),
    }
    (args.output / "anisotropic_dense_audit.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
