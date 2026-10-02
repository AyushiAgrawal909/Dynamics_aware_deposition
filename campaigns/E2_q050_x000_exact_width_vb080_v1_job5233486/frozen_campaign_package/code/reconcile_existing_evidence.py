#!/usr/bin/env python3
"""Reconcile archived q050 intervals and correct interpretation-only metadata."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def ratio_draws(
    baseline: np.ndarray,
    candidate: np.ndarray,
    rng: np.random.Generator,
    draws: int = 5000,
) -> tuple[float, float, float]:
    indices = rng.integers(0, baseline.size, size=(draws, baseline.size))
    values = np.mean(baseline[indices], axis=1) / np.mean(candidate[indices], axis=1)
    return (
        float(np.mean(baseline) / np.mean(candidate)),
        float(np.percentile(values, 2.5)),
        float(np.percentile(values, 97.5)),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retrieval", type=Path, required=True)
    parser.add_argument("--archived-analysis", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    args.output.mkdir(parents=True)

    per_seed = read_csv(args.retrieval / "bandwidth_85_per_seed_objectives.csv")
    candidate_table = read_csv(args.archived_analysis / "anisotropic_dense_candidates.csv")
    labels = [row["label"] for row in candidate_table]
    selected_label = "q050_x000"
    selected_index = labels.index(selected_label)
    test_seeds = [105, 106, 107, 108, 109]
    by_key = {(int(row["seed"]), row["label"]): row for row in per_seed}
    density = np.asarray(
        [
            [float(by_key[(seed, label)]["density_mse"]) for label in labels]
            for seed in test_seeds
        ]
    )
    field = np.asarray(
        [
            [float(by_key[(seed, label)]["field_mse"]) for label in labels]
            for seed in test_seeds
        ]
    )

    # Reproduce the archived reducer exactly: one RNG stream, density then field
    # for every candidate in archived candidate order.
    rng = np.random.default_rng(271828)
    archived_reproduced = {}
    for index, label in enumerate(labels):
        density_interval = ratio_draws(density[:, 0], density[:, index], rng)
        field_interval = ratio_draws(field[:, 0], field[:, index], rng)
        if label == selected_label:
            archived_reproduced = {
                "density": density_interval,
                "field": field_interval,
            }

    # Reproduce the later report convention: restart the RNG on the selected
    # density candidate instead of consuming draws for the earlier candidates.
    selected_only_density = ratio_draws(
        density[:, 0], density[:, selected_index], np.random.default_rng(271828)
    )
    selected_only_field = ratio_draws(
        field[:, 0], field[:, selected_index], np.random.default_rng(271829)
    )
    archived_row = candidate_table[selected_index]
    retrieval_selection = next(
        row
        for row in read_csv(args.retrieval / "bandwidth_complete_rule_selection.csv")
        if row["metric"] == "density"
    )
    rows = []
    for metric, reproduced, selected_only in (
        ("density", archived_reproduced["density"], selected_only_density),
        ("field", archived_reproduced["field"], selected_only_field),
    ):
        archived = (
            float(archived_row[f"test_{metric}_reduction"]),
            float(archived_row[f"test_{metric}_ci_lo"]),
            float(archived_row[f"test_{metric}_ci_hi"]),
        )
        if not np.allclose(archived, reproduced, rtol=0.0, atol=2.0e-15):
            raise AssertionError(f"failed to reproduce archived {metric} interval")
        rows.append(
            {
                "metric": metric,
                "candidate": selected_label,
                "central_ratio": archived[0],
                "archived_reducer_ci_lo": archived[1],
                "archived_reducer_ci_hi": archived[2],
                "selected_only_rerun_ci_lo": selected_only[1],
                "selected_only_rerun_ci_hi": selected_only[2],
                "difference_source": "Monte Carlo bootstrap stream position; same five held-out seeds and same central estimator",
                "reporting_decision": "use archived reducer interval for the archived fixed-candidate result",
            }
        )
    write_csv(args.output / "density_ci_reconciliation.csv", rows)

    landau = json.loads(
        (args.retrieval / "landau_three_arm_summary.json").read_text(encoding="utf-8")
    )
    accounting = landau["wide_filter_excess_decay_phase_attribution"]
    accounting.pop("hypothesized_0p732_supported", None)
    accounting["interpretation"] = (
        "Descriptive additive log-slope accounting for this saved three-arm ensemble. "
        "It is not a causal attribution and is not compared with the old 73.2% toy "
        "calculation because the denominators and scientific questions differ."
    )
    landau["wide_filter_excess_decay_phase_accounting"] = accounting
    del landau["wide_filter_excess_decay_phase_attribution"]
    landau["estimand_note"] = (
        "The coherent observable is the slope of the logarithm of the magnitude of "
        "the complex seed mean after the frozen envelope. The mean-seed-magnitude "
        "term is the slope of the logarithm of the arithmetic mean of seed magnitudes. "
        "Neither is the arithmetic mean of the individual-seed log-amplitude slopes."
    )
    (args.output / "landau_three_arm_summary_corrected_interpretation.json").write_text(
        json.dumps(landau, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    loading_scope = {
        "priority": "SECOND_PRIORITY_NOT_SUBMITTED",
        "existing_correlation": (
            "The observed CIC rate/loading-covariance correlation is descriptive and "
            "does not establish that the covariance causes the rate spread."
        ),
        "proposed_modified_loading_study": (
            "A PPC scan using nested positions and antithetic per-cell beam/thermal "
            "quartets tests convergence of that modified loading scheme. It does not "
            "directly explain the spread of the original global-shuffle scheme unless "
            "an additional bridge comparison holds PPC and all other settings fixed "
            "while changing only the loading construction."
        ),
        "decision": "keep separate from the exact-width validation and do not submit now",
    }
    (args.output / "loading_convergence_scope.json").write_text(
        json.dumps(loading_scope, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    reconciliation = {
        "density_report_csv_discrepancy_resolved": True,
        "cause": (
            "The archived reducer advances one NumPy RNG stream through density and "
            "field intervals for every candidate. The retrieval report restarted a "
            "fresh RNG at q050_x000. Bootstrap percentile Monte Carlo endpoints therefore "
            "differ slightly, while inputs, central ratio and estimator are identical."
        ),
        "authoritative_archived_density_interval": [
            float(archived_row["test_density_ci_lo"]),
            float(archived_row["test_density_ci_hi"]),
        ],
        "selected_only_recomputed_density_interval": [
            selected_only_density[1],
            selected_only_density[2],
        ],
        "nested_complete_rule_density_interval": [
            float(retrieval_selection["nested_ci_lo"]),
            float(retrieval_selection["nested_ci_hi"]),
        ],
        "reporting": (
            "Use the archived interval for the archived fixed-candidate claim; retain "
            "the nested interval under its separate complete-rule selection label."
        ),
    }
    (args.output / "reconciliation_summary.json").write_text(
        json.dumps(reconciliation, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(reconciliation, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
