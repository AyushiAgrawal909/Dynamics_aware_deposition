#!/usr/bin/env python3
"""Complete the post-hoc E2 estimator comparison from existing histories.

The registered estimator and its published/registered results remain primary.
This script adds labelled sensitivity results based on individual-seed fits for
the completed drift points, the complete historical Table-5 corner sweep, and
the later compensated-filter corner.  It never changes a registered window or
overwrites an existing result directory.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from diagnose_E2_rate_estimator import (
    analyse_group,
    linear_fit,
    load_corner_group,
    load_current_group,
    select_window,
)


BOOTSTRAP_DRAWS = 5000
BOOTSTRAP_SEED = 20260911
CURRENT_ARMS = ("cic", "fixed_gaussian", "constrained_gaussian")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise AssertionError(f"refusing to write empty table: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def individual_rates(times: np.ndarray, amplitudes: np.ndarray,
                     start: float, stop: float) -> np.ndarray:
    selected_times, selected = select_window(times, amplitudes, start, stop)
    return np.asarray(
        [linear_fit(selected_times, row)["gamma"] for row in selected],
        dtype=np.float64,
    )


def registered_rate(times: np.ndarray, amplitudes: np.ndarray,
                    start: float, stop: float) -> float:
    selected_times, selected = select_window(times, amplitudes, start, stop)
    return float(linear_fit(selected_times, np.mean(selected, axis=0))["gamma"])


def paired_bootstrap_rows(
    campaign: str,
    drift: float,
    window_label: str,
    baseline_label: str,
    group_data: dict[str, dict],
    predicted_ratios: dict[str, float],
) -> list[dict]:
    """Bootstrap ratios of individual-rate means with one paired draw per replicate."""
    baseline = group_data[baseline_label]
    baseline_rates = individual_rates(
        baseline["times"], baseline["amplitudes"], baseline["start"], baseline["stop"]
    )
    rows: list[dict] = []
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    index_draws = rng.integers(
        0, baseline_rates.size, size=(BOOTSTRAP_DRAWS, baseline_rates.size)
    )
    for label, data in group_data.items():
        if label == baseline_label:
            continue
        if data["seeds"] != baseline["seeds"]:
            raise AssertionError(f"unpaired seed order: {campaign}, {label}")
        rates = individual_rates(
            data["times"], data["amplitudes"], data["start"], data["stop"]
        )
        boot = np.mean(rates[index_draws], axis=1) / np.mean(
            baseline_rates[index_draws], axis=1
        )
        paired_seedwise = rates / baseline_rates
        registered_ratio = registered_rate(
            data["times"], data["amplitudes"], data["start"], data["stop"]
        ) / registered_rate(
            baseline["times"], baseline["amplitudes"], baseline["start"], baseline["stop"]
        )
        rows.append(
            {
                "analysis_role": "POST_HOC_SENSITIVITY_REGISTERED_RESULT_UNCHANGED",
                "campaign": campaign,
                "v_beam": drift,
                "window": window_label,
                "baseline": baseline_label,
                "candidate": label,
                "n_paired_seeds": baseline_rates.size,
                "registered_ratio_of_mean_amplitude_rates_primary": registered_ratio,
                "ratio_of_individual_rate_means_sensitivity": float(
                    np.mean(rates) / np.mean(baseline_rates)
                ),
                "paired_bootstrap_ci_lo_sensitivity": float(np.quantile(boot, 0.025)),
                "paired_bootstrap_ci_hi_sensitivity": float(np.quantile(boot, 0.975)),
                "bootstrap_draws": BOOTSTRAP_DRAWS,
                "bootstrap_seed": BOOTSTRAP_SEED,
                "bootstrap_unit": "paired seed; same resampled indices in baseline and candidate",
                "mean_of_paired_seedwise_rate_ratios_diagnostic": float(
                    np.mean(paired_seedwise)
                ),
                "sd_of_paired_seedwise_rate_ratios_diagnostic": float(
                    np.std(paired_seedwise, ddof=1)
                ),
                "predicted_rate_ratio": predicted_ratios.get(label, float("nan")),
            }
        )
    return rows


def leave_one_out_rows(
    campaign: str,
    drift: float,
    window_label: str,
    baseline_label: str,
    group_data: dict[str, dict],
) -> list[dict]:
    baseline = group_data[baseline_label]
    rows: list[dict] = []
    for omitted_index, omitted_seed in enumerate(baseline["seeds"]):
        keep = np.ones(len(baseline["seeds"]), dtype=bool)
        keep[omitted_index] = False
        base_rate = registered_rate(
            baseline["times"], baseline["amplitudes"][keep],
            baseline["start"], baseline["stop"],
        )
        base_individual = individual_rates(
            baseline["times"], baseline["amplitudes"][keep],
            baseline["start"], baseline["stop"],
        )
        for label, data in group_data.items():
            if data["seeds"] != baseline["seeds"]:
                raise AssertionError(f"unpaired leave-one-out group: {campaign}, {label}")
            candidate_rate = registered_rate(
                data["times"], data["amplitudes"][keep], data["start"], data["stop"]
            )
            candidate_individual = individual_rates(
                data["times"], data["amplitudes"][keep], data["start"], data["stop"]
            )
            rows.append(
                {
                    "analysis_role": "POST_HOC_INFLUENCE_DIAGNOSTIC",
                    "campaign": campaign,
                    "v_beam": drift,
                    "window": window_label,
                    "baseline": baseline_label,
                    "arm": label,
                    "omitted_seed": omitted_seed,
                    "leave_one_out_registered_gamma": candidate_rate,
                    "leave_one_out_registered_ratio_to_baseline": candidate_rate / base_rate,
                    "leave_one_out_individual_gamma_mean": float(
                        np.mean(candidate_individual)
                    ),
                    "leave_one_out_ratio_of_individual_rate_means": float(
                        np.mean(candidate_individual) / np.mean(base_individual)
                    ),
                }
            )
    return rows


def add_group(
    summaries: list[dict],
    seed_rows: list[dict],
    contribution_rows: list[dict],
    concentration_rows: list[dict],
    campaign: str,
    drift: float,
    label: str,
    data: dict,
    window_label: str,
    expected_registered: float,
) -> None:
    summary, seeds_out, contributions_out, concentration_out = analyse_group(
        campaign,
        drift,
        label,
        data["seeds"],
        data["times"],
        data["amplitudes"],
        data["start"],
        data["stop"],
        window_label,
    )
    if abs(summary["registered_mean_amplitude_gamma"] - expected_registered) > 5.0e-12:
        raise AssertionError(
            f"registered rate mismatch: {campaign}, {drift}, {label}: "
            f"{summary['registered_mean_amplitude_gamma']} != {expected_registered}"
        )
    summaries.append(summary)
    seed_rows.extend(seeds_out)
    contribution_rows.extend(contributions_out)
    concentration_rows.extend(concentration_out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--scan-root", type=Path, required=True)
    parser.add_argument(
        "--extension-root",
        type=Path,
        help="completed extension tree; used only when a drift is absent from --scan-root",
    )
    parser.add_argument("--registered", type=Path, required=True)
    parser.add_argument("--historical-root", type=Path, required=True)
    parser.add_argument("--compensated-root", type=Path, required=True)
    parser.add_argument("--compensated-predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")

    config = json.loads(args.config.read_text(encoding="utf-8"))
    registered = json.loads(args.registered.read_text(encoding="utf-8"))
    historical_analysis_path = args.historical_root / "analysis" / "joint_corner_sweep.json"
    compensated_analysis_path = (
        args.compensated_root / "analysis" / "E2_compensated_corner.json"
    )
    historical_analysis = json.loads(historical_analysis_path.read_text(encoding="utf-8"))
    compensated_analysis = json.loads(compensated_analysis_path.read_text(encoding="utf-8"))
    if config["fit"]["observable"] != "slope of ln(mean_seed(abs(Ex_hat(12,1))))":
        raise AssertionError("unexpected frozen drift estimator")
    if compensated_analysis["rate_definition"] != config["fit"]["observable"]:
        raise AssertionError("compensated and drift estimators differ")
    if historical_analysis["fit_window"] != [24.32, 59.68]:
        raise AssertionError("unexpected historical fit window")
    if compensated_analysis["fit_window"] != [24.32, 59.68]:
        raise AssertionError("unexpected compensated fit window")

    with args.compensated_predictions.open(newline="", encoding="utf-8") as handle:
        prediction_rows = list(csv.DictReader(handle))
    compensated_predictions = {
        row["label"]: float(row["predicted_rate_ratio"]) for row in prediction_rows
    }

    summaries: list[dict] = []
    seed_rows: list[dict] = []
    contribution_rows: list[dict] = []
    concentration_rows: list[dict] = []
    bootstrap_rows: list[dict] = []
    loo_rows: list[dict] = []
    hashes = {
        str(path): sha256(path)
        for path in (
            args.config,
            args.registered,
            historical_analysis_path,
            compensated_analysis_path,
            args.compensated_predictions,
        )
    }

    registered_lookup = {
        (float(row["v_beam"]), row["arm"]): row
        for row in registered["rows"]
        if row["measurement_status"] == "COMPLETE"
    }
    for scan in config["scan_rows"]:
        drift = float(scan["v_beam"])
        if (drift, "cic") not in registered_lookup:
            continue
        current_root = args.scan_root
        if not (current_root / f"vb{int(round(1000.0 * drift)):04d}" / "cic").is_dir():
            if args.extension_root is None:
                raise FileNotFoundError(f"completed drift {drift} is absent from --scan-root")
            current_root = args.extension_root / "runs"
        group: dict[str, dict] = {}
        for arm in CURRENT_ARMS:
            times, amplitudes, seeds, loaded_hashes = load_current_group(
                current_root, config, scan, arm
            )
            hashes.update(loaded_hashes)
            data = {
                "times": times,
                "amplitudes": amplitudes,
                "seeds": seeds,
                "start": float(scan["fit_t_start"]),
                "stop": float(scan["fit_t_stop"]),
            }
            group[arm] = data
            add_group(
                summaries, seed_rows, contribution_rows, concentration_rows,
                "registered_drift_scan_5201231", drift, arm, data,
                "registered_frozen", float(registered_lookup[(drift, arm)]["gamma"]),
            )
        predicted = {
            "fixed_gaussian": float(scan["fixed_predicted_rate_ratio"]),
            "constrained_gaussian": float(scan["constrained_predicted_rate_ratio"]),
        }
        current_bootstrap = paired_bootstrap_rows(
            "registered_drift_scan_5201231", drift, "registered_frozen",
            "cic", group, predicted,
        )
        for row in current_bootstrap:
            row["baseline_registered_signal_gate"] = registered_lookup[(drift, "cic")][
                "signal_gate"
            ]
            row["candidate_registered_signal_gate"] = registered_lookup[
                (drift, row["candidate"])
            ]["signal_gate"]
            row["scientific_interpretation"] = (
                "descriptive estimator sensitivity only; frozen signal gate failed"
                if row["baseline_registered_signal_gate"] != "PASS"
                or row["candidate_registered_signal_gate"] != "PASS"
                else "sensitivity analysis for a signal-valid registered comparison"
            )
        bootstrap_rows.extend(current_bootstrap)
        loo_rows.extend(
            leave_one_out_rows(
                "registered_drift_scan_5201231", drift, "registered_frozen", "cic", group
            )
        )

    historical_group: dict[str, dict] = {}
    historical_expected = {row["label"]: float(row["gamma"]) for row in historical_analysis["rows"]}
    historical_predicted = {
        row["label"]: float(row["warm_ratio"]) for row in historical_analysis["rows"]
    }
    for label in historical_expected:
        times, amplitudes, seeds, _results, loaded_hashes = load_corner_group(
            args.historical_root, label
        )
        hashes.update(loaded_hashes)
        data = {
            "times": times, "amplitudes": amplitudes, "seeds": seeds,
            "start": 24.32, "stop": 59.68,
        }
        historical_group[label] = data
        add_group(
            summaries, seed_rows, contribution_rows, concentration_rows,
            "historical_table5_corner_4913408", 0.8, label, data,
            "historical_registered", historical_expected[label],
        )
    bootstrap_rows.extend(
        paired_bootstrap_rows(
            "historical_table5_corner_4913408", 0.8, "historical_registered",
            "base", historical_group, historical_predicted,
        )
    )
    loo_rows.extend(
        leave_one_out_rows(
            "historical_table5_corner_4913408", 0.8, "historical_registered",
            "base", historical_group,
        )
    )

    compensated_group: dict[str, dict] = {}
    compensated_expected = {
        row["label"]: float(row["gamma"]) for row in compensated_analysis["rows"]
    }
    for label in compensated_expected:
        times, amplitudes, seeds, _results, loaded_hashes = load_corner_group(
            args.compensated_root, label
        )
        hashes.update(loaded_hashes)
        data = {
            "times": times, "amplitudes": amplitudes, "seeds": seeds,
            "start": 24.32, "stop": 59.68,
        }
        compensated_group[label] = data
        add_group(
            summaries, seed_rows, contribution_rows, concentration_rows,
            "compensated_corner_5195640", 0.8, label, data,
            "compensated_registered", compensated_expected[label],
        )
    bootstrap_rows.extend(
        paired_bootstrap_rows(
            "compensated_corner_5195640", 0.8, "compensated_registered",
            "base", compensated_group, compensated_predictions,
        )
    )
    loo_rows.extend(
        leave_one_out_rows(
            "compensated_corner_5195640", 0.8, "compensated_registered",
            "base", compensated_group,
        )
    )

    # Regression asserts for the independently specified v_b=0.80 bootstrap.
    fixed = next(
        row for row in bootstrap_rows
        if row["campaign"] == "registered_drift_scan_5201231"
        and row["v_beam"] == 0.8 and row["candidate"] == "fixed_gaussian"
    )
    constrained = next(
        row for row in bootstrap_rows
        if row["campaign"] == "registered_drift_scan_5201231"
        and row["v_beam"] == 0.8 and row["candidate"] == "constrained_gaussian"
    )
    np.testing.assert_allclose(
        [fixed["paired_bootstrap_ci_lo_sensitivity"], fixed["paired_bootstrap_ci_hi_sensitivity"]],
        [0.88743222, 0.92665638], rtol=0.0, atol=5.0e-9,
    )
    np.testing.assert_allclose(
        [constrained["paired_bootstrap_ci_lo_sensitivity"], constrained["paired_bootstrap_ci_hi_sensitivity"]],
        [0.94406780, 0.96310258], rtol=0.0, atol=5.0e-9,
    )

    args.output.mkdir(parents=True)
    write_csv(args.output / "estimator_summary.csv", summaries)
    write_csv(args.output / "individual_seed_rates.csv", seed_rows)
    write_csv(args.output / "time_dependent_seed_contributions.csv", contribution_rows)
    write_csv(args.output / "contribution_concentration.csv", concentration_rows)
    write_csv(args.output / "paired_individual_rate_bootstrap_sensitivity.csv", bootstrap_rows)
    write_csv(args.output / "leave_one_seed_out_influence.csv", loo_rows)

    summary = {
        "status": "POST_HOC_SENSITIVITY_COMPLETE_REGISTERED_RESULTS_UNCHANGED",
        "registered_estimator_primary": config["fit"]["observable"],
        "sensitivity_estimator": (
            "fit slope(log(abs(Ex_hat_s))) for each seed on the same frozen window; "
            "compare ratios of seed-mean rates with a paired-seed bootstrap"
        ),
        "coverage": {
            "completed_drift_points": sorted(
                {float(row["v_beam"]) for row in registered["rows"]
                 if row["measurement_status"] == "COMPLETE"}
            ),
            "historical_table5_labels": list(historical_expected),
            "compensated_corner_labels": list(compensated_expected),
        },
        "bootstrap": {
            "draws": BOOTSTRAP_DRAWS,
            "seed": BOOTSTRAP_SEED,
            "unit": "paired seed",
            "shared_draw_rule": (
                "one index matrix per campaign/drift, reused for every candidate arm"
            ),
            "interpretation": "sensitivity analysis, not a replacement for registered intervals",
        },
        "input_sha256": dict(sorted(hashes.items())),
        "tables": {
            "estimator_summary": "estimator_summary.csv",
            "individual_seed_rates": "individual_seed_rates.csv",
            "time_dependent_seed_contributions": "time_dependent_seed_contributions.csv",
            "contribution_concentration": "contribution_concentration.csv",
            "paired_individual_rate_bootstrap_sensitivity": "paired_individual_rate_bootstrap_sensitivity.csv",
            "leave_one_seed_out_influence": "leave_one_seed_out_influence.csv",
        },
    }
    (args.output / "diagnostic_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "status": summary["status"],
        "n_summary_rows": len(summaries),
        "n_bootstrap_rows": len(bootstrap_rows),
        "vbeam_080_fixed_sensitivity_ci": [
            fixed["paired_bootstrap_ci_lo_sensitivity"],
            fixed["paired_bootstrap_ci_hi_sensitivity"],
        ],
        "vbeam_080_constrained_sensitivity_ci": [
            constrained["paired_bootstrap_ci_lo_sensitivity"],
            constrained["paired_bootstrap_ci_hi_sensitivity"],
        ],
    }, indent=2))


if __name__ == "__main__":
    main()
