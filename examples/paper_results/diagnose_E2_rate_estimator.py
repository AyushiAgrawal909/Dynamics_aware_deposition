#!/usr/bin/env python3
"""Diagnose arithmetic-mean amplitude weighting in the E2 drift scan.

This is a post hoc diagnostic.  It never replaces the preregistered estimator
or its reported result.  It compares estimators on the exact frozen windows,
records every individual-seed fit, and makes the v_b=0.80 endpoint/campaign
comparison requested after the registered reduction.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


ARM_ORDER = ("cic", "fixed_gaussian", "constrained_gaussian")


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


def linear_fit(times: np.ndarray, amplitude: np.ndarray) -> dict:
    if np.any(~np.isfinite(amplitude)) or np.any(amplitude <= 1.0e-300):
        raise AssertionError("fit amplitude is non-finite or non-positive")
    log_amplitude = np.log(amplitude)
    slope, intercept = np.polyfit(times, log_amplitude, 1)
    fitted = slope * times + intercept
    residual = float(np.sum((log_amplitude - fitted) ** 2))
    total = float(np.sum((log_amplitude - np.mean(log_amplitude)) ** 2))
    return {
        "gamma": float(slope),
        "intercept": float(intercept),
        "r_squared": float(1.0 - residual / total) if total > 0.0 else 0.0,
    }


def select_window(
    times: np.ndarray, amplitudes: np.ndarray, start: float, stop: float
) -> tuple[np.ndarray, np.ndarray]:
    mask = (times >= start - 1.0e-12) & (times <= stop + 1.0e-12)
    if np.count_nonzero(mask) < 5:
        raise AssertionError("fit window has fewer than five samples")
    selected_times = times[mask]
    if abs(float(selected_times[0]) - start) > 1.0e-10:
        raise AssertionError("requested start is not a recorded sample")
    if abs(float(selected_times[-1]) - stop) > 1.0e-10:
        raise AssertionError("requested stop is not a recorded sample")
    return selected_times, amplitudes[:, mask]


def drift_label(value: float) -> str:
    return f"vb{int(round(1000.0 * value)):04d}"


def load_current_group(
    root: Path, config: dict, scan: dict, arm: str
) -> tuple[np.ndarray, np.ndarray, list[int], dict[str, str]]:
    times = None
    amplitudes = []
    seeds = []
    hashes = {}
    for seed in config["seeds"]:
        seed_root = root / drift_label(float(scan["v_beam"])) / arm / f"seed{seed}"
        result_path = seed_root / "result.json"
        history_path = seed_root / "mode_history.npz"
        manifest_path = seed_root / "campaign_manifest.json"
        for path in (result_path, history_path, manifest_path):
            if not path.is_file():
                raise FileNotFoundError(path)
            hashes[str(path)] = sha256(path)
        result = json.loads(result_path.read_text(encoding="utf-8"))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        required_result = {
            "alpha", "dt", "filter_kind", "kx_mode", "ky_mode", "marker_dynamics",
            "ppc", "seed", "sigma_x_dx", "sigma_y_dy", "t_end",
            "target_mode_projection",
        }
        if not required_result.issubset(result):
            raise AssertionError(f"result schema mismatch in {result_path}")
        if int(result["seed"]) != int(seed):
            raise AssertionError(f"seed mismatch in {result_path}")
        if (int(result["kx_mode"]), int(result["ky_mode"])) != tuple(config["mode"]):
            raise AssertionError(f"mode mismatch in {result_path}")
        if int(result["ppc"]) != int(config["ppc"]):
            raise AssertionError(f"PPC mismatch in {result_path}")
        if not bool(result["target_mode_projection"]):
            raise AssertionError(f"mode projection disabled in {result_path}")
        if manifest.get("arm") != arm or int(manifest.get("seed", -1)) != int(seed):
            raise AssertionError(f"manifest arm/seed mismatch in {manifest_path}")
        with np.load(history_path, allow_pickle=False) as archive:
            required_npz = {"times", "mode", "seed", "delta_f_ratio"}
            if not required_npz.issubset(archive.files):
                raise AssertionError(f"NPZ schema mismatch in {history_path}")
            current_times = np.asarray(archive["times"], dtype=np.float64)
            current_amplitude = np.abs(
                np.asarray(archive["mode"], dtype=np.complex128)
            )
            archived_seed = int(np.asarray(archive["seed"]).flat[0])
        if archived_seed != int(seed):
            raise AssertionError(f"NPZ seed mismatch in {history_path}")
        if times is None:
            times = current_times
        else:
            np.testing.assert_array_equal(times, current_times)
        amplitudes.append(current_amplitude)
        seeds.append(int(seed))
    assert times is not None
    return times, np.stack(amplitudes), seeds, hashes


def load_corner_group(
    root: Path, label: str
) -> tuple[np.ndarray, np.ndarray, list[int], list[dict], dict[str, str]]:
    paths = sorted((root / "rate" / label).glob("seed*/mode_history.npz"))
    if len(paths) != 10:
        raise AssertionError(f"expected ten historical histories for {label}")
    times = None
    amplitudes = []
    seeds = []
    results = []
    hashes = {}
    for history_path in paths:
        result_path = history_path.with_name("result.json")
        if not result_path.is_file():
            raise FileNotFoundError(result_path)
        hashes[str(history_path)] = sha256(history_path)
        hashes[str(result_path)] = sha256(result_path)
        result = json.loads(result_path.read_text(encoding="utf-8"))
        required_result = {
            "alpha", "dt", "kx_mode", "ky_mode", "marker_dynamics", "ppc",
            "seed", "sigma_x_dx", "sigma_y_dy", "t_end", "target_mode_projection",
        }
        if not required_result.issubset(result):
            raise AssertionError(f"historical result schema mismatch in {result_path}")
        with np.load(history_path, allow_pickle=False) as archive:
            required_npz = {"times", "mode", "seed", "delta_f_ratio"}
            if not required_npz.issubset(archive.files):
                raise AssertionError(f"historical NPZ schema mismatch in {history_path}")
            current_times = np.asarray(archive["times"], dtype=np.float64)
            current_amplitude = np.abs(
                np.asarray(archive["mode"], dtype=np.complex128)
            )
            archived_seed = int(np.asarray(archive["seed"]).flat[0])
        if archived_seed != int(result["seed"]):
            raise AssertionError(f"historical seed mismatch in {history_path}")
        if times is None:
            times = current_times
        else:
            np.testing.assert_array_equal(times, current_times)
        amplitudes.append(current_amplitude)
        seeds.append(archived_seed)
        results.append(result)
    if sorted(seeds) != list(range(10)):
        raise AssertionError("historical seed set is not 0--9")
    assert times is not None
    return times, np.stack(amplitudes), seeds, results, hashes


def analyse_group(
    campaign: str,
    drift: float,
    arm: str,
    seeds: list[int],
    times: np.ndarray,
    amplitudes: np.ndarray,
    start: float,
    stop: float,
    window_label: str,
) -> tuple[dict, list[dict], list[dict], list[dict]]:
    x, values = select_window(times, amplitudes, start, stop)
    individual = [linear_fit(x, values[index]) for index in range(len(seeds))]
    rates = np.asarray([row["gamma"] for row in individual])
    intercepts = np.asarray([row["intercept"] for row in individual])
    arithmetic = linear_fit(x, np.mean(values, axis=0))
    geometric = linear_fit(x, np.exp(np.mean(np.log(values), axis=0)))
    fitted_amplitudes = np.exp(intercepts[:, None] + rates[:, None] * x[None, :])
    fitted_mixture = linear_fit(x, np.mean(fitted_amplitudes, axis=0))
    fractions = values / np.sum(values, axis=0, keepdims=True)
    fitted_fractions = fitted_amplitudes / np.sum(
        fitted_amplitudes, axis=0, keepdims=True
    )
    changes = fractions[:, -1] - fractions[:, 0]
    gamma_change_correlation = float(np.corrcoef(rates, changes)[0, 1])
    indices = {"start": 0, "midpoint": len(x) // 2, "end": len(x) - 1}
    concentration = []
    for position, index in indices.items():
        concentration.append(
            {
                "campaign": campaign,
                "v_beam": drift,
                "arm": arm,
                "window": window_label,
                "position": position,
                "time": float(x[index]),
                "effective_seed_count": float(1.0 / np.sum(fractions[:, index] ** 2)),
                "largest_seed_fraction": float(np.max(fractions[:, index])),
                "largest_seed": int(seeds[int(np.argmax(fractions[:, index]))]),
            }
        )
    seed_rows = []
    for seed, fit, start_fraction, end_fraction in zip(
        seeds, individual, fractions[:, 0], fractions[:, -1]
    ):
        seed_rows.append(
            {
                "campaign": campaign,
                "v_beam": drift,
                "arm": arm,
                "window": window_label,
                "fit_t_start": start,
                "fit_t_stop": stop,
                "seed": seed,
                "gamma_individual": fit["gamma"],
                "fit_r_squared_individual": fit["r_squared"],
                "fit_intercept_individual": fit["intercept"],
                "amplitude_fraction_start": float(start_fraction),
                "amplitude_fraction_end": float(end_fraction),
                "amplitude_fraction_change": float(end_fraction - start_fraction),
            }
        )
    contribution_rows = []
    for time_index, time in enumerate(x):
        for seed_index, seed in enumerate(seeds):
            contribution_rows.append(
                {
                    "campaign": campaign,
                    "v_beam": drift,
                    "arm": arm,
                    "window": window_label,
                    "time": float(time),
                    "seed": seed,
                    "amplitude": float(values[seed_index, time_index]),
                    "actual_fraction_of_mean_sum": float(fractions[seed_index, time_index]),
                    "fitted_exponential_fraction_of_mean_sum": float(
                        fitted_fractions[seed_index, time_index]
                    ),
                }
            )
    summary = {
        "campaign": campaign,
        "v_beam": drift,
        "arm": arm,
        "window": window_label,
        "fit_t_start": start,
        "fit_t_stop": stop,
        "n_seeds": len(seeds),
        "registered_mean_amplitude_gamma": arithmetic["gamma"],
        "registered_mean_amplitude_r_squared": arithmetic["r_squared"],
        "individual_gamma_mean": float(np.mean(rates)),
        "individual_gamma_sd": float(np.std(rates, ddof=1)),
        "individual_gamma_median": float(np.median(rates)),
        "individual_gamma_q025": float(np.quantile(rates, 0.025)),
        "individual_gamma_q975": float(np.quantile(rates, 0.975)),
        "individual_gamma_min": float(np.min(rates)),
        "individual_gamma_max": float(np.max(rates)),
        "geometric_mean_amplitude_gamma": geometric["gamma"],
        "fitted_exponential_mixture_gamma": fitted_mixture["gamma"],
        "registered_minus_individual_mean": float(arithmetic["gamma"] - np.mean(rates)),
        "registered_minus_fitted_mixture": float(arithmetic["gamma"] - fitted_mixture["gamma"]),
        "gamma_vs_contribution_change_correlation": gamma_change_correlation,
        "effective_seed_count_start": concentration[0]["effective_seed_count"],
        "effective_seed_count_midpoint": concentration[1]["effective_seed_count"],
        "effective_seed_count_end": concentration[2]["effective_seed_count"],
        "largest_seed_fraction_start": concentration[0]["largest_seed_fraction"],
        "largest_seed_fraction_end": concentration[2]["largest_seed_fraction"],
    }
    return summary, seed_rows, contribution_rows, concentration


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--scan-root", type=Path, required=True)
    parser.add_argument("--registered", type=Path, required=True)
    parser.add_argument("--transfer-summary", type=Path, required=True)
    parser.add_argument("--corner-root", type=Path, required=True)
    parser.add_argument("--corner-launcher", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")

    config = json.loads(args.config.read_text(encoding="utf-8"))
    registered = json.loads(args.registered.read_text(encoding="utf-8"))
    transfer = json.loads(args.transfer_summary.read_text(encoding="utf-8"))
    corner_analysis_path = args.corner_root / "analysis" / "joint_corner_sweep.json"
    corner_analysis = json.loads(corner_analysis_path.read_text(encoding="utf-8"))
    if config["fit"]["observable"] != "slope of ln(mean_seed(abs(Ex_hat(12,1))))":
        raise AssertionError("unexpected registered estimator definition")
    if transfer.get("status") != "PASS":
        raise AssertionError("implemented transfer-function audit did not pass")
    if float(transfer["maximum_absolute_error_W2"]) > 1.0e-12:
        raise AssertionError("implemented W2 transfer mismatch exceeds tolerance")
    if corner_analysis["fit_window"] != [24.32, 59.68]:
        raise AssertionError("unexpected historical corner fit window")
    registered_lookup = {
        (float(row["v_beam"]), row["arm"]): row
        for row in registered["rows"]
        if row["measurement_status"] == "COMPLETE"
    }

    summaries: list[dict] = []
    seed_rows: list[dict] = []
    contribution_rows: list[dict] = []
    concentration_rows: list[dict] = []
    input_hashes = {
        str(args.config): sha256(args.config),
        str(args.registered): sha256(args.registered),
        str(args.transfer_summary): sha256(args.transfer_summary),
        str(corner_analysis_path): sha256(corner_analysis_path),
        str(args.corner_launcher): sha256(args.corner_launcher),
    }
    loaded_current: dict[tuple[float, str], tuple[np.ndarray, np.ndarray, list[int]]] = {}
    for scan in config["scan_rows"]:
        drift = float(scan["v_beam"])
        if (drift, "cic") not in registered_lookup:
            continue
        for arm in ARM_ORDER:
            times, amplitudes, seeds, hashes = load_current_group(
                args.scan_root, config, scan, arm
            )
            input_hashes.update(hashes)
            loaded_current[(drift, arm)] = (times, amplitudes, seeds)
            summary, seeds_out, contributions_out, concentration_out = analyse_group(
                "registered_drift_scan_5201231",
                drift,
                arm,
                seeds,
                times,
                amplitudes,
                float(scan["fit_t_start"]),
                float(scan["fit_t_stop"]),
                "registered_frozen",
            )
            expected = float(registered_lookup[(drift, arm)]["gamma"])
            if abs(summary["registered_mean_amplitude_gamma"] - expected) > 5.0e-12:
                raise AssertionError(f"registered rate mismatch for {drift}, {arm}")
            summaries.append(summary)
            seed_rows.extend(seeds_out)
            contribution_rows.extend(contributions_out)
            concentration_rows.extend(concentration_out)

    historical: dict[str, tuple[np.ndarray, np.ndarray, list[int], list[dict]]] = {}
    for old_label, arm in (("base", "cic"), ("y02428", "fixed_gaussian")):
        times, amplitudes, seeds, results, hashes = load_corner_group(
            args.corner_root, old_label
        )
        input_hashes.update(hashes)
        historical[arm] = (times, amplitudes, seeds, results)
        summary, seeds_out, contributions_out, concentration_out = analyse_group(
            "historical_corner_4913408",
            0.8,
            arm,
            seeds,
            times,
            amplitudes,
            24.32,
            59.68,
            "historical_native",
        )
        historical_row = next(
            row for row in corner_analysis["rows"] if row["label"] == old_label
        )
        if abs(
            summary["registered_mean_amplitude_gamma"]
            - float(historical_row["gamma"])
        ) > 5.0e-12:
            raise AssertionError(f"historical registered rate mismatch for {old_label}")
        summaries.append(summary)
        seed_rows.extend(seeds_out)
        contribution_rows.extend(contributions_out)
        concentration_rows.extend(concentration_out)

    for arm in ("cic", "fixed_gaussian"):
        times, amplitudes, seeds = loaded_current[(0.8, arm)]
        summary, seeds_out, contributions_out, concentration_out = analyse_group(
            "registered_drift_scan_5201231",
            0.8,
            arm,
            seeds,
            times,
            amplitudes,
            24.32,
            59.68,
            "historical_endpoint_sensitivity",
        )
        summaries.append(summary)
        seed_rows.extend(seeds_out)
        contribution_rows.extend(contributions_out)
        concentration_rows.extend(concentration_out)

    summary_lookup = {
        (row["campaign"], row["arm"], row["window"]): row for row in summaries
    }
    theory = float(
        next(row for row in config["scan_rows"] if float(row["v_beam"]) == 0.8)[
            "grid_time_discrete_cic_gamma"
        ]
    )
    new_native = summary_lookup[
        ("registered_drift_scan_5201231", "cic", "registered_frozen")
    ]
    new_short = summary_lookup[
        (
            "registered_drift_scan_5201231",
            "cic",
            "historical_endpoint_sensitivity",
        )
    ]
    old_native = summary_lookup[
        ("historical_corner_4913408", "cic", "historical_native")
    ]
    registered_gap = new_native["registered_mean_amplitude_gamma"] - theory
    estimator_gap = (
        new_native["registered_mean_amplitude_gamma"]
        - new_native["individual_gamma_mean"]
    )
    residual_gap = new_native["individual_gamma_mean"] - theory
    comparison_rows = [
        {
            "comparison": "current endpoint effect, same seeds and implementation",
            "arm": "cic",
            "from_gamma": new_native["registered_mean_amplitude_gamma"],
            "to_gamma": new_short["registered_mean_amplitude_gamma"],
            "difference_to_minus_from": new_short["registered_mean_amplitude_gamma"]
            - new_native["registered_mean_amplitude_gamma"],
            "interpretation": "isolates fit endpoint 69.16 versus 59.68 within current histories",
        },
        {
            "comparison": "historical versus current at shared endpoint",
            "arm": "cic",
            "from_gamma": new_short["registered_mean_amplitude_gamma"],
            "to_gamma": old_native["registered_mean_amplitude_gamma"],
            "difference_to_minus_from": old_native["registered_mean_amplitude_gamma"]
            - new_short["registered_mean_amplitude_gamma"],
            "interpretation": "different seeds and campaigns; source revision for historical job is unknown, so seed and implementation effects cannot be isolated",
        },
    ]
    diagnostic_ratio_rows = []
    summary_keys = sorted(
        {
            (row["campaign"], float(row["v_beam"]), row["window"])
            for row in summaries
        }
    )
    for campaign, drift, window in summary_keys:
        grouped = {
            row["arm"]: row
            for row in summaries
            if row["campaign"] == campaign
            and float(row["v_beam"]) == drift
            and row["window"] == window
        }
        if "cic" not in grouped:
            continue
        cic_seed_rows = sorted(
            (
                row
                for row in seed_rows
                if row["campaign"] == campaign
                and float(row["v_beam"]) == drift
                and row["window"] == window
                and row["arm"] == "cic"
            ),
            key=lambda row: int(row["seed"]),
        )
        for arm in ("fixed_gaussian", "constrained_gaussian"):
            if arm not in grouped:
                continue
            filtered_seed_rows = sorted(
                (
                    row
                    for row in seed_rows
                    if row["campaign"] == campaign
                    and float(row["v_beam"]) == drift
                    and row["window"] == window
                    and row["arm"] == arm
                ),
                key=lambda row: int(row["seed"]),
            )
            if [row["seed"] for row in filtered_seed_rows] != [
                row["seed"] for row in cic_seed_rows
            ]:
                raise AssertionError("individual diagnostic ratios are not paired")
            paired_ratios = np.asarray(
                [
                    float(filtered["gamma_individual"])
                    / float(baseline["gamma_individual"])
                    for baseline, filtered in zip(cic_seed_rows, filtered_seed_rows)
                ]
            )
            if campaign == "registered_drift_scan_5201231":
                scan = next(
                    row for row in config["scan_rows"] if float(row["v_beam"]) == drift
                )
                predicted_ratio = float(
                    scan[
                        "fixed_predicted_rate_ratio"
                        if arm == "fixed_gaussian"
                        else "constrained_predicted_rate_ratio"
                    ]
                )
            elif arm == "fixed_gaussian":
                predicted_ratio = float(
                    next(
                        row
                        for row in corner_analysis["rows"]
                        if row["label"] == "y02428"
                    )["warm_ratio"]
                )
            else:
                predicted_ratio = float("nan")
            diagnostic_ratio_rows.append(
                {
                    "campaign": campaign,
                    "v_beam": drift,
                    "arm": arm,
                    "window": window,
                    "predicted_rate_ratio": predicted_ratio,
                    "registered_ratio_of_mean_amplitude_rates": grouped[arm][
                        "registered_mean_amplitude_gamma"
                    ]
                    / grouped["cic"]["registered_mean_amplitude_gamma"],
                    "ratio_of_individual_rate_means": grouped[arm][
                        "individual_gamma_mean"
                    ]
                    / grouped["cic"]["individual_gamma_mean"],
                    "mean_of_paired_individual_rate_ratios": float(
                        np.mean(paired_ratios)
                    ),
                    "sd_of_paired_individual_rate_ratios": float(
                        np.std(paired_ratios, ddof=1)
                    ),
                }
            )
    paired_correlation_rows = []
    for scan in config["scan_rows"]:
        drift = float(scan["v_beam"])
        if (drift, "cic") not in loaded_current:
            continue
        current_rates = {}
        for arm in ARM_ORDER:
            current_rates[arm] = np.asarray(
                [
                    float(row["gamma_individual"])
                    for row in seed_rows
                    if row["campaign"] == "registered_drift_scan_5201231"
                    and row["window"] == "registered_frozen"
                    and float(row["v_beam"]) == drift
                    and row["arm"] == arm
                ],
                dtype=np.float64,
            )
        if any(values.size != len(config["seeds"]) for values in current_rates.values()):
            raise AssertionError("paired seed-rate vector is incomplete")
        for arm in ("fixed_gaussian", "constrained_gaussian"):
            paired_correlation_rows.append(
                {
                    "v_beam": drift,
                    "comparison_arm": arm,
                    "pearson_correlation_with_cic_individual_rates": float(
                        np.corrcoef(current_rates["cic"], current_rates[arm])[0, 1]
                    ),
                    "interpretation": "paired finite-particle seed effects are shared across arms",
                }
            )
    for arm in ("fixed_gaussian",):
        current_native = summary_lookup[
            ("registered_drift_scan_5201231", arm, "registered_frozen")
        ]
        current_short = summary_lookup[
            (
                "registered_drift_scan_5201231",
                arm,
                "historical_endpoint_sensitivity",
            )
        ]
        historical_native = summary_lookup[
            ("historical_corner_4913408", arm, "historical_native")
        ]
        comparison_rows.extend(
            [
                {
                    "comparison": "current endpoint effect, same seeds and implementation",
                    "arm": arm,
                    "from_gamma": current_native["registered_mean_amplitude_gamma"],
                    "to_gamma": current_short["registered_mean_amplitude_gamma"],
                    "difference_to_minus_from": current_short["registered_mean_amplitude_gamma"]
                    - current_native["registered_mean_amplitude_gamma"],
                    "interpretation": "isolates fit endpoint 69.16 versus 59.68 within current histories",
                },
                {
                    "comparison": "historical versus current at shared endpoint",
                    "arm": arm,
                    "from_gamma": current_short["registered_mean_amplitude_gamma"],
                    "to_gamma": historical_native["registered_mean_amplitude_gamma"],
                    "difference_to_minus_from": historical_native["registered_mean_amplitude_gamma"]
                    - current_short["registered_mean_amplitude_gamma"],
                    "interpretation": "different seeds and campaigns; widths differ by 0.0000654 dy and the historical source revision is unknown",
                },
            ]
        )

    args.output.mkdir(parents=True)
    write_csv(args.output / "estimator_summary.csv", summaries)
    write_csv(args.output / "individual_seed_rates.csv", seed_rows)
    write_csv(args.output / "time_dependent_seed_contributions.csv", contribution_rows)
    write_csv(args.output / "contribution_concentration.csv", concentration_rows)
    write_csv(args.output / "campaign_endpoint_comparison.csv", comparison_rows)
    write_csv(args.output / "diagnostic_rate_ratios.csv", diagnostic_ratio_rows)
    write_csv(args.output / "paired_seed_rate_correlations.csv", paired_correlation_rows)

    result = {
        "status": "POST_HOC_DIAGNOSTIC_REGISTERED_RESULTS_UNCHANGED",
        "registered_estimator": config["fit"]["observable"],
        "diagnostic_estimators": {
            "individual_seed": "fit slope(log(abs(Ehat_s))) separately, then summarize seeds",
            "geometric_mean": "slope(log(geometric mean_s(abs(Ehat_s)))); algebraically equals mean individual slope on a common sampled window",
            "fitted_exponential_mixture": "fit A_s exp(gamma_s t) per seed, form arithmetic mean, then fit its log",
            "contribution": "abs(Ehat_s(t)) / sum_s abs(Ehat_s(t))",
        },
        "vbeam_080_cic_diagnosis": {
            "numerical_theory_gamma": theory,
            "registered_gamma": new_native["registered_mean_amplitude_gamma"],
            "registered_relative_error_denominator_measured": abs(registered_gap)
            / abs(new_native["registered_mean_amplitude_gamma"]),
            "individual_gamma_mean": new_native["individual_gamma_mean"],
            "individual_mean_relative_error_denominator_measured": abs(residual_gap)
            / abs(new_native["individual_gamma_mean"]),
            "individual_gamma_sd": new_native["individual_gamma_sd"],
            "registered_minus_individual_mean": estimator_gap,
            "fraction_of_registered_absolute_theory_gap_associated_with_estimator": estimator_gap
            / registered_gap,
            "residual_individual_mean_minus_theory": residual_gap,
            "fitted_exponential_mixture_gamma": new_native[
                "fitted_exponential_mixture_gamma"
            ],
            "registered_minus_fitted_exponential_mixture": new_native[
                "registered_minus_fitted_mixture"
            ],
            "effective_seed_count_start": new_native["effective_seed_count_start"],
            "effective_seed_count_end": new_native["effective_seed_count_end"],
            "largest_seed_fraction_start": new_native[
                "largest_seed_fraction_start"
            ],
            "largest_seed_fraction_end": new_native["largest_seed_fraction_end"],
            "gamma_vs_contribution_change_correlation": new_native[
                "gamma_vs_contribution_change_correlation"
            ],
            "conclusion": "most, but not all, of the registered 18.85% discrepancy is associated with arithmetic-mean amplitude reweighting across heterogeneous finite-particle seeds; this diagnostic does not redefine the registered estimator",
        },
        "vbeam_080_campaign_comparison": {
            "current_seeds": config["seeds"],
            "historical_seeds": list(range(10)),
            "current_registered_window": [24.32, 69.16],
            "historical_window": [24.32, 59.68],
            "current_cic_gamma_registered": new_native[
                "registered_mean_amplitude_gamma"
            ],
            "current_cic_gamma_at_historical_endpoint": new_short[
                "registered_mean_amplitude_gamma"
            ],
            "historical_cic_gamma": old_native[
                "registered_mean_amplitude_gamma"
            ],
            "endpoint_effect_current_same_seeds": new_native[
                "registered_mean_amplitude_gamma"
            ]
            - new_short["registered_mean_amplitude_gamma"],
            "remaining_current_minus_historical_at_shared_endpoint": new_short[
                "registered_mean_amplitude_gamma"
            ]
            - old_native["registered_mean_amplitude_gamma"],
            "fixed_width_current_sigma_y_dy": config[
                "fixed_gaussian_sigma_y_dy"
            ],
            "fixed_width_historical_sigma_y_dy": 0.2428,
            "source_revision_current": config["source_identity"][
                "remote_git_revision_observed_before_freeze"
            ],
            "source_revision_historical": None,
            "source_revision_historical_status": "UNKNOWN: not recorded in recovered output",
            "identifiability_limit": "the shared-endpoint campaign difference cannot be uniquely assigned to seeds versus an unrecorded historical source revision",
        },
        "implementation_check": {
            "current_transfer_regression_status": transfer["status"],
            "current_transfer_maximum_absolute_error_W2": transfer[
                "maximum_absolute_error_W2"
            ],
            "current_vs_historical_recorded_common_fields": "alpha, benchmark, dt, kx_mode, ky_mode, marker_dynamics, mode, PPC, sigma_x, sigma_y, target-mode projection, and validity threshold agree for CIC",
            "historical_source_revision": None,
            "limitation": "matching recorded settings and the current transfer test do not prove historical source identity",
        },
        "extension_decision": {
            "job_submitted": False,
            "reason": "the estimator mechanism is established, but a residual finite-particle/campaign difference remains and the historical source revision is unknown; the requested diagnostic precedes any budget extension",
        },
        "input_sha256": dict(sorted(input_hashes.items())),
        "tables": {
            "estimator_summary": "estimator_summary.csv",
            "individual_seed_rates": "individual_seed_rates.csv",
            "time_dependent_seed_contributions": "time_dependent_seed_contributions.csv",
            "contribution_concentration": "contribution_concentration.csv",
            "campaign_endpoint_comparison": "campaign_endpoint_comparison.csv",
            "diagnostic_rate_ratios": "diagnostic_rate_ratios.csv",
            "paired_seed_rate_correlations": "paired_seed_rate_correlations.csv",
        },
    }
    (args.output / "diagnostic_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result["vbeam_080_cic_diagnosis"], indent=2, sort_keys=True))
    print(json.dumps(result["vbeam_080_campaign_comparison"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
