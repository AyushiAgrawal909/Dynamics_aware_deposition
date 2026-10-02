#!/usr/bin/env python3
"""Analyze the paired projected-delta-f m=12 smoothing sweep."""

from __future__ import annotations

import argparse
import csv
import json
import math
import pathlib
import sys

import numpy as np

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from analyze_E2_mode_growth import select_growth_window
from pic_kde_2d.benchmarks.theory import (
    matched_cic_force_coupling,
    warm_two_stream_growth,
)
from pic_kde_2d.validation.landau_fit import fit_gamma_ensemble_fixed_window


def load_sigma_group(path: pathlib.Path, expected_seeds: int) -> dict:
    histories = []
    seed_ids = []
    times = None
    sigma_dx = None
    validity = []
    for seed_path in sorted(path.glob("seed*")):
        result_path = seed_path / "result.json"
        history_path = seed_path / "mode_history.npz"
        if not result_path.exists() or not history_path.exists():
            continue
        metadata = json.loads(result_path.read_text(encoding="utf-8"))
        this_sigma = float(metadata["sigma_dx"])
        if sigma_dx is None:
            sigma_dx = this_sigma
        elif this_sigma != sigma_dx:
            raise RuntimeError(f"mixed sigma values in {path}")
        with np.load(history_path) as data:
            this_times = np.asarray(data["times"], dtype=np.float64)
            histories.append(np.asarray(data["mode"], dtype=np.complex128))
            seed_ids.append(int(np.asarray(data["seed"]).reshape(-1)[0]))
            validity.append(float(np.max(np.asarray(data["delta_f_ratio"]))))
        if times is None:
            times = this_times
        else:
            np.testing.assert_allclose(times, this_times, rtol=0.0, atol=0.0)
    if len(histories) != expected_seeds:
        raise RuntimeError(f"{path}: expected {expected_seeds} seeds, found {len(histories)}")
    if sorted(seed_ids) != list(range(expected_seeds)):
        raise RuntimeError(f"{path}: seed IDs are not 0..{expected_seeds - 1}")
    assert times is not None and sigma_dx is not None
    order = np.argsort(seed_ids)
    return {
        "sigma_dx": sigma_dx,
        "times": times,
        "amplitudes": np.abs(np.stack(histories)[order]),
        "max_delta_f_ratio": max(validity),
    }


def fit_rate(amplitudes: np.ndarray, times: np.ndarray, t_start: float, t_stop: float) -> float:
    fit = fit_gamma_ensemble_fixed_window(
        [row for row in amplitudes],
        times,
        t_fit_start=t_start,
        t_fit_stop=t_stop,
        amp_floor=1.0e-30,
    )
    return float(fit["gamma"])


def paired_bootstrap(
    groups: list[dict],
    t_start: float,
    t_stop: float,
    n_boot: int,
    seed: int,
) -> tuple[list[dict], list[np.ndarray]]:
    rng = np.random.default_rng(seed)
    n_seeds = groups[0]["amplitudes"].shape[0]
    ratios = [[] for _ in groups]
    rates = [[] for _ in groups]
    for _ in range(n_boot):
        indices = rng.integers(0, n_seeds, size=n_seeds)
        boot_rates = [
            fit_rate(group["amplitudes"][indices], group["times"], t_start, t_stop)
            for group in groups
        ]
        baseline = boot_rates[0]
        if (
            not np.isfinite(baseline)
            or baseline <= 0.0
            or not all(np.isfinite(rate) for rate in boot_rates)
        ):
            continue
        for index, rate in enumerate(boot_rates):
            rates[index].append(rate)
            ratios[index].append(rate / baseline)

    output = []
    for rate_values, ratio_values in zip(rates, ratios):
        rate_array = np.asarray(rate_values, dtype=np.float64)
        ratio_array = np.asarray(ratio_values, dtype=np.float64)
        output.append(
            {
                "gamma_sem": float(np.std(rate_array)),
                "gamma_ci_lo": float(np.percentile(rate_array, 2.5)),
                "gamma_ci_hi": float(np.percentile(rate_array, 97.5)),
                "ratio_sem": float(np.std(ratio_array)),
                "ratio_ci_lo": float(np.percentile(ratio_array, 2.5)),
                "ratio_ci_hi": float(np.percentile(ratio_array, 97.5)),
                "bootstrap_valid": int(ratio_array.size),
            }
        )
    return output, [np.asarray(values, dtype=np.float64) for values in ratios]


def central_crossing(rows: list[dict], threshold: float = 0.9) -> float:
    for left, right in zip(rows[:-1], rows[1:]):
        y0 = float(left["gamma_ratio"])
        y1 = float(right["gamma_ratio"])
        if y0 >= threshold and y1 <= threshold:
            fraction = (threshold - y0) / (y1 - y0)
            return float(left["sigma_dx"] + fraction * (right["sigma_dx"] - left["sigma_dx"]))
    return float("nan")


def estimate_growing_floor(
    times: np.ndarray,
    amplitudes: np.ndarray,
    fit_start: float,
    window_duration: float = 2.0,
) -> dict:
    """Measure the quietest pre-emergence plateau before the growth fit."""
    ensemble = np.mean(amplitudes, axis=0)
    dt = float(np.median(np.diff(times)))
    width = max(5, int(round(window_duration / dt)))
    stop_limit = int(np.searchsorted(times, fit_start, side="left"))
    if stop_limit < width:
        raise RuntimeError("insufficient pre-fit samples for the growing-mode floor")
    best = None
    for start in range(0, stop_limit - width + 1):
        stop = start + width
        floor = float(np.median(ensemble[start:stop]))
        candidate = (floor, start, stop)
        if best is None or candidate[0] < best[0]:
            best = candidate
    assert best is not None
    floor, start, stop = best
    fit_index = int(np.searchsorted(times, fit_start, side="left"))
    return {
        "noise_floor": floor,
        "floor_t_start": float(times[start]),
        "floor_t_stop": float(times[stop - 1]),
        "fit_start_amplitude": float(ensemble[fit_index]),
        "fit_start_snr": float(ensemble[fit_index] / max(floor, 1.0e-300)),
        "floor_scope": "per-cell",
        "floor_definition": "quietest rolling-median pre-emergence window",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--nboot", type=int, default=2000)
    parser.add_argument("--raw-prior", type=float, default=0.19872036040909125)
    parser.add_argument("--corrected-prior", type=float, default=0.18445488487247555)
    parser.add_argument("--damage-calibration", type=float, default=1.1606583770168997)
    parser.add_argument("--safe-margin-sigma", type=float, default=0.175)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    groups = [
        load_sigma_group(path, args.seeds)
        for path in args.input.glob("sigma_*")
        if path.is_dir()
    ]
    groups.sort(key=lambda group: group["sigma_dx"])
    if not groups or groups[0]["sigma_dx"] != 0.0:
        raise RuntimeError("the sweep must contain a sigma=0 paired baseline")
    for group in groups[1:]:
        np.testing.assert_allclose(groups[0]["times"], group["times"], rtol=0.0, atol=0.0)

    times = groups[0]["times"]
    baseline_mean = np.mean(groups[0]["amplitudes"], axis=0)
    window = select_growth_window(
        times,
        baseline_mean,
        t_min=0.3,
        t_start_max=min(30.0, float(times[-1]) - 5.0),
        t_max=float(times[-1]),
        min_duration=5.0,
        min_points=30,
        target_r2=0.98,
        saturation_fraction=0.95,
    )
    t_start = float(window["t_start"])
    t_stop = float(window["t_stop"])
    if not np.isfinite(t_start) or not np.isfinite(t_stop):
        raise RuntimeError("could not select the common baseline growth window")

    central_rates = [
        fit_rate(group["amplitudes"], group["times"], t_start, t_stop)
        for group in groups
    ]
    boot, ratio_samples = paired_bootstrap(groups, t_start, t_stop, args.nboot, seed=53120)
    baseline_rate = central_rates[0]
    k = 2.0 * math.pi * 12.0 / 64.0

    def predicted_ratio(sigma_dx: float) -> float:
        baseline_coupling = matched_cic_force_coupling(12, 128, 0.0)
        coupling = matched_cic_force_coupling(12, 128, sigma_dx)
        baseline = warm_two_stream_growth(k, 0.8, 0.05, baseline_coupling)
        return warm_two_stream_growth(k, 0.8, 0.05, coupling) / baseline

    rows = []
    for group, rate, uncertainty in zip(groups, central_rates, boot):
        ratio = rate / baseline_rate
        raw_prediction = predicted_ratio(group["sigma_dx"])
        corrected_prediction = 1.0 - args.damage_calibration * (1.0 - raw_prediction)
        floor = estimate_growing_floor(
            group["times"],
            group["amplitudes"],
            t_start,
        )
        rows.append(
            {
                "sigma_dx": group["sigma_dx"],
                "gamma": rate,
                **uncertainty,
                "gamma_ratio": ratio,
                "raw_closed_form_ratio": raw_prediction,
                "raw_closed_form_residual": ratio - raw_prediction,
                "bias_corrected_ratio": corrected_prediction,
                "bias_corrected_residual": ratio - corrected_prediction,
                "ci_supported_10pct": (
                    uncertainty["ratio_ci_lo"] >= 0.9
                    and uncertainty["ratio_ci_hi"] <= 1.1
                ),
                **floor,
                "max_delta_f_ratio": group["max_delta_f_ratio"],
                "n_seeds": args.seeds,
                "fit_t_start": t_start,
                "fit_t_stop": t_stop,
            }
        )

    with (args.output / "m12_sigma_sweep.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    crossing = central_crossing(rows)
    supported = [row["sigma_dx"] for row in rows if row["ci_supported_10pct"]]
    raw_index = next(
        index
        for index, row in enumerate(rows)
        if np.isclose(row["sigma_dx"], args.raw_prior, rtol=0.0, atol=1.0e-12)
    )
    corrected_index = next(
        index
        for index, row in enumerate(rows)
        if np.isclose(row["sigma_dx"], args.corrected_prior, rtol=0.0, atol=1.0e-12)
    )
    candidate_difference = ratio_samples[corrected_index] - ratio_samples[raw_index]
    candidate_difference_ci = [
        float(np.percentile(candidate_difference, 2.5)),
        float(np.percentile(candidate_difference, 97.5)),
    ]
    raw_interval = (rows[raw_index]["ratio_ci_lo"], rows[raw_index]["ratio_ci_hi"])
    corrected_interval = (
        rows[corrected_index]["ratio_ci_lo"],
        rows[corrected_index]["ratio_ci_hi"],
    )
    candidate_intervals_overlap = not (
        corrected_interval[0] > raw_interval[1]
        or raw_interval[0] > corrected_interval[1]
    )
    measured_ratios = np.asarray([row["gamma_ratio"] for row in rows])
    raw_predictions = np.asarray([row["raw_closed_form_ratio"] for row in rows])
    corrected_predictions = np.asarray([row["bias_corrected_ratio"] for row in rows])
    ratio_sample_matrix = np.column_stack(ratio_samples)

    # The four nonzero widths jointly discriminate the raw and calibrated
    # curves. Their paired bootstrap preserves all cross-width correlations.
    measured_nonzero = measured_ratios[1:]
    raw_nonzero = raw_predictions[1:]
    corrected_nonzero = corrected_predictions[1:]
    samples_nonzero = ratio_sample_matrix[:, 1:]
    raw_rmse_samples = np.sqrt(np.mean((samples_nonzero - raw_nonzero) ** 2, axis=1))
    corrected_rmse_samples = np.sqrt(
        np.mean((samples_nonzero - corrected_nonzero) ** 2, axis=1)
    )
    rmse_difference_samples = corrected_rmse_samples - raw_rmse_samples
    rmse_difference_ci = [
        float(np.percentile(rmse_difference_samples, 2.5)),
        float(np.percentile(rmse_difference_samples, 97.5)),
    ]
    if rmse_difference_ci[0] > 0.0:
        favored_model = "raw_closed_form"
    elif rmse_difference_ci[1] < 0.0:
        favored_model = "bias_corrected"
    else:
        favored_model = "unresolved"

    central_decreases = measured_ratios[:-1] - measured_ratios[1:]
    bootstrap_decreases = ratio_sample_matrix[:, :-1] - ratio_sample_matrix[:, 1:]
    monotone_pairwise = []
    for index, central_decrease in enumerate(central_decreases):
        values = bootstrap_decreases[:, index]
        monotone_pairwise.append(
            {
                "sigma_left": rows[index]["sigma_dx"],
                "sigma_right": rows[index + 1]["sigma_dx"],
                "central_decrease": float(central_decrease),
                "ci_lo": float(np.percentile(values, 2.5)),
                "ci_hi": float(np.percentile(values, 97.5)),
            }
        )
    monotone_central = bool(np.all(central_decreases > 0.0))
    monotone_ci_supported = bool(all(item["ci_lo"] > 0.0 for item in monotone_pairwise))
    monotone_bootstrap_fraction = float(
        np.mean(np.all(bootstrap_decreases > 0.0, axis=1))
    )

    corrected_row = rows[corrected_index]
    corrected_width_clears_with_margin = bool(corrected_row["ratio_ci_lo"] > 0.9)
    if (
        monotone_ci_supported
        and favored_model == "raw_closed_form"
        and corrected_width_clears_with_margin
    ):
        recommended_sigma = args.corrected_prior
        recommendation_reason = "raw model favored and corrected-prior CI clears the gate"
    else:
        recommended_sigma = args.safe_margin_sigma
        recommendation_reason = (
            "safe-margin default: model unresolved/corrected, monotonicity unsupported, "
            "or corrected-prior CI lacks margin"
        )
    recommended_row = next(
        row
        for row in rows
        if np.isclose(row["sigma_dx"], recommended_sigma, rtol=0.0, atol=1.0e-12)
    )
    result = {
        "mode": 12,
        "paired_common_random_numbers": True,
        "common_fit_window": [t_start, t_stop],
        "fit_window_r2": float(window["r2"]),
        "fit_window_efoldings": float(window["efoldings"]),
        "central_10pct_crossing_dx": crossing,
        "largest_ci_supported_sigma_dx": max(supported) if supported else None,
        "recommended_conservative_sigma_dx": recommended_sigma,
        "recommended_rate_ci_supported": bool(recommended_row["ci_supported_10pct"]),
        "recommendation_reason": recommendation_reason,
        "raw_closed_form_prior_dx": args.raw_prior,
        "bias_corrected_prior_dx": args.corrected_prior,
        "raw_curve_rmse": float(np.sqrt(np.mean((measured_nonzero - raw_nonzero) ** 2))),
        "bias_corrected_curve_rmse": float(
            np.mean((measured_nonzero - corrected_nonzero) ** 2) ** 0.5
        ),
        "rmse_difference_corrected_minus_raw_ci": rmse_difference_ci,
        "favored_model": favored_model,
        "model_comparison_primary": True,
        "monotone_central": monotone_central,
        "monotone_ci_supported": monotone_ci_supported,
        "monotone_bootstrap_fraction": monotone_bootstrap_fraction,
        "monotone_pairwise": monotone_pairwise,
        "power_precheck_pass": monotone_ci_supported,
        "candidate_ratio_difference_corrected_minus_raw": float(
            rows[corrected_index]["gamma_ratio"] - rows[raw_index]["gamma_ratio"]
        ),
        "candidate_ratio_difference_ci": candidate_difference_ci,
        "candidate_intervals_overlap": candidate_intervals_overlap,
        "candidate_discrimination_supported": bool(candidate_difference_ci[0] > 0.0),
        "selection_policy": (
            "model comparison is primary; default to 0.175 dx unless the raw model "
            "is favored and the 0.1845 dx lower CI exceeds 0.9"
        ),
        "closer_prior": (
            "bias_corrected"
            if abs(crossing - args.corrected_prior) < abs(crossing - args.raw_prior)
            else "raw_closed_form"
        )
        if np.isfinite(crossing)
        else "unresolved",
        "rows": rows,
    }
    (args.output / "m12_sigma_sweep.json").write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
