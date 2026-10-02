#!/usr/bin/env python3
"""Retrieve and reduce the remaining Paper-1 validation data without new runs.

This script is deliberately read-only with respect to all source campaigns.  It
creates a new results directory containing:

* the complete 85-point E2 anisotropic candidate table augmented with the
  5%-operating/10%-acceptance model flags and the fixed seed split;
* the selected-candidate held-out error calculation under the complete rule;
* a paired three-arm weak-Landau common-window analysis over [5, 28]; and
* the existing v_b=0.80 initialization-only finite-marker predictions joined
  to fitted rates and reconstructed loading diagnostics.

No simulation is launched and no manuscript source is touched.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Iterable

import numpy as np


HERE = Path(__file__).resolve().parent
DOWNLOADS = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import analyze_weak_landau_confirmatory as landau_base  # noqa: E402
import analyze_weak_landau_confirmatory_v2 as landau_v2  # noqa: E402
from recompute_dispersion import E2_M12N1, solve_level  # noqa: E402


BANDWIDTH_ROOT = (
    DOWNLOADS
    / "pic_kde_2d/results_raw_rate_audit/recovered_meluxina_20260909"
    / "e2_anisotropic_dense_5197174"
)
BANDWIDTH_ANALYSIS = BANDWIDTH_ROOT / "analysis"
LANDAU_CONFIG = HERE / "weak_landau_confirmatory_v2_global_velocity.json"
LANDAU_LOW = (
    HERE
    / "results/weak_landau_confirmatory_v2_production_20260911/low"
)
FINITE_MARKER_ROOT = HERE / "results/e2_seed_rate_variation_20260912_with_extension"
E2_ESTIMAND_ROOT = (
    HERE / "results/bounded_floor_estimand_reference_correction_20260914_r12"
)
DEFAULT_OUTPUT = HERE / "results/missing_validation_data_retrieval_20260921"

TRAIN_SEEDS = tuple(range(100, 105))
TEST_SEEDS = tuple(range(105, 110))
OPERATING_MIN_RATIO = 0.95
ACCEPTANCE_MIN_RATIO = 0.90
# Root residuals are converged to 2e-11.  Classification at an analytically
# constructed boundary must not turn that solver tolerance into a false fail.
MODEL_RATIO_FLAG_TOLERANCE = 1.0e-8
BOOTSTRAP_DRAWS = 5000
BOOTSTRAP_SEED = 20260911
COMMON_START = 5.0
COMMON_STOP = 28.0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Iterable[dict]) -> None:
    materialized = list(rows)
    if not materialized:
        raise ValueError(f"refusing to write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(materialized[0]))
        writer.writeheader()
        writer.writerows(materialized)


def percentile_interval(values: np.ndarray) -> tuple[float, float]:
    lo, hi = np.percentile(np.asarray(values, dtype=np.float64), [2.5, 97.5])
    return float(lo), float(hi)


def ratio_interval(
    baseline: np.ndarray,
    candidate: np.ndarray,
    *,
    seed: int,
    draws: int = BOOTSTRAP_DRAWS,
) -> tuple[float, float, float]:
    baseline = np.asarray(baseline, dtype=np.float64)
    candidate = np.asarray(candidate, dtype=np.float64)
    if baseline.shape != candidate.shape or baseline.ndim != 1:
        raise ValueError("paired baseline/candidate arrays must be one-dimensional")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, baseline.size, size=(draws, baseline.size))
    values = np.mean(baseline[indices], axis=1) / np.mean(candidate[indices], axis=1)
    central = float(np.mean(baseline) / np.mean(candidate))
    lo, hi = percentile_interval(values)
    return central, lo, hi


def configure_matplotlib() -> None:
    import matplotlib as mpl

    # The analysis is routinely executed in a non-interactive shell on macOS
    # and on MeluXina.  Select a file-only backend before pyplot is imported.
    mpl.use("Agg", force=True)
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 7.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "savefig.bbox": "tight",
        }
    )


def bandwidth_analysis(output: Path) -> dict:
    source_candidates = BANDWIDTH_ANALYSIS / "anisotropic_dense_candidates.csv"
    rows = read_csv(source_candidates)
    if len(rows) != 85 or len({row["label"] for row in rows}) != 85:
        raise AssertionError("expected exactly 85 unique bandwidth candidates")
    labels = [row["label"] for row in rows]
    if labels[0] != "base":
        raise AssertionError("candidate ordering changed")

    by_seed: dict[int, dict[str, dict[str, str]]] = {}
    per_seed_rows: list[dict] = []
    input_hashes: dict[str, str] = {str(source_candidates): sha256(source_candidates)}
    for seed in (*TRAIN_SEEDS, *TEST_SEEDS):
        path = BANDWIDTH_ROOT / f"seed{seed}/result/anisotropic_dense_noise_seed.csv"
        current = read_csv(path)
        if [row["label"] for row in current] != labels:
            raise AssertionError(f"candidate ordering changed for seed {seed}")
        by_seed[seed] = {row["label"]: row for row in current}
        input_hashes[str(path)] = sha256(path)
        split = "selection" if seed in TRAIN_SEEDS else "independent_evaluation"
        for row in current:
            per_seed_rows.append(
                {
                    "seed": seed,
                    "split": split,
                    "label": row["label"],
                    "density_mse": float(row["density_mse"]),
                    "field_mse": float(row["field_mse"]),
                    "max_neutrality_defect": float(row["max_neutrality_defect"]),
                    "max_relative_charge_error": float(row["max_relative_charge_error"]),
                }
            )
    write_csv(output / "bandwidth_85_per_seed_objectives.csv", per_seed_rows)

    density = np.asarray(
        [
            [float(by_seed[seed][label]["density_mse"]) for label in labels]
            for seed in (*TRAIN_SEEDS, *TEST_SEEDS)
        ],
        dtype=np.float64,
    )
    field = np.asarray(
        [
            [float(by_seed[seed][label]["field_mse"]) for label in labels]
            for seed in (*TRAIN_SEEDS, *TEST_SEEDS)
        ],
        dtype=np.float64,
    )
    matrices = {"density": density, "field": field}

    baseline = solve_level(E2_M12N1, "grid_time_discrete", qmax=8, lmax=1024)
    q_values = sorted({float(row["q_total"]) for row in rows})
    roots: dict[float, complex] = {}
    rate_ratios: dict[float, float] = {}
    previous = baseline
    for q_total in q_values:
        root = solve_level(
            E2_M12N1,
            "grid_time_discrete",
            filter_force_transfer=math.exp(-q_total),
            initial=previous,
            qmax=8,
            lmax=1024,
        )
        roots[q_total] = root
        rate_ratios[q_total] = float(root.real / baseline.real)
        previous = root

    feasible_indices: list[int] = []
    enriched: list[dict] = []
    for index, row in enumerate(rows):
        q_total = float(row["q_total"])
        ratio = rate_ratios[q_total]
        operating = bool(ratio >= OPERATING_MIN_RATIO - MODEL_RATIO_FLAG_TOLERANCE)
        acceptance = bool(ratio >= ACCEPTANCE_MIN_RATIO - MODEL_RATIO_FLAG_TOLERANCE)
        if operating:
            feasible_indices.append(index)
        enriched.append(
            {
                **row,
                "predicted_grid_time_gamma": float(roots[q_total].real),
                "predicted_rate_ratio_to_cic": ratio,
                "operating_5pct_model_feasible": operating,
                "acceptance_10pct_model_feasible": acceptance,
                "selection_seeds": ";".join(map(str, TRAIN_SEEDS)),
                "independent_evaluation_seeds": ";".join(map(str, TEST_SEEDS)),
            }
        )
    if not feasible_indices:
        raise AssertionError("5% operating set is empty")

    selected: dict[str, int] = {}
    summaries: list[dict] = []
    nested_details: dict[str, dict] = {}
    for metric_index, (metric, matrix) in enumerate(matrices.items()):
        train_mean = np.mean(matrix[: len(TRAIN_SEEDS)], axis=0)
        selected_index = min(feasible_indices, key=lambda idx: train_mean[idx])
        selected[metric] = selected_index
        candidate = matrix[len(TRAIN_SEEDS) :, selected_index]
        reference = matrix[len(TRAIN_SEEDS) :, 0]
        reduction, ci_lo, ci_hi = ratio_interval(
            reference, candidate, seed=271828 + metric_index
        )

        rng = np.random.default_rng(314159 + metric_index)
        nested_reductions = np.empty(BOOTSTRAP_DRAWS, dtype=np.float64)
        nested_selected = np.empty(BOOTSTRAP_DRAWS, dtype=np.int64)
        for draw in range(BOOTSTRAP_DRAWS):
            train_draw = rng.integers(0, len(TRAIN_SEEDS), size=len(TRAIN_SEEDS))
            test_draw = rng.integers(
                len(TRAIN_SEEDS),
                len(TRAIN_SEEDS) + len(TEST_SEEDS),
                size=len(TEST_SEEDS),
            )
            draw_selected = min(
                feasible_indices,
                key=lambda idx: float(np.mean(matrix[train_draw, idx])),
            )
            nested_selected[draw] = draw_selected
            nested_reductions[draw] = float(
                np.mean(matrix[test_draw, 0]) / np.mean(matrix[test_draw, draw_selected])
            )
        nested_lo, nested_hi = percentile_interval(nested_reductions)
        values, counts = np.unique(nested_selected, return_counts=True)
        frequencies = sorted(
            (
                {
                    "label": labels[int(value)],
                    "fraction": float(count / BOOTSTRAP_DRAWS),
                }
                for value, count in zip(values, counts)
            ),
            key=lambda item: item["fraction"],
            reverse=True,
        )
        nested_details[metric] = {
            "nested_ci_lo": nested_lo,
            "nested_ci_hi": nested_hi,
            "selection_frequencies": frequencies,
        }
        row = enriched[selected_index]
        summaries.append(
            {
                "metric": metric,
                "selected_label": labels[selected_index],
                "sigma_x_dx": float(row["sigma_x_dx"]),
                "sigma_y_dy": float(row["sigma_y_dy"]),
                "q_fraction": float(row["q_fraction"]),
                "x_allocation": float(row["x_allocation"]),
                "predicted_rate_ratio": float(row["predicted_rate_ratio_to_cic"]),
                "heldout_error_reduction": reduction,
                "heldout_ci_lo": ci_lo,
                "heldout_ci_hi": ci_hi,
                "nested_ci_lo": nested_lo,
                "nested_ci_hi": nested_hi,
                "selection_seeds": ";".join(map(str, TRAIN_SEEDS)),
                "evaluation_seeds": ";".join(map(str, TEST_SEEDS)),
                "exact_selected_width_dynamics_available": False,
                "complete_rule_status": "PARTIAL_ERROR_PASS_DYNAMICS_UNRESOLVED",
            }
        )

    for index, row in enumerate(enriched):
        row["selected_by_complete_density_rule"] = index == selected["density"]
        row["selected_by_complete_field_rule"] = index == selected["field"]
    write_csv(output / "bandwidth_85_complete_rule.csv", enriched)
    write_csv(output / "bandwidth_complete_rule_selection.csv", summaries)

    # The nearest qualified dynamics setting is retained only as context, not
    # substituted for the exact discrete-design selection.
    exact_width = float(enriched[selected["density"]]["sigma_y_dy"])
    nearby_width = 0.1738592384458977
    estimator_rows = read_csv(E2_ESTIMAND_ROOT / "e2_estimand_ratios_like_for_like.csv")
    nearby_measurement = next(
        row
        for row in estimator_rows
        if row["v_beam"] == "0.8" and row["candidate"] == "constrained_gaussian"
    )
    dynamics_context = {
        "exact_selected_label": labels[selected["density"]],
        "exact_selected_sigma_y_dy": exact_width,
        "exact_width_history_found": False,
        "nearest_current_qualified_setting": {
            "sigma_y_dy": nearby_width,
            "relative_width_difference": nearby_width / exact_width - 1.0,
            "role": "nearby R=0.95 constrained setting; context only, not substituted",
            "measured_individual_rate_mean_ratio": float(
                nearby_measurement["measured_ratio_of_individual_rate_means"]
            ),
            "measured_individual_ratio_ci": [
                float(nearby_measurement["measured_ratio_ci_lo"]),
                float(nearby_measurement["measured_ratio_ci_hi"]),
            ],
            "registered_ensemble_amplitude_ratio": float(
                nearby_measurement["registered_ensemble_amplitude_ratio"]
            ),
            "registered_ratio_ci": [
                float(nearby_measurement["registered_ci_lo"]),
                float(nearby_measurement["registered_ci_hi"]),
            ],
        },
        "conclusion": (
            "The selected 85-grid candidate has independently measured held-out error, "
            "but no exact-width qualified dynamics history. The complete rule is therefore "
            "not demonstrated end to end by this dataset."
        ),
    }

    configure_matplotlib()
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(7.0, 2.8), sharex=True)
    colors = np.asarray([float(row["x_allocation"]) for row in enriched])
    x = np.asarray([float(row["predicted_rate_ratio_to_cic"]) for row in enriched])
    for axis, metric in zip(axes, ("density", "field")):
        matrix = matrices[metric]
        objective = np.mean(matrix[: len(TRAIN_SEEDS)], axis=0)
        objective /= objective[0]
        axis.axvspan(OPERATING_MIN_RATIO, 1.005, color="#dff0d8", alpha=0.75, zorder=0)
        axis.axvspan(ACCEPTANCE_MIN_RATIO, OPERATING_MIN_RATIO, color="#fff0c2", alpha=0.65, zorder=0)
        axis.axvline(ACCEPTANCE_MIN_RATIO, color="0.45", ls=":", lw=1.0)
        axis.axvline(OPERATING_MIN_RATIO, color="#346a34", ls="--", lw=1.0)
        scatter = axis.scatter(
            x,
            objective,
            c=colors,
            cmap="viridis",
            s=18,
            alpha=0.85,
            edgecolors="none",
            zorder=2,
        )
        selected_index = selected[metric]
        axis.scatter(
            [x[selected_index]],
            [objective[selected_index]],
            marker="*",
            s=100,
            color="#b2182b",
            edgecolor="white",
            linewidth=0.7,
            zorder=4,
            label=f"selected: {labels[selected_index]}",
        )
        boundary_index = labels.index("q100_x000")
        axis.scatter(
            [x[boundary_index]],
            [objective[boundary_index]],
            marker="o",
            facecolor="none",
            edgecolor="0.25",
            linewidth=1.0,
            s=52,
            zorder=3,
            label="10% boundary optimum (excluded)",
        )
        axis.set_title(f"({ 'a' if metric == 'density' else 'b' }) {metric} objective")
        axis.set_xlabel(r"predicted $\gamma/\gamma_{\rm CIC}$")
        axis.set_ylabel(r"selection MSE / CIC MSE")
        axis.set_xlim(0.895, 1.005)
        axis.legend(loc="best", frameon=False)
    colorbar = figure.colorbar(scatter, ax=axes, pad=0.015, fraction=0.04)
    colorbar.set_label("x allocation of protected-mode budget")
    figure.suptitle("85-candidate rule: 5% operating margin, 10% acceptance boundary", y=1.03)
    for suffix in ("pdf", "svg"):
        figure.savefig(output / f"fig_bandwidth_85_complete_rule.{suffix}")
    figure.savefig(output / "fig_bandwidth_85_complete_rule.png", dpi=300)
    plt.close(figure)

    result = {
        "status": "PARTIAL_ERROR_PASS_DYNAMICS_UNRESOLVED",
        "candidate_count": len(enriched),
        "filter_family": "diagonal anisotropic Gaussian",
        "protected_mode": [12, 1],
        "operating_tolerance": 0.05,
        "acceptance_tolerance": 0.10,
        "operating_model_rule": "predicted gamma/gamma_CIC >= 0.95",
        "acceptance_model_rule": "predicted gamma/gamma_CIC >= 0.90",
        "dispersion_level": "grid/time-discrete continuum-velocity response, qmax=8, lmax=1024",
        "cic_predicted_gamma": float(baseline.real),
        "fixed_seed_split": {"selection": list(TRAIN_SEEDS), "evaluation": list(TEST_SEEDS)},
        "selected": summaries,
        "nested_bootstrap": nested_details,
        "dynamics_context": dynamics_context,
        "source_hashes": input_hashes,
    }
    (output / "bandwidth_complete_rule_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def log_slope(times: np.ndarray, positive: np.ndarray, start: float, stop: float) -> dict:
    mask = (
        (times >= start)
        & (times <= stop)
        & np.isfinite(positive)
        & (positive > 0.0)
    )
    x = times[mask]
    y = np.log(positive[mask])
    coefficients = np.polyfit(x, y, 1)
    predicted = np.polyval(coefficients, x)
    residual = float(np.sum((y - predicted) ** 2))
    total = float(np.sum((y - np.mean(y)) ** 2))
    return {
        "slope": float(coefficients[0]),
        "intercept": float(coefficients[1]),
        "r_squared": float(1.0 - residual / total) if total > 0.0 else float("nan"),
        "n_points": int(x.size),
    }


def landau_decomposition(
    histories: np.ndarray, times: np.ndarray, config: dict
) -> tuple[dict, dict[str, np.ndarray]]:
    coherent = np.abs(np.mean(histories, axis=0))
    mean_magnitude = np.mean(np.abs(histories), axis=0)
    coherence = coherent / np.maximum(mean_magnitude, 1.0e-300)
    unweighted_phase = np.abs(
        np.mean(histories / np.maximum(np.abs(histories), 1.0e-300), axis=0)
    )
    period = config["fit_rule"]["envelope_period"]
    coherent_envelope, points = landau_base.geometric_envelope(coherent, times, period)
    magnitude_envelope, points_2 = landau_base.geometric_envelope(mean_magnitude, times, period)
    coherence_envelope, points_3 = landau_base.geometric_envelope(coherence, times, period)
    if len({points, points_2, points_3}) != 1:
        raise AssertionError("envelope point count changed between diagnostics")
    coherent_fit = log_slope(times, coherent_envelope, COMMON_START, COMMON_STOP)
    magnitude_fit = log_slope(times, magnitude_envelope, COMMON_START, COMMON_STOP)
    coherence_fit = log_slope(times, coherence_envelope, COMMON_START, COMMON_STOP)
    if not math.isclose(
        coherent_fit["slope"],
        magnitude_fit["slope"] + coherence_fit["slope"],
        abs_tol=2.0e-13,
    ):
        raise AssertionError("coherent-amplitude slope decomposition failed")
    mask = (times >= COMMON_START) & (times <= COMMON_STOP)
    frequencies = np.asarray(
        [
            np.polyfit(times[mask], np.unwrap(np.angle(history))[mask], 1)[0]
            for history in histories
        ],
        dtype=np.float64,
    )
    start_index = int(np.flatnonzero(np.isclose(times, COMMON_START))[0])
    stop_index = int(np.flatnonzero(np.isclose(times, COMMON_STOP))[0])
    summary = {
        "coherent_gamma": coherent_fit["slope"],
        "coherent_r_squared": coherent_fit["r_squared"],
        "mean_seed_magnitude_gamma": magnitude_fit["slope"],
        "mean_seed_magnitude_r_squared": magnitude_fit["r_squared"],
        "coherence_log_slope": coherence_fit["slope"],
        "coherence_r_squared": coherence_fit["r_squared"],
        "additive_identity_residual": coherent_fit["slope"]
        - magnitude_fit["slope"]
        - coherence_fit["slope"],
        "coherence_at_start": float(coherence[start_index]),
        "coherence_at_stop": float(coherence[stop_index]),
        "coherence_envelope_at_start": float(coherence_envelope[start_index]),
        "coherence_envelope_at_stop": float(coherence_envelope[stop_index]),
        "seed_phase_slope_mean": float(np.mean(frequencies)),
        "seed_phase_slope_sd": float(np.std(frequencies, ddof=1)),
        "seed_phase_slope_min": float(np.min(frequencies)),
        "seed_phase_slope_max": float(np.max(frequencies)),
    }
    arrays = {
        "coherent": coherent,
        "coherent_envelope": coherent_envelope,
        "mean_magnitude": mean_magnitude,
        "magnitude_envelope": magnitude_envelope,
        "coherence": coherence,
        "coherence_envelope": coherence_envelope,
        "unweighted_phase_concentration": unweighted_phase,
    }
    return summary, arrays


def landau_analysis(output: Path) -> dict:
    config, config_path, config_hash = landau_v2.load_config(LANDAU_CONFIG)
    data = landau_base.verify_low_tree(config, config_hash, LANDAU_LOW.resolve())
    times = data["times"]
    arms = tuple(arm["id"] for arm in config["arms"])
    if arms != ("cic", "c4_sigma1", "lscv_job5008392"):
        raise AssertionError(f"unexpected arm order: {arms}")
    if not math.isclose(float(times[-1]), 90.0, abs_tol=1.0e-12):
        raise AssertionError("Landau histories do not reach the frozen tail")

    central = {
        arm: landau_base.fit_fixed_window(
            data["histories"][arm], times, COMMON_START, COMMON_STOP, config
        )
        for arm in arms
    }
    decomposition: dict[str, dict] = {}
    arrays: dict[str, dict[str, np.ndarray]] = {}
    for arm in arms:
        decomposition[arm], arrays[arm] = landau_decomposition(
            data["histories"][arm], times, config
        )
        if not math.isclose(
            central[arm]["gamma"], decomposition[arm]["coherent_gamma"], abs_tol=2.0e-13
        ):
            raise AssertionError(f"coherent fit mismatch for {arm}")

    n_seeds = len(data["seeds"])
    n_draws = int(config["fit_rule"]["bootstrap_draws"])
    rng = np.random.default_rng(int(config["fit_rule"]["bootstrap_seed"]))
    gamma_draws = {arm: np.empty(n_draws, dtype=np.float64) for arm in arms}
    ratio_draws = {
        arm: np.empty(n_draws, dtype=np.float64) for arm in arms if arm != "cic"
    }
    magnitude_draws = {arm: np.empty(n_draws, dtype=np.float64) for arm in arms}
    coherence_draws = {arm: np.empty(n_draws, dtype=np.float64) for arm in arms}
    attribution_draws = np.empty(n_draws, dtype=np.float64)
    for draw in range(n_draws):
        indices = rng.integers(0, n_seeds, size=n_seeds)
        draw_decomposition: dict[str, dict] = {}
        for arm in arms:
            history = data["histories"][arm][indices]
            fit = landau_base.fit_fixed_window(
                history, times, COMMON_START, COMMON_STOP, config
            )
            gamma_draws[arm][draw] = fit["gamma"]
            draw_decomposition[arm], _ = landau_decomposition(history, times, config)
            magnitude_draws[arm][draw] = draw_decomposition[arm][
                "mean_seed_magnitude_gamma"
            ]
            coherence_draws[arm][draw] = draw_decomposition[arm][
                "coherence_log_slope"
            ]
        for arm in ratio_draws:
            ratio_draws[arm][draw] = gamma_draws[arm][draw] / gamma_draws["cic"][draw]
        numerator = (
            draw_decomposition["lscv_job5008392"]["coherence_log_slope"]
            - draw_decomposition["cic"]["coherence_log_slope"]
        )
        denominator = (
            draw_decomposition["lscv_job5008392"]["coherent_gamma"]
            - draw_decomposition["cic"]["coherent_gamma"]
        )
        attribution_draws[draw] = numerator / denominator

    summary_rows: list[dict] = []
    phase_rows: list[dict] = []
    for arm in arms:
        gamma_lo, gamma_hi = percentile_interval(gamma_draws[arm])
        if arm == "cic":
            ratio = ratio_lo = ratio_hi = 1.0
        else:
            ratio = central[arm]["gamma"] / central["cic"]["gamma"]
            ratio_lo, ratio_hi = percentile_interval(ratio_draws[arm])
        fit = central[arm]
        duration_pass = bool(
            fit["efoldings"] >= config["fit_rule"]["min_fitted_amplitude_efoldings"]
        )
        non_duration_quality = bool(
            fit["gamma"] < 0.0
            and fit["n_points"] >= config["fit_rule"]["min_points"]
            and fit["duration"] >= config["fit_rule"]["min_duration"]
            and fit["r_squared"] >= config["fit_rule"]["min_fit_r_squared"]
            and fit["fit_start_snr"]
            >= config["fit_rule"]["min_fit_start_signal_to_floor"]
        )
        summary_rows.append(
            {
                "arm": arm,
                "n_paired_seeds": n_seeds,
                "fit_t_start": COMMON_START,
                "fit_t_stop": COMMON_STOP,
                "gamma": fit["gamma"],
                "gamma_ci_lo": gamma_lo,
                "gamma_ci_hi": gamma_hi,
                "ratio_to_cic": ratio,
                "ratio_ci_lo": ratio_lo,
                "ratio_ci_hi": ratio_hi,
                "r_squared": fit["r_squared"],
                "fitted_efoldings": fit["efoldings"],
                "fit_start_snr": fit["fit_start_snr"],
                "non_duration_signal_fit_gate": "PASS" if non_duration_quality else "FAIL",
                "two_efolding_duration_gate": "PASS" if duration_pass else "FAIL",
                "overall_fixed_window_qualification": "PASS"
                if non_duration_quality and duration_pass
                else "FAIL",
            }
        )
        magnitude_lo, magnitude_hi = percentile_interval(magnitude_draws[arm])
        coherence_lo, coherence_hi = percentile_interval(coherence_draws[arm])
        phase_rows.append(
            {
                "arm": arm,
                **decomposition[arm],
                "mean_seed_magnitude_gamma_ci_lo": magnitude_lo,
                "mean_seed_magnitude_gamma_ci_hi": magnitude_hi,
                "coherence_log_slope_ci_lo": coherence_lo,
                "coherence_log_slope_ci_hi": coherence_hi,
            }
        )
    write_csv(output / "landau_three_arm_common_window.csv", summary_rows)
    write_csv(output / "landau_phase_decomposition.csv", phase_rows)

    aggregate_rows: list[dict] = []
    for index, time in enumerate(times):
        row: dict[str, float] = {"time": float(time)}
        for arm in arms:
            mean_complex = np.mean(data["histories"][arm], axis=0)[index]
            row[f"{arm}_mean_complex_real"] = float(mean_complex.real)
            row[f"{arm}_mean_complex_imag"] = float(mean_complex.imag)
            for name, values in arrays[arm].items():
                row[f"{arm}_{name}"] = float(values[index])
        aggregate_rows.append(row)
    write_csv(output / "landau_histories_and_envelopes.csv", aggregate_rows)

    # CSV export of every retrieved complex history. The authoritative binary
    # paths and hashes are also retained in the manifest below.
    complex_rows: list[dict] = []
    for arm in arms:
        for seed_index, seed in enumerate(data["seeds"]):
            history = data["histories"][arm][seed_index]
            for time, value in zip(times, history):
                complex_rows.append(
                    {
                        "arm": arm,
                        "seed": seed,
                        "time": float(time),
                        "mode_real": float(value.real),
                        "mode_imag": float(value.imag),
                    }
                )
    write_csv(output / "landau_individual_complex_histories.csv", complex_rows)

    np.savez_compressed(
        output / "landau_paired_bootstrap_draws.npz",
        cic_gamma=gamma_draws["cic"],
        c4_sigma1_gamma=gamma_draws["c4_sigma1"],
        lscv_job5008392_gamma=gamma_draws["lscv_job5008392"],
        c4_sigma1_to_cic=ratio_draws["c4_sigma1"],
        lscv_job5008392_to_cic=ratio_draws["lscv_job5008392"],
        phase_attribution_fraction=attribution_draws,
    )

    configure_matplotlib()
    import matplotlib.pyplot as plt

    colors = {
        "cic": "#14213d",
        "c4_sigma1": "#b2182b",
        "lscv_job5008392": "#6a3d9a",
    }
    labels = {
        "cic": "CIC",
        "c4_sigma1": r"Gaussian $\sigma=1\,\Delta x$",
        "lscv_job5008392": r"fixed CV-derived $\sigma=7.2443\,\Delta x$",
    }
    figure, axes = plt.subplots(1, 2, figsize=(7.0, 2.8))
    axis = axes[0]
    axis.axvspan(COMMON_START, COMMON_STOP, color="0.8", alpha=0.2, zorder=0)
    for arm in arms:
        axis.plot(
            times,
            arrays[arm]["coherent"],
            color=colors[arm],
            lw=0.65,
            alpha=0.35,
        )
        axis.plot(
            times,
            arrays[arm]["coherent_envelope"],
            color=colors[arm],
            lw=1.35,
            label=labels[arm],
        )
        fit = central[arm]
        mask = (times >= COMMON_START) & (times <= COMMON_STOP)
        start_index = int(np.flatnonzero(mask)[0])
        fit_line = arrays[arm]["coherent_envelope"][start_index] * np.exp(
            fit["gamma"] * (times[mask] - times[start_index])
        )
        axis.plot(times[mask], fit_line, color=colors[arm], ls="--", lw=0.9)
    axis.set_yscale("log")
    axis.set_xlim(0.0, 45.0)
    axis.set_xlabel(r"$t\omega_{pe}$")
    axis.set_ylabel(r"$|\langle \widehat E_x\rangle_s|$")
    axis.set_title(r"(a) Coherent histories and one-period envelopes")
    axis.legend(frameon=False, loc="upper right")

    axis = axes[1]
    axis.axvspan(COMMON_START, COMMON_STOP, color="0.8", alpha=0.2, zorder=0)
    for arm in arms:
        axis.plot(
            times,
            arrays[arm]["coherence_envelope"],
            color=colors[arm],
            lw=1.35,
            label=labels[arm],
        )
    axis.set_xlim(0.0, 45.0)
    axis.set_ylim(0.0, 1.05)
    axis.set_xlabel(r"$t\omega_{pe}$")
    axis.set_ylabel(r"$|\langle E_s\rangle|/\langle|E_s|\rangle$")
    axis.set_title("(b) Amplitude-weighted phase coherence")
    figure.tight_layout()
    for suffix in ("pdf", "svg"):
        figure.savefig(output / f"fig_landau_three_arm_common_window.{suffix}")
    figure.savefig(output / "fig_landau_three_arm_common_window.png", dpi=300)
    plt.close(figure)

    attr_lo, attr_hi = percentile_interval(attribution_draws)
    attr_central = (
        decomposition["lscv_job5008392"]["coherence_log_slope"]
        - decomposition["cic"]["coherence_log_slope"]
    ) / (
        decomposition["lscv_job5008392"]["coherent_gamma"]
        - decomposition["cic"]["coherent_gamma"]
    )
    config_by_arm = {arm["id"]: arm for arm in config["arms"]}
    result = {
        "analysis_role": "post-hoc paired common-window reanalysis",
        "window": [COMMON_START, COMMON_STOP],
        "observable": config["fit_rule"]["observable"],
        "operation_order": config["fit_rule"]["operation_order"],
        "n_paired_seeds": n_seeds,
        "seeds": data["seeds"],
        "bootstrap": {
            "draws": n_draws,
            "seed": int(config["fit_rule"]["bootstrap_seed"]),
            "unit": "paired seed index shared across all three arms",
        },
        "arms": config_by_arm,
        "configuration": {
            "ppc": config["benchmark"]["ppc"],
            "loading": config["benchmark"]["velocity_loading"],
            "spatial_loading": config["benchmark"]["spatial_particle_loading"],
            "normalization": "omega_pe=1; field Fourier coefficient from numpy rfft2 / (Nx*Ny)",
            "envelope": (
                "49-sample centered geometric moving mean of the positive amplitude; "
                "edge padded exactly as pic_kde_2d.validation.landau_censored_fit.geometric_envelope"
            ),
            "floor": config["fit_rule"]["noise_floor_estimator"],
        },
        "central_rows": summary_rows,
        "phase_decomposition_rows": phase_rows,
        "wide_filter_excess_decay_phase_attribution": {
            "definition": (
                "(coherence-slope_wide - coherence-slope_CIC) / "
                "(coherent-gamma_wide - coherent-gamma_CIC)"
            ),
            "central": float(attr_central),
            "paired_bootstrap_ci_lo": attr_lo,
            "paired_bootstrap_ci_hi": attr_hi,
            "hypothesized_0p732_supported": bool(attr_lo <= 0.732 <= attr_hi),
            "interpretation": (
                "This is an additive log-slope decomposition of the saved ensemble, not a "
                "causal intervention. It tests, but does not assume, the illustrative 73.2% attribution."
            ),
        },
        "config_path": str(config_path),
        "config_sha256": config_hash,
        "reducer_paths": [
            str(HERE / "analyze_weak_landau_confirmatory.py"),
            str(HERE / "analyze_weak_landau_confirmatory_v2.py"),
        ],
        "reducer_sha256": {
            "analyze_weak_landau_confirmatory.py": sha256(
                HERE / "analyze_weak_landau_confirmatory.py"
            ),
            "analyze_weak_landau_confirmatory_v2.py": sha256(
                HERE / "analyze_weak_landau_confirmatory_v2.py"
            ),
        },
        "input_root": str(LANDAU_LOW.resolve()),
        "input_file_sha256": data["file_hashes"],
        "all_pairing_hashes_verified": True,
    }
    (output / "landau_three_arm_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def finite_marker_analysis(output: Path) -> dict:
    response_path = FINITE_MARKER_ROOT / "finite_marker_linear_response.csv"
    loading_path = FINITE_MARKER_ROOT / "reconstructed_loading_moments.csv"
    summary_path = FINITE_MARKER_ROOT / "finite_marker_response_summary.csv"
    response = read_csv(response_path)
    loading = read_csv(loading_path)
    response_summary = read_csv(summary_path)

    exact_loading = {
        int(row["seed"]): row
        for row in loading
        if row["marker_group"] == "current_drift_seeds_200_209"
        and row["v_beam"] == "0.8"
    }
    if sorted(exact_loading) != list(range(200, 210)):
        raise AssertionError("missing exact current v_b=0.80 loading rows")
    exact_response = [
        row
        for row in response
        if row["marker_group"] == "current_drift_seeds_200_209"
        and row["v_beam"] == "0.8"
    ]
    if len(exact_response) != 30:
        raise AssertionError("expected ten seeds times three v_b=0.80 arms")

    joined: list[dict] = []
    for row in exact_response:
        moment = exact_loading[int(row["seed"])]
        joined.append(
            {
                "v_beam": float(row["v_beam"]),
                "arm": row["arm"],
                "seed": int(row["seed"]),
                "source_status": row["source_status"],
                "particles": int(moment["particles"]),
                "filter_force_transfer_W2": float(row["filter_force_transfer_W2"]),
                "continuum_velocity_grid_time_discrete_gamma": float(
                    row["continuum_velocity_grid_time_discrete_gamma"]
                ),
                "finite_marker_diagonal_gamma": float(row["finite_marker_diagonal_gamma"]),
                "finite_marker_diagonal_frequency": float(
                    row["finite_marker_diagonal_frequency"]
                ),
                "finite_marker_root_status": row["finite_marker_root_status"],
                "observed_individual_gamma": float(row["observed_individual_gamma"]),
                "observed_minus_finite_marker_gamma": float(
                    row["observed_minus_finite_marker_gamma"]
                ),
                "positive_beam_fraction": float(moment["positive_beam_fraction"]),
                "thermal_vx_mean": float(moment["thermal_vx_mean"]),
                "thermal_vx_std": float(moment["thermal_vx_std"]),
                "vy_mean": float(moment["vy_mean"]),
                "vy_std": float(moment["vy_std"]),
                "beam_thermal_vx_covariance": float(
                    moment["beam_thermal_vx_covariance"]
                ),
                "per_cell_beam_imbalance_rms": float(
                    moment["per_cell_beam_imbalance_rms"]
                ),
                "per_cell_beam_imbalance_max_abs": float(
                    moment["per_cell_beam_imbalance_max_abs"]
                ),
                "per_cell_thermal_vx_mean_rms": float(
                    moment["per_cell_thermal_vx_mean_rms"]
                ),
                "per_cell_vy_mean_rms": float(moment["per_cell_vy_mean_rms"]),
                "predictor_limit": row["approximation_limit"],
            }
        )
    write_csv(output / "e2_vb080_finite_marker_loading_joined.csv", joined)

    historical = [
        row
        for row in response
        if row["marker_group"] == "historical_seeds_0_9_reconstructed"
        and row["v_beam"] == "0.8"
    ]
    write_csv(output / "e2_vb080_historical_nominal_finite_marker.csv", historical)

    exact_summary = [
        row
        for row in response_summary
        if row["marker_group"] == "current_drift_seeds_200_209"
        and row["v_beam"] == "0.8"
    ]
    write_csv(output / "e2_vb080_finite_marker_summary.csv", exact_summary)

    inventory_rows = [
        {
            "campaign": "e2_beam_drift_scan_5201231",
            "histories": 120,
            "ppc": 512,
            "scope": "projected E2 drift scan, four completed drift values",
            "loading_covariance_control_across_ppc": "NO_SINGLE_PPC_ONLY",
        },
        {
            "campaign": "e2_beam_drift_scan_extension_5202227",
            "histories": 30,
            "ppc": 512,
            "scope": "projected E2 v_b=0.818 extension",
            "loading_covariance_control_across_ppc": "NO_SINGLE_PPC_ONLY",
        },
        {
            "campaign": "e2_compensated_corner_5195640",
            "histories": 50,
            "ppc": 512,
            "scope": "projected E2 compensated-filter corner",
            "loading_covariance_control_across_ppc": "NO_SINGLE_PPC_ONLY",
        },
        {
            "campaign": "e2_new_mode_m10n1_5197185",
            "histories": 20,
            "ppc": 512,
            "scope": "projected new-mode comparison",
            "loading_covariance_control_across_ppc": "NO_SINGLE_PPC_ONLY",
        },
        {
            "campaign": "e2_anisotropic_dense_5197174",
            "histories": 0,
            "ppc": "32 candidate / 1024 independent static reference",
            "scope": "t=0 density and pre-return-grid-field error only; no rate history",
            "loading_covariance_control_across_ppc": "NOT_A_DYNAMICS_CONVERGENCE_TEST",
        },
        {
            "campaign": "E2_m8_sigma_sweep_4863559",
            "histories": "historical aggregate only",
            "ppc": 2048,
            "scope": "different mode/provenance; no paired v_b=0.80 (12,1) loading control",
            "loading_covariance_control_across_ppc": "NOT_COMPARABLE",
        },
    ]
    write_csv(output / "e2_existing_ppc_loading_inventory.csv", inventory_rows)

    cic_summary = next(row for row in exact_summary if row["arm"] == "cic")
    study = {
        "status": "SPECIFICATION_ONLY_NOT_SUBMITTED",
        "question": (
            "Does the v_b=0.80 individual-rate distribution converge with marker count "
            "when beam-sign/thermal-vx covariance is held at zero rather than changing randomly?"
        ),
        "scope": "CIC only; target mode (12,1); projected linearized delta-f; existing grid and dt",
        "ppc_levels": [128, 512, 2048],
        "fixed_seeds_per_level": 10,
        "histories": 30,
        "loading_control": (
            "Use nested regular position lattices and antithetic per-cell quartets with exactly "
            "balanced beam signs and paired +/- thermal-vx samples, so global and per-cell "
            "mean(sign*thermal_vx) are zero to roundoff at every PPC. Keep the same base uniforms "
            "and perturbation phase across PPC levels."
        ),
        "record_before_run": [
            "particle-array hashes and exact initialization arrays",
            "beam-sign/thermal-vx covariance globally and per cell",
            "initialization-only finite-marker eigenvalue and convergence residual",
            "individual complex mode history through t=71.16",
            "fixed [24.32,69.16] fitted rate and signal/linearity diagnostics",
        ],
        "analysis": (
            "paired seed bootstrap of means and variances across PPC; regress finite-marker "
            "prediction and fitted-rate spread against 1/sqrt(Np); do not add seeds after inspection"
        ),
        "empirical_cost_basis": (
            "Slurm job 5195640: each ten-seed PPC-512 t=60 arm used 27.0--30.6 min "
            "on one CPU node (five completed array tasks)."
        ),
        "estimated_cpu_node_hours": 2.90,
        "recommended_hard_cap_cpu_node_hours": 3.65,
        "cost_scaling": (
            "linear in PPC and duration: 0.466 node-h * (71.16/60) * "
            "[(128+512+2048)/512] = 2.90 node-h; cap includes about 26% margin"
        ),
        "why_minimal": (
            "One arm isolates loading/PPC convergence; three PPC levels distinguish a trend from "
            "a two-point difference; ten seeds matches the existing uncertainty estimate."
        ),
    }
    result = {
        "status": "NO_EXISTING_CONTROLLED_MULTI_PPC_RATE_COMPARISON",
        "current_exact_vb080": {
            "marker_group": "current_drift_seeds_200_209",
            "source_status": "EXACT_CURRENT_AUDITED_INITIALIZER",
            "ppc": 512,
            "particles": 1_048_576,
            "seeds": list(range(200, 210)),
            "arms": sorted({row["arm"] for row in joined}),
            "cic_observed_gamma_correlation_beam_thermal_vx_covariance": float(
                cic_summary["observed_gamma_correlation_beam_thermal_vx_covariance"]
            ),
            "predictor": (
                "initialization-only principal represented-mode, diagonal +k finite-marker root"
            ),
            "predictor_limit": (
                "spatial aliases and finite-sample +/-k mixing omitted; not exact response of full update"
            ),
        },
        "historical_status": (
            "Seeds 0--9 can only be reconstructed with the nominal current audited initializer; "
            "historical particle arrays and source revision are unknown."
        ),
        "meluxina_readonly_inventory": {
            "root": "/mnt/tier2/users/u103576/paper1_validation",
            "two_stream_result_counts": {
                "e2_beam_drift_scan_5201231": 120,
                "e2_beam_drift_scan_extension_5202227": 30,
                "e2_compensated_corner_5195640": 50,
                "e2_new_mode_m10n1_5197185": 20,
            },
            "all_dynamic_ppc_values": [512],
            "checked_utc_date": "2026-09-21",
        },
        "smallest_useful_convergence_study": study,
        "source_hashes": {
            str(response_path): sha256(response_path),
            str(loading_path): sha256(loading_path),
            str(summary_path): sha256(summary_path),
            str(HERE / "diagnose_E2_seed_rate_variation.py"): sha256(
                HERE / "diagnose_E2_seed_rate_variation.py"
            ),
        },
    }
    (output / "e2_finite_marker_loading_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "e2_smallest_useful_convergence_study.json").write_text(
        json.dumps(study, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing result: {output}")
    output.mkdir(parents=True)

    bandwidth = bandwidth_analysis(output)
    landau = landau_analysis(output)
    finite_marker = finite_marker_analysis(output)
    manifest = {
        "analysis": "missing validation data retrieval",
        "date": "2026-09-21",
        "no_simulations_launched": True,
        "manuscripts_edited": False,
        "script": str(Path(__file__).resolve()),
        "script_sha256": sha256(Path(__file__).resolve()),
        "bandwidth_status": bandwidth["status"],
        "landau_status": {
            row["arm"]: row["overall_fixed_window_qualification"]
            for row in landau["central_rows"]
        },
        "finite_marker_loading_status": finite_marker["status"],
        "outputs": sorted(path.name for path in output.iterdir()),
    }
    (output / "analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
