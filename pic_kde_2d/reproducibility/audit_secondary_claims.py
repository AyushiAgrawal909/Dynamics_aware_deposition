#!/usr/bin/env python3
"""Independently regenerate the numerical evidence for Paper 1 secondary claims."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


DELTA_RESULT_KEYS = {
    "alpha",
    "background",
    "benchmark",
    "dt",
    "marker_dynamics",
    "max_delta_f_ratio",
    "mode",
    "ppc",
    "seed",
    "sigma_dx",
    "t_end",
    "target_mode_projection",
    "valid_through_run",
    "validity_threshold",
}
COST_KEYS = {
    "Nx",
    "Ny",
    "deposit_ms_per_step",
    "filter_ms_per_step",
    "gather_ms_per_step",
    "method",
    "n_particles",
    "persistent_array_bytes",
    "persistent_bytes_per_particle",
    "ppc",
    "process_peak_rss_bytes",
    "protocol",
    "push_ms_per_step",
    "solve_ms_per_step",
    "steps",
    "total_ms_per_step",
    "total_ms_step_std",
    "unattributed_ms_per_step",
    "warmup_steps",
}


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def bootstrap_mean(values: np.ndarray, *, seed: int, nboot: int) -> dict:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or values.size < 2:
        raise ValueError("bootstrap requires at least two independent seeds")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, values.size, size=(nboot, values.size))
    draws = np.mean(values[indices], axis=1)
    return {
        "value": float(np.mean(values)),
        "ci_lo": float(np.percentile(draws, 2.5)),
        "ci_hi": float(np.percentile(draws, 97.5)),
        "n_resampling_units": int(values.size),
    }


def bootstrap_ratio_of_means(
    numerator: np.ndarray,
    denominator: np.ndarray,
    *,
    seed: int,
    nboot: int,
) -> dict:
    numerator = np.asarray(numerator, dtype=np.float64)
    denominator = np.asarray(denominator, dtype=np.float64)
    if numerator.shape != denominator.shape or numerator.ndim != 1:
        raise ValueError("paired vectors must be one-dimensional and equally sized")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, numerator.size, size=(nboot, numerator.size))
    draws = np.mean(numerator[indices], axis=1) / np.mean(
        denominator[indices], axis=1
    )
    return {
        "value": float(np.mean(numerator) / np.mean(denominator)),
        "ci_lo": float(np.percentile(draws, 2.5)),
        "ci_hi": float(np.percentile(draws, 97.5)),
        "n_paired_units": int(numerator.size),
    }


def load_histories(root: Path) -> tuple[np.ndarray, np.ndarray, list[int]]:
    histories: list[np.ndarray] = []
    seeds: list[int] = []
    times: np.ndarray | None = None
    for path in sorted(root.glob("seed*/mode_history.npz")):
        with np.load(path) as data:
            assert set(data.keys()) >= {"times", "mode"}
            this_times = np.asarray(data["times"], dtype=np.float64)
            histories.append(np.asarray(data["mode"], dtype=np.complex128))
            if "seed" in data:
                seeds.append(int(np.asarray(data["seed"]).reshape(-1)[0]))
            else:
                seeds.append(int(path.parent.name.removeprefix("seed")))
        if times is None:
            times = this_times
        else:
            np.testing.assert_array_equal(times, this_times)
    if times is None or not histories:
        raise RuntimeError(f"no mode histories in {root}")
    order = np.argsort(seeds)
    return times, np.stack(histories)[order], list(np.asarray(seeds)[order])


def window_energy(times: np.ndarray, histories: np.ndarray, lo: float, hi: float) -> np.ndarray:
    mask = (times >= lo) & (times <= hi)
    if np.count_nonzero(mask) < 2:
        raise RuntimeError(f"empty or singleton energy window [{lo}, {hi}]")
    return np.mean(np.abs(histories[:, mask]) ** 2, axis=1)


def persistent_crossing(values: np.ndarray, threshold: float, persistence: int) -> int | None:
    above = values > threshold
    for index in range(len(values) - persistence + 1):
        if np.all(above[index : index + persistence]):
            return index
    return None


def audit_delta_f(recovered: Path, *, nboot: int) -> dict:
    root = recovered / "df_breakdown_4873222"
    rows = []
    maxima_by_alpha: dict[float, np.ndarray] = {}
    for alpha_index, alpha_token in enumerate(("0p001", "0p01", "0p1")):
        alpha_root = root / f"alpha_{alpha_token}"
        maxima = []
        crossings = []
        initial = []
        for path in sorted(alpha_root.glob("seed*/mode_history.npz")):
            result = load_json(path.parent / "result.json")
            assert set(result.keys()) == DELTA_RESULT_KEYS
            assert result["marker_dynamics"] == "free_streaming_frozen_equilibrium"
            assert result["target_mode_projection"] is True
            with np.load(path) as data:
                assert set(data.keys()) == {"times", "mode", "delta_f_ratio", "seed"}
                times = np.asarray(data["times"], dtype=np.float64)
                ratios = np.asarray(data["delta_f_ratio"], dtype=np.float64)
            maxima.append(float(np.max(ratios)))
            initial.append(float(ratios[0]))
            found = np.flatnonzero(ratios >= 0.5)
            crossings.append(float(times[found[0]]) if found.size else np.nan)
        maxima_array = np.asarray(maxima)
        maxima_by_alpha[float(result["alpha"])] = maxima_array
        maximum_summary = bootstrap_mean(
            maxima_array, seed=9_102_026 + alpha_index, nboot=nboot
        )
        finite_crossings = np.asarray(crossings)[np.isfinite(crossings)]
        rows.append(
            {
                "alpha": float(result["alpha"]),
                "maximum_ratio": maximum_summary,
                "mean_initial_ratio": float(np.mean(initial)),
                "fraction_crossing_0p5": float(np.mean(maxima_array >= 0.5)),
                "median_first_crossing_time": (
                    float(np.median(finite_crossings)) if finite_crossings.size else None
                ),
            }
        )

    base = maxima_by_alpha[0.001]
    np.testing.assert_allclose(maxima_by_alpha[0.01], 10.0 * base, rtol=2.0e-14, atol=2.0e-14)
    np.testing.assert_allclose(maxima_by_alpha[0.1], 100.0 * base, rtol=2.0e-14, atol=2.0e-13)
    return {
        "production_operator": "linearized delta-f on frozen/free-streaming equilibrium characteristics",
        "deposition_projection": "real conjugate pair of the requested target mode only",
        "validity_metric": "max_p |w_delta,p| / mean_p |w_full,p|",
        "validity_metric_scope": (
            "heuristic marker-maximum guard; not an L2 error norm and dependent on "
            "marker sampling of velocity tails"
        ),
        "validity_threshold": 0.5,
        "rows": rows,
        "alpha_linearity_check": "PASS: every per-seed maximum scales exactly with alpha",
    }


def audit_nonlinear(recovered: Path, *, nboot: int) -> dict:
    full_root = recovered / "df_fullf_overlay_4873285" / "fullf"
    delta_root = recovered / "df_breakdown_4873222" / "alpha_0p1"
    times, full, full_seeds = load_histories(full_root)
    times_delta, delta, delta_seeds = load_histories(delta_root)
    np.testing.assert_array_equal(times, times_delta)
    assert full_seeds == delta_seeds == list(range(5))
    for path in sorted(full_root.glob("seed*/run_parameters.json")):
        parameters = load_json(path)
        assert parameters["alpha"] == 0.1
        assert parameters["ppc"] == 128
        assert parameters["Nx"] == 64 and parameters["Ny"] == 16
        assert parameters["dt"] == 0.05 and parameters["n_steps"] == 1200

    early_full = window_energy(times, full, 12.0, 18.0)
    early_delta = window_energy(times, delta, 12.0, 18.0)
    rebound_full = window_energy(times, full, 30.0, 40.0)
    rebound_delta = window_energy(times, delta, 30.0, 40.0)
    late_full = window_energy(times, full, 40.0, 60.0)
    late_delta = window_energy(times, delta, 40.0, 60.0)

    full_amplitude = np.mean(np.abs(full), axis=0)
    delta_amplitude = np.mean(np.abs(delta), axis=0)
    relative_difference = np.abs(full_amplitude - delta_amplitude) / np.maximum(
        delta_amplitude, 1.0e-300
    )
    eligible = np.flatnonzero(times >= 5.0)
    crossing = persistent_crossing(relative_difference[eligible], 0.25, 5)
    divergence_time = float(times[eligible[crossing]]) if crossing is not None else None

    return {
        "observable": "paired seed-wise target-mode electric-field energy proxy |E_hat_x(1,0)|^2",
        "full_f_rebound_energy_t30_40_over_t12_18": bootstrap_ratio_of_means(
            rebound_full, early_full, seed=9_102_101, nboot=nboot
        ),
        "linearized_delta_f_energy_t30_40_over_t12_18": bootstrap_ratio_of_means(
            rebound_delta, early_delta, seed=9_102_102, nboot=nboot
        ),
        "late_energy_full_f_over_delta_f_t40_60": bootstrap_ratio_of_means(
            late_full, late_delta, seed=9_102_103, nboot=nboot
        ),
        "late_amplitude_ratio_of_ensemble_means_t40_60": float(
            np.mean(full_amplitude[times >= 40.0])
            / np.mean(delta_amplitude[times >= 40.0])
        ),
        "first_persistent_25pct_amplitude_divergence_time": divergence_time,
        "scope": (
            "the recovered histories contain only the target Fourier mode; no particle "
            "phase space or harmonic spectra are available, so trapping itself is not directly measured"
        ),
    }


def audit_cost(recovered: Path, *, nboot: int) -> dict:
    root = recovered / "component_cost_5049224"
    records: list[dict] = []
    for path in sorted(root.glob("rep*/cost_*.json")):
        row = load_json(path)
        assert set(row.keys()) == COST_KEYS
        row["repeat"] = path.parent.name
        records.append(row)
    assert len(records) == 15
    repeats = sorted({row["repeat"] for row in records})
    assert repeats == [f"rep{index}" for index in range(5)]
    by_key = {(row["repeat"], row["method"]): row for row in records}
    c4_ratios = np.asarray(
        [
            by_key[(repeat, "c4")]["total_ms_per_step"]
            / by_key[(repeat, "cic")]["total_ms_per_step"]
            for repeat in repeats
        ]
    )
    delta_ratios = np.asarray(
        [
            by_key[(repeat, "deltaf")]["total_ms_per_step"]
            / by_key[(repeat, "cic")]["total_ms_per_step"]
            for repeat in repeats
        ]
    )
    c4_rows = [by_key[(repeat, "c4")] for repeat in repeats]
    cic_rows = [by_key[(repeat, "cic")] for repeat in repeats]
    component_residual = np.asarray(
        [
            row["total_ms_per_step"]
            - sum(row[f"{name}_ms_per_step"] for name in ("deposit", "solve", "filter", "gather", "push"))
            for row in records
        ]
    )
    np.testing.assert_allclose(
        component_residual,
        np.asarray([row["unattributed_ms_per_step"] for row in records]),
        rtol=0.0,
        atol=2.0e-12,
    )
    c4_total_mean = float(np.mean([row["total_ms_per_step"] for row in c4_rows]))
    c4_filter_mean = float(np.mean([row["filter_ms_per_step"] for row in c4_rows]))
    return {
        "protocol": {
            "grid": [64, 16],
            "ppc": 1024,
            "n_particles": 1_048_576,
            "measured_steps_per_repeat": 200,
            "warmup_steps": 10,
            "same_node_order_rotated_repeats": 5,
        },
        "c4_total_ratio_vs_cic": bootstrap_mean(c4_ratios, seed=99_127, nboot=nboot),
        "delta_f_total_ratio_vs_cic": bootstrap_mean(delta_ratios, seed=99_128, nboot=nboot),
        "c4_filter_ms_per_step": c4_filter_mean,
        "c4_filter_fraction_of_total": c4_filter_mean / c4_total_mean,
        "c4_persistent_filter_bytes": int(
            c4_rows[0]["persistent_array_bytes"] - cic_rows[0]["persistent_array_bytes"]
        ),
        "scope": {
            "filter_application_included": True,
            "bandwidth_selection_included": False,
            "filter_construction_included": False,
            "particle_and_grid_initialization_included": False,
            "process_peak_rss_is_high_water_mark_not_incremental_memory": True,
            "extra_memory_column_is_explicit_persistent_numpy_arrays_only": True,
        },
        "interpretation": (
            "fixed-width steady-state time-stepping benchmark only; it is not an "
            "end-to-end selector-efficiency measurement"
        ),
    }


def audit_h1(h1_root: Path, learned_root: Path, *, nboot: int) -> dict:
    rows = []
    for path in sorted(h1_root.glob("D*/seed*/result.json")):
        result = load_json(path)
        assert result["status"] == "complete"
        rows.extend(result["rows"])
    assert len(rows) == 60

    def paired_ratio(dynamic_range: float, metric: str, seed: int) -> dict:
        global_rows = sorted(
            (row for row in rows if row["dynamic_range"] == dynamic_range and row["method"] == "global_c4"),
            key=lambda row: row["seed"],
        )
        local_rows = sorted(
            (row for row in rows if row["dynamic_range"] == dynamic_range and row["method"] == "abramson_c4_capped"),
            key=lambda row: row["seed"],
        )
        assert [row["seed"] for row in global_rows] == [row["seed"] for row in local_rows] == list(range(10))
        return bootstrap_ratio_of_means(
            np.asarray([row[metric] for row in local_rows]),
            np.asarray([row[metric] for row in global_rows]),
            seed=seed,
            nboot=nboot,
        )

    model_suffixes = {".pt", ".pth", ".onnx", ".ckpt", ".pkl", ".joblib", ".h5"}
    learned_artifacts = sorted(
        str(path.relative_to(learned_root))
        for path in learned_root.rglob("*")
        if path.is_file() and path.suffix.lower() in model_suffixes
    )
    decision = load_json(h1_root / "analysis" / "H1_result.json")
    assert decision["decision"] == "STOP_LEARNING_ANALYTIC_H1_NEGATIVE"
    assert not decision["locality_gate_passed"]
    assert not learned_artifacts
    return {
        "analytic_controller": "frozen initial-density, C4-capped Abramson widths in 24 bins",
        "D100_density_mise_local_over_global": paired_ratio(
            100.0, "density_mise", 61_112
        ),
        "D100_field_mse_local_over_global": paired_ratio(
            100.0, "field_mse", 61_109
        ),
        "D100_protected_error_local_over_global": paired_ratio(
            100.0, "protected_relative_l2", 61_121
        ),
        "decision": decision["decision"],
        "learned_model_artifact_count": len(learned_artifacts),
        "learned_controller_was_trained": False,
        "scope": (
            "the negative H1 result applies to this frozen analytic density-following "
            "controller and benchmark, not to learned controllers generally"
        ),
    }


def audit_g0(recovered: Path) -> dict:
    summary = load_json(
        recovered / "G0_locality_headroom_ceiling_baseline" / "g0_summary.json"
    )
    precheck = load_json(recovered / "G0_conservation_precheck" / "g0_conservation_precheck.json")
    candidate = next(
        row
        for row in summary["summaries"]
        if row.get("geometry") == "narrowband_H2_candidate"
    )
    assert candidate["best_admissible_sigma_free_cells"] == 384.0
    assert precheck["requested_profile"]["sigma_free_cells"] == 384.0
    return {
        "requested_free_width_cells": 384.0,
        "predicted_packet_field_noise_gain": float(candidate["best_admissible_field_noise_gain"]),
        "implemented_width_cap_cells": float(precheck["production_variational_chain"]["kde_max_cells"]),
        "predicted_gain_at_cap": float(
            precheck["production_variational_chain"]["field_gain_at_largest_supported_width"]
        ),
        "status": precheck["decision"],
        "scope": (
            "analytic covariance/headroom calculation plus implementation precheck; "
            "the requested profile was not run in dynamics"
        ),
    }


def write_summary_csv(path: Path, result: dict) -> None:
    rows = []
    for section, values in result.items():
        if not isinstance(values, dict):
            continue
        for name, value in values.items():
            if isinstance(value, dict) and "value" in value:
                rows.append(
                    {
                        "section": section,
                        "quantity": name,
                        "value": value["value"],
                        "ci_lo": value.get("ci_lo", ""),
                        "ci_hi": value.get("ci_hi", ""),
                    }
                )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("section", "quantity", "value", "ci_lo", "ci_hi"))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recovered", type=Path, required=True)
    parser.add_argument("--h1", type=Path, required=True)
    parser.add_argument("--learned-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--nboot", type=int, default=100_000)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    result = {
        "uncertainty": {
            "unit": "paired independent seed where a ratio is reported",
            "estimator": "percentile bootstrap of paired ratio of seed means",
            "nboot": args.nboot,
        },
        "delta_f": audit_delta_f(args.recovered, nboot=args.nboot),
        "finite_amplitude": audit_nonlinear(args.recovered, nboot=args.nboot),
        "cost": audit_cost(args.recovered, nboot=args.nboot),
        "adaptive_and_learned": audit_h1(args.h1, args.learned_root, nboot=args.nboot),
        "g0_locality": audit_g0(args.recovered),
    }
    with (args.output / "secondary_claims_audit.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    write_summary_csv(args.output / "secondary_claims_audit.csv", result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
