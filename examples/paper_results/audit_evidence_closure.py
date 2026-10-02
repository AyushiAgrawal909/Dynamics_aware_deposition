#!/usr/bin/env python3
"""Close loading, E2 signal, and prediction-calibration evidence questions.

The audit reads only recovered histories and existing reductions.  It runs no
particle simulation and refuses to overwrite an earlier audit directory.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np


HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[1]
PROJECT = WORKSPACE / "pic_kde_2d"
for candidate in (PROJECT, PROJECT / "scripts"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from analyze_linearized_delta_f_m12_sigma import estimate_growing_floor  # noqa: E402
from pic_kde_2d.conventions import marker_grid_shape  # noqa: E402
from pic_kde_2d.core.particles2d import quiet_start_maxwellian_2d  # noqa: E402


RECOVERED = PROJECT / "results_raw_rate_audit" / "recovered_meluxina_20260909"
NOISE_AUDIT = HERE / "results/noise_metric_audit/noise_metric_audit.json"
JOINT_REDUCTION = HERE / "results/rate_uncertainty_audit_v8/e2_gaussian_joint_corner_recovered/joint_corner_sweep.csv"
JOINT_RAW = RECOVERED / "e2_joint_corner_4913408/rate"
COMPENSATED_REDUCTION = RECOVERED / "e2_compensated_corner_5195640/analysis/E2_compensated_corner.csv"
PREDICTIONS = HERE / "results/dispersion_audit/filter_rate_predictions.csv"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def one(rows: list[dict[str, str]], **criteria: str) -> dict[str, str]:
    matches = [row for row in rows if all(row.get(key) == value for key, value in criteria.items())]
    if len(matches) != 1:
        raise AssertionError(f"expected one row for {criteria}, found {len(matches)}")
    return matches[0]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def reconstruct_per_cell_support() -> dict[str, Any]:
    nx_marker, ny_marker = marker_grid_shape(64, 16, 1024)
    _, _, vx, _ = quiet_start_maxwellian_2d(
        nx_marker,
        ny_marker,
        2.0 * math.pi / 0.4,
        8.0,
        1.0,
        np.random.default_rng(0),
        grid_shape=(64, 16),
    )
    phase_velocity = 3.2126424241344536
    result = {
        "field_grid": [64, 16],
        "ppc": 1024,
        "marker_grid": [nx_marker, ny_marker],
        "per_cell_velocity_tensor": [nx_marker // 64, ny_marker // 16],
        "distinct_longitudinal_nodes": int(len(np.unique(vx))),
        "longitudinal_velocity_min": float(np.min(vx)),
        "longitudinal_velocity_max": float(np.max(vx)),
        "resonant_phase_velocity": phase_velocity,
        "nodes_at_or_above_resonance": int(np.count_nonzero(vx >= phase_velocity)),
        "resonance_covered": bool(np.max(vx) >= phase_velocity),
    }
    if result["distinct_longitudinal_nodes"] != 32 or result["resonance_covered"]:
        raise AssertionError("the reconstructed per-cell support is not the diagnosed 32-node case")
    return result


def historical_loading_rows() -> list[dict[str, Any]]:
    noise = json.loads(NOISE_AUDIT.read_text(encoding="utf-8"))
    table4 = one(noise["weak_landau_sigma1_comparison"], job="4867515")
    aligned = one(noise["weak_landau_sigma1_comparison"], job="4868286")
    if not math.isclose(float(table4["noise_power_reduction"]), 2.559393052975873):
        raise AssertionError("Table-4 C4 reduction changed")
    if not math.isclose(float(aligned["noise_power_reduction"]), 5.643223264073615):
        raise AssertionError("aligned Gaussian reduction changed")

    explicit_global_roots = {
        "weak raw-binomial rate, job 5048412": PROJECT / "results_raw_rate_audit/landau_binomial_weak_5048412",
        "weak compensated rate families": PROJECT / "results_raw_rate_audit/compensated_binomial_20260811",
        "weak higher-order compensated rate families": PROJECT / "results_raw_rate_audit/higher_order_compensation_20260811",
        "strong Gaussian rates, job 5048278": PROJECT / "results_raw_rate_audit/strong_landau_fidelity_5048278",
        "strong raw-binomial rates, job 5048414": RECOVERED / "landau_binomial_strong_5048414",
    }
    rows: list[dict[str, Any]] = [
        {
            "result_group": "Table 4 Gaussian C4 and Silverman density-error powers",
            "job_or_campaign": "4867515",
            "quantity": "density squared-error power; not a rate history",
            "loading_classification": "per-cell 32x32 velocity tensor",
            "evidence_strength": "contemporaneous audit; raw JSON predates the layout field",
            "same_as_rejected_confirmation": "YES",
            "consequence": (
                "retain as a measurement for that constructed ensemble only; do not combine "
                "with global-loading rates as a same-campaign physical validation"
            ),
        },
        {
            "result_group": "aligned Gaussian density-error powers / Fig. 7a overlay",
            "job_or_campaign": "4868286",
            "quantity": "density squared-error power; not a rate history",
            "loading_classification": "global inverse-normal shuffle",
            "evidence_strength": "quiet_start_layout=global serialized in every seed result",
            "same_as_rejected_confirmation": "NO",
            "consequence": "no loading-support reassessment",
        },
        {
            "result_group": "weak Gaussian CIC/C4 rate sweep and derived Silverman/MISE baselines",
            "job_or_campaign": "4864819 plus baseline-linked 4866573/4872322/5007754",
            "quantity": "complex Fourier-mode rate histories",
            "loading_classification": "global branch of matching rate entry point",
            "evidence_strength": (
                "matching run_landau_sigma_refinement.py omits grid_shape; oldest metadata "
                "predate quiet_start_layout and the exact historical revision remains unknown"
            ),
            "same_as_rejected_confirmation": "NO EVIDENCE OF PER-CELL; EXACT REVISION UNKNOWN",
            "consequence": (
                "do not transfer the rejected v1 failure automatically; retain the existing "
                "rate-specific duration and signal assessments"
            ),
        },
    ]
    for label, root in explicit_global_roots.items():
        metadata_paths = list(root.rglob("run_parameters.json"))
        if not metadata_paths:
            raise AssertionError(f"no metadata below {root}")
        layouts = {
            json.loads(path.read_text(encoding="utf-8")).get("quiet_start_layout")
            for path in metadata_paths
        }
        if layouts != {"global"}:
            raise AssertionError(f"non-global or missing layout below {root}: {layouts}")
        rows.append(
            {
                "result_group": label,
                "job_or_campaign": root.name,
                "quantity": "Landau rate histories",
                "loading_classification": "global inverse-normal shuffle",
                "evidence_strength": f"quiet_start_layout=global in {len(metadata_paths)} run records",
                "same_as_rejected_confirmation": "NO",
                "consequence": "no loading-support reassessment",
            }
        )
    return rows


def load_joint_amplitudes(label: str) -> tuple[np.ndarray, np.ndarray, list[Path]]:
    histories = []
    times = None
    paths = []
    for seed in range(10):
        path = JOINT_RAW / label / f"seed{seed}/mode_history.npz"
        with np.load(path, allow_pickle=False) as archive:
            current_times = np.asarray(archive["times"], dtype=np.float64)
            mode = np.asarray(archive["mode"], dtype=np.complex128)
        if times is None:
            times = current_times
        else:
            np.testing.assert_array_equal(times, current_times)
        histories.append(np.abs(mode))
        paths.append(path)
    assert times is not None
    return times, np.stack(histories), paths


def e2_signal_evidence() -> dict[str, Any]:
    rows = read_csv(JOINT_REDUCTION)
    baseline = one(rows, label="base")
    candidate = one(rows, label="y02428")
    times, amplitudes, raw_paths = load_joint_amplitudes("y02428")
    floor = estimate_growing_floor(times, amplitudes, 24.32)
    checks = {
        "fit_start_snr_at_least_5": floor["fit_start_snr"] >= 5.0,
        "fit_r_squared_at_least_0p98": float(candidate["fit_r2"]) >= 0.98,
        "candidate_efoldings_at_least_2": float(candidate["fit_efoldings"]) >= 2.0,
        "all_5000_paired_resamples_valid": int(candidate["bootstrap_valid"]) == 5000,
        "delta_f_guard_below_0p5": float(candidate["max_delta_f_ratio_fit_window"]) < 0.5,
        "shared_cic_baseline_efoldings_at_least_2": float(baseline["fit_efoldings"]) >= 2.0,
    }
    if not all(checks.values()):
        raise AssertionError(f"E2 C4 signal evidence does not pass: {checks}")
    return {
        "condition": "E2 Gaussian C4 binding corner (12,1), sigma_x=0, sigma_y=0.2428 dy",
        "fit_window": [24.32, 59.68],
        "floor": floor,
        "fit_r_squared": float(candidate["fit_r2"]),
        "candidate_fitted_efoldings": float(candidate["fit_efoldings"]),
        "shared_cic_baseline_fitted_efoldings": float(baseline["fit_efoldings"]),
        "paired_bootstrap_valid": int(candidate["bootstrap_valid"]),
        "max_delta_f_ratio_fit_window": float(candidate["max_delta_f_ratio_fit_window"]),
        "checks": checks,
        "signal_resolvability": "PASS",
        "historical_reducer_gap": (
            "the original joint reducer stored no floor; this audit applies the existing "
            "quietest two-time-unit rolling-median pre-emergence estimator to the recovered histories"
        ),
        "raw_history_count": len(raw_paths),
        "raw_history_sha256": {str(path.relative_to(WORKSPACE)): sha256(path) for path in raw_paths},
        "reduction_sha256": sha256(JOINT_REDUCTION),
    }


def compensated_prediction_rows() -> list[dict[str, Any]]:
    measured = read_csv(COMPENSATED_REDUCTION)
    predicted = read_csv(PREDICTIONS)
    output = []
    for label, order, passes in (
        ("p1n1", "1", "1"),
        ("p2n2", "2", "2"),
        ("p2n4", "2", "4"),
        ("p3n4", "3", "4"),
    ):
        actual = one(measured, label=label)
        theory = one(
            predicted,
            evaluated_mode="(12,1)",
            compensation_order=order,
            passes=passes,
            theory_level="grid_time_discrete",
        )
        y = float(actual["gamma_ratio_to_cic"])
        lo = float(actual["ratio_ci_lo"])
        hi = float(actual["ratio_ci_hi"])
        model = float(theory["predicted_rate_ratio"])
        output.append(
            {
                "label": label,
                "compensation_order": int(order),
                "passes": int(passes),
                "predicted_rate_ratio": model,
                "measured_rate_ratio": y,
                "measured_ci_lo": lo,
                "measured_ci_hi": hi,
                "prediction_minus_measurement": model - y,
                "absolute_discrepancy_percentage_points": 100.0 * abs(model - y),
                "relative_discrepancy_percent": 100.0 * abs(model / y - 1.0),
                "prediction_inside_measured_95pct_interval": lo <= model <= hi,
                "measured_rate_preservation_ci_pass": lo >= 0.90 and hi <= 1.10,
            }
        )
    if any(row["prediction_inside_measured_95pct_interval"] for row in output):
        raise AssertionError("expected all four disputed corner predictions outside measured intervals")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")

    loading = historical_loading_rows()
    support = reconstruct_per_cell_support()
    signal = e2_signal_evidence()
    prediction = compensated_prediction_rows()
    result = {
        "status": "PASS",
        "new_simulations_run": False,
        "historical_loading_conclusion": (
            "Yes: job 4867515, which supplied the Table-4 C4 and Silverman density-error "
            "powers, used the same repeated per-cell 32x32 velocity construction.  No "
            "recovered Landau rate history is identified as per-cell.  The oldest rate "
            "metadata predate the layout field, so their exact historical revision remains unknown."
        ),
        "historical_loading_claim_narrowing": (
            "The 4867515 powers remain measurements of density error for that constructed "
            "ensemble, but they cannot be combined with global-loading rates as a same-campaign "
            "physical Landau validation or used to claim resonant-dynamics support."
        ),
        "reconstructed_per_cell_support": support,
        "e2_c4_signal_resolvability": signal,
        "baseline_duration_policy": {
            "older_matrix_NOT_APPLICABLE": (
                "The retrospective matrix originally defined duration for the candidate arm "
                "only; NOT_APPLICABLE was a schema marker, not evidence that a baseline check failed."
            ),
            "e2_c4_actual_shared_baseline": (
                "The Gaussian C4 comparison uses the same [24.32,59.68] window as CIC; CIC "
                "spans 4.483681 fitted e-foldings, so the newer baseline-duration criterion passes."
            ),
            "new_confirmation_required": (
                "The v2 confirmatory campaign explicitly preregistered separate arm-specific "
                "censoring windows and required both CIC and C4 to span two e-foldings; its CIC "
                "window spans only 1.721386 and therefore fails."
            ),
            "scope_warning": (
                "NOT_APPLICABLE on other retrospective rows must not be read as an unmeasured "
                "baseline pass; claims there remain limited to the stated candidate-arm test."
            ),
        },
        "compensated_E2_prediction_conclusion": (
            "All four grid/time-discrete point predictions lie below and outside the direct "
            "measured 95% intervals.  The p2/n2 and p3/n4 settings nevertheless pass the "
            "separate 10% measured rate-preservation criterion."
        ),
        "input_sha256": {
            str(path.relative_to(WORKSPACE)): sha256(path)
            for path in (NOISE_AUDIT, JOINT_REDUCTION, COMPENSATED_REDUCTION, PREDICTIONS)
        },
    }

    args.output.mkdir(parents=True)
    write_csv(args.output / "historical_landau_loading.csv", loading)
    write_csv(args.output / "e2_compensated_prediction_accuracy.csv", prediction)
    (args.output / "e2_c4_signal_resolvability.json").write_text(
        json.dumps(signal, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output / "evidence_closure.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
