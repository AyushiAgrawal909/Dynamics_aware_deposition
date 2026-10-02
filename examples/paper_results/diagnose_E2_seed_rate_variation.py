#!/usr/bin/env python3
"""Diagnose E2 individual-seed rate variation using saved complex histories.

This is a post-hoc diagnostic.  Registered estimators and frozen windows remain
unchanged.  The analysis tests initial amplitude/phase, a fixed window grid,
within-window transients, loading moments, and a principal-mode finite-marker
linear-response approximation.  Historical marker samples are reconstructed
with the current audited initializer and are labelled nominal because the
historical source revision was not recorded.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

from complete_E2_estimator_comparison import sha256, write_csv
from diagnose_E2_rate_estimator import (
    linear_fit,
    load_corner_group,
    load_current_group,
    select_window,
)
from recompute_dispersion import (
    E2_M12N1,
    cic_shape,
    compensated_axis,
    complex_newton,
    gaussian_force_transfer,
    solve_level,
)


HERE = Path(__file__).resolve().parent
DOWNLOADS = HERE.parents[1]
AUDITED_PROJECT = DOWNLOADS / "pic_kde_2d_landau_cv_stage"
if str(AUDITED_PROJECT) not in sys.path:
    sys.path.insert(0, str(AUDITED_PROJECT))

from pic_kde_2d.conventions import marker_grid_shape  # noqa: E402
from pic_kde_2d.core.particles2d import init_two_stream_2d  # noqa: E402


EXPECTED_INITIAL_PHASE = -0.5 * math.pi
WINDOW_VARIANTS = (
    ("registered", 0.0, 0.0),
    ("start_plus_2", 2.0, 0.0),
    ("start_plus_5", 5.0, 0.0),
    ("stop_minus_5", 0.0, 5.0),
    ("stop_minus_10", 0.0, 10.0),
    ("start_plus_2_stop_minus_5", 2.0, 5.0),
    ("start_plus_5_stop_minus_10", 5.0, 10.0),
)


def wrap_phase(value: np.ndarray | float) -> np.ndarray | float:
    return (np.asarray(value) + math.pi) % (2.0 * math.pi) - math.pi


def safe_correlation(x: np.ndarray, y: np.ndarray) -> float:
    if np.std(x) <= 1.0e-15 or np.std(y) <= 1.0e-15:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def phase_multiple_r2(phase: np.ndarray, rates: np.ndarray) -> float:
    design = np.column_stack((np.ones(phase.size), np.sin(phase), np.cos(phase)))
    coefficients, *_ = np.linalg.lstsq(design, rates, rcond=None)
    fitted = design @ coefficients
    residual = float(np.sum((rates - fitted) ** 2))
    total = float(np.sum((rates - np.mean(rates)) ** 2))
    return float(1.0 - residual / total) if total > 0.0 else float("nan")


def finite_marker_residual_filtered(
    lam: complex,
    case,
    vx: np.ndarray,
    vy: np.ndarray,
    filter_force_transfer: float,
) -> complex:
    """Principal-mode finite-marker response of the explicit-midpoint update."""
    if not 0.0 < filter_force_transfer <= 1.0 + 1.0e-12:
        raise ValueError("filter-force transfer must lie in (0,1]")
    d2 = case.kx**2 + case.ky**2
    shape_squared = cic_shape(case.kx, case.ky, case) ** 2
    conditional_center = case.vbeam * np.tanh(
        vx * case.vbeam / case.vthermal**2
    )
    score_x = -(vx - conditional_center) / case.vthermal**2
    score_y = -vy / case.vthermal**2
    phase_rate = case.kx * vx + case.ky * vy
    source = (
        filter_force_transfer
        * 1j
        * shape_squared
        * (case.kx * score_x + case.ky * score_y)
        / d2
        / vx.size
    )
    streaming = np.exp(-1j * phase_rate * case.dt)
    half_streaming = np.exp(-0.5j * phase_rate * case.dt)
    denominator = np.exp(complex(lam) * case.dt) - streaming
    c0 = np.sum(half_streaming * source)
    return complex(
        1.0
        - case.dt * np.sum(streaming * source / denominator)
        - 0.5 * case.dt**2 * c0 * np.sum(half_streaming * source / denominator)
    )


def seed_history_rows(campaign: str, drift: float, arm: str, data: dict) -> tuple[list[dict], list[dict]]:
    times = data["times"]
    amplitudes = data["amplitudes"]
    modes = data["modes"]
    start = data["start"]
    stop = data["stop"]
    x, selected = select_window(times, amplitudes, start, stop)
    mask = (times >= start - 1.0e-12) & (times <= stop + 1.0e-12)
    selected_modes = modes[:, mask]
    midpoint = len(x) // 2
    seed_rows: list[dict] = []
    window_rows: list[dict] = []
    for index, seed in enumerate(data["seeds"]):
        native = linear_fit(x, selected[index])
        early = linear_fit(x[: midpoint + 1], selected[index, : midpoint + 1])
        late = linear_fit(x[midpoint:], selected[index, midpoint:])
        centered = x - np.mean(x)
        quadratic, linear, intercept = np.polyfit(
            centered, np.log(selected[index]), 2
        )
        initial_phase_delta = float(wrap_phase(np.angle(modes[index, 0]) - EXPECTED_INITIAL_PHASE))
        start_phase_delta = float(
            wrap_phase(np.angle(selected_modes[index, 0]) - EXPECTED_INITIAL_PHASE)
        )
        phase_unwrapped = np.unwrap(np.angle(selected_modes[index]))
        phase_fit = linear_fit(x, np.exp(phase_unwrapped))
        seed_rows.append(
            {
                "analysis_role": "POST_HOC_SEED_VARIATION_DIAGNOSTIC",
                "campaign": campaign,
                "v_beam": drift,
                "arm": arm,
                "seed": seed,
                "fit_t_start": start,
                "fit_t_stop": stop,
                "initial_amplitude": float(amplitudes[index, 0]),
                "initial_phase": float(np.angle(modes[index, 0])),
                "initial_phase_minus_imposed_minus_pi_over_2": initial_phase_delta,
                "fit_start_amplitude": float(selected[index, 0]),
                "fit_start_phase": float(np.angle(selected_modes[index, 0])),
                "fit_start_phase_minus_imposed_minus_pi_over_2": start_phase_delta,
                "fit_end_phase": float(np.angle(selected_modes[index, -1])),
                "native_gamma": native["gamma"],
                "native_r_squared": native["r_squared"],
                "early_half_gamma": early["gamma"],
                "late_half_gamma": late["gamma"],
                "late_minus_early_gamma": late["gamma"] - early["gamma"],
                "quadratic_log_amplitude_coefficient": float(quadratic),
                "quadratic_implied_end_minus_start_slope": float(
                    2.0 * quadratic * (stop - start)
                ),
                "phase_drift_slope": phase_fit["gamma"],
                "phase_drift_r_squared": phase_fit["r_squared"],
            }
        )
        for label, start_offset, stop_retreat in WINDOW_VARIANTS:
            variant_start = start + start_offset
            variant_stop = stop - stop_retreat
            if variant_stop <= variant_start:
                continue
            vx, va = select_window(times, amplitudes[index : index + 1], variant_start, variant_stop)
            fit = linear_fit(vx, va[0])
            window_rows.append(
                {
                    "analysis_role": "POST_HOC_FIXED_WINDOW_GRID",
                    "campaign": campaign,
                    "v_beam": drift,
                    "arm": arm,
                    "seed": seed,
                    "window_variant": label,
                    "fit_t_start": variant_start,
                    "fit_t_stop": variant_stop,
                    "gamma": fit["gamma"],
                    "r_squared": fit["r_squared"],
                    "fitted_efoldings": abs(fit["gamma"]) * (variant_stop - variant_start),
                    "at_least_two_fitted_efoldings": bool(
                        abs(fit["gamma"]) * (variant_stop - variant_start) >= 2.0
                    ),
                }
            )
    return seed_rows, window_rows


def group_diagnostic_rows(seed_rows: list[dict]) -> list[dict]:
    groups: dict[tuple[str, float, str], list[dict]] = {}
    for row in seed_rows:
        groups.setdefault((row["campaign"], float(row["v_beam"]), row["arm"]), []).append(row)
    output = []
    for (campaign, drift, arm), rows in groups.items():
        rows.sort(key=lambda row: int(row["seed"]))
        rates = np.asarray([row["native_gamma"] for row in rows], dtype=np.float64)
        initial_amplitude = np.asarray([row["initial_amplitude"] for row in rows])
        start_amplitude = np.asarray([row["fit_start_amplitude"] for row in rows])
        initial_phase = np.asarray(
            [row["initial_phase_minus_imposed_minus_pi_over_2"] for row in rows]
        )
        start_phase = np.asarray(
            [row["fit_start_phase_minus_imposed_minus_pi_over_2"] for row in rows]
        )
        transient = np.asarray([row["late_minus_early_gamma"] for row in rows])
        output.append(
            {
                "campaign": campaign,
                "v_beam": drift,
                "arm": arm,
                "n_seeds": len(rows),
                "initial_amplitude_relative_range": float(
                    (np.max(initial_amplitude) - np.min(initial_amplitude))
                    / np.mean(initial_amplitude)
                ),
                "maximum_absolute_initial_phase_error": float(
                    np.max(np.abs(initial_phase))
                ),
                "gamma_correlation_log_initial_amplitude": safe_correlation(
                    rates, np.log(initial_amplitude)
                ),
                "gamma_correlation_log_fit_start_amplitude": safe_correlation(
                    rates, np.log(start_amplitude)
                ),
                "gamma_correlation_abs_fit_start_phase_error": safe_correlation(
                    rates, np.abs(start_phase)
                ),
                "gamma_fit_start_phase_sine_cosine_multiple_r2": phase_multiple_r2(
                    start_phase, rates
                ),
                "mean_late_minus_early_gamma": float(np.mean(transient)),
                "sd_late_minus_early_gamma": float(np.std(transient, ddof=1)),
                "gamma_correlation_late_minus_early_gamma": safe_correlation(
                    rates, transient
                ),
                "initial_condition_conclusion": (
                    "saved t=0 field amplitude and phase are seed-invariant"
                    if np.ptp(initial_amplitude) / np.mean(initial_amplitude) < 1.0e-12
                    and np.max(np.abs(initial_phase)) < 1.0e-12
                    else "saved t=0 field contains seed variation"
                ),
            }
        )
    return output


def load_modes(paths: list[Path]) -> np.ndarray:
    rows = []
    for path in paths:
        with np.load(path, allow_pickle=False) as archive:
            if "mode" not in archive.files:
                raise AssertionError(f"mode missing from {path}")
            rows.append(np.asarray(archive["mode"], dtype=np.complex128))
    return np.stack(rows)


def attach_modes(data: dict, root: Path, label: str) -> None:
    paths = sorted((root / label).glob("seed*/mode_history.npz"))
    if len(paths) != len(data["seeds"]):
        raise AssertionError(f"complex-history count mismatch: {root / label}")
    data["modes"] = load_modes(paths)


def filter_force_for_historical(row: dict, case) -> float:
    return gaussian_force_transfer(
        case, float(row["sigma_x_dx"]), float(row["sigma_y_dy"])
    )


def filter_force_for_compensated(row: dict, case) -> float:
    if row["label"] == "base":
        return 1.0
    wx = compensated_axis(
        2.0 * math.pi * case.mx / case.nx,
        int(row["passes_x"]), int(row["compensation_order"]),
    )
    wy = compensated_axis(
        2.0 * math.pi * case.my / case.ny,
        int(row["passes_y"]), int(row["compensation_order"]),
    )
    return float((wx * wy) ** 2)


def marker_and_root_rows(
    marker_group: str,
    source_status: str,
    drift: float,
    seeds: list[int],
    arm_transfers: dict[str, float],
    observed_rates: dict[tuple[str, int], float],
    qmax: int,
    lmax: int,
) -> tuple[list[dict], list[dict]]:
    case = dataclasses.replace(E2_M12N1, vbeam=drift)
    nx_marker, ny_marker = marker_grid_shape(case.nx, case.ny, 512)
    if (nx_marker, ny_marker) != (2048, 512):
        raise AssertionError("unexpected marker grid")
    moment_rows: list[dict] = []
    root_rows: list[dict] = []
    baseline_nominal = solve_level(
        case, "grid_time_discrete", filter_force_transfer=1.0,
        qmax=qmax, lmax=lmax,
    )
    for seed in seeds:
        particles = init_two_stream_2d(
            nx_marker, ny_marker, case.lx, case.ly, drift, case.vthermal,
            0.0, case.mx, case.my, np.random.default_rng(seed),
        )
        beam_sign = np.where(particles.vx >= 0.0, 1.0, -1.0)
        if np.any(np.abs(particles.vx) < 0.25):
            raise AssertionError("beam sign cannot be inferred unambiguously")
        thermal_x = particles.vx - drift * beam_sign
        px = nx_marker // case.nx
        py = ny_marker // case.ny
        cell_sign = beam_sign.reshape(case.nx, px, case.ny, py).mean(axis=(1, 3))
        cell_thermal_x = thermal_x.reshape(case.nx, px, case.ny, py).mean(axis=(1, 3))
        cell_vy = particles.vy.reshape(case.nx, px, case.ny, py).mean(axis=(1, 3))
        moment_rows.append(
            {
                "marker_group": marker_group,
                "source_status": source_status,
                "v_beam": drift,
                "seed": seed,
                "particles": particles.vx.size,
                "positive_beam_fraction": float(np.mean(beam_sign > 0.0)),
                "thermal_vx_mean": float(np.mean(thermal_x)),
                "thermal_vx_std": float(np.std(thermal_x)),
                "vy_mean": float(np.mean(particles.vy)),
                "vy_std": float(np.std(particles.vy)),
                "beam_thermal_vx_covariance": float(np.mean(beam_sign * thermal_x)),
                "per_cell_beam_imbalance_rms": float(np.sqrt(np.mean(cell_sign**2))),
                "per_cell_beam_imbalance_max_abs": float(np.max(np.abs(cell_sign))),
                "per_cell_thermal_vx_mean_rms": float(
                    np.sqrt(np.mean(cell_thermal_x**2))
                ),
                "per_cell_vy_mean_rms": float(np.sqrt(np.mean(cell_vy**2))),
            }
        )
        for arm, force in arm_transfers.items():
            try:
                nominal = solve_level(
                    case, "grid_time_discrete", filter_force_transfer=force,
                    qmax=qmax, lmax=lmax,
                )
                nominal_status = "CONVERGED_MODEL_BRANCH"
                initial = nominal
            except RuntimeError:
                # At the frozen v_b=0.818 fixed-filter point the audited
                # continuum-velocity model has no converged growing branch.
                # Continue the finite-sample equation from the unfiltered
                # model root; this choice is theory-only and does not inspect
                # the measured seed rate.
                nominal = complex(float("nan"), float("nan"))
                nominal_status = "NO_CONVERGED_GROWING_MODEL_BRANCH"
                initial = baseline_nominal
            try:
                root = complex_newton(
                    lambda value: finite_marker_residual_filtered(
                        value, case, particles.vx, particles.vy, force
                    ),
                    initial,
                )
                finite_marker_status = "CONVERGED"
            except RuntimeError:
                root = complex(float("nan"), float("nan"))
                finite_marker_status = "NO_CONVERGED_TRACKED_ROOT"
            observed = observed_rates.get((arm, seed), float("nan"))
            root_rows.append(
                {
                    "analysis_role": "POST_HOC_FINITE_MARKER_LINEAR_RESPONSE_APPROXIMATION",
                    "marker_group": marker_group,
                    "source_status": source_status,
                    "v_beam": drift,
                    "arm": arm,
                    "seed": seed,
                    "filter_force_transfer_W2": force,
                    "continuum_velocity_model_status": nominal_status,
                    "continuum_velocity_grid_time_discrete_gamma": float(nominal.real),
                    "finite_marker_diagonal_gamma": float(root.real),
                    "finite_marker_diagonal_frequency": float(root.imag),
                    "finite_marker_root_status": finite_marker_status,
                    "observed_individual_gamma": observed,
                    "observed_minus_finite_marker_gamma": float(observed - root.real),
                    "approximation_limit": (
                        "principal represented mode and diagonal +k response; spatial aliases "
                        "and finite-sample +/-k mixing omitted"
                    ),
                }
            )
        del particles
    return moment_rows, root_rows


def response_summary_rows(moment_rows: list[dict], root_rows: list[dict]) -> list[dict]:
    groups: dict[tuple[str, float, str], list[dict]] = {}
    for row in root_rows:
        groups.setdefault(
            (row["marker_group"], float(row["v_beam"]), row["arm"]), []
        ).append(row)
    moments = {
        (row["marker_group"], float(row["v_beam"]), int(row["seed"])): row
        for row in moment_rows
    }
    output = []
    for (marker_group, drift, arm), rows in groups.items():
        rows.sort(key=lambda row: int(row["seed"]))
        predicted = np.asarray([row["finite_marker_diagonal_gamma"] for row in rows])
        observed = np.asarray([row["observed_individual_gamma"] for row in rows])
        valid = np.isfinite(observed) & np.isfinite(predicted)
        moment_vectors = {
            name: np.asarray([
                moments[(marker_group, drift, int(row["seed"]))][name] for row in rows
            ])
            for name in (
                "beam_thermal_vx_covariance",
                "per_cell_beam_imbalance_rms",
                "per_cell_thermal_vx_mean_rms",
                "per_cell_vy_mean_rms",
            )
        }
        output.append(
            {
                "marker_group": marker_group,
                "v_beam": drift,
                "arm": arm,
                "n_seeds": len(rows),
                "finite_marker_roots_converged": int(np.count_nonzero(np.isfinite(predicted))),
                "finite_marker_gamma_mean": float(np.nanmean(predicted)),
                "finite_marker_gamma_sd": float(np.nanstd(predicted, ddof=1)),
                "observed_gamma_mean": float(np.mean(observed[valid])) if np.any(valid) else float("nan"),
                "observed_gamma_sd": float(np.std(observed[valid], ddof=1)) if np.count_nonzero(valid) > 1 else float("nan"),
                "observed_vs_finite_marker_gamma_correlation": safe_correlation(
                    observed[valid], predicted[valid]
                ) if np.count_nonzero(valid) > 1 else float("nan"),
                "observed_vs_finite_marker_rmse": float(
                    np.sqrt(np.mean((observed[valid] - predicted[valid]) ** 2))
                ) if np.any(valid) else float("nan"),
                **{
                    f"observed_gamma_correlation_{name}": safe_correlation(
                        observed[valid], values[valid]
                    ) if np.count_nonzero(valid) > 1 else float("nan")
                    for name, values in moment_vectors.items()
                },
            }
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--scan-root", type=Path, required=True)
    parser.add_argument(
        "--extension-root",
        type=Path,
        help="completed extension tree used only for a drift absent from --scan-root",
    )
    parser.add_argument("--registered", type=Path, required=True)
    parser.add_argument("--historical-root", type=Path, required=True)
    parser.add_argument("--compensated-root", type=Path, required=True)
    parser.add_argument("--estimator-results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")

    config = json.loads(args.config.read_text(encoding="utf-8"))
    registered = json.loads(args.registered.read_text(encoding="utf-8"))
    historical_path = args.historical_root / "analysis" / "joint_corner_sweep.json"
    compensated_path = args.compensated_root / "analysis" / "E2_compensated_corner.json"
    historical = json.loads(historical_path.read_text(encoding="utf-8"))
    compensated = json.loads(compensated_path.read_text(encoding="utf-8"))
    with (args.estimator_results / "individual_seed_rates.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        estimator_rows = list(csv.DictReader(handle))
    observed = {
        (row["campaign"], float(row["v_beam"]), row["arm"], int(row["seed"])):
        float(row["gamma_individual"])
        for row in estimator_rows
    }

    all_groups: list[tuple[str, float, str, dict]] = []
    registered_lookup = {
        (float(row["v_beam"]), row["arm"]): row
        for row in registered["rows"] if row["measurement_status"] == "COMPLETE"
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
        for arm in ("cic", "fixed_gaussian", "constrained_gaussian"):
            times, amplitudes, seeds, _ = load_current_group(
                current_root, config, scan, arm
            )
            data = {
                "times": times, "amplitudes": amplitudes, "seeds": seeds,
                "start": float(scan["fit_t_start"]), "stop": float(scan["fit_t_stop"]),
            }
            attach_modes(data, current_root / f"vb{int(round(1000 * drift)):04d}", arm)
            all_groups.append(("registered_drift_scan_5201231", drift, arm, data))

    historical_rows = {row["label"]: row for row in historical["rows"]}
    for label in historical_rows:
        times, amplitudes, seeds, _results, _ = load_corner_group(args.historical_root, label)
        data = {
            "times": times, "amplitudes": amplitudes, "seeds": seeds,
            "start": 24.32, "stop": 59.68,
        }
        attach_modes(data, args.historical_root / "rate", label)
        all_groups.append(("historical_table5_corner_4913408", 0.8, label, data))

    compensated_rows = {row["label"]: row for row in compensated["rows"]}
    for label in compensated_rows:
        times, amplitudes, seeds, _results, _ = load_corner_group(args.compensated_root, label)
        data = {
            "times": times, "amplitudes": amplitudes, "seeds": seeds,
            "start": 24.32, "stop": 59.68,
        }
        attach_modes(data, args.compensated_root / "rate", label)
        all_groups.append(("compensated_corner_5195640", 0.8, label, data))

    seed_rows: list[dict] = []
    window_rows: list[dict] = []
    for campaign, drift, arm, data in all_groups:
        these_seeds, these_windows = seed_history_rows(campaign, drift, arm, data)
        seed_rows.extend(these_seeds)
        window_rows.extend(these_windows)
    group_rows = group_diagnostic_rows(seed_rows)

    # The imposed delta-f density mode is deterministic on the marker lattice.
    if max(float(row["initial_amplitude_relative_range"]) for row in group_rows) > 1.0e-12:
        raise AssertionError("unexpected seed variation in initial mode amplitude")
    if max(float(row["maximum_absolute_initial_phase_error"]) for row in group_rows) > 1.0e-12:
        raise AssertionError("unexpected seed variation in initial mode phase")

    qmax = int(config["model"]["spatial_alias_qmax"])
    lmax = int(config["model"]["temporal_alias_lmax"])
    moment_rows: list[dict] = []
    root_rows: list[dict] = []
    for scan in config["scan_rows"]:
        drift = float(scan["v_beam"])
        if (drift, "cic") not in registered_lookup:
            continue
        transfers = {
            "cic": 1.0,
            "fixed_gaussian": float(scan["fixed_particle_force_W2"]),
            "constrained_gaussian": float(scan["constrained_particle_force_W2"]),
        }
        observed_rates = {
            (arm, seed): observed[("registered_drift_scan_5201231", drift, arm, seed)]
            for arm in transfers for seed in config["seeds"]
        }
        moments, roots = marker_and_root_rows(
            "current_drift_seeds_200_209", "EXACT_CURRENT_AUDITED_INITIALIZER",
            drift, list(config["seeds"]), transfers, observed_rates, qmax, lmax,
        )
        moment_rows.extend(moments)
        root_rows.extend(roots)

    historical_case = dataclasses.replace(E2_M12N1, vbeam=0.8)
    historical_transfers = {
        label: filter_force_for_historical(row, historical_case)
        for label, row in historical_rows.items()
    }
    historical_transfers.update({
        label: filter_force_for_compensated(row, historical_case)
        for label, row in compensated_rows.items() if label != "base"
    })
    historical_observed = {
        (label, seed): observed[(
            "compensated_corner_5195640" if label in compensated_rows and label != "base"
            else "historical_table5_corner_4913408",
            0.8, label, seed,
        )]
        for label in historical_transfers for seed in range(10)
    }
    moments, roots = marker_and_root_rows(
        "historical_seeds_0_9_reconstructed", (
            "NOMINAL_CURRENT_AUDITED_INITIALIZER; HISTORICAL SOURCE REVISION UNKNOWN"
        ), 0.8, list(range(10)), historical_transfers, historical_observed, qmax, lmax,
    )
    moment_rows.extend(moments)
    root_rows.extend(roots)
    response_rows = response_summary_rows(moment_rows, root_rows)

    args.output.mkdir(parents=True)
    write_csv(args.output / "per_seed_history_diagnostics.csv", seed_rows)
    write_csv(args.output / "fixed_window_sensitivity_grid.csv", window_rows)
    write_csv(args.output / "group_history_diagnostics.csv", group_rows)
    write_csv(args.output / "reconstructed_loading_moments.csv", moment_rows)
    write_csv(args.output / "finite_marker_linear_response.csv", root_rows)
    write_csv(args.output / "finite_marker_response_summary.csv", response_rows)

    inputs = (
        args.config, args.registered, historical_path, compensated_path,
        args.estimator_results / "individual_seed_rates.csv",
        AUDITED_PROJECT / "pic_kde_2d/core/particles2d.py",
        HERE / "recompute_dispersion.py",
    )
    current_v080 = next(
        row for row in response_rows
        if row["marker_group"] == "current_drift_seeds_200_209"
        and float(row["v_beam"]) == 0.8 and row["arm"] == "cic"
    )
    result = {
        "status": "POST_HOC_DIAGNOSTIC_COMPLETE_REGISTERED_RESULTS_UNCHANGED",
        "initial_condition_result": (
            "Every arm has seed-invariant saved t=0 amplitude and phase; therefore "
            "initial field amplitude/phase cannot explain within-arm seed-rate spread."
        ),
        "historical_loading_status": (
            "Historical marker arrays were not saved. Seeds 0--9 are reconstructed with "
            "the current audited initializer only; exact historical source identity is unknown."
        ),
        "finite_marker_scope": (
            "Principal represented mode, diagonal +k finite-marker response of the "
            "explicit-midpoint update; spatial aliases and finite-sample +/-k mixing omitted."
        ),
        "current_vbeam_080_cic": current_v080,
        "higher_ppc_decision_rule": (
            "No higher-PPC run is launched by this diagnostic. It would be justified only "
            "if a remaining finite-particle question is both specific and not identifiable "
            "from the saved histories/loading reconstruction."
        ),
        "input_sha256": {str(path): sha256(path) for path in inputs},
        "tables": {
            "per_seed_history_diagnostics": "per_seed_history_diagnostics.csv",
            "fixed_window_sensitivity_grid": "fixed_window_sensitivity_grid.csv",
            "group_history_diagnostics": "group_history_diagnostics.csv",
            "reconstructed_loading_moments": "reconstructed_loading_moments.csv",
            "finite_marker_linear_response": "finite_marker_linear_response.csv",
            "finite_marker_response_summary": "finite_marker_response_summary.csv",
        },
    }
    (args.output / "diagnostic_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "status": result["status"],
        "initial_condition_result": result["initial_condition_result"],
        "current_vbeam_080_cic": current_v080,
        "rows": {
            "per_seed": len(seed_rows), "windows": len(window_rows),
            "marker_roots": len(root_rows),
        },
    }, indent=2))


if __name__ == "__main__":
    main()
