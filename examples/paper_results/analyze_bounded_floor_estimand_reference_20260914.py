#!/usr/bin/env python3
"""Bounded Paper-1 correction using only already archived histories.

The script compares the PPC-1024 and PPC-4096 weak-Landau CIC floors,
constructs like-for-like E2 estimand tables, bootstraps the complete finite-
reference decomposition, and makes corrected vector figures.  It neither
runs a simulation nor edits a manuscript.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np


HERE = Path(__file__).resolve().parent
DOWNLOADS = HERE.parents[1]
PROJECT = DOWNLOADS / "pic_kde_2d"
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from pic_kde_2d.validation.landau_censored_fit import (  # noqa: E402
    coherent_amplitude,
    fit_censored_mode,
    geometric_envelope,
)


RESULTS = HERE / "results"
LANDAU = RESULTS / "weak_landau_confirmatory_v2_production_20260911"
LANDAU_CONFIG = HERE / "weak_landau_confirmatory_v2_global_velocity.json"
MARKER = RESULTS / "e2_seed_rate_variation_20260912_with_extension" / "finite_marker_linear_response.csv"
ESTIMANDS = RESULTS / "scientific_evidence_completion_20260914_r3" / "e2_estimand_comparison.csv"
PRACTICAL = RESULTS / "scientific_evidence_completion_20260914_r3" / "practical_benefit_comparison.csv"
SEEDS = tuple(range(202609300, 202609340))
ARMS = ("cic", "c4_sigma1", "lscv_job5008392")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise AssertionError(f"missing CSV schema: {path}")
        return list(reader)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise AssertionError(f"refusing to write empty table: {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def f(row: dict[str, str], key: str) -> float:
    return float(row[key])


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def interval(values: np.ndarray) -> tuple[float, float]:
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def log_bootstrap_se_interval(theta: float, bootstrap: np.ndarray) -> tuple[float, float, float]:
    """Positive interval centered on the plug-in ratio using bootstrap log-SE.

    Population-variance functionals shrink when an empirical distribution is
    resampled with replacement.  The raw percentile distribution is therefore
    also retained, rather than silently presenting it as centered on theta.
    """
    if theta <= 0.0 or np.any(bootstrap <= 0.0):
        return float("nan"), float("nan"), float("nan")
    standard_error = float(np.std(np.log(bootstrap), ddof=1))
    return theta * math.exp(-1.96 * standard_error), theta * math.exp(1.96 * standard_error), standard_error


def fit_kwargs(config: dict) -> dict:
    rule = config["fit_rule"]
    return {
        "fit_t_start": rule["fit_t_start"],
        "fit_t_stop_max": rule["fit_t_stop_max"],
        "noise_floor_start": rule["noise_floor_t_start"],
        "noise_floor_stop": rule["noise_floor_t_stop"],
        "noise_factor": rule["noise_threshold_factor"],
        "envelope_period": rule["envelope_period"],
        "persistence_points": rule["persistence_points"],
        "min_points": rule["min_points"],
        "min_duration": rule["min_duration"],
        "min_model_efoldings": rule["min_fitted_amplitude_efoldings"],
        "min_r_squared": rule["min_fit_r_squared"],
    }


def archive_path(ppc: int, seed: int, arm: str = "cic") -> Path:
    if ppc == 4096:
        return LANDAU / "reference" / f"seed{seed}" / "cic_reference" / "history_and_noise_snapshots.npz"
    return LANDAU / "low" / f"seed{seed}" / arm / "history_and_noise_snapshots.npz"


def parameters_path(ppc: int, seed: int, arm: str = "cic") -> Path:
    return archive_path(ppc, seed, arm).with_name("run_parameters.json")


def load_landau(ppc: int, arm: str = "cic") -> tuple[np.ndarray, dict[str, np.ndarray]]:
    data: dict[str, list[np.ndarray]] = {}
    times = None
    for seed in SEEDS:
        with np.load(archive_path(ppc, seed, arm), allow_pickle=False) as archive:
            current_times = np.asarray(archive["times"], dtype=float)
            if times is None:
                times = current_times
            else:
                np.testing.assert_array_equal(times, current_times)
            for key in ("mode", "snapshot_times", "rho_snapshots", "Ex_snapshots", "Ey_snapshots"):
                data.setdefault(key, []).append(np.asarray(archive[key]))
    assert times is not None
    return times, {key: np.stack(value) for key, value in data.items()}


def linear_phase_frequency(histories: np.ndarray, times: np.ndarray, start: float, stop: float) -> np.ndarray:
    mask = (times >= start) & (times <= stop)
    frequencies = []
    for history in histories:
        phase = np.unwrap(np.angle(history[mask]))
        frequencies.append(float(np.polyfit(times[mask], phase, 1)[0]))
    return np.asarray(frequencies)


def first_recurrence(envelope: np.ndarray, times: np.ndarray, fit_stop: float, config: dict) -> float:
    rule = config["fit_rule"]
    period_points = int(rule["envelope_points_at_frozen_cadence"])
    indices = np.flatnonzero(
        (times > fit_stop)
        & (times < rule["noise_floor_t_start"])
        & (np.arange(times.size) >= period_points)
    )
    delta = np.log(np.maximum(envelope[indices], 1e-300)) - np.log(
        np.maximum(envelope[indices - period_points], 1e-300)
    )
    count = int(rule["persistence_points"])
    runs = np.convolve((delta >= 0).astype(int), np.ones(count, dtype=int), mode="valid")
    hits = np.flatnonzero(runs == count)
    return float(times[indices[hits[0]]]) if hits.size else float("nan")


def build_floor_comparison(output: Path, config: dict) -> tuple[list[dict], dict[int, dict]]:
    rows: list[dict] = []
    plotted: dict[int, dict] = {}
    reference_params = None
    for ppc in (1024, 4096):
        times, arrays = load_landau(ppc)
        histories = arrays["mode"].astype(complex)
        fit = fit_censored_mode(histories, times, **fit_kwargs(config))
        coherent_complex = histories.mean(axis=0)
        raw = np.abs(coherent_complex)
        envelope, points = geometric_envelope(raw, times, config["fit_rule"]["envelope_period"])
        mean_abs = np.mean(np.abs(histories), axis=0)
        coherence = raw / np.maximum(mean_abs, 1e-300)
        incoherent_standard_error = np.sqrt(
            np.mean(np.abs(histories - coherent_complex[None, :]) ** 2, axis=0) / histories.shape[0]
        )
        i0 = int(np.flatnonzero(np.isclose(times, 0.0))[0])
        i5 = int(np.flatnonzero(np.isclose(times, config["fit_rule"]["fit_t_start"]))[0])
        floor_mask = (times >= fit["floor_t_start"]) & (times <= fit["floor_t_stop"])
        early_freq = linear_phase_frequency(histories, times, 5.0, 20.0)
        params = json.loads(parameters_path(ppc, SEEDS[0]).read_text(encoding="utf-8"))
        if reference_params is None:
            reference_params = params
        else:
            for key in ("Nx", "Ny", "Lx", "Ly", "dx", "dy", "alpha", "dt", "n_steps", "simulation_t_stop", "solver", "velocity_assignment_scope", "velocity_loading"):
                if params[key] != reference_params[key]:
                    raise AssertionError(f"PPC histories differ in {key}")
        rows.append(
            {
                "ppc": ppc,
                "n_seeds": histories.shape[0],
                "fourier_field_unit": "abs(rfft2(Ex)[1,0]/(Nx*Ny)); normalized electric-field amplitude",
                "noise_floor": fit["noise_floor"],
                "floor_t_start": fit["floor_t_start"],
                "floor_t_stop": fit["floor_t_stop"],
                "initial_coherent_amplitude_t0": raw[i0],
                "raw_coherent_amplitude_at_fit_start": raw[i5],
                "envelope_amplitude_at_fit_start": envelope[i5],
                "fit_start_signal_to_floor": fit["fit_start_snr"],
                "crossing_time_5x_floor": fit["censor_t"],
                "accepted_t_start": fit["fit_t_start"],
                "accepted_t_stop": fit["fit_t_stop"],
                "fitted_gamma": fit["gamma"],
                "fitted_efoldings": fit["model_efoldings"],
                "r_squared": fit["r_squared"],
                "floor_window_mean_seed_magnitude": float(np.mean(mean_abs[floor_mask])),
                "floor_window_mean_amplitude_weighted_coherence": float(np.mean(coherence[floor_mask])),
                "floor_window_median_amplitude_weighted_coherence": float(np.median(coherence[floor_mask])),
                "floor_window_mean_incoherent_standard_error_amplitude": float(np.mean(incoherent_standard_error[floor_mask])),
                "floor_window_coherent_amplitude_over_incoherent_standard_error": float(np.mean(raw[floor_mask] / np.maximum(incoherent_standard_error[floor_mask], 1e-300))),
                "early_seed_frequency_mean_5_20": float(np.mean(early_freq)),
                "early_seed_frequency_sd_5_20": float(np.std(early_freq, ddof=1)),
                "first_postfit_recurrence_time": first_recurrence(envelope, times, fit["fit_t_stop"], config),
                "envelope_definition": "49-sample centered geometric mean of abs(coherent complex seed mean), edge padded",
                "floor_definition": "minimum rolling median of envelope over complete 5-time-unit windows stepped by 1 in [70,90]",
                "loading": params["velocity_loading"],
                "normalization": "omega_p=n0=epsilon0=me=1; mode divided by Nx*Ny",
                "particle_relation": "same seed identifiers but independently generated global shuffles; not nested particles",
            }
        )
        plotted[ppc] = {
            "times": times,
            "raw": raw,
            "envelope": envelope,
            "mean_abs": mean_abs,
            "coherence": coherence,
            "incoherent_standard_error": incoherent_standard_error,
            "fit": fit,
            "histories": histories,
            "envelope_points": points,
        }

    low, high = rows
    comparison = {
        "ppc4096_over_ppc1024_floor": high["noise_floor"] / low["noise_floor"],
        "ppc4096_over_ppc1024_initial_t0": high["initial_coherent_amplitude_t0"] / low["initial_coherent_amplitude_t0"],
        "ppc4096_over_ppc1024_fit_start_envelope": high["envelope_amplitude_at_fit_start"] / low["envelope_amplitude_at_fit_start"],
        "ppc4096_over_ppc1024_floor_window_mean_seed_magnitude": high["floor_window_mean_seed_magnitude"] / low["floor_window_mean_seed_magnitude"],
        "ppc4096_over_ppc1024_floor_window_coherence": high["floor_window_mean_amplitude_weighted_coherence"] / low["floor_window_mean_amplitude_weighted_coherence"],
        "ppc4096_over_ppc1024_incoherent_standard_error": high["floor_window_mean_incoherent_standard_error_amplitude"] / low["floor_window_mean_incoherent_standard_error_amplitude"],
        "demonstrated": (
            "Normalization, solver, loading protocol, initial amplitude, and fit-start envelope are comparable. "
            "The PPC-4096 individual-seed late magnitude is lower, but its phases remain much more coherent, "
            "so abs(mean complex mode) has a higher reducer floor."
        ),
        "not_demonstrated": (
            "The archives do not isolate why the independently generated PPC-4096 loading retains more late-time "
            "coherence; marker count, recurrence, and loading-realization effects are confounded."
        ),
    }
    write_csv(output / "landau_ppc_floor_comparison.csv", rows)
    (output / "landau_ppc_floor_comparison.json").write_text(
        json.dumps({"rows": rows, "comparison": comparison}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    history_rows = []
    for index, time in enumerate(plotted[1024]["times"]):
        history_rows.append(
            {
                "time": float(time),
                "ppc1024_coherent_amplitude": plotted[1024]["raw"][index],
                "ppc1024_envelope": plotted[1024]["envelope"][index],
                "ppc1024_mean_seed_magnitude": plotted[1024]["mean_abs"][index],
                "ppc1024_coherence": plotted[1024]["coherence"][index],
                "ppc1024_incoherent_standard_error": plotted[1024]["incoherent_standard_error"][index],
                "ppc4096_coherent_amplitude": plotted[4096]["raw"][index],
                "ppc4096_envelope": plotted[4096]["envelope"][index],
                "ppc4096_mean_seed_magnitude": plotted[4096]["mean_abs"][index],
                "ppc4096_coherence": plotted[4096]["coherence"][index],
                "ppc4096_incoherent_standard_error": plotted[4096]["incoherent_standard_error"][index],
            }
        )
    write_csv(output / "landau_ppc_floor_histories.csv", history_rows)
    return rows, plotted


def metric_matrix(arrays: dict[str, np.ndarray], metric: str) -> np.ndarray:
    if metric == "density_error_power":
        return arrays["rho_snapshots"].astype(float).reshape(len(SEEDS), -1)
    return np.concatenate(
        [arrays["Ex_snapshots"].astype(float).reshape(len(SEEDS), -1),
         arrays["Ey_snapshots"].astype(float).reshape(len(SEEDS), -1)], axis=1
    )


def decomposition_from_grams(gxx: np.ndarray, grr: np.ndarray, gxr: np.ndarray, weights: np.ndarray) -> dict[str, np.ndarray]:
    qxx = np.einsum("bi,ij,bj->b", weights, gxx, weights, optimize=True)
    qrr = np.einsum("bi,ij,bj->b", weights, grr, weights, optimize=True)
    qxr = np.einsum("bi,ij,bj->b", weights, gxr, weights, optimize=True)
    ex2 = weights @ np.diag(gxx)
    er2 = weights @ np.diag(grr)
    exr = weights @ np.diag(gxr)
    mean_difference_squared = qxx + qrr - 2.0 * qxr
    candidate_variance = ex2 - qxx
    reference_variance = er2 - qrr
    covariance = exr - qxr
    pair_mse = ex2 + er2 - 2.0 * exr
    adjusted_q = pair_mse - reference_variance + 2.0 * covariance
    # The Gram-matrix form avoids materialising every bootstrap ensemble.  Its
    # two algebraically identical forms differ only at double-precision roundoff.
    np.testing.assert_allclose(adjusted_q, candidate_variance + mean_difference_squared, rtol=1e-10, atol=1e-15)
    return {
        "pair_mse": pair_mse,
        "candidate_variance": candidate_variance,
        "reference_variance": reference_variance,
        "covariance": covariance,
        "plus_two_covariance": 2.0 * covariance,
        "squared_candidate_reference_mean_difference": mean_difference_squared,
        "adjusted_q": adjusted_q,
    }


def build_reference_correction(output: Path, rng: np.random.Generator, n_boot: int) -> list[dict]:
    _, ref_arrays = load_landau(4096)
    draws = rng.multinomial(len(SEEDS), np.full(len(SEEDS), 1.0 / len(SEEDS)), size=n_boot) / len(SEEDS)
    central_weights = np.full((1, len(SEEDS)), 1.0 / len(SEEDS))
    store: dict[tuple[str, str], dict[str, np.ndarray]] = {}
    central: dict[tuple[str, str], dict[str, float]] = {}
    for metric in ("density_error_power", "pre_return_grid_field_error_power"):
        r = metric_matrix(ref_arrays, metric)
        # Subtract one common featurewise origin before forming Gram matrices.
        # Every decomposition term is translation invariant when the same
        # origin is removed from X and R, and this avoids cancellation of the
        # O(1) density mean when resolving O(1e-5) adjusted errors.
        origin = r.mean(axis=0)
        rc = r - origin
        grr = (rc @ rc.T) / rc.shape[1]
        for arm in ARMS:
            _, arm_arrays = load_landau(1024, arm)
            x = metric_matrix(arm_arrays, metric)
            if x.shape != r.shape:
                raise AssertionError("candidate/reference snapshot shape mismatch")
            xc = x - origin
            gxx = (xc @ xc.T) / xc.shape[1]
            gxr = (xc @ rc.T) / xc.shape[1]
            store[(metric, arm)] = decomposition_from_grams(gxx, grr, gxr, draws)
            central[(metric, arm)] = {
                key: float(value[0]) for key, value in decomposition_from_grams(gxx, grr, gxr, central_weights).items()
            }

    rows = []
    for metric in ("density_error_power", "pre_return_grid_field_error_power"):
        base_c = central[(metric, "cic")]
        base_d = store[(metric, "cic")]
        for arm in ARMS:
            c = central[(metric, arm)]
            d = store[(metric, arm)]
            original_draw = base_d["pair_mse"] / d["pair_mse"]
            adjusted_draw = base_d["adjusted_q"] / d["adjusted_q"]
            original_ratio = base_c["pair_mse"] / c["pair_mse"]
            adjusted_ratio = base_c["adjusted_q"] / c["adjusted_q"]
            original_percentile_lo, original_percentile_hi = interval(original_draw)
            adjusted_percentile_lo, adjusted_percentile_hi = interval(adjusted_draw)
            original_lo, original_hi, original_log_se = log_bootstrap_se_interval(original_ratio, original_draw)
            adjusted_lo, adjusted_hi, adjusted_log_se = log_bootstrap_se_interval(adjusted_ratio, adjusted_draw)
            row = {
                "metric": metric,
                "arm": arm,
                "pair_mse": c["pair_mse"],
                "reference_variance": c["reference_variance"],
                "covariance": c["covariance"],
                "plus_two_covariance": c["plus_two_covariance"],
                "candidate_variance": c["candidate_variance"],
                "squared_candidate_reference_mean_difference": c["squared_candidate_reference_mean_difference"],
                "adjusted_q": c["adjusted_q"],
                "identity_residual": c["adjusted_q"] - c["candidate_variance"] - c["squared_candidate_reference_mean_difference"],
                "covariance_adjustment_fraction_of_pair_mse": c["plus_two_covariance"] / c["pair_mse"],
                "original_cic_over_arm_ratio": original_ratio,
                "original_ratio_ci_lo": original_lo,
                "original_ratio_ci_hi": original_hi,
                "original_ratio_percentile_ci_lo": original_percentile_lo,
                "original_ratio_percentile_ci_hi": original_percentile_hi,
                "original_ratio_bootstrap_log_se": original_log_se,
                "original_ratio_bootstrap_mean_minus_central": float(np.mean(original_draw) - original_ratio),
                "adjusted_cic_over_arm_ratio": adjusted_ratio,
                "adjusted_ratio_ci_lo": adjusted_lo,
                "adjusted_ratio_ci_hi": adjusted_hi,
                "adjusted_ratio_percentile_ci_lo": adjusted_percentile_lo,
                "adjusted_ratio_percentile_ci_hi": adjusted_percentile_hi,
                "adjusted_ratio_bootstrap_log_se": adjusted_log_se,
                "adjusted_ratio_bootstrap_mean_minus_central": float(np.mean(adjusted_draw) - adjusted_ratio),
                "reported_ci_method": "paired-seed log-scale bootstrap-SE interval centered on plug-in ratio; raw percentile distribution retained to expose finite-sample bias",
                "bootstrap_draws": n_boot,
                "bootstrap_unit": "paired seed identity; candidate and common reference resampled together",
                "interpretation": "Q_H = Var_seed(X_H) + mean_grid,time[(mean_seed X_H - mean_seed R)^2]; finite-reference target, not continuum error",
            }
            rows.append(row)
    density = {row["arm"]: row for row in rows if row["metric"] == "density_error_power"}
    if not math.isclose(density["c4_sigma1"]["adjusted_cic_over_arm_ratio"], 7.218131520858337, rel_tol=2e-12):
        raise AssertionError("the approximate 7.2 adjusted ratio was not reproduced")
    if not math.isclose(density["lscv_job5008392"]["adjusted_cic_over_arm_ratio"], 36.8581291454626, rel_tol=2e-12):
        raise AssertionError("the approximate 36.9 adjusted ratio was not reproduced")
    write_csv(output / "landau_finite_reference_correction.csv", rows)
    return rows


def bootstrap_mean_pair(pred: np.ndarray, obs: np.ndarray, draws: np.ndarray) -> dict:
    pred_draw = pred[draws].mean(axis=1)
    obs_draw = obs[draws].mean(axis=1)
    pred_lo, pred_hi = interval(pred_draw)
    obs_lo, obs_hi = interval(obs_draw)
    residual_lo, residual_hi = interval(obs_draw - pred_draw)
    return {
        "predicted_mean": float(np.mean(pred)),
        "predicted_mean_ci_lo": pred_lo,
        "predicted_mean_ci_hi": pred_hi,
        "fitted_individual_rate_mean": float(np.mean(obs)),
        "fitted_mean_ci_lo": obs_lo,
        "fitted_mean_ci_hi": obs_hi,
        "observed_minus_predicted_mean": float(np.mean(obs - pred)),
        "residual_ci_lo": residual_lo,
        "residual_ci_hi": residual_hi,
    }


def build_e2_tables(output: Path, rng: np.random.Generator, n_boot: int) -> tuple[list[dict], list[dict]]:
    marker = read_csv(MARKER)
    registered = read_csv(ESTIMANDS)
    current = [
        row for row in marker
        if row["source_status"] == "EXACT_CURRENT_AUDITED_INITIALIZER" and f(row, "v_beam") <= 0.8
    ]
    summary: list[dict] = []
    ratios: list[dict] = []
    for vb in (0.60, 0.70, 0.76, 0.80):
        arm_arrays: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for arm in ("cic", "fixed_gaussian", "constrained_gaussian"):
            group = sorted(
                [row for row in current if abs(f(row, "v_beam") - vb) < 1e-12 and row["arm"] == arm],
                key=lambda row: int(row["seed"]),
            )
            pred = np.asarray([f(row, "finite_marker_diagonal_gamma") for row in group])
            obs = np.asarray([f(row, "observed_individual_gamma") for row in group])
            draws = rng.integers(0, len(group), size=(n_boot, len(group)))
            summary.append(
                {
                    "v_beam": vb,
                    "arm": arm,
                    "n_seeds": len(group),
                    **bootstrap_mean_pair(pred, obs, draws),
                    "predictor": "initialization-only principal-mode diagonal finite-marker midpoint response",
                    "predictor_limit": group[0]["approximation_limit"],
                    "measurement": "mean of individual fitted rates on frozen window",
                }
            )
            arm_arrays[arm] = (pred, obs)
        for candidate in ("fixed_gaussian", "constrained_gaussian"):
            pred0, obs0 = arm_arrays["cic"]
            pred1, obs1 = arm_arrays[candidate]
            draws = rng.integers(0, pred0.size, size=(n_boot, pred0.size))
            pred_ratio_draw = pred1[draws].mean(axis=1) / pred0[draws].mean(axis=1)
            obs_ratio_draw = obs1[draws].mean(axis=1) / obs0[draws].mean(axis=1)
            pred_lo, pred_hi = interval(pred_ratio_draw)
            obs_lo, obs_hi = interval(obs_ratio_draw)
            reg = next(
                row for row in registered
                if row["campaign"] == "registered_drift_scan_5201231"
                and abs(f(row, "v_beam") - vb) < 1e-12 and row["candidate"] == candidate
            )
            ratios.append(
                {
                    "v_beam": vb,
                    "candidate": candidate,
                    "predicted_ratio_of_individual_rate_means": float(np.mean(pred1) / np.mean(pred0)),
                    "predicted_ratio_ci_lo": pred_lo,
                    "predicted_ratio_ci_hi": pred_hi,
                    "measured_ratio_of_individual_rate_means": f(reg, "ratio_of_individual_rate_means"),
                    "measured_ratio_ci_lo": f(reg, "individual_rate_paired_ci_lo"),
                    "measured_ratio_ci_hi": f(reg, "individual_rate_paired_ci_hi"),
                    "registered_ensemble_amplitude_ratio": f(reg, "registered_mean_amplitude_ratio"),
                    "registered_ci_lo": f(reg, "registered_paired_ci_lo"),
                    "registered_ci_hi": f(reg, "registered_paired_ci_hi"),
                    "registered_status": "preserved original observable and verdict",
                    "individual_analysis_status": "post-hoc scientifically motivated sensitivity",
                    "arm_role": "theory-boundary control" if candidate == "fixed_gaussian" else "R=0.95 interior constrained setting",
                }
            )
    write_csv(output / "e2_individual_prediction_measurement.csv", summary)
    write_csv(output / "e2_estimand_ratios_like_for_like.csv", ratios)

    conventions: list[dict] = []
    scopes = {
        "current resolved, all arm/seed rows": current,
        "current v_b=0.80, all arm/seed rows": [row for row in current if abs(f(row, "v_beam") - 0.8) < 1e-12],
        "historical nominal, all arm/seed rows": [row for row in marker if row["source_status"] != "EXACT_CURRENT_AUDITED_INITIALIZER"],
        "current plus historical rows": current + [row for row in marker if row["source_status"] != "EXACT_CURRENT_AUDITED_INITIALIZER"],
    }
    for name, rows in scopes.items():
        obs = np.asarray([f(row, "observed_individual_gamma") for row in rows])
        equilibrium = np.asarray([f(row, "continuum_velocity_grid_time_discrete_gamma") for row in rows])
        finite = np.asarray([f(row, "finite_marker_diagonal_gamma") for row in rows])
        eq_res = obs - equilibrium
        fm_res = obs - finite
        conventions.append(
            {
                "scope": name,
                "averaging_convention": "each arm/seed row weighted equally",
                "n_rows": len(rows),
                "grid_time_equilibrium_rmse": float(np.sqrt(np.mean(eq_res ** 2))),
                "finite_marker_rmse": float(np.sqrt(np.mean(fm_res ** 2))),
                "rmse_agreement_factor": float(np.sqrt(np.mean(eq_res ** 2)) / np.sqrt(np.mean(fm_res ** 2))),
                "mae_agreement_factor": float(np.mean(np.abs(eq_res)) / np.mean(np.abs(fm_res))),
                "dependence": "arms share each seed loading; drift rows reuse paired loading templates; rows are not independent observations",
            }
        )
    for vb_scope in (None, 0.8):
        rows = current if vb_scope is None else [row for row in current if abs(f(row, "v_beam") - vb_scope) < 1e-12]
        grouped = []
        for key in sorted({(row["v_beam"], row["arm"]) for row in rows}):
            group = [row for row in rows if (row["v_beam"], row["arm"]) == key]
            grouped.append(
                tuple(np.mean([f(row, col) for row in group]) for col in
                      ("observed_individual_gamma", "continuum_velocity_grid_time_discrete_gamma", "finite_marker_diagonal_gamma"))
            )
        values = np.asarray(grouped)
        eq_res = values[:, 0] - values[:, 1]
        fm_res = values[:, 0] - values[:, 2]
        conventions.append(
            {
                "scope": "current resolved arm/drift means" if vb_scope is None else "current v_b=0.80 arm means",
                "averaging_convention": "first average ten seeds within each arm/drift, then weight arm/drift means equally",
                "n_rows": len(values),
                "grid_time_equilibrium_rmse": float(np.sqrt(np.mean(eq_res ** 2))),
                "finite_marker_rmse": float(np.sqrt(np.mean(fm_res ** 2))),
                "rmse_agreement_factor": float(np.sqrt(np.mean(eq_res ** 2)) / np.sqrt(np.mean(fm_res ** 2))),
                "mae_agreement_factor": float(np.mean(np.abs(eq_res)) / np.mean(np.abs(fm_res))),
                "dependence": "arm means share particles within drift; drift means reuse paired loading templates",
            }
        )
    write_csv(output / "e2_reported_agreement_improvement_check.csv", conventions)
    return summary, ratios


def configure_plots() -> None:
    import matplotlib as mpl
    mpl.rcParams.update({
        "font.family": "serif", "font.size": 9, "axes.titlesize": 9,
        "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "legend.fontsize": 7, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.25, "savefig.dpi": 300,
    })


def save_figure(fig, output: Path, stem: str) -> None:
    fig.savefig(output / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(output / f"{stem}.svg", bbox_inches="tight")
    fig.savefig(output / f"{stem}.png", dpi=300, bbox_inches="tight")


def make_figures(output: Path, floors: list[dict], plotted: dict[int, dict], e2_summary: list[dict], e2_ratios: list[dict]) -> None:
    configure_plots()
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.0, 3.2))
    colors = {1024: "#17365d", 4096: "#a61c2d"}
    for ppc in (1024, 4096):
        data = plotted[ppc]
        row = next(item for item in floors if item["ppc"] == ppc)
        ax.semilogy(data["times"], data["raw"], color=colors[ppc], alpha=0.28, lw=0.65)
        ax.semilogy(data["times"], data["envelope"], color=colors[ppc], lw=1.45, label=f"CIC, PPC {ppc}")
        ax.hlines(row["noise_floor"], 70, 90, color=colors[ppc], ls=":", lw=1.1)
        ax.plot(row["crossing_time_5x_floor"], 5 * row["noise_floor"], marker="o", mfc="white", mec=colors[ppc], ms=4)
        fit_mask = (data["times"] >= row["accepted_t_start"]) & (data["times"] <= row["accepted_t_stop"])
        intercept = np.polyfit(data["times"][fit_mask], np.log(data["envelope"][fit_mask]), 1)[1]
        ax.semilogy(data["times"][fit_mask], np.exp(intercept + row["fitted_gamma"] * data["times"][fit_mask]),
                    color=colors[ppc], ls="--", lw=1.0)
        ax.axvline(row["accepted_t_stop"], color=colors[ppc], lw=0.7, alpha=0.55)
    ax.axvspan(85, 90, color="#999999", alpha=0.14, label="selected late-floor interval")
    ax.text(86.0, 1.6e-3, "floor windows", color="#555555", fontsize=7)
    ax.set(xlim=(0, 90), ylim=(3e-5, 1.8e-2), xlabel=r"time $t\omega_p$",
           ylabel=r"$|\widehat E_x(m=1)|$", title="Weak-Landau CIC: identical coherent-floor reducer")
    ax.legend(frameon=False, ncol=3, loc="lower left")
    fig.tight_layout()
    save_figure(fig, output, "fig_landau_ppc_floor_comparison")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.35))
    colors = {"cic": "#17365d", "fixed_gaussian": "#b2182b", "constrained_gaussian": "#2166ac"}
    labels = {"cic": "CIC", "fixed_gaussian": "fixed boundary control", "constrained_gaussian": "R=0.95 interior"}
    raw = read_csv(MARKER)
    current = [row for row in raw if row["source_status"] == "EXACT_CURRENT_AUDITED_INITIALIZER" and f(row, "v_beam") <= 0.8]
    for arm in colors:
        group = [row for row in current if row["arm"] == arm]
        axes[0].scatter([f(row, "finite_marker_diagonal_gamma") for row in group],
                        [f(row, "observed_individual_gamma") for row in group],
                        s=13, alpha=0.72, color=colors[arm], label=labels[arm])
    axes[0].plot((0.08, 0.34), (0.08, 0.34), "--", color="#777777", lw=0.9)
    axes[0].set(xlim=(0.08, 0.34), ylim=(0.08, 0.34), xlabel="initialization-only predicted rate",
                ylabel="individual fitted rate", title="(a) Realised principal-mode rates")
    axes[0].legend(frameon=False)
    axes[1].axhspan(0.9, 1.1, color="#4daf4a", alpha=0.10)
    axes[1].text(0.594, 0.999, "rate-ratio tolerance", color="#287a2c", fontsize=7, va="top")
    for candidate, color, marker in (("fixed_gaussian", "#b2182b", "o"), ("constrained_gaussian", "#2166ac", "s")):
        group = sorted([row for row in e2_ratios if row["candidate"] == candidate], key=lambda row: row["v_beam"])
        x = np.asarray([row["v_beam"] for row in group])
        pred = np.asarray([row["predicted_ratio_of_individual_rate_means"] for row in group])
        obs = np.asarray([row["measured_ratio_of_individual_rate_means"] for row in group])
        olo = np.asarray([row["measured_ratio_ci_lo"] for row in group]); ohi = np.asarray([row["measured_ratio_ci_hi"] for row in group])
        reg = np.asarray([row["registered_ensemble_amplitude_ratio"] for row in group])
        rlo = np.asarray([row["registered_ci_lo"] for row in group]); rhi = np.asarray([row["registered_ci_hi"] for row in group])
        axes[1].plot(x, pred, ls="--", color=color, lw=1.0)
        axes[1].errorbar(x - 0.0015, obs, yerr=(obs - olo, ohi - obs), fmt=marker, color=color, ms=4,
                         capsize=2)
        axes[1].errorbar(x + 0.0015, reg, yerr=(reg - rlo, rhi - reg), fmt=marker, mfc="white", mec=color,
                         ecolor=color, ms=4, capsize=2)
    axes[1].set(xlim=(0.585, 0.812), ylim=(0.875, 1.01), xlabel=r"beam drift $v_b$",
                ylabel="filtered/CIC rate ratio", title="(b) Distinct rate estimands")
    from matplotlib.lines import Line2D
    estimator_handles = [
        Line2D([], [], color="#444444", ls="--", lw=1, label="initialization-only predicted means"),
        Line2D([], [], color="#444444", marker="o", ls="none", ms=4, label="mean individual rates (post hoc)"),
        Line2D([], [], color="#444444", marker="o", mfc="white", ls="none", ms=4, label="registered ensemble amplitude"),
        Line2D([], [], color="#b2182b", lw=2, label="fixed boundary control"),
        Line2D([], [], color="#2166ac", lw=2, label="R=0.95 interior"),
    ]
    fig.legend(handles=estimator_handles, frameon=False, fontsize=6.6, ncol=3,
               loc="lower center", bbox_to_anchor=(0.5, -0.005))
    fig.tight_layout(rect=(0, 0.16, 1, 1))
    save_figure(fig, output, "fig_e2_estimands_corrected")
    plt.close(fig)

    practical = read_csv(PRACTICAL)
    fig, ax = plt.subplots(figsize=(7.0, 3.25))
    provenance = {
        "same campaign/configuration and paired seeds": ("o", "same campaign"),
        "compatible settings but separate projected-rate/full-f-error campaigns": ("s", "separate compatible campaigns"),
        "separate historical projected-rate and full-f density-error campaigns": ("D", "separate historical campaigns"),
    }
    status_color = {
        "FAIL_CIC_DURATION": "#e08214",
        "DYNAMICS_NOT_PRESERVED": "#b2182b",
        "RATE_PRESERVED; MODEST_GRID_ERROR_BENEFIT; NOT_END_TO_END": "#2166ac",
        "RATE_PRESERVED_IN_PROJECTED_TEST; NOT_SAME_CAMPAIGN_PRACTICAL_PROOF": "#5e3c99",
    }
    short = {
        "weak Landau C4 sigma=1": r"Landau C4 $\sigma=1$†",
        "weak Landau fixed CV-derived 7.2443": "Landau fixed CV-derived 7.2443",
        "E2 Gaussian R=0.95 interior": "E2 Gaussian R=0.95",
        "E2 compensated p2n2": "E2 compensated p2/n2",
        "E2 compensated p3n4": "E2 compensated p3/n4",
    }
    label_positions = {
        "weak Landau C4 sigma=1": (3.34, 1.18),
        "weak Landau fixed CV-derived 7.2443": (4.70, 1.84),
        "E2 Gaussian R=0.95 interior": (1.18, 0.86),
        "E2 compensated p2n2": (2.42, 0.82),
        "E2 compensated p3n4": (3.72, 1.10),
    }
    for index, row in enumerate(practical):
        x, y = f(row, "density_error_power_reduction"), f(row, "rate_ratio")
        xlo, xhi = f(row, "noise_ci_lo"), f(row, "noise_ci_hi")
        ylo, yhi = f(row, "rate_ci_lo"), f(row, "rate_ci_hi")
        marker, _ = provenance[row["comparison_type"]]
        color = status_color[row["scientific_status"]]
        ax.errorbar(x, y, xerr=((x - xlo,), (xhi - x,)), yerr=((y - ylo,), (yhi - y,)),
                    fmt=marker, mfc=color, mec=color, ecolor=color, ms=5, capsize=2, lw=0.9)
        ax.annotate(short[row["label"]], (x, y), xytext=label_positions[row["label"]], textcoords="data", fontsize=7,
                    arrowprops={"arrowstyle": "-", "lw": 0.5, "color": color})
    ax.axhspan(0.90, 1.10, color="#4daf4a", alpha=0.10)
    ax.axhline(1.0, color="#777777", ls="--", lw=0.8)
    ax.text(2.05, 1.105, "rate-ratio tolerance only", color="#287a2c", fontsize=7, va="bottom")
    for relation, (marker, label) in provenance.items():
        ax.plot([], [], marker=marker, color="#444444", ls="none", ms=5, label=label)
    ax.plot([], [], marker="o", color="#e08214", ls="none", label="duration limitation")
    ax.plot([], [], marker="o", color="#b2182b", ls="none", label="rate not preserved")
    ax.plot([], [], marker="o", color="#2166ac", ls="none", label="conditional/split evidence")
    ax.plot([], [], marker="o", color="#5e3c99", ls="none", label="historical split evidence")
    ax.text(0.86, 0.775, "† CIC fitted duration: 1.721 e-foldings (<2)", fontsize=7, color="#9a5c00")
    ax.set(xlim=(0.75, 4.90), ylim=(0.75, 2.19), xlabel="density-error-power reduction (CIC/candidate)",
           ylabel="filtered/CIC rate ratio", title="Practical evidence: provenance and validation status kept separate")
    ax.legend(frameon=False, ncol=2, fontsize=6.4, loc="upper left")
    fig.tight_layout()
    save_figure(fig, output, "fig_practical_benefit_vs_rate_corrected")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=10000)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.mkdir(parents=True)
    config = json.loads(LANDAU_CONFIG.read_text(encoding="utf-8"))
    rng = np.random.default_rng(20260914)
    floors, plotted = build_floor_comparison(output, config)
    corrections = build_reference_correction(output, rng, args.bootstrap)
    e2_summary, e2_ratios = build_e2_tables(output, rng, args.bootstrap)
    make_figures(output, floors, plotted, e2_summary, e2_ratios)
    inputs = [LANDAU_CONFIG, MARKER, ESTIMANDS, PRACTICAL]
    inputs.extend(archive_path(ppc, seed, arm) for ppc, arm in ((1024, "cic"), (1024, "c4_sigma1"), (1024, "lscv_job5008392"), (4096, "cic")) for seed in SEEDS)
    manifest = {
        "scope": "bounded existing-data correction; no simulations and no manuscript edits",
        "bootstrap_draws": args.bootstrap,
        "bootstrap_seed": 20260914,
        "inputs_sha256": {str(path.relative_to(DOWNLOADS)): sha256(path) for path in inputs},
        "outputs_sha256": {},
        "key_results": {
            "floor_rows": len(floors), "finite_reference_rows": len(corrections),
            "e2_individual_rows": len(e2_summary), "e2_ratio_rows": len(e2_ratios),
        },
    }
    for path in sorted(output.iterdir()):
        if path.name != "analysis_manifest.json":
            manifest["outputs_sha256"][path.name] = sha256(path)
    (output / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
