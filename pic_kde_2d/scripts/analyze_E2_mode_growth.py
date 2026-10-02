#!/usr/bin/env python3
"""Analyze the frozen-CIC E2 growth-rate sweep over modes 1 through 14."""

from __future__ import annotations

import argparse
import csv
import json
import math
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from pic_kde_2d.validation.landau_fit import fit_gamma_ensemble_fixed_window
from validate_E2_warm_surrogate import explicit_sinc, warm_growth_rate


FROZEN_ANCHOR_M12 = 0.15162775514242874

# Frozen after inspection of the 30-seed ensemble traces.  Modes with three or
# more resolved oscillatory maxima use a peak-envelope fit; modes whose growing
# branch dominates use a direct late-time linear window.  This avoids fitting
# the cancellation-zero recovery that biased the first automatic pass high.
PEAK_ENVELOPE_MODES = {1, 2, 3, 4, 5, *range(13, 65)}
INSPECTED_DIRECT_WINDOWS = {
    6: (6.5, 10.5),
    7: (6.0, 10.0),
    8: (5.5, 10.0),
    9: (5.0, 11.0),
    10: (5.0, 11.5),
    11: (5.5, 11.5),
    12: (6.5, 12.5),
}


def linear_fit(times: np.ndarray, log_amp: np.ndarray, start: int, stop: int) -> dict:
    x = times[start:stop]
    y = log_amp[start:stop]
    coeff = np.polyfit(x, y, 1)
    fitted = np.polyval(coeff, x)
    residual = float(np.sum((y - fitted) ** 2))
    total = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - residual / total if total > 0.0 else float("nan")
    duration = float(x[-1] - x[0])
    return {
        "gamma": float(coeff[0]),
        "r2": r2,
        "efoldings": float(coeff[0] * duration),
        "duration": duration,
        "start": start,
        "stop": stop,
        "t_start": float(x[0]),
        "t_stop": float(x[-1]),
        "n_points": int(x.size),
    }


def moving_average(values: np.ndarray, width: int = 5) -> np.ndarray:
    if width <= 1:
        return values.copy()
    pad_left = width // 2
    pad_right = width - 1 - pad_left
    padded = np.pad(values, (pad_left, pad_right), mode="edge")
    return np.convolve(padded, np.ones(width) / width, mode="valid")


def find_envelope_peaks(times: np.ndarray, amplitude: np.ndarray) -> np.ndarray:
    """Find separated maxima of the smoothed log-amplitude envelope."""
    smooth = moving_average(np.log(np.maximum(amplitude, 1.0e-300)), width=7)
    candidates = np.flatnonzero((smooth[1:-1] > smooth[:-2]) & (smooth[1:-1] >= smooth[2:])) + 1
    peaks: list[int] = []
    min_separation = max(1, int(round(1.0 / float(np.median(np.diff(times))))))
    for index in candidates:
        if times[index] < 0.3:
            continue
        if not peaks or index - peaks[-1] >= min_separation:
            peaks.append(int(index))
        elif smooth[index] > smooth[peaks[-1]]:
            peaks[-1] = int(index)
    return np.asarray(peaks, dtype=np.int64)


def fit_peak_envelope(times: np.ndarray, amplitude: np.ndarray, peaks: np.ndarray) -> dict:
    if peaks.size < 3:
        return {"gamma": float("nan"), "r2": float("nan"), "efoldings": 0.0,
                "n_points": int(peaks.size), "t_start": float("nan"), "t_stop": float("nan"),
                "window_strategy": "peak_envelope_unavailable"}
    x = times[peaks]
    y = np.log(np.maximum(amplitude[peaks], 1.0e-300))
    coeff = np.polyfit(x, y, 1)
    fitted = np.polyval(coeff, x)
    total = float(np.sum((y - np.mean(y)) ** 2))
    residual = float(np.sum((y - fitted) ** 2))
    r2 = 1.0 - residual / total if total > 0.0 else float("nan")
    return {
        "gamma": float(coeff[0]),
        "r2": r2,
        "efoldings": float(coeff[0] * (x[-1] - x[0])),
        "n_points": int(peaks.size),
        "t_start": float(x[0]),
        "t_stop": float(x[-1]),
        "window_strategy": "inspected_peak_envelope",
    }


def peak_envelope_seed_rates(
    amplitudes: np.ndarray,
    times: np.ndarray,
    peaks: np.ndarray,
    half_width: int = 3,
) -> np.ndarray:
    """Fit each seed at local maxima near the frozen ensemble peak times."""
    rates = []
    for amplitude in amplitudes:
        peak_values = []
        peak_times = []
        for peak in peaks:
            lo = max(0, int(peak) - half_width)
            hi = min(amplitude.size, int(peak) + half_width + 1)
            local = lo + int(np.argmax(amplitude[lo:hi]))
            peak_values.append(float(amplitude[local]))
            peak_times.append(float(times[local]))
        rates.append(float(np.polyfit(peak_times, np.log(np.maximum(peak_values, 1.0e-300)), 1)[0]))
    return np.asarray(rates, dtype=np.float64)


def bootstrap_seed_rate_mean(seed_rates: np.ndarray, n_boot: int, seed: int) -> dict:
    finite = seed_rates[np.isfinite(seed_rates)]
    if finite.size == 0:
        return {"sem": float("nan"), "ci_lo": float("nan"), "ci_hi": float("nan")}
    rng = np.random.default_rng(seed)
    values = np.asarray(
        [np.mean(rng.choice(finite, size=finite.size, replace=True)) for _ in range(n_boot)],
        dtype=np.float64,
    )
    return {
        "sem": float(np.std(values)),
        "ci_lo": float(np.percentile(values, 2.5)),
        "ci_hi": float(np.percentile(values, 97.5)),
    }


def select_growth_window(
    times: np.ndarray,
    amplitude: np.ndarray,
    t_min: float,
    t_start_max: float,
    t_max: float,
    min_duration: float,
    min_points: int,
    target_r2: float,
    saturation_fraction: float,
) -> dict:
    """Select the longest high-linearity pre-saturation growth interval.

    The stop search is capped before the ensemble amplitude reaches a fixed
    fraction of its post-transient maximum.  Among all candidate intervals,
    the fit capturing the most e-foldings at R^2 >= target_r2 is selected.
    """
    amp = np.asarray(amplitude, dtype=np.float64)
    log_amp = np.log(np.maximum(amp, 1.0e-300))
    smooth_log = moving_average(log_amp, width=5)
    valid = np.flatnonzero((times >= t_min) & (times <= t_max))
    if valid.size < min_points:
        return {"gamma": float("nan"), "r2": float("nan"), "efoldings": 0.0,
                "n_points": 0, "t_start": float("nan"), "t_stop": float("nan"),
                "window_strategy": "unavailable"}

    search_start = int(valid[0])
    search_stop = int(valid[-1]) + 1
    post_amp = amp[search_start:search_stop]
    peak_local = int(np.argmax(post_amp))
    peak_index = search_start + peak_local
    saturation_level = saturation_fraction * float(np.max(post_amp))
    crossing = np.flatnonzero(amp[search_start : peak_index + 1] >= saturation_level)
    saturation_index = search_stop
    if crossing.size:
        saturation_index = search_start + int(crossing[0]) + 1
    saturation_index = max(saturation_index, search_start + min_points)

    # Position/velocity loading excites stable oscillatory branches as well as
    # the growing branch.  Do not fit the steep recovery from an early
    # cancellation zero.  Start only after the last meaningful trough that
    # still leaves a measurable rise before saturation.
    derivative = np.gradient(smooth_log, times)
    minima = np.flatnonzero((derivative[:-1] <= 0.0) & (derivative[1:] > 0.0)) + 1
    meaningful_minima = [
        int(index)
        for index in minima
        if index >= search_start
        and index + min_points <= saturation_index
        and smooth_log[saturation_index - 1] - smooth_log[index] >= 0.5
    ]
    asymptotic_start = search_start
    if meaningful_minima:
        asymptotic_start = meaningful_minima[-1] + 3

    start_indices = np.flatnonzero(
        (times >= t_min) & (times <= t_start_max) & (np.arange(times.size) >= asymptotic_start)
    )
    candidates: list[dict] = []
    for start in start_indices:
        first_stop = start + min_points
        for stop in range(first_stop, min(saturation_index, search_stop) + 1):
            if times[stop - 1] - times[start] < min_duration:
                continue
            fit = linear_fit(times, smooth_log, int(start), int(stop))
            if fit["gamma"] > 0.0 and np.isfinite(fit["r2"]):
                candidates.append(fit)

    if not candidates:
        return {"gamma": float("nan"), "r2": float("nan"), "efoldings": 0.0,
                "n_points": 0, "t_start": float("nan"), "t_stop": float("nan"),
                "window_strategy": "unavailable"}

    clean = [fit for fit in candidates if fit["r2"] >= target_r2]
    if clean:
        selected = max(clean, key=lambda fit: (fit["efoldings"], fit["duration"], fit["r2"]))
        selected["window_strategy"] = "adaptive_pre_saturation_r2"
    else:
        selected = max(candidates, key=lambda fit: (fit["r2"], fit["efoldings"]))
        selected["window_strategy"] = "best_available_short_window"
    selected["saturation_t_cap"] = float(times[min(saturation_index - 1, times.size - 1)])
    selected["saturation_amplitude"] = saturation_level
    selected["asymptotic_start_floor"] = float(times[asymptotic_start])
    return selected


def fixed_window_seed_rates(
    amplitudes: np.ndarray,
    times: np.ndarray,
    t_start: float,
    t_stop: float,
) -> np.ndarray:
    mask = (times >= t_start) & (times <= t_stop)
    if np.count_nonzero(mask) < 5:
        return np.full(amplitudes.shape[0], np.nan)
    rates = []
    for amplitude in amplitudes:
        rates.append(float(np.polyfit(times[mask], np.log(np.maximum(amplitude[mask], 1.0e-300)), 1)[0]))
    return np.asarray(rates, dtype=np.float64)


def bootstrap_ensemble_rate(
    amplitudes: np.ndarray,
    times: np.ndarray,
    t_start: float,
    t_stop: float,
    n_boot: int,
    seed: int,
) -> dict:
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n_boot):
        indices = rng.integers(0, amplitudes.shape[0], size=amplitudes.shape[0])
        fit = fit_gamma_ensemble_fixed_window(
            [row for row in amplitudes[indices]],
            times,
            t_fit_start=t_start,
            t_fit_stop=t_stop,
            amp_floor=1.0e-30,
        )
        if np.isfinite(fit["gamma"]):
            values.append(float(fit["gamma"]))
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {"sem": float("nan"), "ci_lo": float("nan"), "ci_hi": float("nan")}
    return {
        "sem": float(np.std(array)),
        "ci_lo": float(np.percentile(array, 2.5)),
        "ci_hi": float(np.percentile(array, 97.5)),
    }


def warm_prediction(mode: int, Nx: int, Lx: float, v0: float, vth: float) -> float:
    k = 2.0 * math.pi * mode / Lx
    dx = Lx / Nx
    s_cic = float(explicit_sinc(k * dx / 2.0)) ** 2
    coupling = s_cic**2
    return warm_growth_rate(k, v0, vth, coupling)


def contiguous_component(modes: list[int], anchor: int) -> list[int]:
    if anchor not in modes:
        return []
    selected = {anchor}
    value = anchor - 1
    while value in modes:
        selected.add(value)
        value -= 1
    value = anchor + 1
    while value in modes:
        selected.add(value)
        value += 1
    return sorted(selected)


def write_csv(path: pathlib.Path, rows: list[dict]) -> None:
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--modes", default="1,2,3,4,5,6,7,8,9,10,11,12,13,14")
    parser.add_argument("--t-min", type=float, default=0.3)
    parser.add_argument("--t-start-max", type=float, default=8.0)
    parser.add_argument("--t-max", type=float, default=15.0)
    parser.add_argument("--min-duration", type=float, default=1.5)
    parser.add_argument("--min-points", type=int, default=16)
    parser.add_argument("--target-r2", type=float, default=0.98)
    parser.add_argument("--min-efoldings", type=float, default=2.0)
    parser.add_argument("--saturation-fraction", type=float, default=0.90)
    parser.add_argument("--nboot", type=int, default=2000)
    parser.add_argument("--anchor-tolerance", type=float, default=0.10)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    modes = [int(value) for value in args.modes.split(",") if value.strip()]
    rows: list[dict] = []
    trace_data: dict[int, tuple[np.ndarray, np.ndarray, dict]] = {}
    metadata: dict | None = None

    for mode in modes:
        path = args.input / f"mode_{mode:02d}_amplitudes.npz"
        with np.load(path) as data:
            times = np.asarray(data["times"], dtype=np.float64)
            amplitudes = np.asarray(data["amplitudes"], dtype=np.float64)
            mode_metadata = {
                key: data[key].item()
                for key in ("Nx", "Ny", "Lx", "Ly", "ppc", "dt", "n_steps", "diag_stride", "v_beam", "v_thermal", "alpha")
            }
        if metadata is None:
            metadata = mode_metadata
        elif mode_metadata != metadata:
            raise RuntimeError(f"frozen parameters changed at mode {mode}")

        ensemble = np.mean(amplitudes, axis=0)
        if mode in PEAK_ENVELOPE_MODES:
            peaks = find_envelope_peaks(times, ensemble)
            window = fit_peak_envelope(times, ensemble, peaks)
            window["_peak_indices"] = peaks
            seed_rates = peak_envelope_seed_rates(amplitudes, times, peaks) if peaks.size >= 3 else np.full(amplitudes.shape[0], np.nan)
            ensemble_gamma = float(window["gamma"])
            boot = bootstrap_seed_rate_mean(seed_rates, args.nboot, seed=81000 + mode)
        elif mode in INSPECTED_DIRECT_WINDOWS:
            direct_start, direct_stop = INSPECTED_DIRECT_WINDOWS[mode]
            mask = (times >= direct_start) & (times <= direct_stop)
            start_index = int(np.flatnonzero(mask)[0])
            stop_index = int(np.flatnonzero(mask)[-1]) + 1
            window = linear_fit(times, np.log(np.maximum(ensemble, 1.0e-300)), start_index, stop_index)
            window["window_strategy"] = "inspected_direct_asymptotic"
            seed_rates = fixed_window_seed_rates(amplitudes, times, direct_start, direct_stop)
            ensemble_gamma = float(window["gamma"])
            boot = bootstrap_ensemble_rate(
                amplitudes,
                times,
                direct_start,
                direct_stop,
                args.nboot,
                seed=81000 + mode,
            )
        else:
            window = select_growth_window(
                times,
                ensemble,
                args.t_min,
                args.t_start_max,
                args.t_max,
                args.min_duration,
                args.min_points,
                args.target_r2,
                args.saturation_fraction,
            )
            seed_rates = fixed_window_seed_rates(amplitudes, times, window["t_start"], window["t_stop"])
            ensemble_gamma = float(window["gamma"])
            boot = bootstrap_ensemble_rate(
                amplitudes,
                times,
                window["t_start"],
                window["t_stop"],
                args.nboot,
                seed=81000 + mode,
            )
        finite_rates = seed_rates[np.isfinite(seed_rates)]
        gamma_mean = float(np.mean(finite_rates)) if finite_rates.size else float("nan")
        gamma_std = float(np.std(finite_rates, ddof=1)) if finite_rates.size > 1 else float("nan")
        gamma_sem = gamma_std / math.sqrt(finite_rates.size) if finite_rates.size > 1 else float("nan")
        significant_growth = bool(np.isfinite(boot["ci_lo"]) and boot["ci_lo"] > 0.0)
        enough_fit_support = bool(
            window["n_points"] >= (4 if window["window_strategy"] == "inspected_peak_envelope" else 5)
        )
        if not significant_growth or not np.isfinite(window["r2"]) or window["r2"] < 0.90:
            flag = "unresolved"
        elif window["r2"] >= args.target_r2 and window["efoldings"] >= args.min_efoldings and enough_fit_support:
            flag = "clean"
        else:
            flag = "short-window"

        Nx = int(metadata["Nx"])
        Lx = float(metadata["Lx"])
        k = 2.0 * math.pi * mode / Lx
        predicted = warm_prediction(mode, Nx, Lx, float(metadata["v_beam"]), float(metadata["v_thermal"]))
        agreement_tolerance = max(gamma_std if np.isfinite(gamma_std) else 0.0, 0.15 * max(predicted, 1.0e-12))
        rows.append(
            {
                "mode": mode,
                "k": k,
                "k_dx": k * Lx / Nx,
                "mode_over_Nx": mode / Nx,
                "gamma_mean_seed_fits": gamma_mean,
                "gamma_seed_std": gamma_std,
                "gamma_seed_sem": gamma_sem,
                "gamma_ensemble": ensemble_gamma,
                "gamma_bootstrap_ci_lo": boot["ci_lo"],
                "gamma_bootstrap_ci_hi": boot["ci_hi"],
                "gamma_warm_predicted": predicted,
                "warm_agrees_within_seed_spread_or_15pct": bool(abs(gamma_mean - predicted) <= agreement_tolerance),
                "fit_t_start": window["t_start"],
                "fit_t_stop": window["t_stop"],
                "fit_r2_ensemble": window["r2"],
                "fit_efoldings": window["efoldings"],
                "fit_n_points": window["n_points"],
                "fit_window_strategy": window["window_strategy"],
                "significant_growth_95pct": significant_growth,
                "flag": flag,
                "n_seeds": int(amplitudes.shape[0]),
            }
        )
        trace_data[mode] = (times, ensemble, window)

    assert metadata is not None
    row_by_mode = {int(row["mode"]): row for row in rows}
    finite_rows = [row for row in rows if np.isfinite(row["gamma_mean_seed_fits"])]
    if not finite_rows:
        raise RuntimeError("no mode produced a finite growth-rate fit")
    dominant = max(finite_rows, key=lambda row: row["gamma_mean_seed_fits"])
    unstable_candidates = [
        int(row["mode"])
        for row in rows
        if row["significant_growth_95pct"] and row["flag"] != "unresolved"
    ]
    unstable_band = contiguous_component(unstable_candidates, int(dominant["mode"]))
    high_edge_censored = bool(max(modes) in unstable_candidates)
    low_edge_unresolved = bool(row_by_mode[min(modes)]["flag"] == "unresolved")
    observation_time = float(metadata["n_steps"]) * float(metadata["dt"])
    gamma_cut = float(metadata["alpha"]) / observation_time
    alive_band = [
        int(row["mode"])
        for row in rows
        if row["significant_growth_95pct"]
        and row["flag"] != "unresolved"
        and row["gamma_mean_seed_fits"] >= gamma_cut
    ]
    unresolved_modes = [int(row["mode"]) for row in rows if row["flag"] == "unresolved"]
    harmonics = [
        harmonic
        for harmonic in (2 * int(dominant["mode"]), 3 * int(dominant["mode"]))
        if harmonic <= int(metadata["Nx"]) // 2
    ]
    protected_modes = sorted(set(alive_band + harmonics))

    m12_amplitudes_path = args.input / "mode_12_amplitudes.npz"
    with np.load(m12_amplitudes_path) as data:
        m12_times = np.asarray(data["times"], dtype=np.float64)
        m12_amplitudes = np.asarray(data["amplitudes"], dtype=np.float64)
    m12_legacy = fit_gamma_ensemble_fixed_window(
        [row for row in m12_amplitudes],
        m12_times,
        t_fit_start=0.5,
        t_fit_stop=4.0,
        amp_floor=1.0e-30,
    )["gamma"]
    legacy_mask = (m12_times >= 0.5) & (m12_times <= 4.0)
    legacy_y = np.log(np.maximum(np.mean(m12_amplitudes, axis=0)[legacy_mask], 1.0e-300))
    legacy_coeff = np.polyfit(m12_times[legacy_mask], legacy_y, 1)
    legacy_residual = float(np.sum((legacy_y - np.polyval(legacy_coeff, m12_times[legacy_mask])) ** 2))
    legacy_total = float(np.sum((legacy_y - np.mean(legacy_y)) ** 2))
    legacy_r2 = 1.0 - legacy_residual / legacy_total if legacy_total > 0.0 else float("nan")
    anchor_error = abs(float(m12_legacy) - FROZEN_ANCHOR_M12) / FROZEN_ANCHOR_M12

    predicted_unstable = [int(row["mode"]) for row in rows if row["gamma_warm_predicted"] > 0.0]
    predicted_dominant = max(rows, key=lambda row: row["gamma_warm_predicted"])
    dominant_ci_lo = float(dominant["gamma_bootstrap_ci_lo"])
    dominant_ci_hi = float(dominant["gamma_bootstrap_ci_hi"])
    fastest_tied_modes_95pct = [
        int(row["mode"])
        for row in finite_rows
        if float(row["gamma_bootstrap_ci_hi"]) >= dominant_ci_lo
        and float(row["gamma_bootstrap_ci_lo"]) <= dominant_ci_hi
    ]
    measured_vector = np.asarray([row["gamma_mean_seed_fits"] for row in finite_rows], dtype=np.float64)
    predicted_vector = np.asarray([row["gamma_warm_predicted"] for row in finite_rows], dtype=np.float64)
    scale = max(float(np.max(np.abs(measured_vector))), 1.0e-30)
    normalized_rmse = float(np.sqrt(np.mean((measured_vector - predicted_vector) ** 2)) / scale)
    theory_band_close = bool(
        unstable_band
        and predicted_unstable
        and abs(min(unstable_band) - min(predicted_unstable)) <= 1
        and abs(max(unstable_band) - max(predicted_unstable)) <= 1
    )
    theory_fastest_close = bool(int(predicted_dominant["mode"]) in fastest_tied_modes_95pct)
    theory_validated_for_band = bool(theory_band_close and theory_fastest_close and normalized_rmse <= 0.20)

    csv_path = args.output / "E2_gamma0_mode_sweep.csv"
    write_csv(csv_path, rows)
    summary = {
        "configuration": metadata,
        "deposition": "plain CIC deposit and matched CIC gather; sigma=0; no grid post-filter",
        "gamma_cut_alpha_over_Tobs": gamma_cut,
        "measured_fastest_mode": int(dominant["mode"]),
        "measured_fastest_gamma": float(dominant["gamma_mean_seed_fits"]),
        "measured_fastest_modes_tied_95pct": fastest_tied_modes_95pct,
        "measured_unstable_band_95pct_contiguous": unstable_band,
        "unstable_band_high_edge_right_censored": high_edge_censored,
        "unstable_band_low_edge_unresolved": low_edge_unresolved,
        "M_phys_status": "provisional; extend high-mode scan" if high_edge_censored else "measured within scan",
        "measured_dynamically_alive_modes": alive_band,
        "unresolved_modes": unresolved_modes,
        "harmonics_of_dominant_mode": harmonics,
        "M_phys": protected_modes,
        "anisotropy": "protect x modes in M_phys; no measured ky growth band",
        "m12_legacy_gamma_window_0p5_4p0": float(m12_legacy),
        "m12_legacy_fit_r2": legacy_r2,
        "m12_post_transient_gamma": float(row_by_mode[12]["gamma_mean_seed_fits"]),
        "m12_post_transient_flag": row_by_mode[12]["flag"],
        "m12_frozen_anchor": FROZEN_ANCHOR_M12,
        "m12_anchor_relative_error": anchor_error,
        "m12_anchor_pass_10pct": bool(anchor_error <= args.anchor_tolerance),
        "warm_predicted_fastest_mode": int(predicted_dominant["mode"]),
        "warm_predicted_fastest_gamma": float(predicted_dominant["gamma_warm_predicted"]),
        "warm_predicted_unstable_modes": predicted_unstable,
        "warm_normalized_rmse": normalized_rmse,
        "warm_fastest_mode_close": theory_fastest_close,
        "warm_band_close": theory_band_close,
        "warm_theory_validated_for_gamma0_band": theory_validated_for_band,
        "warm_smoothing_response_status": "separately invalid; this verdict applies only to unsmoothed gamma0(k)",
        "csv": str(csv_path),
    }
    with (args.output / "E2_gamma0_mode_sweep.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)

    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    mode_values = np.asarray([row["mode"] for row in rows])
    measured = np.asarray([row["gamma_mean_seed_fits"] for row in rows])
    spread = np.asarray([row["gamma_seed_std"] for row in rows])
    predicted = np.asarray([row["gamma_warm_predicted"] for row in rows])
    ax.errorbar(mode_values, measured, yerr=spread, fmt="o", capsize=3, color="#a33b20", label="PIC mean +/- seed spread")
    ax.plot(mode_values, predicted, "s--", color="#174a7c", label="warm dispersion (zero smoothing)")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.axvline(12, color="#666666", linestyle=":", linewidth=1.0, label="m=12 regression anchor")
    ax.set_xlabel("integer x mode m")
    ax.set_ylabel(r"growth rate $\gamma_0(m)$")
    ax.set_title("Frozen E2 CIC growth-rate band")
    ax.set_xticks(mode_values)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(args.output / "E2_gamma0_mode_sweep.png", dpi=220)
    plt.close(fig)

    ncols = 4
    nrows = int(math.ceil(len(modes) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(13.0, 2.5 * nrows), sharex=True)
    axes = np.atleast_1d(axes)
    for mode, axis in zip(modes, axes.flat):
        times, ensemble, window = trace_data[mode]
        axis.plot(times, np.log(np.maximum(ensemble, 1.0e-300)), color="#30343b", linewidth=1.0)
        mask = (times >= window["t_start"]) & (times <= window["t_stop"])
        if "_peak_indices" in window and len(window["_peak_indices"]) >= 3:
            peaks = np.asarray(window["_peak_indices"], dtype=np.int64)
            peak_y = np.log(np.maximum(ensemble[peaks], 1.0e-300))
            coeff = np.polyfit(times[peaks], peak_y, 1)
            axis.scatter(times[peaks], peak_y, color="#c23b22", s=10, zorder=3)
            axis.plot(times[mask], np.polyval(coeff, times[mask]), color="#c23b22", linewidth=1.6)
        elif np.count_nonzero(mask) >= 2:
            coeff = np.polyfit(times[mask], np.log(np.maximum(ensemble[mask], 1.0e-300)), 1)
            axis.plot(times[mask], np.polyval(coeff, times[mask]), color="#c23b22", linewidth=1.6)
        row = row_by_mode[mode]
        axis.set_title(f"m={mode}, {row['flag']}, R2={row['fit_r2_ensemble']:.3f}", fontsize=8)
    for axis in axes.flat[len(modes):]:
        axis.axis("off")
    fig.supxlabel("time")
    fig.supylabel(r"$\log |\hat E_x(m,0,t)|$")
    fig.suptitle("Per-mode adaptive linear-growth windows", y=0.995)
    fig.tight_layout()
    fig.savefig(args.output / "E2_mode_fit_windows.png", dpi=200)
    plt.close(fig)

    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    if not summary["m12_anchor_pass_10pct"]:
        print("M12 REGRESSION FAILED", flush=True)
        raise SystemExit(2)
    print("M12 REGRESSION PASS", flush=True)
    print("WARM GAMMA0 BAND THEORY " + ("VALIDATED" if theory_validated_for_band else "DISCARDED"), flush=True)


if __name__ == "__main__":
    main()
