"""Censored ensemble fits for damped electrostatic modes.

The fitter operates on complex Fourier-mode histories. Paired seeds are
coherently averaged before taking the magnitude, which preserves the seeded
mode while cancelling incoherent particle noise. The fit is a contiguous
prefix terminated at the first persistent crossing of a measured late-time
noise threshold; recurrence points cannot re-enter the fit.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np


def coherent_amplitude(mode_histories: np.ndarray) -> np.ndarray:
    """Return ``|mean_seed(E_k)|`` for phase-aligned paired histories."""
    histories = np.asarray(mode_histories, dtype=np.complex128)
    if histories.ndim != 2 or histories.shape[0] == 0:
        raise ValueError("mode_histories must have shape (n_seeds, n_times)")
    return np.abs(np.mean(histories, axis=0))


def geometric_envelope(amplitude: np.ndarray, times: np.ndarray, period: float) -> tuple[np.ndarray, int]:
    """Average ``log|E_k|`` over one oscillation period.

    The moment-matched full-f initialization projects onto a small continuum
    component, producing cycle-scale beating in ``|E_k|``. A geometric moving
    mean removes that modulation without changing an exponential damping slope.
    """
    amplitude = np.asarray(amplitude, dtype=np.float64)
    times = np.asarray(times, dtype=np.float64)
    if len(times) < 3:
        raise ValueError("at least three diagnostics are required")
    dt = float(np.median(np.diff(times)))
    points = max(3, int(round(period / dt)))
    if points % 2 == 0:
        points += 1
    half = points // 2
    log_amplitude = np.log(np.maximum(amplitude, 1.0e-300))
    padded = np.pad(log_amplitude, (half, half), mode="edge")
    smoothed = np.convolve(padded, np.full(points, 1.0 / points), mode="valid")
    return np.exp(smoothed), points


def quiet_tail_floor(
    amplitude: np.ndarray,
    times: np.ndarray,
    tail_start: float = 70.0,
    tail_stop: float | None = None,
    window_width: float = 5.0,
    window_step: float = 1.0,
) -> dict:
    """Measure the minimum rolling-median tail floor.

    Taking the quietest tail window avoids treating a recurrence bump as the
    particle-noise floor. The selected window is returned for auditability.
    """
    amplitude = np.asarray(amplitude, dtype=np.float64)
    times = np.asarray(times, dtype=np.float64)
    stop = float(times[-1]) if tail_stop is None else min(float(tail_stop), float(times[-1]))
    candidates: list[tuple[float, float, float]] = []
    for start in np.arange(tail_start, stop - window_width + 0.5 * window_step, window_step):
        mask = (times >= start) & (times <= start + window_width) & np.isfinite(amplitude)
        if np.count_nonzero(mask) >= 3:
            candidates.append((float(np.median(amplitude[mask])), float(start), float(start + window_width)))
    if not candidates:
        raise ValueError(
            f"no complete floor window in [{tail_start}, {stop}]; extend the simulation tail"
        )
    floor, start, end = min(candidates, key=lambda item: item[0])
    return {"noise_floor": floor, "floor_t_start": start, "floor_t_stop": end}


def _first_persistent_crossing(
    amplitude: np.ndarray,
    candidate_indices: np.ndarray,
    threshold: float,
    persistence_points: int,
) -> int | None:
    below = amplitude[candidate_indices] <= threshold
    if persistence_points <= 1:
        matches = np.flatnonzero(below)
    else:
        run = np.convolve(below.astype(np.int64), np.ones(persistence_points, dtype=np.int64), mode="valid")
        matches = np.flatnonzero(run == persistence_points)
    if matches.size == 0:
        return None
    return int(candidate_indices[int(matches[0])])


def fit_censored_mode(
    mode_histories: np.ndarray,
    times: np.ndarray,
    fit_t_start: float = 5.0,
    fit_t_stop_max: float = 30.0,
    noise_floor_start: float = 70.0,
    noise_floor_stop: float | None = 90.0,
    noise_factor: float = 5.0,
    envelope_period: float = 4.889,
    persistence_points: int = 5,
    min_points: int = 30,
    min_duration: float = 5.0,
    min_model_efoldings: float = 0.5,
    min_r_squared: float = 0.75,
) -> dict:
    """Fit ``log|E_k|`` on a contiguous, floor-censored damping interval."""
    times = np.asarray(times, dtype=np.float64)
    raw_amplitude = coherent_amplitude(mode_histories)
    if raw_amplitude.shape != times.shape:
        raise ValueError("mode history time dimension does not match times")
    amplitude, envelope_points = geometric_envelope(raw_amplitude, times, envelope_period)
    floor_info = quiet_tail_floor(
        amplitude,
        times,
        tail_start=noise_floor_start,
        tail_stop=noise_floor_stop,
    )
    floor = float(floor_info["noise_floor"])
    threshold = noise_factor * floor
    candidates = np.flatnonzero(
        (times >= fit_t_start)
        & (times <= fit_t_stop_max)
        & np.isfinite(amplitude)
        & (amplitude > 0.0)
    )
    if candidates.size == 0:
        raise ValueError("no candidate fit points")

    crossing = _first_persistent_crossing(amplitude, candidates, threshold, persistence_points)
    floor_censored = crossing is not None
    if crossing is None:
        fit_indices = candidates
        crossing_time = float("nan")
    else:
        crossing_time = float(times[crossing])
        # The centered envelope contains half a window of future samples.
        # Stop before that look-ahead can mix floor samples into the fit.
        protected_stop = crossing - envelope_points // 2
        fit_indices = candidates[candidates < protected_stop]

    result = {
        **floor_info,
        "threshold": threshold,
        "envelope_period": envelope_period,
        "envelope_points": envelope_points,
        "floor_censored": floor_censored,
        "censor_t": crossing_time,
        "n_points": int(fit_indices.size),
        "fit_t_start": float("nan"),
        "fit_t_stop": float("nan"),
        "fit_duration": 0.0,
        "gamma": float("nan"),
        "r_squared": float("nan"),
        "model_efoldings": 0.0,
        "observed_log_drop": float("nan"),
        "initial_snr": float(amplitude[0] / max(floor, 1.0e-300)),
        "fit_start_snr": float("nan"),
        "quality_pass": False,
        "status": "censored-before-fit",
    }
    if fit_indices.size < 2:
        return result

    x = times[fit_indices]
    y = np.log(amplitude[fit_indices])
    coefficients = np.polyfit(x, y, 1)
    predicted = np.polyval(coefficients, x)
    residual = float(np.sum((y - predicted) ** 2))
    total = float(np.sum((y - np.mean(y)) ** 2))
    gamma = float(coefficients[0])
    r_squared = float(1.0 - residual / total) if total > 0.0 else float("nan")
    duration = float(x[-1] - x[0])
    model_efoldings = float(max(0.0, -gamma * duration))
    observed_drop = float(y[0] - y[-1])
    quality = bool(
        gamma < 0.0
        and fit_indices.size >= min_points
        and duration >= min_duration
        and model_efoldings >= min_model_efoldings
        and np.isfinite(r_squared)
        and r_squared >= min_r_squared
    )
    result.update(
        {
            "fit_t_start": float(x[0]),
            "fit_t_stop": float(x[-1]),
            "fit_duration": duration,
            "gamma": gamma,
            "r_squared": r_squared,
            "model_efoldings": model_efoldings,
            "observed_log_drop": observed_drop,
            "fit_start_snr": float(amplitude[fit_indices[0]] / max(floor, 1.0e-300)),
            "quality_pass": quality,
            "status": "clean" if quality else "fit-quality-failed",
        }
    )
    return result


def paired_bootstrap_ratio(
    baseline_histories: np.ndarray,
    current_histories: np.ndarray,
    times: np.ndarray,
    n_boot: int = 1000,
    seed: int = 12345,
    **fit_kwargs,
) -> dict:
    """Bootstrap paired damping-rate ratios with common seed resampling."""
    baseline = np.asarray(baseline_histories, dtype=np.complex128)
    current = np.asarray(current_histories, dtype=np.complex128)
    if baseline.shape != current.shape:
        raise ValueError("paired histories must have identical shapes")
    rng = np.random.default_rng(seed)
    ratios: list[float] = []
    gammas: list[float] = []
    quality_pairs = 0
    for _ in range(n_boot):
        indices = rng.integers(0, baseline.shape[0], size=baseline.shape[0])
        base_fit = fit_censored_mode(baseline[indices], times, **fit_kwargs)
        current_fit = fit_censored_mode(current[indices], times, **fit_kwargs)
        gamma0 = float(base_fit["gamma"])
        gamma = float(current_fit["gamma"])
        if np.isfinite(gamma0) and gamma0 < 0.0 and np.isfinite(gamma) and gamma < 0.0:
            ratios.append(gamma / gamma0)
            gammas.append(gamma)
            quality_pairs += int(base_fit["quality_pass"] and current_fit["quality_pass"])
    if not ratios:
        return {
            "ratio_sem": float("nan"),
            "ratio_ci_lo": float("nan"),
            "ratio_ci_hi": float("nan"),
            "gamma_ci_lo": float("nan"),
            "gamma_ci_hi": float("nan"),
            "bootstrap_valid_fraction": 0.0,
            "bootstrap_quality_fraction": 0.0,
        }
    ratio_values = np.asarray(ratios, dtype=np.float64)
    gamma_values = np.asarray(gammas, dtype=np.float64)
    return {
        "ratio_sem": float(np.std(ratio_values)),
        "ratio_ci_lo": float(np.percentile(ratio_values, 2.5)),
        "ratio_ci_hi": float(np.percentile(ratio_values, 97.5)),
        "gamma_ci_lo": float(np.percentile(gamma_values, 2.5)),
        "gamma_ci_hi": float(np.percentile(gamma_values, 97.5)),
        "bootstrap_valid_fraction": len(ratios) / float(n_boot),
        "bootstrap_quality_fraction": quality_pairs / float(n_boot),
    }
