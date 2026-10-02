"""Bounce-censored paired fits for finite-amplitude Landau damping."""

from __future__ import annotations

import numpy as np

from pic_kde_2d.validation.landau_censored_fit import (
    coherent_amplitude,
    geometric_envelope,
)


def rolling_log_slopes(
    amplitude: np.ndarray,
    times: np.ndarray,
    window: float,
) -> np.ndarray:
    """Return centered local slopes of ``log(amplitude)``."""
    amplitude = np.asarray(amplitude, dtype=np.float64)
    times = np.asarray(times, dtype=np.float64)
    slopes = np.full(times.shape, np.nan, dtype=np.float64)
    half = 0.5 * float(window)
    log_amplitude = np.log(np.maximum(amplitude, 1.0e-300))
    for index, center in enumerate(times):
        mask = (times >= center - half) & (times <= center + half)
        if np.count_nonzero(mask) >= 3:
            slopes[index] = np.polyfit(times[mask], log_amplitude[mask], 1)[0]
    return slopes


def detect_first_bounce(
    mode_histories: np.ndarray,
    times: np.ndarray,
    *,
    envelope_period: float,
    search_start: float = 8.0,
    search_stop: float = 35.0,
    slope_window: float = 2.0,
    persistence_points: int = 5,
) -> dict:
    """Detect the first persistent decay-to-recovery slope reversal.

    The reported bounce is the local envelope minimum immediately surrounding
    the first persistent non-negative running slope. The safe fit stop is moved
    back by half an envelope period so the centered envelope cannot mix recovery
    samples into the damping fit.
    """
    times = np.asarray(times, dtype=np.float64)
    raw = coherent_amplitude(mode_histories)
    envelope, envelope_points = geometric_envelope(raw, times, envelope_period)
    slopes = rolling_log_slopes(envelope, times, slope_window)
    eligible = np.flatnonzero(
        (times >= search_start)
        & (times <= search_stop)
        & np.isfinite(slopes)
    )
    bounce_index = None
    for offset in range(0, max(0, eligible.size - persistence_points + 1)):
        run = eligible[offset : offset + persistence_points]
        if not np.all(np.diff(run) == 1):
            continue
        if not np.all(slopes[run] >= 0.0):
            continue
        prior = eligible[:offset]
        if prior.size == 0 or not np.any(slopes[prior] < 0.0):
            continue
        left_time = max(search_start, float(times[run[0]]) - slope_window)
        local = np.flatnonzero(
            (times >= left_time)
            & (times <= float(times[run[-1]]) + 0.5 * slope_window)
        )
        bounce_index = int(local[np.argmin(envelope[local])])
        break

    bounce_time = float(times[bounce_index]) if bounce_index is not None else float("nan")
    safe_fit_stop = (
        bounce_time - 0.5 * envelope_period
        if np.isfinite(bounce_time)
        else float(search_stop)
    )
    return {
        "bounce_detected": bounce_index is not None,
        "bounce_time": bounce_time,
        "safe_fit_stop": safe_fit_stop,
        "envelope": envelope,
        "raw_amplitude": raw,
        "rolling_slope": slopes,
        "envelope_points": envelope_points,
    }


def fit_initial_decay(
    mode_histories: np.ndarray,
    times: np.ndarray,
    *,
    fit_start: float,
    fit_stop: float,
    envelope_period: float,
    min_duration: float = 3.0,
    min_r_squared: float = 0.90,
) -> dict:
    """Fit a common, explicitly pre-bounce interval."""
    amplitude, envelope_points = geometric_envelope(
        coherent_amplitude(mode_histories),
        np.asarray(times, dtype=np.float64),
        envelope_period,
    )
    times = np.asarray(times, dtype=np.float64)
    mask = (
        (times >= fit_start)
        & (times <= fit_stop)
        & np.isfinite(amplitude)
        & (amplitude > 0.0)
    )
    result = {
        "gamma": float("nan"),
        "r_squared": float("nan"),
        "fit_t_start": float("nan"),
        "fit_t_stop": float("nan"),
        "fit_duration": 0.0,
        "model_efoldings": 0.0,
        "n_points": int(np.count_nonzero(mask)),
        "envelope_points": envelope_points,
        "quality_pass": False,
    }
    if np.count_nonzero(mask) < 3:
        return result
    x = times[mask]
    y = np.log(amplitude[mask])
    coefficient = np.polyfit(x, y, 1)
    predicted = np.polyval(coefficient, x)
    residual = float(np.sum((y - predicted) ** 2))
    total = float(np.sum((y - np.mean(y)) ** 2))
    gamma = float(coefficient[0])
    r_squared = float(1.0 - residual / total) if total > 0.0 else float("nan")
    duration = float(x[-1] - x[0])
    quality = bool(
        gamma < 0.0
        and duration >= min_duration
        and np.isfinite(r_squared)
        and r_squared >= min_r_squared
    )
    result.update(
        {
            "gamma": gamma,
            "r_squared": r_squared,
            "fit_t_start": float(x[0]),
            "fit_t_stop": float(x[-1]),
            "fit_duration": duration,
            "model_efoldings": float(max(0.0, -gamma * duration)),
            "quality_pass": quality,
        }
    )
    return result


def paired_bootstrap_ratio_fixed_window(
    baseline: np.ndarray,
    current: np.ndarray,
    times: np.ndarray,
    *,
    fit_start: float,
    fit_stop: float,
    envelope_period: float,
    n_boot: int,
    seed: int,
    min_duration: float = 3.0,
    min_r_squared: float = 0.90,
) -> dict:
    """Bootstrap a paired rate ratio on one common pre-bounce window."""
    baseline = np.asarray(baseline, dtype=np.complex128)
    current = np.asarray(current, dtype=np.complex128)
    if baseline.shape != current.shape:
        raise ValueError("paired histories must have identical shapes")
    rng = np.random.default_rng(seed)
    ratios: list[float] = []
    gammas: list[float] = []
    quality = 0
    for _ in range(n_boot):
        indices = rng.integers(0, baseline.shape[0], size=baseline.shape[0])
        base_fit = fit_initial_decay(
            baseline[indices],
            times,
            fit_start=fit_start,
            fit_stop=fit_stop,
            envelope_period=envelope_period,
            min_duration=min_duration,
            min_r_squared=min_r_squared,
        )
        current_fit = fit_initial_decay(
            current[indices],
            times,
            fit_start=fit_start,
            fit_stop=fit_stop,
            envelope_period=envelope_period,
            min_duration=min_duration,
            min_r_squared=min_r_squared,
        )
        gamma0 = float(base_fit["gamma"])
        gamma = float(current_fit["gamma"])
        if gamma0 < 0.0 and gamma < 0.0 and np.isfinite(gamma0) and np.isfinite(gamma):
            ratios.append(gamma / gamma0)
            gammas.append(gamma)
            quality += int(base_fit["quality_pass"] and current_fit["quality_pass"])
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
    ratios_array = np.asarray(ratios)
    gammas_array = np.asarray(gammas)
    return {
        "ratio_sem": float(np.std(ratios_array)),
        "ratio_ci_lo": float(np.percentile(ratios_array, 2.5)),
        "ratio_ci_hi": float(np.percentile(ratios_array, 97.5)),
        "gamma_ci_lo": float(np.percentile(gammas_array, 2.5)),
        "gamma_ci_hi": float(np.percentile(gammas_array, 97.5)),
        "bootstrap_valid_fraction": len(ratios) / float(n_boot),
        "bootstrap_quality_fraction": quality / float(n_boot),
    }
