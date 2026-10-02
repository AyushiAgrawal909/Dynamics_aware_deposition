"""
validation/landau_fit.py
========================

Ensemble-averaged damping/growth fitting utilities.

The important Phase 4 fix is to average linear amplitudes across seeds before
fitting. Fitting each noisy seed independently and averaging gamma biases the
result when late-time trajectories approach the particle-noise floor.
"""

from __future__ import annotations

import numpy as np


def ensemble_average_amplitude(amplitude_list: list[np.ndarray]) -> np.ndarray:
    """Return the seed-ensemble average of linear amplitudes."""
    if not amplitude_list:
        raise ValueError("amplitude_list must not be empty")
    stack = np.stack([np.asarray(amp, dtype=np.float64) for amp in amplitude_list], axis=0)
    return np.mean(stack, axis=0)


def estimate_noise_floor(amp_avg: np.ndarray, times: np.ndarray, t_floor_start: float) -> float:
    """Estimate asymptotic amplitude floor from late-time median."""
    amp_avg = np.asarray(amp_avg, dtype=np.float64)
    times = np.asarray(times, dtype=np.float64)
    mask = times > t_floor_start
    if np.count_nonzero(mask) < 3:
        return float(amp_avg[-1])
    return float(np.median(amp_avg[mask]))


def fit_gamma_ensemble(
    amplitude_list: list[np.ndarray],
    times: np.ndarray,
    t_fit_start: float = 2.0,
    t_fit_stop_max: float = 30.0,
    noise_floor_start: float = 40.0,
    noise_factor: float = 8.0,
    fallback_t_fit_start: float = 5.0,
) -> dict:
    """Fit gamma with adaptive floor truncation and a recorded fixed-window fallback.

    If the estimated late-time floor is larger than the earlier damping signal,
    the adaptive threshold can reject every point. In that case the result
    falls back to the validated transient-skipping window rather than silently
    returning an unusable NaN.
    """
    times = np.asarray(times, dtype=np.float64)
    amp_avg = ensemble_average_amplitude(amplitude_list)
    floor = estimate_noise_floor(amp_avg, times, noise_floor_start)
    threshold = noise_factor * floor
    mask = (
        (times >= t_fit_start)
        & (times <= t_fit_stop_max)
        & np.isfinite(amp_avg)
        & (amp_avg > threshold)
        & (amp_avg > 0.0)
    )
    strategy = "adaptive_floor"
    if np.count_nonzero(mask) < 5:
        mask = (
            (times >= fallback_t_fit_start)
            & (times <= t_fit_stop_max)
            & np.isfinite(amp_avg)
            & (amp_avg > 0.0)
        )
        strategy = "fixed_window_fallback"
    if np.count_nonzero(mask) < 5:
        return {
            "gamma": float("nan"),
            "n_points": int(np.count_nonzero(mask)),
            "noise_floor": floor,
            "t_fit_start": float("nan"),
            "t_fit_stop": float("nan"),
            "fit_strategy": strategy,
        }
    coeffs = np.polyfit(times[mask], np.log(amp_avg[mask]), 1)
    return {
        "gamma": float(coeffs[0]),
        "n_points": int(np.count_nonzero(mask)),
        "noise_floor": floor,
        "t_fit_start": float(times[mask][0]),
        "t_fit_stop": float(times[mask][-1]),
        "fit_strategy": strategy,
    }


def fit_gamma_ensemble_fixed_window(
    amplitude_list: list[np.ndarray],
    times: np.ndarray,
    t_fit_start: float,
    t_fit_stop: float,
    amp_floor: float = 1.0e-30,
) -> dict:
    """Fit gamma on a fixed window; useful for growing two-stream modes."""
    times = np.asarray(times, dtype=np.float64)
    amp_avg = ensemble_average_amplitude(amplitude_list)
    mask = (
        (times >= t_fit_start)
        & (times <= t_fit_stop)
        & np.isfinite(amp_avg)
        & (amp_avg > amp_floor)
    )
    if np.count_nonzero(mask) < 5:
        return {"gamma": float("nan"), "n_points": int(np.count_nonzero(mask))}
    coeffs = np.polyfit(times[mask], np.log(amp_avg[mask]), 1)
    return {"gamma": float(coeffs[0]), "n_points": int(np.count_nonzero(mask))}


def bootstrap_gamma_ci(
    amplitude_list: list[np.ndarray],
    times: np.ndarray,
    n_boot: int = 1000,
    seed: int = 12345,
    fixed_window: bool = False,
    **fit_kw,
) -> dict:
    """Bootstrap gamma by resampling seeds with replacement."""
    rng = np.random.default_rng(seed)
    n_series = len(amplitude_list)
    if n_series == 0:
        return {
            "gamma_mean": float("nan"),
            "gamma_sem": float("nan"),
            "ci_lo": float("nan"),
            "ci_hi": float("nan"),
        }
    gammas: list[float] = []
    fitter = fit_gamma_ensemble_fixed_window if fixed_window else fit_gamma_ensemble
    for _ in range(n_boot):
        indices = rng.integers(0, n_series, size=n_series)
        resampled = [amplitude_list[int(index)] for index in indices]
        gamma = fitter(resampled, times, **fit_kw)["gamma"]
        if np.isfinite(gamma):
            gammas.append(float(gamma))
    if not gammas:
        return {
            "gamma_mean": float("nan"),
            "gamma_sem": float("nan"),
            "ci_lo": float("nan"),
            "ci_hi": float("nan"),
        }
    values = np.asarray(gammas, dtype=np.float64)
    return {
        "gamma_mean": float(np.mean(values)),
        "gamma_sem": float(np.std(values)),
        "ci_lo": float(np.percentile(values, 2.5)),
        "ci_hi": float(np.percentile(values, 97.5)),
    }
