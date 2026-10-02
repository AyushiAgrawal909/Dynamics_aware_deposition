"""Causal controller features, including empirical instability marginality.

The existing density-only controller cannot distinguish a robust damped mode
from a near-marginal instability.  This module adds particle phase-space
moments and a past-only measured mode-growth feature.  It intentionally does
not call a warm dispersion model: that model failed the frozen E2 smoothing
response gate and is not a trusted runtime input.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np


LOCAL_FEATURE_NAMES = (
    "marker_density_over_mean",
    "shot_noise_proxy",
    "effective_weighted_marker_count",
    "weighted_shot_noise_proxy",
    "mean_vx_over_vrms",
    "mean_vy_over_vrms",
    "thermal_vx_over_vrms",
    "thermal_vy_over_vrms",
    "velocity_covariance_over_vrms2",
    "counterstream_balance",
    "counterstream_separation",
    "counterstream_drive_proxy",
)

HISTORY_FEATURE_NAMES = (
    "mode_history_valid",
    "empirical_gamma",
    "empirical_gamma_stderr",
    "empirical_gamma_significance",
    "empirical_growth_r2",
    "empirical_growth_efolds",
    "log_mode_amplitude",
)


@dataclass(frozen=True)
class ModeHistoryFeatures:
    valid: float
    gamma: float
    gamma_stderr: float
    significance: float
    r2: float
    growth_efolds: float
    log_amplitude: float

    def as_array(self) -> np.ndarray:
        return np.array(
            [
                self.valid,
                self.gamma,
                self.gamma_stderr,
                self.significance,
                self.r2,
                self.growth_efolds,
                self.log_amplitude,
            ],
            dtype=np.float64,
        )


@dataclass(frozen=True)
class ControllerFeatureField:
    values: np.ndarray
    names: Tuple[str, ...]
    controller_shape: Tuple[int, int]


def estimate_causal_mode_features(
    times: Sequence[float],
    amplitudes: Sequence[float],
    max_points: int = 24,
    min_points: int = 6,
) -> ModeHistoryFeatures:
    """Fit the trailing log-amplitude history using only data available now."""
    t = np.asarray(times, dtype=np.float64)
    amplitude = np.asarray(amplitudes, dtype=np.float64)
    valid = np.isfinite(t) & np.isfinite(amplitude) & (amplitude > 0.0)
    t = t[valid]
    amplitude = amplitude[valid]
    if t.size < min_points:
        return ModeHistoryFeatures(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -70.0)
    if max_points > 0 and t.size > max_points:
        t = t[-max_points:]
        amplitude = amplitude[-max_points:]
    if np.any(np.diff(t) <= 0.0):
        raise ValueError("mode-history times must be strictly increasing")

    log_amp = np.log(amplitude)
    design = np.column_stack([t, np.ones_like(t)])
    gamma, intercept = np.linalg.lstsq(design, log_amp, rcond=None)[0]
    fitted = gamma * t + intercept
    residual = log_amp - fitted
    ss_res = float(np.sum(residual * residual))
    centered = log_amp - np.mean(log_amp)
    ss_tot = float(np.sum(centered * centered))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 0.0
    dof = max(t.size - 2, 1)
    centered_t = t - np.mean(t)
    denom = float(np.sum(centered_t * centered_t))
    stderr = np.sqrt((ss_res / dof) / denom) if denom > 0.0 else np.inf
    significance = gamma / stderr if np.isfinite(stderr) and stderr > 0.0 else 0.0
    return ModeHistoryFeatures(
        valid=1.0,
        gamma=float(gamma),
        gamma_stderr=float(stderr),
        significance=float(significance),
        r2=float(r2),
        growth_efolds=float(gamma * (t[-1] - t[0])),
        log_amplitude=float(log_amp[-1]),
    )


def _cic_accumulate(
    x: np.ndarray,
    y: np.ndarray,
    values: np.ndarray,
    shape: Tuple[int, int],
    lengths: Tuple[float, float],
) -> np.ndarray:
    nx, ny = shape
    Lx, Ly = lengths
    ux = np.mod(x, Lx) * nx / Lx
    uy = np.mod(y, Ly) * ny / Ly
    i0_raw = np.floor(ux).astype(np.int64)
    j0_raw = np.floor(uy).astype(np.int64)
    fx = ux - i0_raw
    fy = uy - j0_raw
    i0 = i0_raw % nx
    j0 = j0_raw % ny
    i1 = (i0 + 1) % nx
    j1 = (j0 + 1) % ny
    result = np.zeros((nx, ny), dtype=np.float64)
    for ii, jj, weight in (
        (i0, j0, (1.0 - fx) * (1.0 - fy)),
        (i1, j0, fx * (1.0 - fy)),
        (i0, j1, (1.0 - fx) * fy),
        (i1, j1, fx * fy),
    ):
        np.add.at(result, (ii, jj), values * weight)
    return result


def build_controller_feature_field(
    particles,
    controller_shape: Tuple[int, int],
    mode_history: Optional[Tuple[Sequence[float], Sequence[float]]] = None,
    velocity_split_temperature: Optional[float] = None,
) -> ControllerFeatureField:
    """Build nodal, causal features from a particle structure-of-arrays object.

    The counterstream features are empirical local phase-space descriptors.
    They are proxies, not a dispersion-theory label.  The measured growth
    history is broadcast to all nodes and supplies the global distance-to-
    marginality information unavailable to density-only selectors.
    """
    x = np.asarray(particles.x, dtype=np.float64)
    y = np.asarray(particles.y, dtype=np.float64)
    vx = np.asarray(particles.vx, dtype=np.float64)
    vy = np.asarray(particles.vy, dtype=np.float64)
    if not (x.shape == y.shape == vx.shape == vy.shape) or x.ndim != 1:
        raise ValueError("particle x, y, vx, and vy arrays must be equal-length vectors")
    nx, ny = controller_shape
    if nx < 2 or ny < 2:
        raise ValueError("controller_shape requires at least two nodes per direction")
    lengths = (float(particles.Lx), float(particles.Ly))
    ones = np.ones_like(x)
    particle_weight = np.asarray(
        getattr(particles, "w", np.ones_like(x)), dtype=np.float64
    )
    if particle_weight.shape != x.shape or np.any(~np.isfinite(particle_weight)):
        raise ValueError("particle weights must be a finite vector matching positions")
    count = _cic_accumulate(x, y, ones, controller_shape, lengths)
    safe_count = np.maximum(count, 1.0e-12)
    sum_abs_weight = _cic_accumulate(
        x, y, np.abs(particle_weight), controller_shape, lengths
    )
    sum_weight_sq = _cic_accumulate(
        x, y, particle_weight * particle_weight, controller_shape, lengths
    )
    effective_count = sum_abs_weight * sum_abs_weight / np.maximum(sum_weight_sq, 1.0e-30)
    weighted_noise = 1.0 / np.sqrt(np.maximum(effective_count, 1.0))

    sum_vx = _cic_accumulate(x, y, vx, controller_shape, lengths)
    sum_vy = _cic_accumulate(x, y, vy, controller_shape, lengths)
    sum_vx2 = _cic_accumulate(x, y, vx * vx, controller_shape, lengths)
    sum_vy2 = _cic_accumulate(x, y, vy * vy, controller_shape, lengths)
    sum_vxvy = _cic_accumulate(x, y, vx * vy, controller_shape, lengths)
    mean_vx = sum_vx / safe_count
    mean_vy = sum_vy / safe_count
    var_vx = np.maximum(sum_vx2 / safe_count - mean_vx * mean_vx, 0.0)
    var_vy = np.maximum(sum_vy2 / safe_count - mean_vy * mean_vy, 0.0)
    cov_v = sum_vxvy / safe_count - mean_vx * mean_vy
    vrms = max(float(np.sqrt(np.mean(vx * vx + vy * vy))), 1.0e-12)

    split_temperature = (
        max(0.05 * vrms, 1.0e-12)
        if velocity_split_temperature is None
        else float(velocity_split_temperature)
    )
    if not np.isfinite(split_temperature) or split_temperature <= 0.0:
        raise ValueError("velocity_split_temperature must be finite and positive")
    # A smooth beam-membership split keeps the feature differentiable. A hard
    # sign split would inject a second discontinuity into the controller.
    positive = 0.5 * (1.0 + np.tanh(vx / split_temperature))
    negative = 1.0 - positive
    n_pos = _cic_accumulate(x, y, positive, controller_shape, lengths)
    n_neg = _cic_accumulate(x, y, negative, controller_shape, lengths)
    pos_safe = np.maximum(n_pos, 1.0e-12)
    neg_safe = np.maximum(n_neg, 1.0e-12)
    mu_pos = _cic_accumulate(x, y, vx * positive, controller_shape, lengths) / pos_safe
    mu_neg = _cic_accumulate(x, y, vx * negative, controller_shape, lengths) / neg_safe
    var_pos = np.maximum(
        _cic_accumulate(x, y, vx * vx * positive, controller_shape, lengths) / pos_safe
        - mu_pos * mu_pos,
        0.0,
    )
    var_neg = np.maximum(
        _cic_accumulate(x, y, vx * vx * negative, controller_shape, lengths) / neg_safe
        - mu_neg * mu_neg,
        0.0,
    )
    balance = 4.0 * n_pos * n_neg / (safe_count * safe_count)
    separation = np.abs(mu_pos - mu_neg) / np.sqrt(var_pos + var_neg + 1.0e-12 * vrms * vrms)
    separation[(n_pos < 0.25) | (n_neg < 0.25)] = 0.0
    drive_proxy = balance * separation * separation

    mean_count = max(float(np.mean(count)), 1.0e-12)
    local = np.stack(
        [
            count / mean_count,
            1.0 / np.sqrt(np.maximum(count, 1.0)),
            effective_count,
            weighted_noise,
            mean_vx / vrms,
            mean_vy / vrms,
            np.sqrt(var_vx) / vrms,
            np.sqrt(var_vy) / vrms,
            cov_v / (vrms * vrms),
            balance,
            separation,
            drive_proxy,
        ],
        axis=-1,
    )

    if mode_history is None:
        history = ModeHistoryFeatures(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -70.0)
    else:
        history = estimate_causal_mode_features(*mode_history)
    history_values = np.broadcast_to(history.as_array(), (nx, ny, len(HISTORY_FEATURE_NAMES)))
    values = np.concatenate([local, history_values], axis=-1)
    return ControllerFeatureField(
        values=values,
        names=LOCAL_FEATURE_NAMES + HISTORY_FEATURE_NAMES,
        controller_shape=controller_shape,
    )
