"""
attractor_analysis/forecast.py
==============================

Frozen-parameter forecasts and the baselines they must beat (T1 section 6).

Every baseline receives exactly the training information the scaling
forecast receives: snapshots and budgets up to the end of the training
window.  The Maxwellian built from *future* kinetic energy is an oracle
diagnostic, labelled ``ORACLE`` and excluded from ranking.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional

import numpy as np
from scipy import special

from .energy import maxwellian_1v_bin_content


ORACLE = "ORACLE"


def _loglog_interp(x_new, x, y):
    """Interpolate ln y vs ln x; NaN outside [x[0], x[-1]] or where y <= 0."""
    ok = (x > 0) & (y > 0)
    x, y = x[ok], y[ok]
    out = np.full(np.shape(x_new), np.nan)
    if x.size < 2:
        return out
    inside = (x_new >= x[0]) & (x_new <= x[-1])
    out[inside] = np.exp(np.interp(np.log(x_new[inside]), np.log(x), np.log(y)))
    return out


def scaling_forecast(eps: np.ndarray, N_ref: np.ndarray, support_ref: np.ndarray, t_ref: float,
                     t: float, alpha: float, beta: float, t0: float) -> np.ndarray:
    """``N_pred(eps, t) = r^alpha N(r^beta eps, t_ref)``, ``r = (t - t0)/(t_ref - t0)``.

    Energies that would need the template outside the supported range of the
    reference snapshot are returned as NaN (masked), never extrapolated.
    """
    r = (t - t0) / (t_ref - t0)
    m = support_ref & (N_ref > 0)
    return r**alpha * _loglog_interp(r**beta * eps, eps[m], N_ref[m])


def stationary_forecast(eps, N_last, support_last, t):
    out = np.where(support_last & (N_last > 0), N_last, np.nan)
    return out


def maxwellian_bins(edges: np.ndarray, number: float, kinetic: float) -> np.ndarray:
    """Per-bin N(eps) of a 1V Maxwellian with given number and kinetic energy."""
    T = 2.0 * kinetic / number
    nb, _ = maxwellian_1v_bin_content(edges, T, number)
    return nb / np.diff(edges)


def matched_maxwellian_forecast(edges, number_train, kinetic_train):
    """Maxwellian matched to the last training-window number and energy (constant in time)."""
    return maxwellian_bins(edges, number_train, kinetic_train)


def oracle_maxwellian(edges, number_future, kinetic_future):
    """Maxwellian at the *future* energy: oracle, excluded from ranking."""
    return maxwellian_bins(edges, number_future, kinetic_future)


def fit_heating_law(times: np.ndarray, kinetic: np.ndarray) -> dict:
    """One-scale heating: K(t) = K0 (t / t_ref)^a fitted on training data."""
    ok = (times > 0) & (kinetic > 0)
    a, lnK0 = np.polyfit(np.log(times[ok]), np.log(kinetic[ok]), 1)
    return {"a": float(a), "lnK0": float(lnK0)}


def heating_forecast(eps, N_ref, support_ref, t_ref, t, law: dict):
    """Number-conserving one-scale stretch: N(eps, t) = (1/g) N_ref(eps/g), g = K(t)/K(t_ref)."""
    g = (t / t_ref) ** law["a"]
    m = support_ref & (N_ref > 0)
    return _loglog_interp(eps / g, eps[m], N_ref[m]) / g


def quasilinear_forecast(*args, **kwargs):
    """Quasilinear velocity-diffusion baseline (interface stub).

    Requires a justified diffusion coefficient D(v, t) from the measured
    field spectrum (resonant, random-phase, weak-turbulence assumptions).
    Those assumptions are not justified a priori for strong beam turbulence
    with trapping; T3 must document them before using this baseline, and
    report it NOT_TESTED otherwise.
    """
    raise NotImplementedError("quasilinear baseline requires justified D(v,t); report NOT_TESTED")


def forecast_error(N_pred: np.ndarray, N_obs: np.ndarray, sigma_ln: np.ndarray, support_obs: np.ndarray) -> dict:
    """Error of a forecast against an observed snapshot on supported bins.

    ``rms_ln`` is the RMS of ln(N_pred/N_obs); ``chi2_red`` weights by the
    observation uncertainty.  ``coverage`` is the fraction of supported
    observed bins the forecast covers (masked bins do not count as errors but
    lower coverage).
    """
    sup = support_obs & (N_obs > 0)
    m = sup & np.isfinite(N_pred) & (N_pred > 0)
    if m.sum() == 0:
        return {"rms_ln": np.nan, "chi2_red": np.nan, "coverage": 0.0, "n": 0}
    d = np.log(N_pred[m]) - np.log(N_obs[m])
    return {"rms_ln": float(np.sqrt(np.mean(d * d))),
            "chi2_red": float(np.mean(d * d / sigma_ln[m] ** 2)),
            "coverage": float(m.sum() / max(sup.sum(), 1)), "n": int(m.sum())}


def rank_forecasts(errors: Dict[str, dict], key: str = "rms_ln", min_coverage: float = 0.5) -> list:
    """Rank non-oracle forecasts by ``key``; low-coverage forecasts are listed last."""
    names = [n for n in errors if not n.startswith(ORACLE)]
    good = sorted([n for n in names if errors[n]["coverage"] >= min_coverage],
                  key=lambda n: errors[n][key])
    bad = [n for n in names if errors[n]["coverage"] < min_coverage]
    return good + bad
