"""
attractor_analysis/convergence.py
=================================

Nested-subsample convergence gate (T1 section 4), onset detection, and the
regulator-scaling model comparison T2 pre-registers.

Conventions: regulator *level* 0 is the finest (most particles, or smallest
Delta v); level l is coarser by a factor 2**l.  Feature trajectories are
arrays of shape (n_seeds, n_times) on a common time grid.

Gate verdicts
-------------
``PASS``
    Paired differences shrink with refinement (log-slope test), the
    finest difference is below ``tau`` times the feature's own motion over the
    analysis window, and (if a reference is given) the finest trajectory
    agrees with the independent reference within its uncertainty.
``PASS_AFTER_ONSET_ALIGNMENT``
    Raw trajectories fail only because the onset time depends on the
    regulator; after aligning each run at its own onset, the trajectories pass.
    The onset-scaling model is reported alongside (ln N onset dependence is
    the expected Casimir-breaking signature and does not by itself make the
    post-onset evolution unphysical).
``FAIL_REGULATOR_DEPENDENT``
    Paired differences do not shrink and exceed ``tau`` x motion, or the
    finest difference exceeds the feature's whole motion in time (the feature
    is set by the regulator).
``STATIONARY_CONVERGED`` (protocol v2)
    The feature does not drift over the window (fitted drift below
    ``motion_floor``), the finest paired difference is below the absolute
    tolerance ``abs_tol`` (seed-bootstrap upper bound), and the finest level
    agrees with the reference within ``abs_tol``.  A converged equilibrium:
    nothing moves, so there is nothing for T3 to scale, but it is not a
    regulator effect.  (v1 normalised by the feature's own motion, which
    makes any stationary feature non-PASS: review F3.)
``INCONCLUSIVE``
    Too few levels or seeds, non-contiguous levels, or confidence intervals
    compatible with both.

FAIL results carry ``fail_kind``: ``"regulator"`` (the PIC ladder itself does
not converge) or ``"reference"`` (it converges, but to a value the
independent reference does not share).
"""

from __future__ import annotations

from typing import Dict, Mapping, Optional, Sequence

import numpy as np


PASS = "PASS"
PASS_ALIGNED = "PASS_AFTER_ONSET_ALIGNMENT"
FAIL = "FAIL_REGULATOR_DEPENDENT"
STATIONARY = "STATIONARY_CONVERGED"
INCONCLUSIVE = "INCONCLUSIVE"


def _bootstrap_mean_ci(samples: np.ndarray, n_boot: int = 2000, alpha: float = 0.05,
                       rng: Optional[np.random.Generator] = None):
    """Seed-bootstrap CI of the mean along axis 0."""
    rng = rng or np.random.default_rng(12345)
    samples = np.asarray(samples, dtype=np.float64)
    n = samples.shape[0]
    idx = rng.integers(0, n, size=(n_boot, n))
    means = samples[idx].mean(axis=1)
    lo, hi = np.nanquantile(means, [alpha / 2, 1 - alpha / 2], axis=0)
    return samples.mean(axis=0), lo, hi


def onset_time(times: np.ndarray, signal: np.ndarray, threshold: float) -> float:
    """First time ``signal`` crosses ``threshold`` upward (linear interpolation).

    NaN if it never does.  The threshold is frozen in the protocol.
    """
    s = np.asarray(signal, dtype=np.float64)
    above = np.flatnonzero(s >= threshold)
    if above.size == 0:
        return np.nan
    j = int(above[0])
    if j == 0:
        return float(times[0])
    t0, t1 = times[j - 1], times[j]
    s0, s1 = s[j - 1], s[j]
    return float(t0 + (threshold - s0) / (s1 - s0) * (t1 - t0))


def paired_differences(feature_by_level: Mapping[int, np.ndarray], window: np.ndarray) -> Dict[int, np.ndarray]:
    """Per-seed RMS over the window of F_l minus the next finer retained level, keyed by the coarser level l."""
    out = {}
    levels = sorted(feature_by_level)
    for lf, l in zip(levels[:-1], levels[1:]):
        d = np.asarray(feature_by_level[l])[:, window] - np.asarray(feature_by_level[lf])[:, window]
        out[l] = np.sqrt(np.nanmean(d * d, axis=1))
    return out


def _shift_to_onset(times: np.ndarray, traj: np.ndarray, t_on: float, grid: np.ndarray) -> np.ndarray:
    if not np.isfinite(t_on):
        return np.full_like(grid, np.nan, dtype=np.float64)
    return np.interp(grid, times - t_on, traj, left=np.nan, right=np.nan)


def convergence_gate(times: np.ndarray, feature_by_level: Mapping[int, np.ndarray],
                     window: tuple[float, float], tau: float = 0.1,
                     reference: Optional[np.ndarray] = None,
                     reference_err: Optional[np.ndarray] = None,
                     onset_by_level: Optional[Mapping[int, np.ndarray]] = None,
                     regulator_by_level: Optional[Mapping[int, float]] = None,
                     min_levels: int = 3, min_seeds: int = 3,
                     shrink_slope: float = 0.25,
                     abs_tol: Optional[float] = None, motion_floor: Optional[float] = None,
                     reference_onset: Optional[float] = None) -> dict:
    """Decide whether a feature trajectory converges as the regulator is removed.

    Parameters
    ----------
    times : common diagnostic times.
    feature_by_level : level -> array (n_seeds, n_times), level 0 finest.
    window : analysis window (t_start, t_end), frozen in the protocol.
    tau : tolerance on finest paired difference relative to feature motion.
    reference, reference_err : independent reference trajectory on ``times``
        (e.g. Vlasov at its finest Delta v) and its uncertainty.
    onset_by_level : level -> array (n_seeds,) of onset times; enables the
        onset-aligned retry.
    regulator_by_level : level -> N (or 1/Delta v) for the onset-scaling fit.
    shrink_slope : minimum slope of log2(paired difference) against level
        (pure nested Monte-Carlo noise gives 0.5; a non-vanishing
        regulator-dependent offset gives ~0).  The seed-bootstrap 5% quantile
        of the slope must also be positive.
    abs_tol, motion_floor : protocol-v2 stationary branch (both in feature
        units).  Without them the gate is the v1 gate.
    reference_onset : onset time of the reference run.  The onset-aligned
        retry shifts the reference by it; without it, the retry cannot pass a
        feature whose raw comparison with the reference failed (review F2).
    """
    times = np.asarray(times, dtype=np.float64)
    levels = sorted(feature_by_level)
    n_seeds = min(np.asarray(feature_by_level[l]).shape[0] for l in levels)
    win = (times >= window[0]) & (times <= window[1])
    result = {"n_levels": len(levels), "n_seeds": int(n_seeds)}
    if len(levels) < min_levels or n_seeds < min_seeds or win.sum() < 3:
        result.update(verdict=INCONCLUSIVE, reason="too few levels, seeds or window samples")
        return result
    if levels != list(range(levels[0], levels[0] + len(levels))):
        # the shrinkage slope is fitted against the level index, so every
        # adjacent pair must be one factor of 2 apart (review F7)
        result.update(verdict=INCONCLUSIVE, reason=f"levels {levels} are not contiguous")
        return result
    stat = dict(abs_tol=abs_tol, motion_floor=motion_floor)

    raw = _gate_core(times, feature_by_level, win, tau, reference, reference_err, shrink_slope, **stat)
    result["raw"] = raw
    if raw["verdict"] in (PASS, STATIONARY) or onset_by_level is None:
        result["verdict"] = raw["verdict"]
        result["reason"] = raw["reason"]
        if "fail_kind" in raw:
            result["fail_kind"] = raw["fail_kind"]
    else:
        # retry with every run aligned at its own onset
        span = window[1] - window[0]
        tons = np.concatenate([np.asarray(onset_by_level[l]) for l in levels])
        t_ref = np.nanmedian(np.asarray(onset_by_level[levels[0]]))
        grid = times[win] - t_ref
        grid = grid[(grid >= 0) & (grid <= span)]
        aligned = {}
        for l in levels:
            F = np.asarray(feature_by_level[l])
            aligned[l] = np.vstack([_shift_to_onset(times, F[s], onset_by_level[l][s], grid)
                                    for s in range(n_seeds)])
        # the reference is aligned at its own onset; it is never dropped
        ref_al = err_al = None
        ref_blocked = False
        if reference is not None:
            if reference_onset is not None and np.isfinite(reference_onset):
                ref_al = _shift_to_onset(times, np.asarray(reference, float), reference_onset, grid)
                if reference_err is not None:
                    err_al = _shift_to_onset(times, np.asarray(reference_err, float), reference_onset, grid)
            else:
                ref_blocked = True
        al = _gate_core(grid, aligned, np.ones_like(grid, dtype=bool), tau, ref_al, err_al, shrink_slope, **stat)
        result["aligned"] = al
        if (al["verdict"] == PASS and np.all(np.isfinite(tons)) and not ref_blocked):
            result["verdict"] = PASS_ALIGNED
            result["reason"] = ("raw trajectories depend on the regulator only through the onset time; "
                                "aligned at onset (reference included) they converge")
        else:
            result["verdict"] = raw["verdict"]
            result["reason"] = raw["reason"]
            if "fail_kind" in raw:
                result["fail_kind"] = raw["fail_kind"]
            if ref_blocked and al["verdict"] == PASS:
                result["reason"] += ("; the aligned retry passed without a reference onset, "
                                     "which cannot cure a reference disagreement")

    if onset_by_level is not None and regulator_by_level is not None:
        Ns, ts = [], []
        for l in levels:
            for t in np.asarray(onset_by_level[l]):
                Ns.append(regulator_by_level[l])
                ts.append(t)
        result["onset_scaling"] = onset_scaling_fit(np.array(Ns), np.array(ts))
    return result


def _gate_core(times, feature_by_level, win, tau, reference, reference_err, shrink_slope,
               abs_tol=None, motion_floor=None) -> dict:
    levels = sorted(feature_by_level)
    pdiff = paired_differences(feature_by_level, win)
    D = {l: float(np.nanmean(pdiff[l])) for l in pdiff}
    finest = np.asarray(feature_by_level[levels[0]])[:, win]
    mean_traj = np.nanmean(finest, axis=0)
    finite = np.isfinite(mean_traj)
    if finite.sum() < 3:
        return {"verdict": INCONCLUSIVE, "reason": "feature undefined in window", "D": D}
    motion = float(np.nanmax(mean_traj) - np.nanmin(mean_traj))
    # Shrinkage: slope of log2(D_l) against level l.  Pure Monte-Carlo noise
    # in nested subsets gives D_l ~ 2^(l/2) (slope 0.5); a bias ~ 1/N gives
    # slope 1; a regulator-dependent offset that does not vanish (e.g. ~ ln N)
    # gives slope ~ 0.  Seed bootstrap gives the CI of the slope.
    lv = np.array(sorted(D), dtype=np.float64)
    ordered = [D[l] for l in sorted(D)]
    ratios = [ordered[i] / ordered[i + 1] if ordered[i + 1] > 0 else np.inf
              for i in range(len(ordered) - 1)]
    P = np.vstack([pdiff[l] for l in sorted(pdiff)])  # (n_pairs, n_seeds)
    def _slope(Dv):
        Dv = np.maximum(Dv, 1e-300)
        return float(np.polyfit(lv, np.log2(Dv), 1)[0])
    slope = _slope(P.mean(axis=1))
    rng = np.random.default_rng(777)
    boots = [_slope(P[:, rng.integers(0, P.shape[1], P.shape[1])].mean(axis=1)) for _ in range(500)]
    slope_lo = float(np.quantile(boots, 0.05))
    shrinking = bool(slope >= shrink_slope and slope_lo > 0.0)
    rel = ordered[0] / motion if motion > 0 else np.inf
    # seed-bootstrap CI for the finest difference
    _, lo, hi = _bootstrap_mean_ci(pdiff[min(pdiff)][:, None])
    out = {"D": D, "ratios": ratios, "shrink_slope": slope, "shrink_slope_lo90": slope_lo, "motion": motion, "finest_rel_diff": rel,
           "finest_rel_diff_ci": [float(lo[0] / motion) if motion > 0 else np.inf,
                                  float(hi[0] / motion) if motion > 0 else np.inf]}
    ref_ok = None
    if reference is not None:
        ref = np.asarray(reference, dtype=np.float64)[win]
        err = (np.zeros_like(ref) if reference_err is None
               else np.asarray(reference_err, dtype=np.float64)[win])
        sd = np.nanstd(finest, axis=0, ddof=1) / np.sqrt(finest.shape[0])
        z = np.abs(mean_traj - ref) / np.sqrt(sd**2 + err**2 + (tau * motion) ** 2)
        ref_ok = bool(np.nanmax(z) < 3.0)
        out["reference_max_z"] = float(np.nanmax(z))
    out["reference_ok"] = ref_ok

    if abs_tol is not None and motion_floor is not None:
        tw = np.asarray(times, dtype=np.float64)[win][finite]
        drift = float(abs(np.polyfit(tw, mean_traj[finite], 1)[0]) * (tw[-1] - tw[0]))
        out["drift"] = drift
        out["stationary"] = bool(drift < motion_floor)
        if out["stationary"]:
            return _stationary_verdict(out, finest, mean_traj, reference, reference_err, win,
                                       lo[0], hi[0], ordered, shrinking, abs_tol)

    if shrinking and rel < tau:
        if ref_ok is False:
            out.update(verdict=FAIL, fail_kind="reference",
                       reason="converged across levels but disagrees with the independent reference")
        else:
            out.update(verdict=PASS, reason="paired differences shrink and the finest is below tau x motion")
    elif out["finest_rel_diff_ci"][0] >= 1.0:
        out.update(verdict=FAIL, fail_kind="regulator", reason="finest paired difference exceeds the feature's own motion in time "
                                       "(the feature is set by the regulator, not by the dynamics)")
    elif not shrinking and out["finest_rel_diff_ci"][0] >= tau:
        out.update(verdict=FAIL, fail_kind="regulator", reason="paired differences do not shrink and exceed tau x motion "
                                       "(includes features that differ between levels but barely move in time)")
    else:
        out.update(verdict=INCONCLUSIVE, reason="shrinkage or tolerance not established within CI")
    return out


def _stationary_verdict(out, finest, mean_traj, reference, reference_err, win, d_lo, d_hi,
                        ordered, shrinking, abs_tol) -> dict:
    """Protocol-v2 branch for a feature that does not drift over the window:
    convergence is judged against the absolute tolerance, not the motion."""
    out["abs_tol"] = abs_tol
    out["finest_diff_ci"] = [float(d_lo), float(d_hi)]
    ref_ok = None
    if reference is not None:
        ref = np.asarray(reference, dtype=np.float64)[win]
        err = (np.zeros_like(ref) if reference_err is None
               else np.asarray(reference_err, dtype=np.float64)[win])
        sd = np.nanstd(finest, axis=0, ddof=1) / np.sqrt(finest.shape[0])
        z = np.abs(mean_traj - ref) / np.sqrt(sd**2 + err**2 + abs_tol**2)
        ref_ok = bool(np.nanmax(z) < 3.0)
        out["reference_max_z"] = float(np.nanmax(z))
    out["reference_ok"] = ref_ok
    within = d_hi < abs_tol and (shrinking or max(ordered) < abs_tol)
    if within:
        if ref_ok is False:
            out.update(verdict=FAIL, fail_kind="reference",
                       reason="stationary and converged across levels, but disagrees with the independent reference")
        else:
            out.update(verdict=STATIONARY,
                       reason="stationary over the window; finest paired difference below the absolute tolerance")
    elif d_lo >= abs_tol and not shrinking:
        out.update(verdict=FAIL, fail_kind="regulator",
                   reason="stationary, but the levels differ by more than the absolute tolerance and do not converge")
    else:
        out.update(verdict=INCONCLUSIVE, reason="stationary; convergence within the absolute tolerance not established")
    return out


def onset_scaling_fit(N: np.ndarray, t_on: np.ndarray) -> dict:
    """Compare t_on = a + b ln N against t_on = a + b N (and a constant).

    Gaussian-residual AIC; the preferred model must beat the next by
    Delta AIC >= 4, otherwise the comparison is INCONCLUSIVE.
    """
    N = np.asarray(N, dtype=np.float64)
    t = np.asarray(t_on, dtype=np.float64)
    ok = np.isfinite(N) & np.isfinite(t) & (N > 0)
    N, t = N[ok], t[ok]
    n = len(t)
    if n < 4 or len(np.unique(N)) < 3:
        return {"preferred": INCONCLUSIVE, "reason": "need >= 3 distinct regulator values"}
    models = {}
    for name, X in (("const", np.ones((n, 1))),
                    ("lnN", np.column_stack([np.ones(n), np.log(N)])),
                    ("N", np.column_stack([np.ones(n), N]))):
        coef, *_ = np.linalg.lstsq(X, t, rcond=None)
        rss = float(np.sum((t - X @ coef) ** 2))
        k = X.shape[1] + 1
        aic = n * np.log(max(rss, 1e-300) / n) + 2 * k
        models[name] = {"coef": coef.tolist(), "rss": rss, "aic": float(aic)}
    order = sorted(models, key=lambda m: models[m]["aic"])
    best, second = order[0], order[1]
    daic = models[second]["aic"] - models[best]["aic"]
    preferred = best if daic >= 4.0 else INCONCLUSIVE
    return {"preferred": preferred, "delta_aic": float(daic), "models": models, "ranking": order}
