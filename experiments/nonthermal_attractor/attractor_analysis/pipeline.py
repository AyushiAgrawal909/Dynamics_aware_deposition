"""
attractor_analysis/pipeline.py
==============================

End-to-end analysis used by the synthetic suite and, unchanged, by T2/T3:
seed merging, tail detection, scaling fit on training snapshots, frozen
forecasts against baselines, the regulator gate on a moving feature, and the
claim logic that turns those into a verdict.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

from . import convergence, forecast, scaling
from .energy import (EnergyHistogram, core_tail_excess, histogram_moments, maxwellian_tail_excess,
                     quantile_feature, tail_moments)


TAIL_EXCESS_MIN = 3.0       # tail number / Maxwellian-equivalent tail number
FORECAST_MARGIN = 0.8       # scaling rms_ln must be <= margin x best baseline
MIN_COUNTS_PER_SEED = 10.0  # mean effective counts per bin per seed for support


def merge_seeds(hists: Sequence[EnergyHistogram]) -> EnergyHistogram:
    """Pool seeds (weights averaged so the result is a per-run estimate)."""
    n = len(hists)
    w = sum(h.weight for h in hists) / n
    e = sum(h.energy for h in hists) / n
    neff = sum(h.neff for h in hists)
    return EnergyHistogram(hists[0].edges, w / np.diff(hists[0].edges), w, e, neff,
                           sum(h.overflow_weight for h in hists) / n,
                           sum(h.overflow_energy for h in hists) / n)


def snapshot_set(hists_by_seed: List[List[EnergyHistogram]], times: np.ndarray, fit_window,
                 select: Optional[np.ndarray] = None) -> scaling.SnapshotSet:
    """Seed-mean snapshots with seed-scatter uncertainties."""
    sel = np.ones(len(times), bool) if select is None else select
    edges = hists_by_seed[0][0].edges
    eps = np.sqrt(np.where(edges[:-1] > 0, edges[:-1], 0.5 * edges[1]) * edges[1:])
    Ns = np.array([[h.N for h, keep in zip(row, sel) if keep] for row in hists_by_seed])
    neffs = np.array([[h.neff for h, keep in zip(row, sel) if keep] for row in hists_by_seed])
    n_seeds = Ns.shape[0]
    N = Ns.mean(axis=0)
    if n_seeds >= 2:
        sig = scaling.sigma_ln_from_seeds(Ns)
    else:
        sig = scaling.sigma_ln_from_neff(neffs[0])
    support = (neffs.mean(axis=0) >= MIN_COUNTS_PER_SEED) & (neffs.min(axis=0) >= 2)
    fit_bins = (edges[:-1] >= fit_window[0]) & (edges[1:] <= fit_window[1])
    return scaling.SnapshotSet(times=np.asarray(times)[sel], eps=eps, N=N, sigma_ln=sig,
                               support=support, fit_bins=fit_bins)


def detect_tail(hists_by_seed, t_index: int, kappa: float = 10.0) -> dict:
    """Nonthermal-tail detection on the seed-merged snapshot.

    A tail is detected when the number above ``kappa T_core`` exceeds the
    core-Maxwellian expectation by ``TAIL_EXCESS_MIN`` and is supported by at
    least 100 effective counts (pooled over seeds).
    """
    merged = merge_seeds([row[t_index] for row in hists_by_seed])
    ex = core_tail_excess(merged, kappa)
    ex["tail"] = bool(ex["excess"] > TAIL_EXCESS_MIN and ex["neff_tail"] >= 100)
    return ex


def forecast_suite(hists_by_seed, times, train_mask, fit: dict, fit_window) -> dict:
    """Frozen-parameter forecast vs baselines on every held-out snapshot."""
    full = snapshot_set(hists_by_seed, times, fit_window)
    edges = hists_by_seed[0][0].edges
    eps = full.eps
    tr_idx = np.flatnonzero(train_mask)
    ho_idx = np.flatnonzero(~train_mask)
    i_ref = tr_idx[-1]
    t_ref = times[i_ref]
    merged = [merge_seeds([row[i] for row in hists_by_seed]) for i in range(len(times))]
    moms = [histogram_moments(m) for m in merged]
    K = np.array([m["energy_exact"] for m in moms])
    num = np.array([m["number"] for m in moms])
    law = forecast.fit_heating_law(times[tr_idx], K[tr_idx])
    sup_ref = full.support[i_ref] & full.fit_bins
    errs: Dict[str, List[dict]] = {}
    for i in ho_idx:
        obs, sig, sup = full.N[i], full.sigma_ln[i], full.support[i] & full.fit_bins
        preds = {
            "stationary": forecast.stationary_forecast(eps, full.N[i_ref], sup_ref, times[i]),
            "maxwellian_matched": forecast.matched_maxwellian_forecast(edges, num[i_ref], K[i_ref]),
            "heating_one_scale": forecast.heating_forecast(eps, full.N[i_ref], sup_ref, t_ref, times[i], law),
            "ORACLE_maxwellian": forecast.oracle_maxwellian(edges, num[i], K[i]),
        }
        if fit.get("status") == scaling.IDENTIFIED:
            preds["scaling"] = forecast.scaling_forecast(eps, full.N[i_ref], sup_ref, t_ref, times[i],
                                                         fit["alpha"], fit["beta"], fit["t0"])
        for name, p in preds.items():
            errs.setdefault(name, []).append(forecast.forecast_error(p, obs, sig, sup))
    summary = {}
    for name, lst in errs.items():
        summary[name] = {
            "rms_ln": float(np.nanmean([e["rms_ln"] for e in lst])),
            "chi2_red": float(np.nanmean([e["chi2_red"] for e in lst])),
            "coverage": float(np.mean([e["coverage"] for e in lst])),
        }
    ranking = forecast.rank_forecasts(summary)
    return {"errors": summary, "ranking": ranking, "heating_law": law}


def nonthermal_scaling_claim(fit: dict, fc: dict, tail: dict) -> dict:
    """All three must hold: identified exponents, a forecast that beats every
    non-oracle baseline by the frozen margin, and a nonthermal tail."""
    reasons = []
    if fit.get("status") != scaling.IDENTIFIED:
        reasons.append(f"fit status {fit.get('status')}")
    rank = fc["ranking"]
    if "scaling" not in fc["errors"]:
        reasons.append("no scaling forecast")
    else:
        best_base = min(fc["errors"][n]["rms_ln"] for n in rank if n != "scaling"
                        and fc["errors"][n]["coverage"] >= 0.5)
        if not (rank[0] == "scaling" and fc["errors"]["scaling"]["rms_ln"] <= FORECAST_MARGIN * best_base):
            reasons.append("scaling forecast does not beat baselines by the margin")
    if not tail.get("tail"):
        reasons.append("no nonthermal tail (Maxwellian-equivalent excess below threshold)")
    return {"claim": not reasons, "reasons": reasons}


def feature_trajectories(levels: dict, feature_fn) -> Dict[int, np.ndarray]:
    """level -> (n_seeds, n_times) feature values."""
    return {lvl: np.array([[feature_fn(h) for h in row] for row in ld.hists]) for lvl, ld in levels.items()}


def onset_by_level(levels: dict, eps_threshold: float, frac_threshold: float) -> Dict[int, np.ndarray]:
    out = {}
    for lvl, ld in levels.items():
        ons = []
        for row in ld.hists:
            q = np.array([tail_moments(h, eps_threshold)["Q_N"] for h in row])
            ons.append(convergence.onset_time(ld.times, q, frac_threshold))
        out[lvl] = np.array(ons)
    return out
