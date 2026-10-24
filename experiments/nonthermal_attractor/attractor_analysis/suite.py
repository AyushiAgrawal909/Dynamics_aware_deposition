"""
attractor_analysis/suite.py
===========================

Evaluation of the synthetic cases S1-S7 into verdicts that can be compared
with ``synthetic.EXPECTED``.  Used by ``scripts/run_t1_synthetic_suite.py``
(full sizes) and by the tests (reduced sizes).
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from . import convergence, pipeline, scaling, synthetic as S
from .energy import histogram_moments, quantile_feature


GATE_FEATURE_Q = {"S2": 0.01, "S5": 0.005, "S6": 0.01}
S6_ONSET_QN = 0.03
GATE_WINDOW = {"S2": (1.0, 6.0), "S5": (1.0, 6.0), "S6": (1.5, 6.5)}
T0_BOUNDS = {"S6": (0.5, 1.45)}
DEFAULT_T0_BOUNDS = (-0.5, 0.5)


def evaluate_fit_case(case: str, ppc: int, n_seeds: int) -> dict:
    """S1-S4b: tail detection, scaling fit on training snapshots, forecasts, claim."""
    lv = S.generate_case(case, [ppc], n_seeds)
    ld = lv[0]
    train = ld.times <= S.TRAIN_END
    ss = pipeline.snapshot_set(ld.hists, ld.times, S.FIT_WINDOW[case], select=train)
    fit = scaling.fit_scaling(ss, t0_bounds=T0_BOUNDS.get(case, DEFAULT_T0_BOUNDS))
    tail = pipeline.detect_tail(ld.hists, int(np.flatnonzero(train)[-1]))
    fc = pipeline.forecast_suite(ld.hists, ld.times, train, fit, S.FIT_WINDOW[case])
    claim = pipeline.nonthermal_scaling_claim(fit, fc, tail)

    # estimator checks on the first snapshot of seed 0
    h = ld.hists[0][0]
    mom = histogram_moments(h)
    out = {"case": case, "ppc": ppc, "n_seeds": n_seeds, "fit": _slim_fit(fit), "tail": tail,
           "forecast": {"errors": fc["errors"], "ranking": fc["ranking"]}, "claim": claim,
           "moments_first_snapshot": mom}
    if case == "S1":
        # unbiasedness: mean energy per particle vs T/2 across seeds
        e = np.array([histogram_moments(row[0])["energy_exact"] for row in ld.hists])
        se = 1.0 / np.sqrt(2.0) / np.sqrt(ppc * S.N_CELLS * n_seeds)  # sd(eps)=T/sqrt2
        out["energy_bias_sigma"] = float((e.mean() - 0.5) / se)
        unbiased = abs(out["energy_bias_sigma"]) < 3.0
        out["verdict"] = "NO_TAIL" if (not tail["tail"] and unbiased) else "S1_FAILED"
    elif case == "S2":
        ok = (fit["status"] == scaling.IDENTIFIED and claim["claim"]
              and _covers(fit["alpha"], fit["alpha_err"], S.ALPHA_TRUE)
              and _covers(fit["beta"], fit["beta_err"], S.BETA_TRUE))
        out["verdict"] = "IDENTIFIED_AND_FORECAST_WINS" if ok else f"S2_{fit['status']}"
    elif case == "S3":
        comb = fit.get("identified_combination", {})
        ok = fit["status"] == scaling.DEGENERATE and abs(comb.get("value", np.inf)) < max(3 * comb.get("err", 0), 0.05)
        out["verdict"] = scaling.DEGENERATE if ok else f"S3_{fit['status']}"
    else:  # S4, S4b
        out["verdict"] = "NO_CLAIM" if not claim["claim"] else "FALSE_ATTRACTOR_CLAIM"
    return out


def _covers(value, err, truth, k=3.0, floor=0.05):
    return bool(np.isfinite(value) and abs(value - truth) <= max(k * err, floor))


def _slim_fit(fit: dict) -> dict:
    return {k: v for k, v in fit.items() if k != "landscape"}


def evaluate_gate_case(case: str, ppc_finest: int, n_levels: int, n_seeds: int) -> dict:
    """S2 (positive control), S5, S6: regulator gate on a moving quantile feature."""
    ppcs = [ppc_finest >> l for l in range(n_levels)]
    lv = S.generate_case(case, ppcs, n_seeds)
    q = GATE_FEATURE_Q[case]
    feats = pipeline.feature_trajectories(lv, lambda h: quantile_feature(h, q))
    times = lv[0].times
    onsets = regulators = None
    if case == "S6":
        onsets = pipeline.onset_by_level(lv, S.TAIL_THRESHOLD, S6_ONSET_QN)
        regulators = {l: float(lv[l].ppc) for l in lv}
    gate = convergence.convergence_gate(times, feats, GATE_WINDOW[case], tau=0.1,
                                        onset_by_level=onsets, regulator_by_level=regulators)
    out = {"case": case, "ppc_levels": ppcs, "n_seeds": n_seeds, "gate": _slim_gate(gate)}
    if case == "S2":
        out["verdict"] = gate["verdict"]
    elif case == "S5":
        out["verdict"] = gate["verdict"]
    else:
        # S6 also needs (alpha, beta) recovered at the finest level after t0 profiling
        ld = lv[0]
        train = ld.times <= S.TRAIN_END + 0.5
        ss = pipeline.snapshot_set(ld.hists, ld.times, S.FIT_WINDOW[case], select=train)
        fit = scaling.fit_scaling(ss, t0_bounds=T0_BOUNDS["S6"])
        out["fit"] = _slim_fit(fit)
        out["t0_true_finest"] = S.s6_t0(ppcs[0])
        pref = gate.get("onset_scaling", {}).get("preferred")
        ok_fit = (fit["status"] == scaling.IDENTIFIED and _covers(fit["alpha"], fit["alpha_err"], S.ALPHA_TRUE)
                  and _covers(fit["beta"], fit["beta_err"], S.BETA_TRUE))
        parts = [gate["verdict"], pref if pref else "NO_ONSET_FIT", "IDENTIFIED" if ok_fit else fit["status"]]
        out["verdict"] = "+".join(parts)
    return out


def _slim_gate(g: dict) -> dict:
    return g


def evaluate_s7(ppc_levels: Sequence[int], times=None) -> dict:
    times = np.arange(1.0, 6.5, 1.0) if times is None else times
    rows = S.run_s7(ppc_levels, times)
    v = S.s7_verdict(rows)
    return {"case": "S7", "ppc_levels": list(ppc_levels), "rows": rows, **v}


def onset_power_analysis(b: float = 0.25, a: float = 0.5, sigmas=(0.05, 0.1, 0.2),
                         seeds_list=(2, 3, 4, 6, 8), levels_list=(3, 4, 5), ppc_finest: int = 8192,
                         reps: int = 400, rng_seed: int = 11) -> list:
    """Probability that the onset-scaling comparison prefers ln N when the
    truth is t_on = a + b ln N + sigma xi (xi ~ N(0,1) per run).

    Effect size b = 0.25 matches S6.  sigma is the run-to-run onset scatter,
    which T2 must measure; the table tells T2 how many seeds and nested levels
    it needs at a given scatter.  The false-preference rate (truth = linear
    in N with the same total onset shift over the ladder) is also reported.
    """
    rng = np.random.default_rng(rng_seed)
    rows = []
    for sigma in sigmas:
        for L in levels_list:
            Ns = np.array([ppc_finest >> l for l in range(L)], float)
            for ns in seeds_list:
                hit = wrong = 0
                for _ in range(reps):
                    Nrep = np.repeat(Ns, ns)
                    t = a + b * np.log(Nrep / Ns[-1]) + sigma * rng.standard_normal(Nrep.size)
                    r = convergence.onset_scaling_fit(Nrep, t)
                    hit += r["preferred"] == "lnN"
                    # alternative truth: linear in N with the same total shift
                    slope = b * np.log(Ns[0] / Ns[-1]) / (Ns[0] - Ns[-1])
                    t2 = a + slope * (Nrep - Ns[-1]) + sigma * rng.standard_normal(Nrep.size)
                    r2 = convergence.onset_scaling_fit(Nrep, t2)
                    wrong += r2["preferred"] == "lnN"
                rows.append({"sigma": sigma, "levels": L, "seeds": ns,
                             "p_prefer_lnN_given_lnN": hit / reps,
                             "p_prefer_lnN_given_linearN": wrong / reps})
    return rows
