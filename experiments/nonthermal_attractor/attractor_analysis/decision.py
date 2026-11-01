"""
attractor_analysis/decision.py
==============================

Protocol-v2 T2 decision: prerequisites first, then the scientific gate.

The v1 analysis chose GO_T3 / REGULATOR_RESULT from the feature verdicts
alone and only printed the quality gate (review F1); it took one Vlasov run
as the reference and never checked the campaign against the protocol (F6).
Here every step that can make a verdict meaningless is a veto:

``build_manifest``  -> the declared run matrix, nothing missing, duplicated or
                       unexpected, frozen configs and accepted revisions, one
                       common time grid that reaches t_end;
``run_quality``     -> per-run numerical budgets (PIC energy + stable-control
                       heating vs the tail energy, momentum, number; Vlasov
                       mass identity, clipping, boundary outflow, energy);
``vlasov_ladder``   -> the four Vlasov levels as a ladder: the reference is the
                       finest level, admissible only if the ladder converges,
                       with the finest step as its uncertainty;
``decide_campaign`` -> a pure function of those results and the feature
                       gates.  Missing or failed evidence gives INVALID_* or
                       INCONCLUSIVE with reasons, never a scientific outcome.
"""

from __future__ import annotations

from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from . import convergence as C

GO_T3 = "GO_T3"
STATIONARY_RESULT = "STATIONARY_RESULT"
REGULATOR_RESULT = "REGULATOR_RESULT"
REGULATOR_RESULT_EPS_Q_ONLY = "REGULATOR_RESULT_EPS_Q_ONLY"
INCONCLUSIVE = "INCONCLUSIVE"
INVALID_INPUTS = "INVALID_INPUTS"
INVALID_NUMERICS = "INVALID_NUMERICS"
INCOMPLETE = "INCOMPLETE"

RunKey = Tuple[str, int, int]  # (kind, level, seed); kind in pic, control, vlasov


def run_kind(md: Mapping) -> str:
    if md.get("preparation") == "stable_control":
        return "control"
    return {"pic1d1v": "pic", "vlasov1d1v": "vlasov"}.get(md.get("backend"), "unknown")


def expected_runs(protocol: Mapping) -> Dict[str, set]:
    """The declared matrix: required keys, and optional keys (conditional seeds)."""
    scan = protocol["regulator_scan"]
    req, opt = set(), set()
    for l in scan["pic_levels"]:
        for s in scan["pic_seeds"]:
            req.add(("pic", l, s))
        for s in scan.get("pic_conditional_seeds", []):
            opt.add(("pic", l, s))
        for s in scan["control_seeds"]:
            req.add(("control", l, s))
    for l in scan["vlasov_levels"]:
        req.add(("vlasov", l, 0))
    return {"required": req, "optional": opt}


def build_manifest(bundles: Sequence[Tuple[Mapping, Mapping, str]], protocol: Mapping,
                   configs: Mapping[str, Mapping]) -> dict:
    """Index the bundles and list every way the campaign deviates from the protocol.

    ``bundles``: (metadata, arrays, path).  ``configs``: kind -> the frozen
    config dict that kind must have been run with.
    """
    runs: Dict[RunKey, Tuple[Mapping, Mapping]] = {}
    problems: List[str] = []
    exp = expected_runs(protocol)
    accepted = protocol["accepted_revisions"]
    for md, arr, path in bundles:
        kind = run_kind(md)
        key = (kind, int(md["nested_level"]), int(md["seed"]))
        if kind == "unknown":
            problems.append(f"{path}: backend {md.get('backend')!r} is not part of T2")
            continue
        if key in runs:
            problems.append(f"{path}: duplicate run {key}")
            continue
        if key not in exp["required"] and key not in exp["optional"]:
            problems.append(f"{path}: run {key} is not in the declared matrix")
            continue
        cfg_kind = "vlasov" if kind == "vlasov" else "pic"
        if md.get("config") != configs[cfg_kind]:
            problems.append(f"{path}: config differs from the frozen {cfg_kind} config")
        rev = str(md.get("git_revision", ""))
        if not any(rev.startswith(a) or a.startswith(rev) for a in accepted[cfg_kind] if a):
            problems.append(f"{path}: git revision {rev[:12]} not accepted for {cfg_kind} runs")
        runs[key] = (md, arr)
    missing = sorted(exp["required"] - set(runs))
    for key in missing:
        problems.append(f"missing run {key}")
    # one common time grid, reaching t_end at the declared cadence
    t_end = float(protocol["time"]["t_end"])
    every = float(protocol["time"]["snapshot_every"])
    ref_t = None
    for key, (md, arr) in sorted(runs.items()):
        t = np.asarray(arr["times"], dtype=np.float64)
        if not np.isclose(t[-1], t_end):
            problems.append(f"run {key} ends at t = {t[-1]:g}, not t_end = {t_end:g}")
        if ref_t is None:
            ref_t = t
            if not np.allclose(np.diff(t), every):
                problems.append(f"run {key}: snapshot spacing is not {every:g}")
        elif t.shape != ref_t.shape or not np.allclose(t, ref_t):
            problems.append(f"run {key}: time grid differs from the other runs")
        for name in ("number", "kinetic_energy", "total_energy", "momentum", "N_eps"):
            if not np.all(np.isfinite(arr[name])):
                problems.append(f"run {key}: non-finite {name}")
    return {"runs": runs, "problems": problems, "missing": missing, "times": ref_t}


def end_index(times: np.ndarray, t_end: float) -> int:
    """Index of the snapshot at t_end, inclusive (v1 took the one before)."""
    t = np.asarray(times, dtype=np.float64)
    hit = np.flatnonzero(np.isclose(t, t_end))
    return int(hit[0]) if hit.size else int(np.searchsorted(t, t_end, side="right") - 1)


def run_quality(runs: Mapping[RunKey, Tuple[Mapping, Mapping]], protocol: Mapping,
                f_tail: Mapping[RunKey, float]) -> Dict[RunKey, dict]:
    """Per-run numerical-quality vetoes.  ``f_tail``: run -> tail energy
    fraction above eps_tail at t_end (the scale every energy error is held to)."""
    q = protocol["quality"]
    t_end = float(protocol["time"]["t_end"])
    out: Dict[RunKey, dict] = {}
    controls = {k: v for k, v in runs.items() if k[0] == "control"}
    for key, (md, arr) in runs.items():
        if key[0] == "control":
            continue
        i = end_index(arr["times"], t_end)
        N0 = float(arr["number"][0])
        K0 = float(arr["kinetic_energy"][0])
        dE = abs(float(arr["total_energy"][i]) - float(arr["total_energy"][0])) / K0
        ft = float(f_tail[key])
        row = {"dE_over_K": dE, "f_tail": ft, "fails": []}
        if key[0] == "pic":
            c = [cv for ck, cv in controls.items() if ck[1] == key[1]]
            dKc = (float(np.mean([abs(a["kinetic_energy"][i] / a["kinetic_energy"][0] - 1) for _, a in c]))
                   if c else np.nan)
            row["control_dK_over_K"] = dKc
            if not np.isfinite(dKc):
                row["fails"].append("no stable control at this level")
            elif not dE + dKc < q["energy_factor"] * ft:
                row["fails"].append(f"|dE|/K + control |dK|/K = {dE + dKc:.3g} >= "
                                    f"{q['energy_factor']} f_tail = {q['energy_factor'] * ft:.3g}")
            p = np.asarray(arr["momentum"], dtype=np.float64)
            p_scale = np.sqrt(2.0 * K0 * N0)
            row["momentum_drift"] = float(np.max(np.abs(p - p[0])) / p_scale)
            if not row["momentum_drift"] < q["pic_momentum_rel"]:
                row["fails"].append(f"momentum drift {row['momentum_drift']:.3g} not at round-off")
            dN = float(np.max(np.abs(np.asarray(arr["number"]) / N0 - 1)))
            if not dN < q["pic_number_rel"]:
                row["fails"].append(f"particle number changed by {dN:.3g}")
        else:  # vlasov
            num = np.asarray(arr["number"], dtype=np.float64)
            out_m = np.asarray(arr.get("outflow_mass", np.full_like(num, np.nan)), dtype=np.float64)
            if "clip_mass" in arr and np.all(np.isfinite(arr["clip_mass"])):
                clip = np.asarray(arr["clip_mass"], dtype=np.float64)
                resid = float(np.max(np.abs(num - num[0] + out_m - clip)) / N0)
                row.update(clip_mass=float(clip[i] / N0), outflow_mass=float(out_m[i] / N0),
                           mass_identity_residual=resid)
                if not resid < q["vlasov_identity_rel"]:
                    row["fails"].append(f"mass identity residual {resid:.3g}")
            else:
                # v1 bundles net the clipping gain into outflow_mass, so only
                # the total mass change is meaningful
                row.update(clip_mass=np.nan, outflow_mass=np.nan, legacy_budget=True)
                row["fails"].append("bundle has no separate clipping budget (protocol-v1 solver)")
            dM = float(abs(num[i] / N0 - 1))
            row["mass_change"] = dM
            if np.isfinite(row["clip_mass"]) and not row["clip_mass"] < q["vlasov_clip_rel"]:
                row["fails"].append(f"clipping added {row['clip_mass']:.3g} of the mass")
            if np.isfinite(row["outflow_mass"]) and not row["outflow_mass"] < q["vlasov_outflow_rel"]:
                row["fails"].append(f"boundary outflow {row['outflow_mass']:.3g} of the mass")
            if not dM < q["vlasov_mass_rel"]:
                row["fails"].append(f"total mass changed by {dM:.3g}")
            if not dE < q["energy_factor"] * ft:
                row["fails"].append(f"|dE|/K = {dE:.3g} >= {q['energy_factor']} f_tail = "
                                    f"{q['energy_factor'] * ft:.3g}")
        row["pass"] = not row["fails"]
        out[key] = row
    return out


def usable_pic_levels(quality: Mapping[RunKey, dict], levels: Sequence[int], seeds: Sequence[int]) -> List[int]:
    """Levels where every seed passes, contiguous from the finest level."""
    out = []
    for l in sorted(levels):
        if all(quality.get(("pic", l, s), {}).get("pass", False) for s in seeds):
            out.append(l)
        else:
            break
    return out


def vlasov_ladder(times: np.ndarray, feature_by_level: Mapping[int, np.ndarray], window, tol: float) -> dict:
    """The Vlasov levels as a ladder (finest = level 0, Delta v doubling).

    D_l = RMS over the window of F_l - F_{l-1}.  The finest level is an
    admissible reference only if D_1 < tol and the differences do not grow
    under refinement (D_1 <= D_2); its uncertainty is |F_0 - F_1| per time.
    """
    t = np.asarray(times, dtype=np.float64)
    win = (t >= window[0]) & (t <= window[1])
    levels = sorted(feature_by_level)
    out = {"levels": levels, "tol": tol}
    if len(levels) < 2 or levels[:2] != [0, 1]:
        out.update(converged=False, reason="the two finest Vlasov levels are needed")
        return out
    F = {l: np.asarray(feature_by_level[l], dtype=np.float64) for l in levels}
    D = {}
    for lf, l in zip(levels[:-1], levels[1:]):
        if l != lf + 1:
            break
        D[l] = float(np.sqrt(np.nanmean((F[l][win] - F[lf][win]) ** 2)))
    out["D"] = D
    ok = np.isfinite(D[1]) and D[1] < tol and (2 not in D or D[1] <= D[2])
    out["converged"] = bool(ok)
    out["reason"] = ("finest Vlasov step below tolerance and shrinking" if ok else
                     "Vlasov ladder not converged within tolerance")
    out["reference"] = F[0]
    out["reference_err"] = np.abs(F[0] - F[1])
    return out


def onset_seed_rule(finest_onsets: np.ndarray, onset_fit: Mapping, threshold: float) -> dict:
    """Registered conditional rule: if the onset scatter of the finest-level
    seeds exceeds ``threshold`` x the onset effect size, seeds 4-7 are needed.

    v1 states the effect size as "the S6 effect size" (b = 0.25 in the
    synthetic units of T1).  Protocol v2 reads it as the fitted ln N
    coefficient b of the plasma onsets, the same quantity in plasma units.
    """
    sigma = float(np.nanstd(np.asarray(finest_onsets, dtype=np.float64), ddof=1))
    b = np.nan
    if onset_fit.get("models", {}).get("lnN"):
        b = float(onset_fit["models"]["lnN"]["coef"][1])
    ratio = sigma / abs(b) if np.isfinite(b) and b != 0 else np.inf
    return {"sigma_t_on": sigma, "b_lnN": b, "ratio": float(ratio), "triggered": bool(ratio > threshold)}


def decide_campaign(manifest_problems: Sequence[str], finest_pic_ok: bool, n_usable_levels: int,
                    seed_rule: Optional[Mapping], conditional_seeds_present: bool,
                    features: Mapping[str, Mapping], reference_ok: Mapping[str, bool],
                    min_levels: int = 3) -> dict:
    """Pure campaign decision.

    ``features``: name -> gate result (``verdict``, optional ``unresolved``,
    ``fail_kind``).  ``reference_ok``: name -> whether that feature's Vlasov
    reference is admissible (quality passed and ladder converged).
    """
    reasons: List[str] = []
    if manifest_problems:
        return {"outcome": INVALID_INPUTS, "reasons": list(manifest_problems)}
    if not finest_pic_ok:
        return {"outcome": INVALID_NUMERICS, "reasons": ["the finest PIC level fails the quality gate"]}
    if seed_rule is not None and seed_rule.get("triggered") and not conditional_seeds_present:
        return {"outcome": INCOMPLETE,
                "reasons": [f"onset scatter / effect size = {seed_rule['ratio']:.3g} triggers the "
                            "conditional rule: run seeds 4-7 at every PIC level first"]}
    if n_usable_levels < min_levels:
        return {"outcome": INCONCLUSIVE,
                "reasons": [f"only {n_usable_levels} contiguous PIC level(s) pass the quality gate"]}

    positive, stationary, fails, other = [], [], [], []
    for name, g in features.items():
        v = g["verdict"]
        if g.get("unresolved"):
            reasons.append(f"{name}: unresolved ({g.get('reason', '')})")
            continue
        if v in (C.PASS, C.PASS_ALIGNED, C.STATIONARY) and not reference_ok.get(name, False):
            reasons.append(f"{name}: {v} but the Vlasov reference is not admissible")
            other.append(name)
        elif v in (C.PASS, C.PASS_ALIGNED):
            positive.append(name)
        elif v == C.STATIONARY:
            stationary.append(name)
        elif v == C.FAIL and g.get("fail_kind") == "reference":
            if reference_ok.get(name, False):
                reasons.append(f"{name}: PIC converges but disagrees with an admissible Vlasov reference "
                               "(backend disagreement, not regulator dependence)")
            else:
                reasons.append(f"{name}: disagrees with an inadmissible reference")
            other.append(name)
        elif v == C.FAIL:
            fails.append(name)
        else:
            reasons.append(f"{name}: {v} ({g.get('reason', '')})")
            other.append(name)
    resolved = positive + stationary + fails + other
    if positive:
        outcome = GO_T3
    elif stationary and not fails:
        outcome = STATIONARY_RESULT
    elif fails and not positive and not stationary and not other:
        outcome = REGULATOR_RESULT if len(resolved) == len(features) else (
            REGULATOR_RESULT_EPS_Q_ONLY if resolved == ["eps_q"] else REGULATOR_RESULT)
    else:
        outcome = INCONCLUSIVE
    return {"outcome": outcome, "reasons": reasons, "positive": positive, "stationary": stationary,
            "regulator_fail": fails}
