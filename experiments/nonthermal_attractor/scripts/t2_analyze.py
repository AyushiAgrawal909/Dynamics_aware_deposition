#!/usr/bin/env python3
"""Protocol-v2 T2 analysis: validate the campaign, then apply the gate.

    python3 experiments/nonthermal_attractor/scripts/t2_analyze.py protocol/T2_PROTOCOL_v2.yaml \\
        BUNDLE_DIR [BUNDLE_DIR ...] --out results/... [--label LABEL]

Every bundle in the given directories is read.  The campaign must match the
declared matrix (attractor_analysis.decision.build_manifest); per-run
numerical quality is a veto (run_quality); the four Vlasov levels are used as
a ladder (vlasov_ladder) and the finest is the reference only if it
converges; then each feature goes through convergence_gate with the v2
stationary branch, and decide_campaign issues the outcome.  Writes
T2_DECISION.json and T2_SUMMARY.md.  The v1 analysis is scripts/t2_analyze_v1.py.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

from attractor_analysis import convergence, decision as D, schema  # noqa: E402
from attractor_analysis.energy import EnergyHistogram, quantile_feature, tail_moments  # noqa: E402
from attractor_analysis.waterbag import WaterbagContent, eta_minus1_range  # noqa: E402


def hists(arr):
    out = []
    for i in range(len(arr["times"])):
        out.append(EnergyHistogram(arr["eps_edges"], arr["N_eps"][i], arr["eps_bin_weight"][i],
                                   arr["eps_bin_energy"][i], arr["eps_bin_neff"][i],
                                   float(arr["eps_overflow"][i, 0]), float(arr["eps_overflow"][i, 1])))
    return out


def eta_min_series(arr):
    dxb, dvb, floor = arr["rho_reconstruction"]
    vals, flags = [], []
    for i in range(len(arr["times"])):
        wc = WaterbagContent(arr["eta_edges"], arr["rho_eta"][i], arr["rho_bin_volume"][i],
                             arr["rho_bin_mass"][i], dxb, dvb, floor)
        r = eta_minus1_range(wc)
        vals.append(r["eta_min"] if r["found"] and not r["resolution_limited"] else np.nan)
        flags.append(bool(r["resolution_limited"]) or not r["found"])
    return np.array(vals), np.array(flags)


def nearest_edge(edges, e):
    return float(edges[np.argmin(np.abs(edges - e))])


def run_features(arr, an, eps_tail, grid_backend):
    """eps_q, eta_min (+ floor flags), onset, and f_tail at t_end for one run."""
    H = hists(arr)
    min_counts = 0 if grid_backend else 100
    eq = np.array([quantile_feature(h, an["q"], min_counts=min_counts) for h in H])
    em, fl = eta_min_series(arr)
    n0 = arr["number"][0]
    qn = np.array([tail_moments(h, eps_tail)["Q_N"] for h in H]) / n0
    t_on = convergence.onset_time(arr["times"], qn, an["q_onset"])
    return {"eps_q": eq, "eta_min": em, "eta_flags": fl, "onset": t_on, "H": H}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("protocol")
    ap.add_argument("bundles", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", default="")
    a = ap.parse_args()
    proto = yaml.safe_load(Path(a.protocol).read_text())
    configs = {k: yaml.safe_load((HERE / v).read_text()) for k, v in proto["configs"].items()}
    an = proto["analysis"]
    window = tuple(an["window"])
    t_end = float(proto["time"]["t_end"])
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    bundles = []
    for d in a.bundles:
        for p in sorted(Path(d).glob("*.npz")):
            md, arr = schema.read_bundle(p)
            bundles.append((md, arr, str(p)))
    man = D.build_manifest(bundles, proto, configs)
    runs = man["runs"]
    decision = {"label": a.label, "protocol_version": proto["protocol_version"],
                "protocol_status": proto["status"], "manifest_problems": man["problems"]}
    pic_keys = sorted(k for k in runs if k[0] == "pic")
    if man["problems"] and not pic_keys:
        decision.update(D.decide_campaign(man["problems"], False, 0, None, False, {}, {}))
        return write(out, decision, a.label)

    edges = runs[pic_keys[0]][1]["eps_edges"]
    eps_tail = nearest_edge(edges, an["eps_tail"])
    times = man["times"]
    i_end = D.end_index(times, t_end)
    feats = {}
    f_tail = {}
    for key, (md, arr) in runs.items():
        if key[0] == "control":
            continue
        rf = run_features(arr, an, eps_tail, grid_backend=(key[0] == "vlasov"))
        feats[key] = rf
        f_tail[key] = tail_moments(rf["H"][i_end], eps_tail)["Q_E"] / arr["kinetic_energy"][i_end]
    quality = D.run_quality(runs, proto, f_tail)
    decision["quality"] = {str(k): v for k, v in quality.items()}

    scan = proto["regulator_scan"]
    seeds = list(scan["pic_seeds"])
    cond = [s for s in scan.get("pic_conditional_seeds", []) if all(("pic", l, s) in runs for l in scan["pic_levels"])]
    seeds_used = seeds + cond
    usable = D.usable_pic_levels(quality, scan["pic_levels"], seeds_used)
    decision["usable_pic_levels"] = usable
    finest_ok = bool(usable) and usable[0] == min(scan["pic_levels"])
    ppc = {l: runs[("pic", l, seeds[0])][0]["particles_per_cell"] for l in scan["pic_levels"] if ("pic", l, seeds[0]) in runs}

    def traj(name, l):
        return np.array([feats[("pic", l, s)][name] for s in seeds_used])

    # eta_min enters only at levels resolved in >= 1 - max_limited_fraction of window snapshots
    win_t = (times >= window[0]) & (times <= window[1])
    lim_frac = {l: float(np.mean([feats[("pic", l, s)]["eta_flags"][win_t].mean() for s in seeds_used]))
                for l in usable}
    decision["resolution_limited_fraction_eta"] = lim_frac
    eta_levels = []
    for l in usable:  # contiguous from the finest only (review F7)
        if lim_frac[l] <= an["eta_min_max_limited_fraction"]:
            eta_levels.append(l)
        else:
            break
    level_sets = {"eps_q": usable, "eta_min": eta_levels}
    decision["feature_levels"] = level_sets

    # Vlasov ladder per feature
    vl_levels = sorted(k[1] for k in runs if k[0] == "vlasov" and quality[k]["pass"])
    vl_dv = {k[1]: runs[k][0]["dv"] for k in runs if k[0] == "vlasov"}
    ladders, ref_ok = {}, {}
    gates = {}
    onsets = {l: np.array([feats[("pic", l, s)]["onset"] for s in seeds_used]) for l in usable}
    for name in ("eps_q", "eta_min"):
        ls = level_sets[name]
        tol_rel = an["features"][name]["abs_tol_rel"]
        finest_mean = float(np.nanmean(traj(name, usable[0])[:, win_t])) if usable else np.nan
        abs_tol = tol_rel * abs(finest_mean) if np.isfinite(finest_mean) else None
        floor_rel = an["features"][name]["motion_floor_rel"]
        motion_floor = floor_rel * abs(finest_mean) if np.isfinite(finest_mean) else None
        lad = D.vlasov_ladder(times, {l: feats[("vlasov", l, 0)][name] for l in vl_levels}, window,
                              abs_tol if abs_tol is not None else np.inf)
        ladders[name] = {k: v for k, v in lad.items() if k not in ("reference", "reference_err")}
        ref_ok[name] = bool(lad["converged"])
        ref = lad.get("reference") if lad["converged"] else None
        ref_err = lad.get("reference_err") if lad["converged"] else None
        ref_on = feats[("vlasov", 0, 0)]["onset"] if ("vlasov", 0, 0) in feats else None
        if name == "eta_min" and len(ls) < an["min_levels"]:
            gates[name] = {"verdict": convergence.INCONCLUSIVE, "unresolved": True, "n_levels": len(ls),
                           "reason": f"floor-limited: only {len(ls)} level(s) resolve eta_min"}
            continue
        gates[name] = convergence.convergence_gate(
            times, {l: traj(name, l) for l in ls}, window, tau=an["tau"], reference=ref, reference_err=ref_err,
            onset_by_level={l: onsets[l] for l in ls}, regulator_by_level={l: ppc[l] for l in ls},
            min_levels=an["min_levels"], min_seeds=an["min_seeds"], shrink_slope=an["shrink_slope"],
            abs_tol=abs_tol, motion_floor=motion_floor, reference_onset=ref_on)
        gates[name]["abs_tol"] = abs_tol
        gates[name]["motion_floor"] = motion_floor
    decision["vlasov_ladder"] = ladders
    decision["features"] = gates

    onset_fit = gates["eps_q"].get("onset_scaling", {})
    rule = D.onset_seed_rule(onsets[usable[0]], onset_fit, an["onset_seed_rule_threshold"]) if usable else None
    decision["onset_seed_rule"] = rule
    vl_on = {l: feats[("vlasov", l, 0)]["onset"] for l in vl_levels}
    decision["vlasov_onsets"] = vl_on
    if len(vl_on) >= 3:
        decision["vlasov_onset_scaling"] = convergence.onset_scaling_fit(
            np.array([1.0 / vl_dv[l] for l in vl_on]), np.array(list(vl_on.values())))
    decision["onsets"] = {str(l): v.tolist() for l, v in onsets.items()}
    decision.update(D.decide_campaign(man["problems"], finest_ok, len(usable), rule, bool(cond), gates, ref_ok,
                                      min_levels=an["min_levels"]))
    return write(out, decision, a.label)


def write(out, decision, label):
    (out / "T2_DECISION.json").write_text(json.dumps(
        decision, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
    lines = [f"# T2 analysis (protocol v{decision['protocol_version']}) {label}", "",
             f"Gate outcome: **{decision['outcome']}**", ""]
    for r in decision.get("reasons", []):
        lines.append(f"- {r}")
    if decision.get("features"):
        lines += ["", "| Feature | Verdict | Reason |", "|---|---|---|"]
        for n, g in decision["features"].items():
            lines.append(f"| {n} | {g['verdict']} | {g.get('reason', '')} |")
    if decision.get("usable_pic_levels") is not None:
        lines += ["", f"PIC levels passing the quality gate: {decision['usable_pic_levels']}"]
    (out / "T2_SUMMARY.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return decision


if __name__ == "__main__":
    main()
