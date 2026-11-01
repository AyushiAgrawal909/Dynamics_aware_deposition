#!/usr/bin/env python3
"""Apply the frozen T1 analysis to a directory of T2 bundles.

    python3 experiments/nonthermal_attractor/scripts/t2_analyze.py CONFIG.yaml BUNDLE_DIR --out results/... [--label SMOKE]

Features (from the config's ``analysis`` block, frozen in T2_PROTOCOL.yaml):
* eps_q: energy above which a fraction q of all electrons lies;
* eta_min: lower edge of the eta^-1 range of rho(eta), with resolution flag;
* onset: first time the tail number above eps_tail crosses q_onset.
The PIC nested levels go through ``convergence_gate`` with the finest Vlasov
run as the independent reference; the onset model compares ln N with N.
Writes T2_DECISION.json and a short markdown summary.  Nothing is fitted
for scaling exponents (that is T3).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

from attractor_analysis import convergence, schema  # noqa: E402
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("bundles")
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", default="")
    a = ap.parse_args()
    cfg = yaml.safe_load(Path(a.config).read_text())
    an = cfg["analysis"]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    pic, vl, ctrl = {}, {}, {}
    for p in sorted(Path(a.bundles).glob("*.npz")):
        md, arr = schema.read_bundle(p)
        key = (md["nested_level"], md["seed"])
        if md.get("preparation") == "stable_control":
            ctrl[key] = (md, arr)
        else:
            (pic if md["backend"] == "pic1d1v" else vl)[key] = (md, arr)
    levels = sorted({k[0] for k in pic})
    seeds = sorted({k[1] for k in pic})
    times = pic[(levels[0], seeds[0])][1]["times"]
    edges = pic[(levels[0], seeds[0])][1]["eps_edges"]
    eps_tail = nearest_edge(edges, an["eps_tail"])
    feats = {"eps_q": {}, "eta_min": {}}
    onsets, flags_frac = {}, {}
    for l in levels:
        eq, em, on, fl = [], [], [], []
        for s in seeds:
            md, arr = pic[(l, s)]
            H = hists(arr)
            n0 = arr["number"][0]
            eq.append([quantile_feature(h, an["q"]) for h in H])
            e, f = eta_min_series(arr)
            em.append(e)
            win_t = (arr["times"] >= an["window"][0]) & (arr["times"] <= an["window"][1])
            fl.append(f[win_t].mean())
            qn = np.array([tail_moments(h, eps_tail)["Q_N"] for h in H]) / n0
            on.append(convergence.onset_time(times, qn, an["q_onset"]))
        feats["eps_q"][l] = np.array(eq)
        feats["eta_min"][l] = np.array(em)
        onsets[l] = np.array(on)
        flags_frac[l] = float(np.mean(fl))
    ppc = {l: pic[(l, seeds[0])][0]["particles_per_cell"] for l in levels}
    ref = None
    if vl:
        vmd, varr = vl[min(vl)]
        VH = hists(varr)
        ref = {"eps_q": np.array([quantile_feature(h, an["q"], min_counts=0) for h in VH]),
               "eta_min": eta_min_series(varr)[0]}
    window = tuple(an["window"])
    decision = {"label": a.label, "levels": levels, "ppc": ppc, "seeds": seeds, "eps_tail": eps_tail,
                "resolution_limited_fraction_eta": flags_frac, "features": {}}
    # eta_min enters the gate only at levels where it is resolved (not within
    # floor_factor of the one-particle floor) in all but max_limited_fraction of
    # the window snapshots; this is decided per level from the flags alone.
    lim_max = float(an.get("eta_min_max_limited_fraction", 0.05))
    level_sets = {"eps_q": levels,
                  "eta_min": [l for l in levels if flags_frac[l] <= lim_max]}
    decision["feature_levels"] = level_sets
    for name, F in feats.items():
        ls = level_sets[name]
        if name == "eta_min" and len(ls) < 3:
            g = {"verdict": convergence.INCONCLUSIVE, "unresolved": True, "n_levels": len(ls),
                 "reason": f"floor-limited: only {len(ls)} level(s) resolve eta_min"}
        else:
            g = convergence.convergence_gate(times, {l: F[l] for l in ls}, window, tau=an["tau"],
                                             reference=None if ref is None else ref[name],
                                             onset_by_level={l: onsets[l] for l in ls},
                                             regulator_by_level={l: ppc[l] for l in ls})
        decision["features"][name] = g
    decision["onsets"] = {str(l): onsets[l].tolist() for l in levels}
    # frozen quality gate: |dE|/K + control dK/K < 0.1 f_tail, per PIC level (end of window)
    qg = {}
    for l in levels:
        rows = []
        for s in seeds:
            md, arr = pic[(l, s)]
            i = int(np.searchsorted(times, window[1]) - 1)
            K0 = arr["kinetic_energy"][0]
            dE = abs(arr["total_energy"][i] - arr["total_energy"][0]) / K0
            H = hists(arr)
            f_tail = tail_moments(H[i], eps_tail)["Q_E"] / arr["kinetic_energy"][i]
            c = [ctrl[k][1] for k in ctrl if k[0] == l]
            dKc = float(np.mean([abs(cc["kinetic_energy"][i] / cc["kinetic_energy"][0] - 1) for cc in c])) if c else np.nan
            rows.append({"seed": s, "dE_over_K": float(dE), "control_dK_over_K": dKc, "f_tail": float(f_tail),
                         "pass": bool(np.isfinite(dKc) and dE + dKc < 0.1 * f_tail)})
        qg[l] = rows
    decision["quality_gate"] = {str(k): v for k, v in qg.items()}
    verdicts = [g["verdict"] for g in decision["features"].values()]
    resolved = [g["verdict"] for g in decision["features"].values() if not g.get("unresolved")]
    if any(v.startswith("PASS") for v in verdicts):
        decision["gate_outcome"] = "GO_T3"
    elif all(v == convergence.FAIL for v in verdicts):
        decision["gate_outcome"] = "REGULATOR_RESULT"
    elif resolved and len(resolved) < len(verdicts) and all(v == convergence.FAIL for v in resolved):
        # pre-registered: the energy-tail feature fails and eta_min is floor-limited
        decision["gate_outcome"] = "REGULATOR_RESULT_EPS_Q_ONLY"
    else:
        decision["gate_outcome"] = "INCONCLUSIVE"
    (out / "T2_DECISION.json").write_text(json.dumps(decision, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
    lines = [f"# T2 analysis {a.label}", "", f"Gate outcome: **{decision['gate_outcome']}**", "",
             "| Feature | Verdict | Reason |", "|---|---|---|"]
    for n, g in decision["features"].items():
        lines.append(f"| {n} | {g['verdict']} | {g.get('reason', '')} |")
    os_ = decision["features"]["eps_q"].get("onset_scaling", {})
    lines += ["", f"Onset model preferred: {os_.get('preferred')} (Delta AIC {os_.get('delta_aic')})", "",
              "Quality gate (|dE|/K + control dK/K < 0.1 f_tail) pass fraction per level: "
              + ", ".join(f"L{l}: {np.mean([r['pass'] for r in qg[l]]):.2f}" for l in levels)]
    (out / "T2_SUMMARY.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
