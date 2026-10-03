#!/usr/bin/env python3
"""Pre-freeze pilot checks on one coarse, full-length PIC bundle.

    python3 experiments/nonthermal_attractor/scripts/t2_pilot_check.py CONFIG.yaml BUNDLE.npz [--out JSON]

Closes the protocol items that need a full-length run before the freeze,
without looking at any level-to-level difference (one level, one seed, so
the regulator outcome cannot be read from it):
* eps_q (q from the config) is supported by >= min_counts effective
  particles above it at every snapshot in the analysis window;
* eta_min stays above floor_factor x the one-particle floor (not
  resolution-limited) in the window;
* the mass beyond 0.8 v_max of the Vlasov grid (energy > (0.8 v_max)^2/2,
  read at the bin edge just below, so the estimate is an upper bound);
* onset time and the energy-drift / tail-fraction ratio of the quality gate.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "scripts"))

from attractor_analysis import convergence, schema  # noqa: E402
from attractor_analysis.energy import quantile_feature, tail_moments  # noqa: E402
from attractor_analysis.waterbag import WaterbagContent, eta_minus1_range  # noqa: E402
from t2_analyze import hists, nearest_edge  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("bundle")
    ap.add_argument("--min-counts", type=float, default=100.0)
    ap.add_argument("--floor-factor", type=float, default=10.0)
    ap.add_argument("--out")
    a = ap.parse_args()
    cfg = yaml.safe_load(Path(a.config).read_text())
    an = cfg["analysis"]
    md, arr = schema.read_bundle(Path(a.bundle))
    t = arr["times"]
    H = hists(arr)
    win = (t >= an["window"][0]) & (t <= an["window"][1])

    eq = np.array([quantile_feature(h, an["q"], min_counts=0.0) for h in H])
    neff_above = np.array([float(np.sum(h.neff[np.searchsorted(h.edges, e, side="right") - 1:]))
                           if np.isfinite(e) else np.nan for h, e in zip(H, eq)])

    dxb, dvb, floor = arr["rho_reconstruction"]
    eta_min, eta_floor, limited = [], [], []
    for i in range(len(t)):
        wc = WaterbagContent(arr["eta_edges"], arr["rho_eta"][i], arr["rho_bin_volume"][i],
                             arr["rho_bin_mass"][i], dxb, dvb, floor)
        r = eta_minus1_range(wc, floor_factor=a.floor_factor)
        eta_min.append(r["eta_min"])
        eta_floor.append(r["eta_floor"])
        limited.append(bool(r["resolution_limited"]) or not r["found"])
    eta_min, eta_floor, limited = np.array(eta_min), np.array(eta_floor), np.array(limited)

    e_cut = 0.5 * (0.8 * cfg["vlasov"]["v_max"]) ** 2
    edges = H[0].edges
    j = max(int(np.searchsorted(edges, e_cut, side="right")) - 1, 0)
    n0 = float(arr["number"][0])
    frac_beyond = np.array([(np.sum(h.weight[j:]) + h.overflow_weight) / n0 for h in H])

    eps_tail = nearest_edge(edges, an["eps_tail"])
    qn = np.array([tail_moments(h, eps_tail)["Q_N"] for h in H]) / n0
    t_on = convergence.onset_time(t, qn, an["q_onset"])
    K0 = arr["kinetic_energy"][0]
    dE = np.abs(arr["total_energy"] - arr["total_energy"][0]) / K0
    f_tail = np.array([tail_moments(h, eps_tail)["Q_E"] for h in H]) / arr["kinetic_energy"]

    res = {
        "bundle": str(a.bundle), "backend": md["backend"], "ppc": md.get("particles_per_cell"),
        "seed": md["seed"], "t_end": float(t[-1]), "window": an["window"],
        "eps_q": {"q": an["q"], "min_neff_above_in_window": float(np.nanmin(neff_above[win])),
                  "required": a.min_counts, "range_in_window": [float(np.nanmin(eq[win])), float(np.nanmax(eq[win]))],
                  "pass": bool(np.all(neff_above[win] >= a.min_counts))},
        "eta_min": {"floor_factor": a.floor_factor, "resolution_limited_fraction_in_window": float(limited[win].mean()),
                    "min_ratio_to_floor_in_window": float(np.nanmin(eta_min[win] / eta_floor[win])),
                    "pass": bool(not limited[win].any())},
        "vlasov_vmax": {"v_max": cfg["vlasov"]["v_max"], "eps_cut_edge": float(edges[j]),
                        "max_fraction_beyond_0p8_vmax": float(frac_beyond.max()), "required_below": 1e-6,
                        "pass": bool(frac_beyond.max() < 1e-6)},
        "onset": {"eps_tail": eps_tail, "q_onset": an["q_onset"], "t_on": float(t_on) if t_on is not None else None},
        "quality": {"dE_over_K_end_window": float(dE[win][-1]), "f_tail_end_window": float(f_tail[win][-1]),
                    "max_dE_over_K": float(dE.max())},
    }
    txt = json.dumps(res, indent=1)
    print(txt)
    if a.out:
        Path(a.out).write_text(txt + "\n")


if __name__ == "__main__":
    main()
