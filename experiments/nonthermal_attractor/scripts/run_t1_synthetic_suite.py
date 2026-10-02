#!/usr/bin/env python3
"""Run the T1 synthetic identifiability suite and write the confusion table.

Usage (from the repository root)::

    python3 experiments/nonthermal_attractor/scripts/run_t1_synthetic_suite.py [--quick]

Outputs (small, committed): ``results/t1_synthetic/summary.json``,
``results/t1_synthetic/confusion_table.md``, ``results/t1_synthetic/power_analysis.md``
and one example bundle that round-trips through the data-contract validator.
No plasma simulation is run.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

from attractor_analysis import schema, suite, synthetic as S  # noqa: E402
from attractor_analysis.energy import energy_histogram, histogram_moments  # noqa: E402
from attractor_analysis.waterbag import (particle_eta_floor, reconstruct_f_particles,  # noqa: E402
                                         waterbag_content)

OUT = HERE / "results" / "t1_synthetic"


def git_rev() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=HERE, text=True).strip()
    except Exception:
        return "unknown"


def _default(o):
    if isinstance(o, (np.generic,)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, float) and not np.isfinite(o):
        return str(o)
    raise TypeError(type(o))


def example_bundle() -> dict:
    """Write and re-read one S2 bundle to exercise the data contract."""
    ppc, seed = 512, 0
    times = S.TIMES["S2"]
    N = ppc * S.N_CELLS
    rng = np.random.default_rng(99)
    eta_edges = S.S7_ETA_EDGES
    nb_eps = len(S.EDGES) - 1
    rows = {k: [] for k in ("N_eps", "eps_bin_weight", "eps_bin_energy", "eps_bin_neff", "eps_overflow",
                            "rho_eta", "rho_bin_volume", "rho_bin_mass")}
    number, kin = [], []
    for t in times:
        u, u2 = rng.random(N), rng.random(N)
        eps = S.CASES["S2"](u, u2, t, ppc)
        w = np.full(N, 1.0 / N)
        h = energy_histogram(eps, w, S.EDGES)
        rows["N_eps"].append(h.N)
        rows["eps_bin_weight"].append(h.weight)
        rows["eps_bin_energy"].append(h.energy)
        rows["eps_bin_neff"].append(h.neff)
        rows["eps_overflow"].append([h.overflow_weight, h.overflow_energy])
        v = np.sqrt(2 * eps) * np.where(rng.random(N) < 0.5, -1, 1)
        x = rng.random(N)
        fh = reconstruct_f_particles(x, v, w, 1.0, 40.0, 8, 400)
        dxb, dvb = 1.0 / 8, 80.0 / 400
        wc = waterbag_content(fh, dxb, dvb, eta_edges, particle_eta_floor(w[0], dxb, dvb))
        rows["rho_eta"].append(wc.rho)
        rows["rho_bin_volume"].append(wc.volume)
        rows["rho_bin_mass"].append(wc.mass)
        m = histogram_moments(h)
        number.append(m["number"])
        kin.append(m["energy_exact"])
    nt = len(times)
    arrays = {k: np.asarray(v) for k, v in rows.items()}
    arrays.update(times=times, number=np.asarray(number), total_charge=-np.asarray(number),
                  momentum=np.zeros(nt), kinetic_energy=np.asarray(kin), field_energy=np.zeros(nt),
                  total_energy=np.asarray(kin), eps_edges=S.EDGES, eta_edges=eta_edges,
                  rho_reconstruction=np.array([1.0 / 8, 80.0 / 400, particle_eta_floor(1.0 / N, 1 / 8, 0.2)]),
                  k=np.array([0.0]), Ek2=np.zeros((nt, 1)))
    md = {"backend": "synthetic", "git_revision": git_rev(), "config": {"case": "S2", "ppc": ppc},
          "normalization": "synthetic: total number 1, T_core = 1", "seed": seed, "nested_level": 0,
          "dt": float("nan"), "dx": 1.0 / S.N_CELLS, "staggering": "none (synthetic snapshots)",
          "wall_time_s": 0.0, "synthetic_case": "S2", "n_particles": N}
    path = schema.write_bundle(OUT / "bundles" / "S2_ppc512_seed0.npz", md, arrays)
    md2, arr2 = schema.read_bundle(path)
    return {"path": str(path.relative_to(HERE)), "validated": True, "n_arrays": len(arr2)}


def main(quick: bool) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    ppcs = [512, 2048] if quick else [512, 2048, 8192]
    n_seeds = 3 if quick else 5
    results = []
    for ppc in ppcs:
        for case in ("S1", "S2", "S3", "S4", "S4b"):
            r = suite.evaluate_fit_case(case, ppc, n_seeds)
            print(f"{case:4s} ppc={ppc:5d} -> {r['verdict']}", flush=True)
            results.append(r)
        # coarsest level >= 512 ppc where possible, at least 3 levels, at most 5
        n_levels = int(np.clip(np.log2(ppc / 512) + 1, 3, 5))
        for case in ("S2", "S5", "S6"):
            r = suite.evaluate_gate_case(case, ppc, n_levels, n_seeds)
            r["case"] = case + "-gate" if case == "S2" else case
            print(f"{r['case']:7s} ppc={ppc:5d} levels={n_levels} -> {r['verdict']}", flush=True)
            results.append(r)
    s7 = suite.evaluate_s7(ppcs if not quick else [2048, 512])
    print(f"S7 -> {s7['verdict']}", flush=True)

    expected = dict(S.EXPECTED)
    expected["S2-gate"] = "PASS"
    positive = {"S2", "S2-gate", "S6", "S7"}
    weak = ("INCONCLUSIVE", "UNIDENTIFIABLE_DEGENERATE", "NO_CLAIM", "BOUNDARY_HIT", "NO_COLLAPSE")
    false_pos_markers = ("PASS", "IDENTIFIED_AND_FORECAST_WINS", "FALSE_ATTRACTOR_CLAIM")

    def classify(case, exp, got):
        if got == exp:
            return "match"
        if "INCONCLUSIVE" in got.split("+")[0] or got == "INCONCLUSIVE":
            return "conservative miss"
        if case in positive:
            return "conservative miss" if any(w in got for w in weak) else "**wrong**"
        if any(m in got for m in false_pos_markers) or got.startswith("PASS"):
            return "**FALSE POSITIVE**"
        return "**wrong**"

    table = ["| Case | ppc (finest) | Expected (resolved data) | Returned | Outcome |", "|---|---|---|---|---|"]
    n_match = 0
    counts = {}
    for r in results:
        exp = expected[r["case"]]
        ppc = r.get("ppc", r.get("ppc_levels", [None])[0])
        levels = r.get("ppc_levels")
        cls = classify(r["case"], exp, r["verdict"])
        r["outcome"] = cls
        counts[cls] = counts.get(cls, 0) + 1
        n_match += cls == "match"
        lab = f"{ppc}" + (f" ({len(levels)} levels to {levels[-1]})" if levels else "")
        table.append(f"| {r['case']} | {lab} | `{exp}` | `{r['verdict']}` | {cls} |")
    cls7 = classify("S7", expected["S7"], s7["verdict"])
    counts[cls7] = counts.get(cls7, 0) + 1
    n_match += cls7 == "match"
    table.append(f"| S7 | {', '.join(map(str, s7['ppc_levels']))} | `{expected['S7']}` | `{s7['verdict']}` | {cls7} |")
    total = len(results) + 1
    (OUT / "confusion_table.md").write_text(
        f"# T1 synthetic suite: expected vs returned verdicts\n\n{n_match}/{total} match; outcomes: "
        + ", ".join(f"{k}: {v}" for k, v in sorted(counts.items()))
        + f".\n\n{n_seeds} seeds per case, 64 cells, quick={quick}. 'Expected' is the verdict for data that "
        "resolve the relevant feature; a *conservative miss* (INCONCLUSIVE, DEGENERATE or no claim where a "
        "positive verdict was expected) is the designed behaviour when the feature is not supported by counts. "
        "A FALSE POSITIVE would be a failure of the analysis.\n\n" + "\n".join(table) + "\n")

    power = suite.onset_power_analysis(reps=200 if quick else 400)
    lines = ["# Onset-scaling power analysis (ln N vs N), effect size b = 0.25 per e-fold of N\n",
             "| onset scatter sigma | nested levels | seeds | P(prefer ln N given ln N) | P(prefer ln N given linear N) |",
             "|---|---|---|---|---|"]
    for p in power:
        lines.append(f"| {p['sigma']} | {p['levels']} | {p['seeds']} | {p['p_prefer_lnN_given_lnN']:.2f} "
                     f"| {p['p_prefer_lnN_given_linearN']:.2f} |")
    (OUT / "power_analysis.md").write_text("\n".join(lines) + "\n")

    bundle = example_bundle()
    summary = {"git_revision": git_rev(), "quick": quick, "n_seeds": n_seeds, "ppcs": ppcs,
               "matches": n_match, "total": total, "outcome_counts": counts, "results": results, "S7": s7,
               "power_analysis": power, "example_bundle": bundle,
               "wall_time_s": time.time() - t_start}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1, default=_default))
    print(f"{n_match}/{total} verdicts match; wall {time.time() - t_start:.0f} s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="smaller sizes for a fast check")
    main(ap.parse_args().quick)
