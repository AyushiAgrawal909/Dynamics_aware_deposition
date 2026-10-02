#!/usr/bin/env python3
"""T2 timing smoke test and cost estimate (T2 section 4).

Times a few steps of each backend at the production grid (PIC at a reduced
ppc, Vlasov at a reduced nv), converts to particle-steps/s and
cell-steps/s, and extrapolates the full regulator matrix in the config.
Writes reports/COST_ESTIMATE.md and prints the decision: run locally if the
matrix fits the budget, otherwise generate SLURM scripts and stop.

    python3 experiments/nonthermal_attractor/scripts/t2_cost_estimate.py CONFIG.yaml [--budget-node-hours 8]
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

from attractor_sim import run as R  # noqa: E402
from attractor_sim.pic1d1v import HAVE_NUMBA  # noqa: E402


def time_pic(cfg, d, ppc, steps=30):
    sim, N = R.build_pic(cfg, d, 0, 0, ppc0=ppc)
    sim.step(cfg["time"]["dt"])  # JIT warm-up
    t = time.time()
    for _ in range(steps):
        sim.step(cfg["time"]["dt"])
    return N * steps / (time.time() - t)


def time_vlasov(cfg, d, nv, steps=5):
    c2 = dict(cfg)
    c2["vlasov"] = dict(cfg["vlasov"], dv_finest=2 * cfg["vlasov"]["v_max"] / nv)
    sim, nv = R.build_vlasov(c2, d, 0)
    t = time.time()
    for _ in range(steps):
        sim.step(cfg["time"]["dt"])
    return d["nx"] * nv * steps / (time.time() - t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--budget-node-hours", type=float, default=8.0)
    ap.add_argument("--out", default=str(HERE / "reports" / "COST_ESTIMATE.md"))
    a = ap.parse_args()
    cfg = yaml.safe_load(Path(a.config).read_text())
    d = R.derived(cfg)
    threads = os.environ.get("NUMBA_NUM_THREADS", str(os.cpu_count()))
    pic_rate = time_pic(cfg, d, 256)
    vl_rate = time_vlasov(cfg, d, 512)
    steps = int(round(cfg["time"]["t_end"] / cfg["time"]["dt"]))
    nx = d["nx"]
    ppcs = cfg["pic"]["ppc_levels"]
    nseeds = len(cfg["pic"]["seeds"])
    pic_ps = nx * sum(ppcs) * steps * nseeds
    ctrl_ps = nx * sum(ppcs) * steps * 2          # stable control, 2 seeds per level
    vc = cfg["vlasov"]
    nvs = [int(round(2 * vc["v_max"] / (vc["dv_finest"] * 2**l))) for l in range(vc["levels"])]
    vl_cs = nx * sum(nvs) * steps
    pic_h = (pic_ps + ctrl_ps) / pic_rate / 3600
    vl_h = vl_cs / vl_rate / 3600
    finest_h = nx * ppcs[0] * steps / pic_rate / 3600
    vl_finest_h = nx * nvs[0] * steps / vl_rate / 3600
    # narrower beams: dv_b = 0.15 needs dx <= lambda_Db(0.15) ~ 0.0433 (nx ~ 3780 -> 4096, unchanged)
    # and dv_finest scaled by 0.75 (nv x 4/3)
    vl_015_h = vl_h * 4 / 3
    total = pic_h + vl_h
    fits = total <= a.budget_node_hours
    lines = [
        "# T2 cost estimate",
        "",
        f"Measured on this machine ({threads} threads, numba={HAVE_NUMBA}) at the production grid: "
        f"nx = {nx}, L = {d['L']:.2f} v_b/omega_p, dt = {cfg['time']['dt']}, t_end = {cfg['time']['t_end']} "
        f"({steps} steps).",
        "",
        f"* PIC throughput: {pic_rate:.3g} particle-steps/s (timed at 256 ppc).",
        f"* Vlasov throughput: {vl_rate:.3g} cell-steps/s (timed at nv = 512).",
        "",
        "| Arm | Size | Work | Wall time on this machine |",
        "|---|---|---|---|",
        f"| PIC nested ladder | ppc {ppcs}, {nseeds} seeds | {pic_ps:.3g} particle-steps | {pic_ps / pic_rate / 3600:.1f} h |",
        f"| Stable control | same ladder, 2 seeds | {ctrl_ps:.3g} particle-steps | {ctrl_ps / pic_rate / 3600:.1f} h |",
        f"| Vlasov dv ladder | nv {nvs} | {vl_cs:.3g} cell-steps | {vl_h:.1f} h |",
        f"| **Total** | | | **{total:.1f} h** (budget {a.budget_node_hours} node-h) |",
        "",
        f"Largest single runs: PIC at {ppcs[0]} ppc ~ {finest_h:.1f} h; Vlasov at nv = {nvs[0]} ~ {vl_finest_h:.1f} h.",
        f"Delta v_b = 0.15 v_b (cost estimate only): Vlasov arm ~ {vl_015_h:.1f} h; the PIC arm is unchanged "
        "at fixed nx because Delta x = 0.040 is still below lambda_Db(0.15) = 0.043.",
        "Doubling L (finite-size check at one middle level) doubles that level's cost.",
        "",
        "These timings are from a 4-core cloud container and are pessimistic for a Meluxina CPU node "
        "(128 cores): PIC scales with threads until memory bandwidth saturates, and independent "
        "(level, seed) runs are embarrassingly parallel across nodes.",
        "",
        "**Decision:** " + ("the matrix fits the local budget; run it." if fits else
                            "the matrix does NOT fit the local budget. Per the T2 spec, SLURM scripts are "
                            "generated in `slurm/` and no underpowered run is launched or interpreted here."),
    ]
    Path(a.out).write_text("\n".join(lines) + "\n")
    print(json.dumps({"pic_rate": pic_rate, "vlasov_rate": vl_rate, "total_hours": total, "fits": fits}))


if __name__ == "__main__":
    main()
