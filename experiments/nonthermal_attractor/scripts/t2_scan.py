#!/usr/bin/env python3
"""Run (or list) the T2 regulator matrix for a config.

    python3 experiments/nonthermal_attractor/scripts/t2_scan.py CONFIG.yaml --out DIR [--list] [--only pic|vlasov|control]

``--list`` prints one runner command per (backend, level, seed) for the SLURM
array scripts; without it the matrix runs locally, sequentially.  The
production matrix does not fit a local budget (reports/COST_ESTIMATE.md):
locally, run only the smoke config.
"""
import argparse
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

from attractor_sim import run as R  # noqa: E402


def matrix(cfg, only=None):
    jobs = []
    if only in (None, "pic"):
        for lvl in range(len(cfg["pic"]["ppc_levels"])):
            for s in cfg["pic"]["seeds"]:
                jobs.append(("pic1d1v", lvl, s))
    if only in (None, "vlasov"):
        for lvl in range(cfg["vlasov"]["levels"]):
            jobs.append(("vlasov1d1v", lvl, 0))
    if only in (None, "control"):
        for lvl in range(len(cfg["pic"]["ppc_levels"])):
            for s in cfg["pic"].get("control_seeds", [0, 1]):
                jobs.append(("control", lvl, s))
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--out", required=True)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--only", choices=["pic", "vlasov", "control"], default=None)
    a = ap.parse_args()
    cfg = yaml.safe_load(Path(a.config).read_text())
    jobs = matrix(cfg, a.only)
    if a.list:
        for b, l, s in jobs:
            back, extra = ("pic1d1v", " --control") if b == "control" else (b, "")
            print(f"python3 -m attractor_sim.run {a.config} --backend {back} --level {l} --seed {s} --out {a.out}{extra}")
        return
    for b, l, s in jobs:
        p = R.run(cfg, "pic1d1v" if b == "control" else b, l, s, Path(a.out), control=(b == "control"))
        print(p, flush=True)


if __name__ == "__main__":
    main()
