#!/usr/bin/env python3
"""Box-size check for T2 (protocol v2, box_check): eps_q only.

    python3 experiments/nonthermal_attractor/scripts/t2_box_check.py protocol/T2_PROTOCOL_v2.yaml \\
        BUNDLES_L16 BUNDLES_L32 --out results/...

Compares the 32-wavelength run (1024 ppc, level 4, seed 0) with the four seeds of
level 4 in the 16-wavelength scan.  Same dx, dt, ppc and seeded wavenumbers;
the L32 rho(eta) x-bin is twice as wide, so eta_min is not compared.  Reports
the L32 window mean of eps_q against the L16 seed mean and spread, and the
RMS trajectory difference L32 - L16 seed 0 against the seed-to-seed RMS
differences of L16.  Not part of the gate outcome.
"""
import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "scripts"))

from attractor_analysis import schema  # noqa: E402
from attractor_analysis.energy import quantile_feature  # noqa: E402
from t2_analyze import hists  # noqa: E402


def eps_q(path, q):
    md, arr = schema.read_bundle(path)
    return md, arr["times"], np.array([quantile_feature(h, q) for h in hists(arr)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("protocol")
    ap.add_argument("bundles_l16")
    ap.add_argument("bundles_l32")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    proto = yaml.safe_load(Path(a.protocol).read_text())
    an = proto["analysis"]
    lo, hi = an["window"]
    md32, t32, f32 = eps_q(Path(a.bundles_l32) / "pic1d1v_L4_s0.npz", an["q"])
    l16 = {s: eps_q(Path(a.bundles_l16) / f"pic1d1v_L4_s{s}.npz", an["q"]) for s in proto["regulator_scan"]["pic_seeds"]}
    t16 = l16[0][1]
    if not np.allclose(t16, t32):
        raise SystemExit("time grids differ")
    if md32["particles_per_cell"] != l16[0][0]["particles_per_cell"] or not np.isclose(md32["dx"], l16[0][0]["dx"]):
        raise SystemExit("ppc or dx differ between the boxes")
    w = (t16 >= lo) & (t16 <= hi)
    means16 = np.array([np.nanmean(v[2][w]) for v in l16.values()])
    mean32 = float(np.nanmean(f32[w]))
    rms = lambda x, y: float(np.sqrt(np.nanmean((x[w] - y[w]) ** 2)))
    seed_rms = [rms(l16[i][2], l16[j][2]) for i, j in itertools.combinations(sorted(l16), 2)]
    sd16 = float(np.std(means16, ddof=1))
    res = {"eps_q_window_mean_L32": mean32, "eps_q_window_mean_L16_seeds": means16.tolist(),
           "L16_seed_sd": sd16, "z_L32_vs_L16": float((mean32 - means16.mean()) / (sd16 * np.sqrt(1 + 1 / len(means16)))),
           "rms_L32_minus_L16_s0": rms(f32, l16[0][2]), "rms_L16_seed_pairs": seed_rms,
           "note": "eps_q only; |z| < 3 and an RMS difference within the seed-pair range means no box effect is resolved"}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "T2_BOX_CHECK.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
