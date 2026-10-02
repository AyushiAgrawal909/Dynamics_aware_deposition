#!/usr/bin/env python3
"""Measure the shot-noise floor of the coarsest PIC level (unseeded load).

The deterministic seed must exceed this floor by >= 10x in every seeded mode,
otherwise low-N runs are silently noise-seeded.  Reports per-mode
|delta n_m| / n0 for the modes in the seed spectrum:

* at t = 0 (a Halton quiet start is near-exact on the x lattice);
* after one plasma period of *free streaming* (fields off), which is the
  fluctuation level a quiet start relaxes to by phase mixing and is the
  relevant seed floor;
* after one plasma period with fields on (includes instability growth of
  the noise, an upper bound, not the floor).

    python3 experiments/nonthermal_attractor/scripts/t2_noise_floor.py CONFIG.yaml [--ppc 64] [--seeds 4]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

from attractor_analysis import nested  # noqa: E402
from attractor_sim.pic1d1v import PIC1D1V  # noqa: E402
from attractor_sim.run import derived  # noqa: E402


def mode_amplitudes(sim, modes):
    ne = sim.electron_density()
    nk = np.fft.rfft(ne) / sim.nx
    return np.array([2 * abs(nk[m]) for m in modes])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--ppc", type=int, default=None)
    ap.add_argument("--seeds", type=int, default=4)
    a = ap.parse_args()
    cfg = yaml.safe_load(Path(a.config).read_text())
    d = derived(cfg)
    ppc = a.ppc or int(cfg["pic"]["ppc_levels"][-1])
    prep = cfg["preparation"]
    dt = float(cfg["time"]["dt"])
    res = []
    for s in range(a.seeds):
        x, v, w = nested.quiet_load_1d1v(ppc * d["nx"], d["L"],
                                         nested.waterbag_beams_inv_cdf(prep["v_b"], prep["dv_b"]), seed=s)
        sim = PIC1D1V(d["L"], d["nx"], x, v, w)
        a0 = mode_amplitudes(sim, d["modes"])
        free = PIC1D1V(d["L"], d["nx"], (x + 2 * np.pi * v) % d["L"], v, w)
        af = mode_amplitudes(free, d["modes"])
        for _ in range(int(round(2 * np.pi / dt))):
            sim.step(dt)
        a1 = mode_amplitudes(sim, d["modes"])
        res.append((a0, af, a1))
    rms = lambda i: np.sqrt(np.mean([r[i] ** 2 for r in res], axis=0))
    a0, af, a1 = rms(0), rms(1), rms(2)
    seed_amp = np.asarray(d["amps"])
    ratio = seed_amp / np.maximum(af, a0)
    out = {"ppc": ppc, "modes": d["modes"], "floor_t0": a0.tolist(), "floor_free_streaming_2pi": af.tolist(),
           "with_fields_2pi_upper_bound": a1.tolist(), "seed_amplitude": seed_amp.tolist(),
           "min_seed_to_floor": float(ratio.min()), "required_seed_for_10x": float(10 * np.max(af)),
           "ok": bool(ratio.min() >= 10.0)}
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
