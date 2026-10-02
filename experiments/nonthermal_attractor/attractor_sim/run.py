"""
attractor_sim/run.py
====================

Configuration-driven runner for one T2 run (one backend, one regulator
level, one seed).  Usage::

    python3 -m attractor_sim.run CONFIG.yaml --backend pic1d1v --level 3 --seed 2 --out DIR
    python3 -m attractor_sim.run CONFIG.yaml --backend vlasov1d1v --level 1 --out DIR

PIC levels are nested: the level-l load is the first N0/2**l markers of the
level-0 load (``attractor_analysis.nested``), with weight x 2**l, so every
level of one seed shares its markers.  Vlasov levels halve the velocity
resolution: Delta v = dv_finest * 2**l.

The deterministic seed spectrum (modes, amplitudes, phases) is read from
the config and is identical for every run and both backends.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

from attractor_analysis import nested  # noqa: E402
from attractor_analysis.energy import common_edges  # noqa: E402
from attractor_sim import theory  # noqa: E402
from attractor_sim.diagnostics import SnapshotRecorder  # noqa: E402
from attractor_sim.pic1d1v import HAVE_NUMBA, PIC1D1V, seed_displacement  # noqa: E402
from attractor_sim.vlasov1d1v import Vlasov1D1V, waterbag_beams_f  # noqa: E402


def git_rev() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=HERE, text=True).strip()
    except Exception:
        return "unknown"


def derived(cfg: dict) -> dict:
    """Box, grid and seed spectrum derived from the preparation."""
    prep = cfg["preparation"]
    fm = theory.fastest_mode(prep["v_b"], prep["dv_b"])
    L = cfg["box"]["n_wavelengths"] * fm["lambda_star"]
    nx = int(cfg["grid"]["nx"])
    seed = cfg["seed_spectrum"]
    modes = list(seed["modes"])
    amps = list(seed["amplitudes"])
    phases = list(seed["phases"])
    return {"L": L, "nx": nx, "dx": L / nx, "fastest": fm, "modes": modes, "amps": amps, "phases": phases,
            "lambda_Db": theory.beam_debye_length(prep["dv_b"])}


def make_recorder(cfg, L):
    b = cfg["bins"]
    edges = common_edges(b["eps_lo"], b["eps_hi"], b["n_log"], b["n_lin"])
    eta_edges = np.exp(np.arange(np.log(b["eta_lo"]), np.log(b["eta_hi"]) + 1e-12, b["eta_log_step"]))
    r = cfg["reconstruction"]
    return SnapshotRecorder(edges, eta_edges, r, n_phase_snapshots=cfg["diagnostics"].get("n_phase", 0),
                            phase_decimate=cfg["diagnostics"].get("phase_decimate", 64))


def build_pic(cfg, d, level, seed, ppc0=None, control=False):
    """Nested PIC load.  ``control=True`` gives the stable control: a single
    waterbag at rest with the same width, density and seed spectrum, which
    measures numerical heating/relaxation at the same numerical settings."""
    prep = cfg["preparation"]
    ppc0 = ppc0 or int(cfg["pic"]["ppc_levels"][0])
    N0 = ppc0 * d["nx"]
    if control:
        inv = lambda q: (np.asarray(q) - 0.5) * prep["dv_b"]
    else:
        inv = nested.waterbag_beams_inv_cdf(prep["v_b"], prep["dv_b"])
    x, v, w = nested.quiet_load_1d1v(N0, d["L"], inv, seed=seed)
    x, v, w = nested.subset(x, v, w, level)
    x = seed_displacement(x, d["L"], d["modes"], d["amps"], d["phases"])
    return PIC1D1V(d["L"], d["nx"], x, v, w), N0 >> level


def build_vlasov(cfg, d, level):
    prep = cfg["preparation"]
    vc = cfg["vlasov"]
    dv = vc["dv_finest"] * 2**level
    nv = int(round(2 * vc["v_max"] / dv))
    f = waterbag_beams_f(d["L"], d["nx"], vc["v_max"], nv, prep["v_b"], prep["dv_b"],
                         modes=d["modes"], amplitudes=d["amps"], phases=d["phases"])
    return Vlasov1D1V(d["L"], d["nx"], vc["v_max"], nv, f), nv


def run(cfg: dict, backend: str, level: int, seed: int, out: Path, t_end=None, max_steps=None,
        control: bool = False) -> Path:
    d = derived(cfg)
    dt = float(cfg["time"]["dt"])
    t_end = float(t_end if t_end is not None else cfg["time"]["t_end"])
    n_steps = int(round(t_end / dt))
    if max_steps is not None:
        n_steps = min(n_steps, max_steps)
    every = int(cfg["diagnostics"]["every_steps"])
    rec = make_recorder(cfg, d["L"])
    n_phase = cfg["diagnostics"].get("n_phase", 0)
    phase_every = max(1, n_steps // max(n_phase, 1)) if n_phase else None
    t0 = time.time()
    meta = {"backend": backend, "git_revision": git_rev(), "config": cfg,
            "normalization": ("omega_pe of total n0 = 1, eps0 = 1, q = -1, m = 1; v in units of v_b, x in "
                              f"v_b/omega_pe; beam Debye length lambda_Db = (dv_b/sqrt12)/omega_pe = {d['lambda_Db']:.6g}"),
            "seed": int(seed), "nested_level": int(level), "dt": dt, "dx": d["dx"],
            "staggering": "KDK: x, v, E synchronised at integer steps; diagnostics at t = n dt",
            "derived": {k: v for k, v in d.items() if k != "fastest"} | {"fastest": d["fastest"]},
            "numba": HAVE_NUMBA}
    if backend == "pic1d1v":
        sim, N = build_pic(cfg, d, level, seed, control=control)
        meta.update(particles_per_cell=N / d["nx"], n_particles=int(N), macro_weight=float(sim.w[0]),
                    preparation="stable_control" if control else "waterbag_beams")
        record = lambda t, n: rec.record_pic(t, sim, phase=bool(phase_every and n % phase_every == 0))
    elif backend == "vlasov1d1v":
        sim, nv = build_vlasov(cfg, d, level)
        meta.update(dv=sim.dv, v_max=sim.v_max, nv=nv)
        record = lambda t, n: rec.record_vlasov(t, sim)
    else:
        raise ValueError(backend)
    for n in range(n_steps + 1):
        if n % every == 0:
            record(n * dt, n)
        if n < n_steps:
            sim.step(dt)
    meta["wall_time_s"] = time.time() - t0
    meta["n_steps"] = n_steps
    name = f"{'control_' if control else ''}{backend}_L{level}_s{seed}.npz"
    return rec.to_bundle(Path(out) / name, meta)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--backend", required=True, choices=["pic1d1v", "vlasov1d1v"])
    ap.add_argument("--level", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--t-end", type=float, default=None)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--control", action="store_true", help="stable control (single waterbag at rest)")
    a = ap.parse_args(argv)
    cfg = yaml.safe_load(Path(a.config).read_text())
    p = run(cfg, a.backend, a.level, a.seed, Path(a.out), a.t_end, a.max_steps, control=a.control)
    print(json.dumps({"bundle": str(p)}))


if __name__ == "__main__":
    main()
