#!/usr/bin/env python3
"""Generate the paired density snapshots of the paper's reconstruction test.

Oblique two-stream benchmark, v_b = 0.80, v_th = 0.05, 128 x 16 mesh,
L = 64 x 16.  For every seed a production run (PPC 32) and a reference run
(PPC 1024) start from the same random stream, are evolved with unfiltered CIC
coupling (dt = 0.05, 80 steps), and their CIC densities are saved every second
step on t in [0.5, 4] (36 times).

Uses the repository's PIC code (``pic_kde_2d``).  Output: ``snapshots.npz``
with arrays ``rho`` and ``rho_ref`` of shape (seeds, 36, 128, 16).

    python make_snapshots.py --seeds 100-109 --out snapshots.npz
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "pic_kde_2d"))

from pic_kde_2d.conventions import marker_grid_shape  # noqa: E402
from pic_kde_2d.core.grid2d import Grid2D  # noqa: E402
from pic_kde_2d.core.particles2d import init_two_stream_2d  # noqa: E402
from pic_kde_2d.core.pic_loop import pic_step  # noqa: E402
from pic_kde_2d.methods.cic import deposit_cic_2d  # noqa: E402
from pic_kde_2d.methods.spectral_W_wrappers import make_spectral_W_gaussian_methods  # noqa: E402


def particles(ppc: int, seed: int):
    nxm, nym = marker_grid_shape(128, 16, ppc)
    return init_two_stream_2d(Nx_marker=nxm, Ny_marker=nym, Lx=64.0, Ly=16.0, v_beam=0.8,
                              v_thermal=0.05, alpha=0.001, kx_mode=12, ky_mode=0,
                              rng=np.random.default_rng(seed))


def run_seed(seed: int):
    grid = Grid2D(128, 16, 64.0, 16.0)
    low, ref = particles(32, seed), particles(1024, seed)
    dep, gat = make_spectral_W_gaussian_methods(grid, 0.0, 0.0)
    out_low, out_ref, times = [], [], []
    for step in range(81):
        t = step * 0.05
        if step % 2 == 0 and 0.5 <= t <= 4.0 + 1e-12:
            out_low.append(deposit_cic_2d(low.x, low.y, low.w, grid.Nx, grid.Ny, grid.dx, grid.dy))
            out_ref.append(deposit_cic_2d(ref.x, ref.y, ref.w, grid.Nx, grid.Ny, grid.dx, grid.dy))
            times.append(t)
        if step == 80:
            break
        pic_step(low, grid, 0.05, dep, gat)
        pic_step(ref, grid, 0.05, dep, gat)
    return np.array(out_low), np.array(out_ref), np.array(times)


def parse_seeds(text: str) -> list[int]:
    seeds = []
    for part in text.split(","):
        if "-" in part:
            a, b = part.split("-")
            seeds += list(range(int(a), int(b) + 1))
        else:
            seeds.append(int(part))
    return seeds


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", default="100-109")
    ap.add_argument("--out", type=Path, default=Path("snapshots.npz"))
    a = ap.parse_args()
    seeds = parse_seeds(a.seeds)
    lows, refs = [], []
    for s in seeds:
        t0 = time.time()
        lo, rf, times = run_seed(s)
        lows.append(lo)
        refs.append(rf)
        print(f"seed {s}: {lo.shape[0]} snapshots in {time.time() - t0:.0f} s", flush=True)
    np.savez_compressed(a.out, rho=np.array(lows), rho_ref=np.array(refs), times=times,
                        seeds=np.array(seeds))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
