#!/usr/bin/env python3
"""Check your own filter against the dynamics-aware constraint.

Example: a sharp spectral cut-off at 60% of Nyquist, and a 5-point
smoother used once on the density only (no return filtering).
"""
import numpy as np

from dabs import (Custom, Grid, Kernel, ModelObjective, NoFilter, Plasma, ProtectedMode,
                  Selector, Species)

grid = Grid([128, 16], [64.0, 16.0], shape="cic")
plasma = Plasma([Species(-1, 1, 0.5, [0.8, 0], 0.05),
                 Species(-1, 1, 0.5, [-0.8, 0], 0.05)]).resolve(grid.dim)
objective = ModelObjective(grid, plasma, ppc=32, signal=[{"mode": [12, 0], "amplitude": 1e-3}])


def cutoff(k, g, frac=0.6):
    return np.prod(np.where(np.abs(k) * g.dx <= frac * np.pi, 1.0, 0.0), axis=-1)


def five_point(k, g):
    # taps [-1, 4, 10, 4, -1] / 16 on each axis
    c = np.cos(k * g.dx)
    return np.prod((10 + 8 * c - 2 * np.cos(2 * k * g.dx)) / 16.0, axis=-1)


sel = Selector(grid, plasma, [ProtectedMode((12, 1))], objective, tolerance=0.05, n_alias=8,
               gaussian={"enabled": False}, binomial={"p": [4], "n": [8]})
sel.analyse_modes()
baseline = sel.evaluate(Kernel(grid.order, NoFilter()))
for name, fun, power in (("cut-off 0.6 k_Nyq", cutoff, 2), ("5-point, density only", five_point, 1)):
    c = sel.evaluate(Kernel(grid.order, Custom(fun, name), power))
    R = "undefined" if c.R[0] is None else f"{c.R[0]:.4f}"
    print(f"{name:<24} a = {c.a[0]:.6f}  status {c.status[0]:<8} R_gamma = {R}  "
          f"admissible: {c.admissible}  density MSE reduction: "
          f"{100 * (1 - c.objective / baseline.objective):.1f}%")
