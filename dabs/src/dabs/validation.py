"""Built-in reproduction checks (``dabs validate``).

Reference values come from kinetic theory and from the independently written
dispersion code used for the paper (principal-harmonic level unless stated).
"""

from __future__ import annotations

import math

import numpy as np

from .dispersion import ModeDispersion
from .filters import Gaussian, Kernel
from .grid import Grid
from .objectives import ModelObjective
from .plasma import Plasma, Species
from . import roots as rt
from .selection import ProtectedMode, Selector


def _two_stream(dim2=True):
    grid = Grid([128, 16], [64.0, 16.0], shape="cic")
    plasma = Plasma([Species(-1, 1, 0.5, [0.8, 0], 0.05, "beam+"),
                     Species(-1, 1, 0.5, [-0.8, 0], 0.05, "beam-")]).resolve(2)
    return grid, plasma


def checks():
    out = []

    # 1. Continuum Landau damping, k lambda_D = 0.4 (fine 1D mesh, principal only)
    g = Grid([4096], [2 * math.pi / 0.4])
    p = Plasma([Species(-1, 1, 1, 0.0, 1.0)]).resolve(1)
    d = ModeDispersion(g, p, (1,), n_alias=0)
    w = rt.select_branch(rt.find_roots(d), "least_damped")
    out.append(("Landau k=0.4 continuum: omega_r", w.real, 1.2850569696537832, 2e-6))
    out.append(("Landau k=0.4 continuum: gamma", w.imag, -0.06612795869066214, 2e-6))

    # 2. Landau on the paper mesh, principal CIC, damping ratios of two Gaussians
    g = Grid([64, 16], [2 * math.pi / 0.4, 8.0])
    p = Plasma([Species(-1, 1, 1, 0.0, 1.0)]).resolve(2)
    d = ModeDispersion(g, p, (1, 0), n_alias=0)
    w0 = rt.select_branch(rt.find_roots(d), "least_damped")
    out.append(("Landau paper mesh, principal CIC: gamma", w0.imag, -0.06630795108128433, 1e-9))
    br = rt.continue_branch(d, w0, 1.0, 0.5)
    for a, ref in ((0.9904080138578515, 1.016348411037295), (0.6030132596901013, 1.979865965142041)):
        out.append((f"Landau damping ratio at W^2={a:.4f}", rt.solve_at(d, br, a).imag / w0.imag,
                    ref, 1e-8))

    # 3. Two-stream (12,1), v_b = 0.80, principal CIC
    grid, plasma = _two_stream()
    d = ModeDispersion(grid, plasma, (12, 1), n_alias=0)
    w0 = rt.select_branch(rt.find_roots(d))
    out.append(("two-stream (12,1) principal CIC: gamma", w0.imag, 0.1175192525891037, 1e-6))
    br = rt.continue_branch(d, w0, 1.0, 0.5)
    out.append(("two-stream marginal coupling a_c (principal; Penrose 0.953155)", br.a_marginal,
                0.953155, 2e-6))
    R = rt.solve_at(d, br, 0.99095494).imag / w0.imag
    out.append(("fixed Gaussian W^2=0.99095494: R_gamma (paper model 0.900)", R, 0.900, 2e-3))

    # 4. Joint selection on the paper case, 30 binomials + Gaussians, eps = 0.05
    obj = ModelObjective(grid, plasma, ppc=32, signal=[dict(mode=[12, 0], amplitude=1e-3)])
    res = Selector(grid, plasma, [ProtectedMode((12, 1))], obj, tolerance=0.05, n_alias=8,
                   gaussian=dict(grid=31)).run()
    bins = [c for c in res.candidates if c.family == "binomial"]
    out.append(("binomials with a growing root (paper: 17)", sum(c.status[0] == "growing" for c in bins),
                17, 0))
    out.append(("binomials reaching marginality (paper: 13)", sum(c.status[0] == "marginal" for c in bins),
                13, 0))
    out.append(("admissible binomials at 5% (paper: 11)", sum(c.admissible for c in bins), 11, 0))
    best = res.best.kernel.filter
    out.append(("selected kernel is binomial p=4, n=8 (paper)",
                float(best.family == "binomial" and best.p == 4 and best.n == (8,)), 1.0, 0))
    gb = res.best_by_family["gaussian"].kernel.filter
    out.append(("best admissible Gaussian has h_x = 0 (paper)", gb.h[0], 0.0, 1e-9))

    # 5. Cold leapfrog Langmuir oscillation: sin(omega dt/2) = omega_p dt/2 (S -> 1)
    g = Grid([4096], [2 * math.pi / 0.01])
    p = Plasma([Species(-1, 1, 1, 0.0, 0.0)]).resolve(1)
    dt = 0.5
    d = ModeDispersion(g, p, (1,), n_alias=0, time="leapfrog", dt=dt)
    S = float(g.shape_transform(d.k) ** 1)
    w = max((r for r in rt.find_roots(d) if r.real > 0), key=lambda r: r.real, default=float("nan"))
    out.append(("cold leapfrog Langmuir frequency", w.real, 2 / dt * math.asin(S * 0.5 * dt), 1e-9))

    # 6. Cold symmetric two-stream (two beams of density 1/2, drifts +-1, k u = 0.5):
    #    (x - K^2)^2 = 2 w^2 (x + K^2), x = omega^2, w^2 = S^2/2
    g = Grid([4096], [2 * math.pi / 0.5])
    p = Plasma([Species(-1, 1, 0.5, 1.0, 0.0), Species(-1, 1, 0.5, -1.0, 0.0)]).resolve(1)
    d = ModeDispersion(g, p, (1,), n_alias=0)
    K = float(d.k[0])
    w2 = 0.5 * float(g.shape_transform(d.k)) ** 2
    x = K**2 + w2 - math.sqrt(w2**2 + 4 * K**2 * w2)
    w = rt.select_branch(rt.find_roots(d))
    out.append(("cold symmetric two-stream growth (k u = 0.5)", w.imag, math.sqrt(-x), 1e-9))
    return out


def run_validation(verbose: bool = True) -> bool:
    ok_all = True
    for name, val, ref, tol in checks():
        ok = abs(val - ref) <= tol * max(1.0, abs(ref)) if tol else val == ref
        ok_all &= bool(ok)
        if verbose:
            print(f"[{'PASS' if ok else 'FAIL'}] {name}: {val:.10g} (reference {ref:.10g})")
    if verbose:
        print("all checks passed" if ok_all else "SOME CHECKS FAILED")
    return ok_all
