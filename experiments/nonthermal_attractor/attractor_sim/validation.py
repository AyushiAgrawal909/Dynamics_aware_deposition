"""
attractor_sim/validation.py
===========================

T2 validations V1-V4 for both backends, sized for a laptop and for CI.

V1  neutrality, momentum conservation (PIC: round-off), energy drift on a
    stable Maxwellian.
V2  linear Landau damping at k lambda_D = 0.5 (gamma = -0.1533).
V3  linear growth of the waterbag-beam preparation against the quartic.
V4  Delta x / dt halving on a short nonlinear run (stable-control heating).
"""

from __future__ import annotations

import numpy as np

from . import theory
from .pic1d1v import PIC1D1V, seed_displacement
from .vlasov1d1v import Vlasov1D1V, maxwellian_f, waterbag_beams_f

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from attractor_analysis import nested  # noqa: E402


def _fit_rate(t, amp, t_lo, t_hi, peaks=True):
    """Linear fit of ln|amp| over [t_lo, t_hi] (on local maxima if ``peaks``)."""
    a = np.abs(amp)
    m = (t >= t_lo) & (t <= t_hi)
    tt, aa = t[m], a[m]
    if peaks:
        pk = np.flatnonzero((aa[1:-1] > aa[:-2]) & (aa[1:-1] >= aa[2:])) + 1
        if pk.size >= 3:
            tt, aa = tt[pk], aa[pk]
    return float(np.polyfit(tt, np.log(aa), 1)[0])


def pic_maxwell_load(L, nx, ppc, v_th=1.0, seed=0, mode=1, amp=0.0):
    N = nx * ppc
    x, v, w = nested.quiet_load_1d1v(N, L, nested.maxwellian_inv_cdf(v_th), seed=seed)
    if amp:
        x = seed_displacement(x, L, [mode], [amp], [0.0])
    return PIC1D1V(L, nx, x, v, w)


def v1_pic_conservation(nx=64, ppc=200, steps=400, dt=0.1, seed=0) -> dict:
    L = 4 * np.pi
    sim = pic_maxwell_load(L, nx, ppc, seed=seed, amp=0.01)
    b0 = sim.budgets()
    pmax = 0.0
    for _ in range(steps):
        sim.step(dt)
        b = sim.budgets()
        pmax = max(pmax, abs(b["momentum"] - b0["momentum"]))
    return {"charge": b["total_charge"], "momentum_drift_max": pmax,
            "momentum_scale": float(np.sum(sim.w * np.abs(sim.v))),
            "energy_drift_rel": (b["total_energy"] - b0["total_energy"]) / b0["kinetic_energy"]}


def v2_landau_pic(nx=64, ppc=2000, t_end=20.0, dt=0.05, amp=0.05) -> dict:
    k = 0.5
    L = 2 * np.pi / k
    sim = pic_maxwell_load(L, nx, ppc, amp=amp)
    ts, amps = [], []
    for n in range(int(round(t_end / dt)) + 1):
        if n % 2 == 0:
            ts.append(n * dt)
            amps.append(np.fft.rfft(sim.E)[1])
        sim.step(dt)
    g = _fit_rate(np.array(ts), np.array(amps), 2.0, 18.0)
    return {"gamma_measured": g, "gamma_theory": theory.landau_root(0.5).imag}


def v2_landau_vlasov(nx=32, nv=256, t_end=30.0, dt=0.05, amp=0.01) -> dict:
    k = 0.5
    L = 2 * np.pi / k
    sim = Vlasov1D1V(L, nx, 8.0, nv, maxwellian_f(L, nx, 8.0, nv, amplitude=amp))
    ts, amps = [], []
    for n in range(int(round(t_end / dt)) + 1):
        if n % 2 == 0:
            ts.append(n * dt)
            amps.append(np.fft.rfft(sim.E)[1])
        sim.step(dt)
    g = _fit_rate(np.array(ts), np.array(amps), 2.0, 25.0)
    rec = 2 * np.pi / (k * sim.dv)
    return {"gamma_measured": g, "gamma_theory": theory.landau_root(0.5).imag, "recurrence_time": rec,
            "mass_drift": sim.budgets()["number"] / (L * 1.0) - 1.0}


def v3_waterbag_growth(backend="pic", dv_b=0.2, n_waves=4, cells_per_wave=64, ppc=512, nv=1024,
                       amp=None, dt=0.05, t_end=None) -> dict:
    fm = theory.fastest_mode(1.0, dv_b)
    L = n_waves * fm["lambda_star"]
    nx = n_waves * cells_per_wave
    mode = n_waves
    k = 2 * np.pi * mode / L
    g_th = float(theory.waterbag_growth(k, 1.0, dv_b)[0])
    amp = amp if amp is not None else (1e-4 if backend == "pic" else 1e-7)
    t_end = t_end or (12.0 if backend == "pic" else 20.0) / g_th
    if backend == "pic":
        N = nx * ppc
        x, v, w = nested.quiet_load_1d1v(N, L, nested.waterbag_beams_inv_cdf(1.0, dv_b), seed=1)
        x = seed_displacement(x, L, [mode], [amp], [0.0])
        sim = PIC1D1V(L, nx, x, v, w)
    else:
        f = waterbag_beams_f(L, nx, 2.5, nv, 1.0, dv_b, modes=[mode], amplitudes=[amp], phases=[0.0])
        sim = Vlasov1D1V(L, nx, 2.5, nv, f)
    ts, amps = [], []
    for n in range(int(round(t_end / dt)) + 1):
        ts.append(n * dt)
        amps.append(np.fft.rfft(sim.E)[mode])
        sim.step(dt)
    ts, amps = np.array(ts), np.abs(np.array(amps))
    # fit window: from 3/gamma (damped root gone) until the amplitude reaches 5%
    # of its saturated maximum.  The two neutral roots never decay, so the
    # window must be long compared with their beat period: the noise-free
    # Vlasov run therefore starts from a smaller seed (longer linear phase).
    sat = amps.max()
    hi = ts[np.argmax(amps > 0.05 * sat)] if np.any(amps > 0.05 * sat) else ts[-1]
    g = _fit_rate(ts, amps, 3.0 / g_th, hi, peaks=False)
    return {"backend": backend, "k": k, "gamma_measured": g, "gamma_theory": g_th,
            "rel_err": abs(g - g_th) / g_th, "fit_window": [3.0 / g_th, float(hi)]}


def v4_stable_control_heating(nx, ppc, dv_b=0.2, L=None, dt=0.05, steps=400, seed=0) -> dict:
    """Single waterbag at rest (same width) - stable; fractional kinetic-energy change."""
    L = L or 32.0
    N = nx * ppc

    def inv(q):
        return (np.asarray(q) - 0.5) * dv_b

    x, v, w = nested.quiet_load_1d1v(N, L, inv, seed=seed)
    sim = PIC1D1V(L, nx, x, v, w)
    K0 = sim.budgets()["kinetic_energy"]
    for _ in range(steps):
        sim.step(dt)
    b = sim.budgets()
    return {"dx_over_lambda_Db": (L / nx) / theory.beam_debye_length(dv_b),
            "dK_over_K": (b["kinetic_energy"] - K0) / K0, "t": steps * dt}


def v4_nonlinear_refinement(n_waves=4, cells_per_wave=256, ppc=512, dt=0.05, t_end=60.0, amp=1e-3) -> dict:
    """Short nonlinear beam run (through saturation) at (dx, dt), (dx/2, dt), (dx, dt/2).

    Reports the relative difference of the time-averaged field energy after
    saturation and of the kinetic-energy change: a resolution-sensitive
    result here means finite-grid or time-step effects would set the answer.
    """
    fm = theory.fastest_mode(1.0, 0.2)
    L = n_waves * fm["lambda_star"]
    modes = list(range(1, int(n_waves * fm["k_cut"] / fm["k_star"]) + 1))
    phases = list(np.random.default_rng(5).uniform(0, 2 * np.pi, len(modes)))

    def one(nx, dtt):
        N = (n_waves * cells_per_wave) * ppc
        x, v, w = nested.quiet_load_1d1v(N, L, nested.waterbag_beams_inv_cdf(1.0, 0.2), seed=3)
        x = seed_displacement(x, L, modes, [amp] * len(modes), phases)
        sim = PIC1D1V(L, nx, x, v, w)
        K0 = sim.budgets()["kinetic_energy"]
        n = int(round(t_end / dtt))
        fe = []
        for s in range(n + 1):
            if s * dtt >= 30.0:
                fe.append(sim.budgets()["field_energy"])
            sim.step(dtt)
        b = sim.budgets()
        return {"fe_mean": float(np.mean(fe)), "dK": b["kinetic_energy"] - K0,
                "E_drift_rel": (b["total_energy"] - K0) / K0}

    nx0 = n_waves * cells_per_wave
    base = one(nx0, dt)
    fine_x = one(2 * nx0, dt)
    fine_t = one(nx0, dt / 2)
    rel = lambda a, b: abs(a - b) / abs(b)
    return {"dx": L / nx0, "dx_over_lambda_Db": (L / nx0) / theory.beam_debye_length(0.2),
            "base": base, "dx_half": fine_x, "dt_half": fine_t,
            "fe_rel_change_dx": rel(base["fe_mean"], fine_x["fe_mean"]),
            "fe_rel_change_dt": rel(base["fe_mean"], fine_t["fe_mean"]),
            "dK_rel_change_dx": rel(base["dK"], fine_x["dK"]),
            "dK_rel_change_dt": rel(base["dK"], fine_t["dK"])}
