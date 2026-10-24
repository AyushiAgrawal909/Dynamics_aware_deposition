"""
attractor_sim/theory.py
=======================

Linear theory for the T2 preparations.

Symmetric waterbag beams (each of density n0/2, full width dv_b, centred at
+-v_b).  A waterbag of half-width a contributes the susceptibility
-(w_p^2) / [(w - k u)^2 - k^2 a^2], so with total omega_pe = 1

    eps(k, w) = 1 - (1/2) sum_{+-} 1 / [(w -+ k v_b)^2 - k^2 dv_b^2 / 4] = 0,

a quartic in w.  Cold limit (dv_b -> 0): gamma_max = 1/(2 sqrt 2) at
k v_b = sqrt(3/8).
"""

from __future__ import annotations

import numpy as np
from scipy import optimize, special


def waterbag_roots(k: float, v_b: float = 1.0, dv_b: float = 0.2) -> np.ndarray:
    """All four complex roots w(k) of the waterbag-beam dispersion relation."""
    a2 = (k * dv_b / 2.0) ** 2
    P1 = np.poly1d([1.0, -2 * k * v_b, (k * v_b) ** 2 - a2])  # (w - k v_b)^2 - a^2
    P2 = np.poly1d([1.0, 2 * k * v_b, (k * v_b) ** 2 - a2])
    poly = P1 * P2 - 0.5 * (P1 + P2)
    return np.roots(poly.c)


def waterbag_growth(k, v_b: float = 1.0, dv_b: float = 0.2) -> np.ndarray:
    """Largest imaginary part of w(k) (0 if stable)."""
    k = np.atleast_1d(np.asarray(k, dtype=float))
    return np.array([max(0.0, float(np.max(waterbag_roots(kk, v_b, dv_b).imag))) for kk in k])


def fastest_mode(v_b: float = 1.0, dv_b: float = 0.2) -> dict:
    """k*, gamma_max, lambda* and the unstable band [0, k_c)."""
    res = optimize.minimize_scalar(lambda k: -waterbag_growth(k, v_b, dv_b)[0],
                                   bounds=(0.05 / v_b, 1.2 / v_b), method="bounded",
                                   options={"xatol": 1e-10})
    ks = np.linspace(1e-4, 2.0 / v_b, 4001)
    g = waterbag_growth(ks, v_b, dv_b)
    unstable = ks[g > 1e-10]
    return {"k_star": float(res.x), "gamma_max": float(-res.fun), "lambda_star": float(2 * np.pi / res.x),
            "k_cut": float(unstable.max()) if unstable.size else 0.0}


def beam_debye_length(dv_b: float) -> float:
    """lambda_Db = sigma_v / omega_p with sigma_v = dv_b / sqrt(12) (waterbag rms)."""
    return dv_b / np.sqrt(12.0)


def landau_root(k: float, v_th: float = 1.0, guess=(1.4156, -0.1533)) -> complex:
    """Least-damped Landau root of 1 + (1 + zeta Z(zeta)) / (k lambda_D)^2 = 0 for a Maxwellian.

    Z(zeta) = i sqrt(pi) w(zeta) (Faddeeva).  For k lambda_D = 0.5:
    w = 1.4156 - 0.1533 i.
    """
    def disp(z):
        w = z[0] + 1j * z[1]
        zeta = w / (np.sqrt(2.0) * k * v_th)
        Z = 1j * np.sqrt(np.pi) * special.wofz(zeta)
        d = 1.0 + (1.0 + zeta * Z) / (k * v_th) ** 2
        return [d.real, d.imag]

    sol = optimize.fsolve(disp, guess, xtol=1e-13)
    return complex(sol[0], sol[1])
