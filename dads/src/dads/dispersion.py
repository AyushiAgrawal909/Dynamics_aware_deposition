"""Dielectric function of one grid mode for the implemented coupling.

For a grid mode with wavevector ``k`` the linear response of the
deposit -> filter -> solve -> filter -> gather chain is

    eps(k, omega; a) = 1 + a * G(k, omega),
    G(k, omega)      = sum_q geom(q) sum_s chi_s(q, omega),

where the sum runs over the spatial aliases ``q = k + 2 pi p / dx`` of the
mode, ``geom = S(q)^2 (q . kappa) / K^2`` contains the particle shape and the
field solve, and ``a`` is the round-trip filter transfer at the mode
(``a = W(k)^2`` for a filter applied to the density and again to the field).
Because the filter acts on the mesh, it multiplies every alias term equally,
so the whole filter dependence of one mode's roots enters through the single
number ``a``.

For a drifting Maxwellian species

    chi_s(q, omega) = omega_ps^2 / (|q|^2 v_q^2) [1 + zeta Z(zeta)],
    zeta = (omega - q . u_s) / (sqrt(2) |q| v_q),

with ``v_q^2 = sum_i qhat_i^2 v_th,i^2``; a cold species gives
``-omega_ps^2 / (omega - q . u_s)^2``.

Time integration
----------------
``continuous``: exact-in-time response.
``leapfrog``: the leapfrog (velocity-Verlet) particle push with the field
sampled at integer steps.  Its susceptibility is the sum of the continuous
one over the temporal aliases ``omega - 2 pi l / dt``.  The sum is evaluated
exactly for cold species and, for warm ones, explicitly for |l| <= L with an
analytic tail correct to O((q v)^4).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .grid import Grid
from .plasma import Plasma
from .zfunc import maxwell_response, maxwell_response_derivative

SQRT2 = math.sqrt(2.0)


def _g2(y):
    """csc(y)^2 - 1/y^2, accurate near y = 0."""
    y = np.asarray(y, dtype=np.complex128)
    out = np.empty_like(y)
    small = np.abs(y) < 0.25
    ys = y[small]
    y2 = ys * ys
    out[small] = (1.0 / 3.0 + y2 * (1.0 / 15.0 + y2 * (2.0 / 189.0 + y2 * (1.0 / 675.0 + y2 * (
        2.0 / 10395.0 + y2 * 1382.0 / 58046625.0)))))
    yb = y[~small]
    out[~small] = 1.0 / np.sin(yb) ** 2 - 1.0 / (yb * yb)
    return out


def _r(y):
    """(csc(y)^2 - 1/y^2 - 1/3) / y^2, accurate near y = 0."""
    y = np.asarray(y, dtype=np.complex128)
    out = np.empty_like(y)
    small = np.abs(y) < 0.25
    ys = y[small]
    y2 = ys * ys
    out[small] = (1.0 / 15.0 + y2 * (2.0 / 189.0 + y2 * (1.0 / 675.0 + y2 * (
        2.0 / 10395.0 + y2 * 1382.0 / 58046625.0))))
    yb = y[~small]
    out[~small] = (1.0 / np.sin(yb) ** 2 - 1.0 / (yb * yb) - 1.0 / 3.0) / (yb * yb)
    return out


@dataclass
class _Terms:
    coef: np.ndarray      # warm: geom * wp2 / (|q|^2 v_q^2); cold: geom * wp2
    shift: np.ndarray     # q . u
    scale: np.ndarray     # warm: sqrt(2) |q| v_q
    qv2: np.ndarray       # warm: |q|^2 v_q^2
    geom_wp2: np.ndarray  # geom * wp2


class ModeDispersion:
    """Dielectric function ``1 + a G(omega)`` of one grid mode.

    Parameters
    ----------
    grid, plasma
        Mesh/coupling and equilibrium.
    mode
        Integer mode indices.
    n_alias
        Spatial aliases per axis (0 = principal harmonic only).
    time
        ``continuous`` or ``leapfrog``.
    dt
        Time step (required for ``leapfrog``).
    n_time_alias
        Explicit temporal aliases for warm species in the leapfrog model.
    order
        Particle-shape order override (default: the grid's shape).
    """

    def __init__(self, grid: Grid, plasma: Plasma, mode, *, n_alias: int = 4,
                 time: str = "continuous", dt: float | None = None,
                 n_time_alias: int = 8, order: int | None = None):
        self.grid = grid
        self.plasma = plasma
        self.mode = tuple(int(x) for x in grid.mode_vector(mode))
        for mi, ni in zip(self.mode, grid.N):
            if ni % 2 == 0 and abs(mi) == ni // 2:
                raise ValueError(f"mode {self.mode} sits at the Nyquist index of an axis, where a "
                                 "spectral or centred derivative carries no field; choose another mode")
        self.k = grid.k_of_mode(self.mode)
        self.n_alias = int(n_alias)
        self.order = grid.order if order is None else int(order)
        self.time = time.lower()
        if self.time not in ("continuous", "leapfrog"):
            raise ValueError("time must be 'continuous' or 'leapfrog'")
        if self.time == "leapfrog":
            if dt is None or dt <= 0:
                raise ValueError("the leapfrog model needs a positive dt")
        self.dt = None if dt is None else float(dt)
        self.n_time_alias = int(n_time_alias)
        q, geom = grid.aliases(self.k, self.n_alias, self.order)
        self.q, self.geom = q, geom
        qn = np.sqrt(np.sum(q * q, axis=1))
        warm, cold = [], []
        for s in plasma.mobile:
            wp2 = s.omega_p2(plasma.eps0)
            shift = q @ s.u
            vq = np.sqrt((q * q) @ (s.v**2)) / qn
            hot = vq > 0.0
            if np.any(hot):
                qv2 = (qn[hot] * vq[hot]) ** 2
                warm.append((geom[hot] * wp2 / qv2, shift[hot], SQRT2 * qn[hot] * vq[hot], qv2,
                             geom[hot] * wp2))
            if np.any(~hot):
                c = ~hot
                cold.append((geom[c] * wp2, shift[c], np.zeros(c.sum()), np.zeros(c.sum()),
                             geom[c] * wp2))
        self._warm = _Terms(*[np.concatenate(x) for x in zip(*warm)]) if warm else None
        self._cold = _Terms(*[np.concatenate(x) for x in zip(*cold)]) if cold else None

    # ----------------------------------------------------------- response
    def _G_continuous(self, w: np.ndarray) -> np.ndarray:
        out = np.zeros(w.shape, dtype=np.complex128)
        if self._warm is not None:
            t = self._warm
            zeta = (w[..., None] - t.shift) / t.scale
            out += np.sum(t.coef * maxwell_response(zeta), axis=-1)
        if self._cold is not None:
            t = self._cold
            out -= np.sum(t.coef / (w[..., None] - t.shift) ** 2, axis=-1)
        return out

    def _dG_continuous(self, w: np.ndarray) -> np.ndarray:
        out = np.zeros(w.shape, dtype=np.complex128)
        if self._warm is not None:
            t = self._warm
            zeta = (w[..., None] - t.shift) / t.scale
            out += np.sum(t.coef / t.scale * maxwell_response_derivative(zeta), axis=-1)
        if self._cold is not None:
            t = self._cold
            out += np.sum(2.0 * t.coef / (w[..., None] - t.shift) ** 3, axis=-1)
        return out

    def _G_leapfrog(self, w: np.ndarray) -> np.ndarray:
        h = 0.5 * self.dt
        wg = 2.0 * math.pi / self.dt
        out = np.zeros(w.shape, dtype=np.complex128)
        if self._cold is not None:
            t = self._cold
            y = h * (w[..., None] - t.shift)
            out -= np.sum(t.coef * h * h / np.sin(y) ** 2, axis=-1)
        if self._warm is not None:
            t = self._warm
            L = self.n_time_alias
            l = np.arange(-L, L + 1, dtype=float)
            nz = l[l != 0]
            omega0 = w[..., None] - t.shift                      # (..., T)
            zeta = (omega0[..., None] - l * wg) / t.scale[:, None]
            out += np.sum(t.coef[:, None] * maxwell_response(zeta), axis=(-1, -2))
            y = h * omega0
            g2 = _g2(y)
            g4 = 2.0 * _r(y) + g2 * g2 - (2.0 / 3.0) * g2
            om_l = omega0[..., None] - nz * wg
            tail2 = h**2 * g2 - np.sum(1.0 / om_l**2, axis=-1)
            tail4 = h**4 * g4 - np.sum(1.0 / om_l**4, axis=-1)
            out -= np.sum(t.geom_wp2 * (tail2 + 3.0 * t.qv2 * tail4), axis=-1)
        return out

    def G(self, omega):
        """Unfiltered response sum G(omega); eps = 1 + a G."""
        w = np.asarray(omega, dtype=np.complex128)
        with np.errstate(over="ignore", invalid="ignore"):
            out = self._G_continuous(w) if self.time == "continuous" else self._G_leapfrog(w)
        return out if out.shape else complex(out)

    def dG(self, omega):
        """dG/domega."""
        w = np.asarray(omega, dtype=np.complex128)
        with np.errstate(over="ignore", invalid="ignore"):
            if self.time == "continuous":
                out = self._dG_continuous(w)
            else:
                step = 1e-6 * max(1.0, float(np.max(np.abs(w))))
                out = (self._G_leapfrog(w + step) - self._G_leapfrog(w - step)) / (2.0 * step)
        return out if out.shape else complex(out)

    def eps(self, omega, a: float = 1.0):
        return 1.0 + a * self.G(omega)

    # ------------------------------------------------------------ helpers
    def frequency_window(self, all_terms: bool = False) -> tuple[float, float, float]:
        """(Re-omega min, Re-omega max, Im-omega max) bracketing the roots.

        With ``all_terms`` the window also covers the Doppler-shifted
        resonances q.u of every retained alias, where aliasing instabilities
        of drifting species live.
        """
        wp = self.plasma.omega_p_total
        shift = [float(self.k @ s.u) for s in self.plasma.mobile]
        spread = [float(SQRT2 * np.sqrt((self.k**2) @ s.v**2)) for s in self.plasma.mobile]
        lo = min(shift) - 4.0 * max(spread) - 1.5 * wp
        hi = max(shift) + 4.0 * max(spread) + 1.5 * wp
        if all_terms:
            sh, sc = self.resonances()
            if sh.size:
                lo = min(lo, float(np.min(sh - 4.0 * sc)) - 1.5 * wp)
                hi = max(hi, float(np.max(sh + 4.0 * sc)) + 1.5 * wp)
        return lo, hi, 1.25 * wp

    def resonances(self, rel: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
        """Doppler shifts q.u and thermal widths of the retained response terms.

        ``rel`` keeps only terms whose geometric weight exceeds rel times the
        principal one.
        """
        terms = [t for t in (self._warm, self._cold) if t is not None]
        if not terms:
            return np.zeros(0), np.zeros(0)
        big = max(float(np.max(np.abs(t.geom_wp2))) for t in terms)
        shifts = [t.shift[np.abs(t.geom_wp2) >= rel * big] for t in terms]
        scales = [t.scale[np.abs(t.geom_wp2) >= rel * big] for t in terms]
        return np.concatenate(shifts), np.concatenate(scales)

    def strong_warm_resonances(self, threshold: float = 1e-2) -> tuple[np.ndarray, np.ndarray]:
        """(q.u, sqrt(2)|q|v_q) of warm terms whose response can reach ``threshold``.

        A warm term contributes at most about |coef| to eps, so weaker terms
        cannot turn the phase of eps quickly and need no extra sampling.
        """
        t = self._warm
        if t is None:
            return np.zeros(0), np.zeros(0)
        keep = np.abs(t.coef) > threshold
        return t.shift[keep], t.scale[keep]

    def cold_poles(self) -> np.ndarray:
        """Distinct real frequencies q.u where cold terms have double poles."""
        if self._cold is None:
            return np.zeros(0)
        return np.unique(np.round(self._cold.shift, 12))

    @property
    def omega_grid(self) -> float | None:
        return None if self.dt is None or self.time != "leapfrog" else 2.0 * math.pi / self.dt


class BlendedDispersion:
    """Homotopy between two coupling models: G_t = (1 - t) G_0 + t G_1."""

    def __init__(self, d0: ModeDispersion, d1: ModeDispersion, t: float):
        self.d0, self.d1, self.t = d0, d1, float(t)
        self.k, self.mode, self.plasma = d0.k, d0.mode, d0.plasma
        self.time, self.dt = d0.time, d0.dt

    def G(self, omega):
        return (1.0 - self.t) * self.d0.G(omega) + self.t * self.d1.G(omega)

    def dG(self, omega):
        return (1.0 - self.t) * self.d0.dG(omega) + self.t * self.d1.dG(omega)

    def eps(self, omega, a: float = 1.0):
        return 1.0 + a * self.G(omega)

    def frequency_window(self, all_terms: bool = False):
        lo0, hi0, im0 = self.d0.frequency_window(all_terms)
        lo1, hi1, im1 = self.d1.frequency_window(all_terms)
        return min(lo0, lo1), max(hi0, hi1), max(im0, im1)

    def resonances(self, rel: float = 0.0):
        a, b = self.d0.resonances(rel), self.d1.resonances(rel)
        return np.concatenate([a[0], b[0]]), np.concatenate([a[1], b[1]])

    def cold_poles(self):
        return np.unique(np.concatenate([self.d0.cold_poles(), self.d1.cold_poles()]))

    def strong_warm_resonances(self, threshold: float = 1e-2):
        a, b = self.d0.strong_warm_resonances(threshold), self.d1.strong_warm_resonances(threshold)
        return np.concatenate([a[0], b[0]]), np.concatenate([a[1], b[1]])

    @property
    def omega_grid(self):
        return self.d0.omega_grid
