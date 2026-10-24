"""
attractor_sim/vlasov1d1v.py
===========================

1D1V semi-Lagrangian Vlasov-Poisson reference (Cheng-Knorr Strang splitting).

* x-advection (half steps): exact spectral shift of each velocity column,
  f_k(v) -> f_k(v) exp(-i k v dt/2).  Mass-conserving; not strictly
  positivity-preserving at grid-scale filamentation (the minimum of f is
  recorded so this can be monitored).
* v-advection (full step): conservative and positivity-preserving
  semi-Lagrangian update through the primitive F(v) = int f dv, interpolated
  with monotone piecewise-cubic Hermite (Fritsch-Carlson) slopes.  Monotone
  F gives non-negative cell averages; telescoping fluxes conserve mass
  except for what leaves through +-v_max, which is accumulated and reported.
* Poisson: same spectral solve, background and normalisation as the PIC.

The regulator is filamentation dissipation at the grid scale Delta v.  A
finite-grid run is a *reference calculation*, not collisionless truth.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import fft as sfft

try:
    import numba

    HAVE_NUMBA = True
except Exception:  # pragma: no cover
    HAVE_NUMBA = False


Q = -1.0
M = 1.0


def _pchip_slopes(F: np.ndarray, h: float) -> np.ndarray:
    """Fritsch-Carlson monotone slopes along the last axis (uniform spacing h)."""
    d = np.diff(F, axis=-1) / h
    m = np.zeros_like(F)
    d0, d1 = d[..., :-1], d[..., 1:]
    same = (d0 * d1) > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        hm = np.where(same, 2.0 / (1.0 / np.where(same, d0, 1) + 1.0 / np.where(same, d1, 1)), 0.0)
    m[..., 1:-1] = hm
    m[..., 0] = d[..., 0]
    m[..., -1] = d[..., -1]
    return m


def _hermite(F, m, idx, theta, h):
    """Evaluate the Hermite cubic on interval [idx, idx+1] at fraction theta."""
    n = F.shape[-1]
    i0 = np.clip(idx, 0, n - 1)
    i1 = np.clip(idx + 1, 0, n - 1)
    y0 = np.take_along_axis(F, i0, axis=-1)
    y1 = np.take_along_axis(F, i1, axis=-1)
    m0 = np.take_along_axis(m, i0, axis=-1)
    m1 = np.take_along_axis(m, i1, axis=-1)
    t = theta
    h00 = (1 + 2 * t) * (1 - t) ** 2
    h10 = t * (1 - t) ** 2
    h01 = t * t * (3 - 2 * t)
    h11 = t * t * (t - 1)
    return h00 * y0 + h10 * h * m0 + h01 * y1 + h11 * h * m1


if HAVE_NUMBA:
    @numba.njit(parallel=True)
    def _advect_v_numba(f, shift, dv):
        """Row-parallel conservative PCHIP-primitive shift; returns per-row outflow mass (x dv)."""
        nx, nv = f.shape
        out_mass = np.zeros(nx)
        for r in numba.prange(nx):
            F = np.empty(nv + 1)
            F[0] = 0.0
            for j in range(nv):
                F[j + 1] = F[j] + f[r, j] * dv
            d = np.empty(nv)
            for j in range(nv):
                d[j] = (F[j + 1] - F[j]) / dv
            m = np.empty(nv + 1)
            m[0] = d[0]
            m[nv] = d[nv - 1]
            for j in range(1, nv):
                if d[j - 1] * d[j] > 0.0:
                    m[j] = 2.0 / (1.0 / d[j - 1] + 1.0 / d[j])
                else:
                    m[j] = 0.0
            sh = -shift[r] / dv
            n = int(np.floor(sh))
            t = sh - n
            h00 = (1 + 2 * t) * (1 - t) ** 2
            h10 = t * (1 - t) ** 2
            h01 = t * t * (3 - 2 * t)
            h11 = t * t * (t - 1)
            Fprev = 0.0
            total = F[nv]
            new_total = 0.0
            for j in range(nv + 1):
                s_ = j + sh
                i = j + n
                if s_ <= 0.0:
                    Fn = 0.0
                elif s_ >= nv:
                    Fn = total
                else:
                    i0 = i
                    i1 = i + 1
                    if i1 > nv:
                        i1 = nv
                    Fn = h00 * F[i0] + h10 * dv * m[i0] + h01 * F[i1] + h11 * dv * m[i1]
                if j > 0:
                    val = (Fn - Fprev) / dv
                    if val < 0.0:
                        val = 0.0
                    f[r, j - 1] = val
                    new_total += val * dv
                Fprev = Fn
            out_mass[r] = total - new_total
        return out_mass


@dataclass
class Vlasov1D1V:
    L: float
    nx: int
    v_max: float
    nv: int
    f: np.ndarray            # (nx, nv) cell averages
    n0: float = 1.0
    dx: float = field(init=False)
    dv: float = field(init=False)
    v: np.ndarray = field(init=False)
    k: np.ndarray = field(init=False)
    E: np.ndarray = field(init=False)
    outflow_mass: float = 0.0
    outflow_energy: float = 0.0
    _phase_dt: float = field(init=False, default=np.nan)
    _phase: np.ndarray = field(init=False, default=None)

    def __post_init__(self):
        self.dx = self.L / self.nx
        self.dv = 2 * self.v_max / self.nv
        self.v = -self.v_max + (np.arange(self.nv) + 0.5) * self.dv
        self.k = 2 * np.pi * np.fft.rfftfreq(self.nx, d=self.dx)
        self.f = np.ascontiguousarray(self.f, dtype=np.float64)
        self.E = np.zeros(self.nx)
        self.solve()

    def density(self) -> np.ndarray:
        return self.f.sum(axis=1) * self.dv

    def solve(self) -> None:
        rho = Q * self.density() + self.n0
        rk = np.fft.rfft(rho)
        phik = np.zeros_like(rk)
        phik[1:] = rk[1:] / self.k[1:] ** 2
        Ek = -1j * self.k * phik
        if self.nx % 2 == 0:
            Ek[-1] = 0.0
        self.E = np.fft.irfft(Ek, n=self.nx)

    def _advect_x(self, dt):
        if dt != self._phase_dt:
            self._phase = np.exp(-1j * np.outer(self.k, self.v) * dt)
            self._phase_dt = dt
        fk = sfft.rfft(self.f, axis=0, workers=-1)
        fk *= self._phase
        self.f = np.ascontiguousarray(sfft.irfft(fk, n=self.nx, axis=0, workers=-1))

    def _advect_v(self, dt):
        a = (Q / M) * self.E                      # acceleration per x row
        shift = a * dt                            # v displacement per row
        if HAVE_NUMBA:
            lost = _advect_v_numba(self.f, shift, self.dv)
            self.outflow_mass += float(lost.sum() * self.dx)
            return
        # primitive at cell faces v_{j-1/2}, j = 0..nv
        F = np.concatenate([np.zeros((self.nx, 1)), np.cumsum(self.f, axis=1) * self.dv], axis=1)
        m = _pchip_slopes(F, self.dv)
        # departure points of faces: v_face - shift, in face-index units
        s = -shift[:, None] / self.dv + np.arange(self.nv + 1)[None, :]
        idx = np.floor(s).astype(np.int64)
        theta = s - idx
        Fn = _hermite(F, m, idx, theta, self.dv)
        Fn = np.where(s <= 0, 0.0, Fn)
        Fn = np.where(s >= self.nv, F[:, -1:], Fn)
        fnew = np.diff(Fn, axis=1) / self.dv
        before = self.f.sum() * self.dv * self.dx
        self.f = np.maximum(fnew, 0.0)
        after = self.f.sum() * self.dv * self.dx
        self.outflow_mass += before - after

    def step(self, dt: float) -> None:
        """Strang: x(dt/2) -> Poisson -> v(dt) -> x(dt/2) -> Poisson."""
        self._advect_x(0.5 * dt)
        self.solve()
        self._advect_v(dt)
        self._advect_x(0.5 * dt)
        self.solve()

    def budgets(self) -> dict:
        number = float(self.f.sum() * self.dv * self.dx)
        mom = float(M * np.sum(self.f * self.v[None, :]) * self.dv * self.dx)
        kin = float(0.5 * M * np.sum(self.f * self.v[None, :] ** 2) * self.dv * self.dx)
        fe = float(0.5 * np.sum(self.E ** 2) * self.dx)
        return {"number": number, "total_charge": Q * number + self.n0 * self.L, "momentum": mom,
                "kinetic_energy": kin, "field_energy": fe, "total_energy": kin + fe,
                "f_min": float(self.f.min()), "outflow_mass": self.outflow_mass,
                "L2": float(np.sum(self.f ** 2) * self.dv * self.dx)}

    def field_spectrum(self) -> np.ndarray:
        return np.abs(np.fft.rfft(self.E)) ** 2


def waterbag_beams_f(L, nx, v_max, nv, v_b, dv_b, n0=1.0, modes=(), amplitudes=(), phases=()):
    """Cell-averaged symmetric waterbag beams with a density seed.

    Cell averages integrate the exact waterbag over each velocity cell, so
    partially covered edge cells carry fractional values (no staircase bias
    in the second moment).
    """
    dv = 2 * v_max / nv
    faces = -v_max + np.arange(nv + 1) * dv
    prof = np.zeros(nv)
    for c in (-v_b, v_b):
        lo, hi = c - dv_b / 2, c + dv_b / 2
        overlap = np.clip(np.minimum(faces[1:], hi) - np.maximum(faces[:-1], lo), 0, None)
        prof += (n0 / (2 * dv_b)) * overlap / dv
    x = (np.arange(nx) + 0.5) * (L / nx)
    dens = np.ones(nx)
    for m, a, ph in zip(modes, amplitudes, phases):
        dens += a * np.cos(2 * np.pi * m / L * x + ph)
    return np.outer(dens, prof)


def maxwellian_f(L, nx, v_max, nv, v_th=1.0, n0=1.0, mode=1, amplitude=0.0):
    dv = 2 * v_max / nv
    v = -v_max + (np.arange(nv) + 0.5) * dv
    x = (np.arange(nx) + 0.5) * (L / nx)
    prof = n0 * np.exp(-v * v / (2 * v_th**2)) / np.sqrt(2 * np.pi) / v_th
    return np.outer(1 + amplitude * np.cos(2 * np.pi * mode / L * x), prof)
