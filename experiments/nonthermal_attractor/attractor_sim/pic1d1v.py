"""
attractor_sim/pic1d1v.py
========================

1D1V electrostatic spectral PIC: the matched-CIC chain of ``pic_kde_2d``
(``methods/cic.py`` deposit and gather, ``core/grid2d.py`` FFT Poisson with
E = -i k phi_k at nodes) collapsed to one dimension.

    CIC deposit -> FFT Poisson (k = 0 removed; explicit uniform immobile
    background +n0) -> spectral E at nodes -> matched CIC gather -> KDK push.

Momentum: with identical deposit and gather weights and an antisymmetric
spectral gradient, the total force sum_p q_p E(x_p) w_p = -sum_j rho_j E_j dx
vanishes to round-off (the self-force is zero and pair forces are equal and
opposite).  Energy is not conserved exactly; its drift is a quality-gate
quantity.  No smoothing: fixed CIC is the unsmoothed baseline.

Numba (parallel, thread count from NUMBA_NUM_THREADS) is used when
available (``requirements-simulation.txt``); otherwise vectorised numpy.
Deposition sums per-thread private grids, so results can differ between
thread counts at round-off level only.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

try:
    import numba

    _njit = numba.njit(cache=False, fastmath=False)
    HAVE_NUMBA = True
except Exception:  # pragma: no cover
    HAVE_NUMBA = False

    def _njit(f):
        return f


Q = -1.0   # electron charge
M = 1.0    # electron mass


if HAVE_NUMBA:
    _pjit = numba.njit(parallel=True, fastmath=False)

    @_pjit
    def _deposit(x, w, nx, dx, out):
        """CIC deposit with per-chunk private grids (no write races)."""
        nchunk = numba.get_num_threads()
        n = x.size
        priv = np.zeros((nchunk, nx))
        size = (n + nchunk - 1) // nchunk
        for c in numba.prange(nchunk):
            lo = c * size
            hi = min(n, lo + size)
            for p in range(lo, hi):
                s = x[p] / dx
                i = int(np.floor(s))
                f = s - i
                i0 = i % nx
                i1 = (i0 + 1) % nx
                priv[c, i0] += w[p] * (1.0 - f)
                priv[c, i1] += w[p] * f
        for j in range(nx):
            acc = 0.0
            for c in range(nchunk):
                acc += priv[c, j]
            out[j] = acc / dx

    @_pjit
    def _gather_push_drift(x, v, E, nx, dx, L, half_kick, dt):
        """v += half_kick * E(x); x += dt * v (wrapped)."""
        for p in numba.prange(x.size):
            s = x[p] / dx
            i = int(np.floor(s))
            f = s - i
            i0 = i % nx
            i1 = (i0 + 1) % nx
            v[p] += half_kick * ((1.0 - f) * E[i0] + f * E[i1])
            xn = (x[p] + dt * v[p]) % L
            if xn >= L:
                xn -= L
            x[p] = xn

    @_pjit
    def _gather_kick(x, v, E, nx, dx, half_kick):
        for p in numba.prange(x.size):
            s = x[p] / dx
            i = int(np.floor(s))
            f = s - i
            i0 = i % nx
            i1 = (i0 + 1) % nx
            v[p] += half_kick * ((1.0 - f) * E[i0] + f * E[i1])


def _deposit_np(x, w, nx, dx):
    s = x / dx
    i = np.floor(s).astype(np.int64)
    f = s - i
    i0 = i % nx
    i1 = (i0 + 1) % nx
    return (np.bincount(i0, w * (1 - f), nx) + np.bincount(i1, w * f, nx)) / dx


def _gather_np(x, E, nx, dx):
    s = x / dx
    i = np.floor(s).astype(np.int64)
    f = s - i
    i0 = i % nx
    return (1 - f) * E[i0] + f * E[(i0 + 1) % nx]


@dataclass
class PIC1D1V:
    """State and stepping for one 1D1V run."""

    L: float
    nx: int
    x: np.ndarray
    v: np.ndarray
    w: np.ndarray
    n0: float = 1.0
    dx: float = field(init=False)
    k: np.ndarray = field(init=False)
    E: np.ndarray = field(init=False)
    rho: np.ndarray = field(init=False)

    def __post_init__(self):
        self.dx = self.L / self.nx
        self.k = 2 * np.pi * np.fft.rfftfreq(self.nx, d=self.dx)
        self.x = np.ascontiguousarray(self.x % self.L, dtype=np.float64)
        self.v = np.ascontiguousarray(self.v, dtype=np.float64)
        self.w = np.ascontiguousarray(self.w, dtype=np.float64)
        self.rho = np.zeros(self.nx)
        self.E = np.zeros(self.nx)
        self.solve()

    # ------------------------------------------------------------------
    def electron_density(self) -> np.ndarray:
        if HAVE_NUMBA:
            out = np.empty(self.nx)
            _deposit(self.x, self.w, self.nx, self.dx, out)
            return out
        return _deposit_np(self.x, self.w, self.nx, self.dx)

    def solve(self) -> None:
        """rho = Q n_e + n0 (background); -phi'' = rho; E = -i k phi_k."""
        ne = self.electron_density()
        self.rho = Q * ne + self.n0
        rk = np.fft.rfft(self.rho)
        phik = np.zeros_like(rk)
        phik[1:] = rk[1:] / self.k[1:] ** 2
        Ek = -1j * self.k * phik
        if self.nx % 2 == 0:
            Ek[-1] = 0.0  # Nyquist: no antisymmetric derivative exists
        self.E = np.fft.irfft(Ek, n=self.nx)

    def step(self, dt: float) -> None:
        """Kick-drift-kick; x, v and E are synchronised at integer steps."""
        hk = 0.5 * dt * Q / M
        if HAVE_NUMBA:
            _gather_push_drift(self.x, self.v, self.E, self.nx, self.dx, self.L, hk, dt)
            self.solve()
            _gather_kick(self.x, self.v, self.E, self.nx, self.dx, hk)
        else:
            self.v += hk * _gather_np(self.x, self.E, self.nx, self.dx)
            self.x = (self.x + dt * self.v) % self.L
            self.solve()
            self.v += hk * _gather_np(self.x, self.E, self.nx, self.dx)

    # ------------------------------------------------------------------
    def budgets(self) -> dict:
        number = float(np.sum(self.w))
        kin = float(0.5 * M * np.sum(self.w * self.v * self.v))
        fe = float(0.5 * np.sum(self.E * self.E) * self.dx)
        return {"number": number, "total_charge": Q * number + self.n0 * self.L,
                "momentum": float(M * np.sum(self.w * self.v)), "kinetic_energy": kin,
                "field_energy": fe, "total_energy": kin + fe}

    def field_spectrum(self) -> np.ndarray:
        return np.abs(np.fft.rfft(self.E)) ** 2


def seed_displacement(x: np.ndarray, L: float, modes, amplitudes, phases) -> np.ndarray:
    """Displace a uniform load to impose delta n / n0 = sum a_m cos(k_m x + phi_m).

    Continuity at first order: delta x = -sum (a_m / k_m) sin(k_m x + phi_m).
    """
    dxp = np.zeros_like(x)
    for m, a, ph in zip(modes, amplitudes, phases):
        km = 2 * np.pi * m / L
        dxp -= (a / km) * np.sin(km * x + ph)
    return (x + dxp) % L
