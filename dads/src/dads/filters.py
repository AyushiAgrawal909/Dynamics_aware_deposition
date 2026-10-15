"""Zero-phase mesh filters and their Fourier transfer functions.

A filter is applied to the deposited density and, in the matched scheme,
again to the field before gathering, so the particle coupling contains
``W(k)^2``.  Families:

* ``none``     -- the unfiltered baseline, W = 1;
* ``gaussian`` -- W = exp(-1/2 sum_i (k_i dx_i h_i)^2), widths h_i in cells;
* ``binomial`` -- compensated binomial of degree p after n passes per axis,
  W = (1 - s)^n sum_{j=0}^{p} C(n+j-1, j) s^j,  s = sin^2(k dx / 2).
  p = 0 is the raw n-pass binomial filter [1/4, 1/2, 1/4]^n; the transfer
  error is O((k dx)^(2p+2)).
* ``custom``   -- any user-supplied transfer function (Python API only).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

from .grid import Grid


class Filter:
    family = "base"

    def transfer(self, k: np.ndarray, grid: Grid) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    @property
    def label(self) -> str:  # pragma: no cover
        raise NotImplementedError

    def params(self) -> dict:
        return {}

    def cache_key(self):
        return (self.family, repr(self.params()))

    def at_mode(self, grid: Grid, m) -> float:
        return float(self.transfer(grid.k_of_mode(m), grid))

    def transfer_grid(self, grid: Grid) -> np.ndarray:
        """W on the full ``numpy.fft.fftn`` wavenumber grid."""
        K = np.stack(grid.fft_wavenumbers(), axis=-1)
        return self.transfer(K, grid)

    def apply(self, field_values: np.ndarray, grid: Grid) -> np.ndarray:
        """Filter a periodic mesh quantity of shape ``grid.N`` (or batches)."""
        W = self.transfer_grid(grid)
        axes = tuple(range(-grid.dim, 0))
        return np.real(np.fft.ifftn(np.fft.fftn(field_values, axes=axes) * W, axes=axes))

    def half_power_wavenumber(self, grid: Grid, axis: int) -> float | None:
        """Wavenumber (fraction of Nyquist) along one axis where W = 1/2."""
        def W_axis(frac):
            k = np.zeros(grid.dim)
            k[axis] = frac * math.pi / grid.dx[axis]
            return float(self.transfer(k, grid))
        if W_axis(1.0) > 0.5:
            return None
        lo, hi = 0.0, 1.0
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if W_axis(mid) > 0.5:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    def describe(self, grid: Grid) -> dict:
        d = dict(family=self.family, label=self.label, **self.params())
        d["half_power_k_over_nyquist"] = [self.half_power_wavenumber(grid, i) for i in range(grid.dim)]
        return d


class NoFilter(Filter):
    family = "none"

    def transfer(self, k, grid):
        k = np.asarray(k, dtype=float)
        return np.ones(k.shape[:-1]) if k.ndim > 1 else np.float64(1.0)

    @property
    def label(self):
        return "unfiltered"


@dataclass
class Gaussian(Filter):
    """Gaussian filter with widths ``h`` in cells: sigma_i = h_i dx_i."""

    h: Sequence[float]
    family = "gaussian"

    def __post_init__(self):
        self.h = tuple(float(x) for x in np.atleast_1d(self.h))
        if any(x < 0 for x in self.h):
            raise ValueError("Gaussian widths must be non-negative")

    def transfer(self, k, grid):
        k = np.asarray(k, dtype=float)
        h = np.broadcast_to(np.array(self.h), (grid.dim,))
        return np.exp(-0.5 * np.sum((k * grid.dx * h) ** 2, axis=-1))

    @property
    def label(self):
        return "gaussian(h=" + ",".join(f"{x:.4g}" for x in self.h) + ")"

    def params(self):
        return dict(h_cells=list(self.h))

    def describe(self, grid):
        d = super().describe(grid)
        h = np.broadcast_to(np.array(self.h), (grid.dim,))
        d["sigma"] = (h * grid.dx).tolist()
        return d


def binomial_axis_transfer(s: np.ndarray, n: int, p: int) -> np.ndarray:
    """Compensated binomial transfer on one axis as a function of s = sin^2(k dx/2)."""
    s = np.asarray(s, dtype=float)
    if n == 0:
        return np.ones_like(s)
    comp = np.ones_like(s)
    coef = 1.0
    for j in range(1, p + 1):
        coef *= (n + j - 1) / j
        comp = comp + coef * s**j
    return (1.0 - s) ** n * comp


def binomial_stencil(n: int, p: int) -> np.ndarray:
    """Real-space taps (offsets -(n+p)..(n+p)) of the 1D compensated binomial."""
    one_minus_s = np.array([0.25, 0.5, 0.25])
    s = np.array([-0.25, 0.5, -0.25])
    total = np.zeros(2 * (n + p) + 1)
    base = np.array([1.0])
    for _ in range(n):
        base = np.convolve(base, one_minus_s)
    coef = 1.0
    term = base.copy()
    for j in range(p + 1):
        if j > 0:
            coef *= (n + j - 1) / j
            term = np.convolve(term, s)
        pad = (total.size - term.size) // 2
        total[pad:pad + term.size] += coef * term
    return total


@dataclass
class Binomial(Filter):
    """Separable compensated binomial filter: degree p, passes n (per axis)."""

    p: int
    n: Sequence[int] | int
    family = "binomial"

    def __post_init__(self):
        self.p = int(self.p)
        if self.p < 0:
            raise ValueError("compensation degree p must be >= 0")
        self.n = tuple(int(x) for x in np.atleast_1d(self.n))
        if any(x < 0 for x in self.n):
            raise ValueError("binomial passes must be >= 0")

    def _n_axes(self, dim):
        return self.n * dim if len(self.n) == 1 else self.n

    def transfer(self, k, grid):
        k = np.asarray(k, dtype=float)
        n_axes = self._n_axes(grid.dim)
        if len(n_axes) != grid.dim:
            raise ValueError("binomial passes per axis do not match the grid dimension")
        s = np.sin(0.5 * k * grid.dx) ** 2
        out = np.ones(k.shape[:-1]) if k.ndim > 1 else np.float64(1.0)
        for i, ni in enumerate(n_axes):
            out = out * binomial_axis_transfer(s[..., i], ni, self.p)
        return out

    @property
    def label(self):
        if len(self.n) == 1:
            return f"binomial(p={self.p},n={self.n[0]})"
        return f"binomial(p={self.p},n=" + "x".join(str(x) for x in self.n) + ")"

    def params(self):
        return dict(p=self.p, n=list(self.n))

    def describe(self, grid):
        d = super().describe(grid)
        d["stencil_per_axis"] = [binomial_stencil(ni, self.p).tolist() if ni else [1.0]
                                 for ni in self._n_axes(grid.dim)]
        return d


@dataclass
class Custom(Filter):
    """User-supplied transfer ``fun(k, grid) -> W`` (k has shape (..., dim))."""

    fun: Callable[[np.ndarray, Grid], np.ndarray]
    name: str = "custom"
    family = "custom"

    def transfer(self, k, grid):
        return np.asarray(self.fun(np.asarray(k, dtype=float), grid), dtype=float)

    def cache_key(self):
        return (self.family, self.name, id(self.fun))

    @property
    def label(self):
        return self.name


@dataclass
class Kernel:
    """Deposit/gather kernel: particle shape plus mesh filter."""

    order: int
    filter: Filter
    coupling_power: int = 2       # 2: matched (density and field); 1: density only
    extra: dict = field(default_factory=dict)

    @property
    def family(self) -> str:
        return self.filter.family

    @property
    def label(self) -> str:
        from .grid import shape_name
        base = self.filter.label
        return base if self.extra.get("hide_shape") else f"{shape_name(self.order)}+{base}"

    def coupling(self, grid: Grid, m) -> float:
        return self.filter.at_mode(grid, m) ** self.coupling_power
