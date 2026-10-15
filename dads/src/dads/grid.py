"""Periodic mesh, particle shapes and the discrete field operators.

The mesh is periodic in 1, 2 or 3 dimensions.  A grid mode is identified by
integer indices ``m`` with wavevector ``k_i = 2 pi m_i / L_i``.  A particle
shape of order ``p`` (0 = NGP, 1 = CIC, 2 = TSC, 3 = PQS) has the transform

    S(q) = prod_i sinc(q_i dx_i / 2) ** (p + 1),     sinc z = sin z / z.

Deposition and gathering use the same shape (momentum-conserving scheme).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from typing import Sequence

import numpy as np

SHAPE_ORDERS = {"ngp": 0, "cic": 1, "tsc": 2, "pqs": 3}

# sum_p S(k + 2 pi p / dx)^2 per axis as a polynomial in s = sin^2(k dx / 2)
# (Hockney and Eastwood); coefficients in increasing powers of s.
_ALIAS_POWER = {
    0: (1.0,),
    1: (1.0, -2.0 / 3.0),
    2: (1.0, -1.0, 2.0 / 15.0),
    3: (1.0, -4.0 / 3.0, 2.0 / 5.0, -4.0 / 315.0),
}


def shape_order(shape: str | int) -> int:
    if isinstance(shape, (int, np.integer)):
        if int(shape) not in _ALIAS_POWER:
            raise ValueError(f"shape order must be 0..3, got {shape}")
        return int(shape)
    key = str(shape).strip().lower()
    if key not in SHAPE_ORDERS:
        raise ValueError(f"unknown particle shape '{shape}'; use one of {sorted(SHAPE_ORDERS)}")
    return SHAPE_ORDERS[key]


def shape_name(order: int) -> str:
    return {v: k for k, v in SHAPE_ORDERS.items()}[order]


@dataclass
class Grid:
    """Periodic mesh with its charge-to-field operators.

    Parameters
    ----------
    N, L
        Number of cells and domain length per axis.
    shape
        Deposit/gather particle shape: ``ngp``, ``cic``, ``tsc`` or ``pqs``.
    poisson
        ``spectral`` (K^2 = |k|^2) or ``fd2`` (second-order finite-difference
        Laplacian, K^2 = sum (2 sin(k dx/2)/dx)^2).
    gradient
        ``spectral`` (kappa = k) or ``centered`` (kappa_i = sin(k_i dx_i)/dx_i).
    scheme
        ``momentum``: the gathered field is the interpolated mesh field.
        ``energy``: the force is minus the gradient of the interpolated
        potential (energy-conserving scheme).
    fft
        Transform layout of a spectral field solve, which only decides how
        Nyquist modes enter the field-error metric: ``rfft`` (real-to-complex
        transforms, e.g. ``numpy.fft.rfftn``) or ``complex``.
    """

    N: Sequence[int]
    L: Sequence[float]
    shape: str = "cic"
    poisson: str = "spectral"
    gradient: str = "spectral"
    scheme: str = "momentum"
    fft: str = "rfft"
    dx: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.N = tuple(int(n) for n in np.atleast_1d(self.N))
        self.L = tuple(float(x) for x in np.atleast_1d(self.L))
        if len(self.N) != len(self.L) or not 1 <= len(self.N) <= 3:
            raise ValueError("N and L must have the same length, 1 to 3")
        if any(n < 2 for n in self.N) or any(x <= 0 for x in self.L):
            raise ValueError("N must be >= 2 and L > 0 on every axis")
        self.dx = np.array(self.L) / np.array(self.N)
        self.order = shape_order(self.shape)
        self.shape = shape_name(self.order)
        self.poisson = self.poisson.lower()
        self.gradient = self.gradient.lower()
        self.scheme = self.scheme.lower()
        if self.poisson not in ("spectral", "fd2"):
            raise ValueError("poisson must be 'spectral' or 'fd2'")
        if self.gradient not in ("spectral", "centered"):
            raise ValueError("gradient must be 'spectral' or 'centered'")
        if self.scheme not in ("momentum", "energy"):
            raise ValueError("scheme must be 'momentum' or 'energy'")
        self.fft = self.fft.lower()
        if self.fft not in ("rfft", "complex"):
            raise ValueError("fft must be 'rfft' or 'complex'")

    # ------------------------------------------------------------------ modes
    @property
    def dim(self) -> int:
        return len(self.N)

    @property
    def n_cells(self) -> int:
        return int(np.prod(self.N))

    def mode_vector(self, m: Sequence[int]) -> np.ndarray:
        m = np.atleast_1d(np.asarray(m, dtype=int))
        if m.size != self.dim:
            raise ValueError(f"mode {tuple(m)} has {m.size} indices for a {self.dim}D grid")
        for mi, ni in zip(m, self.N):
            if abs(mi) > ni // 2:
                raise ValueError(f"mode index {mi} exceeds the Nyquist index {ni // 2}")
        if not np.any(m):
            raise ValueError("the k = 0 mode carries no field")
        return m

    def k_of_mode(self, m: Sequence[int]) -> np.ndarray:
        m = self.mode_vector(m)
        return 2.0 * np.pi * m / np.array(self.L)

    def fft_wavenumbers(self) -> list[np.ndarray]:
        """Full-FFT wavenumber arrays (``numpy.fft.fftn`` layout), one per axis."""
        axes = [2.0 * np.pi * np.fft.fftfreq(n, d=d) for n, d in zip(self.N, self.dx)]
        return list(np.meshgrid(*axes, indexing="ij"))

    # ----------------------------------------------------------- transforms
    def shape_transform(self, q: np.ndarray, order: int | None = None) -> np.ndarray:
        """S(q) for wavevectors ``q`` of shape (..., dim)."""
        order = self.order if order is None else shape_order(order)
        q = np.asarray(q, dtype=float)
        arg = q * self.dx / (2.0 * np.pi)          # np.sinc(x) = sin(pi x)/(pi x)
        return np.prod(np.sinc(arg) ** (order + 1), axis=-1)

    def alias_power(self, k: np.ndarray, order: int | None = None) -> np.ndarray:
        """sum over aliases of S^2 (closed form); k has shape (..., dim)."""
        order = self.order if order is None else shape_order(order)
        k = np.asarray(k, dtype=float)
        s = np.sin(0.5 * k * self.dx) ** 2
        coeffs = _ALIAS_POWER[order]
        per_axis = sum(c * s**j for j, c in enumerate(coeffs))
        return np.prod(per_axis, axis=-1)

    def laplacian_symbol(self, k: np.ndarray) -> np.ndarray:
        k = np.asarray(k, dtype=float)
        if self.poisson == "spectral":
            return np.sum(k * k, axis=-1)
        return np.sum((2.0 * np.sin(0.5 * k * self.dx) / self.dx) ** 2, axis=-1)

    def gradient_symbol(self, k: np.ndarray) -> np.ndarray:
        k = np.asarray(k, dtype=float)
        if self.gradient == "spectral":
            return k.copy()
        return np.sin(k * self.dx) / self.dx

    def field_weight(self, k: np.ndarray) -> np.ndarray:
        """|E(k)|^2 / |rho(k)|^2 for the solve (eps0 = 1); zero at k = 0.

        A spectral derivative at an axis's Nyquist wavenumber is lost when the
        transform projects it onto a real field: always for a complex layout,
        and for a real-to-complex layout only where the last-axis index is 0
        or Nyquist (elsewhere the half-spectrum storage keeps its power).
        """
        k = np.asarray(k, dtype=float)
        K2 = self.laplacian_symbol(k)
        kap = self.gradient_symbol(k)
        if self.gradient == "spectral":
            nyq = np.isclose(np.abs(k) * self.dx, np.pi, rtol=0, atol=1e-9)
            if self.fft == "rfft":
                kl = np.abs(k[..., -1]) * self.dx[-1]
                slab = np.isclose(kl, 0.0, atol=1e-12) | np.isclose(kl, np.pi, rtol=0, atol=1e-9)
                nyq = nyq & slab[..., None]
            kap = np.where(nyq, 0.0, kap)
        kap2 = np.sum(kap**2, axis=-1)
        with np.errstate(divide="ignore", invalid="ignore"):
            w = np.where(K2 > 0, kap2 / K2**2, 0.0)
        return w

    def aliases(self, k: np.ndarray, n_alias: int, order: int | None = None,
                drop: float = 1e-16) -> tuple[np.ndarray, np.ndarray]:
        """Alias wavevectors q = k + 2 pi p / dx and the coupling geometry.

        Returns ``(q, geom)`` where ``geom = S(q)^2 (q . kappa(k)) / K^2(k)``
        (momentum scheme) or ``S(q)^2 |q|^2 / K^2(k)`` (energy scheme), so that
        the dielectric function is ``1 + a * sum_q geom(q) chi(q, omega)``.
        """
        k = np.asarray(k, dtype=float)
        rng = range(-n_alias, n_alias + 1)
        p = np.array(list(product(rng, repeat=self.dim)), dtype=float)
        q = k[None, :] + 2.0 * np.pi * p / self.dx[None, :]
        S2 = self.shape_transform(q, order) ** 2
        K2 = self.laplacian_symbol(k)
        if self.scheme == "momentum":
            geom = S2 * (q @ self.gradient_symbol(k)) / K2
        else:
            geom = S2 * np.sum(q * q, axis=-1) / K2
        principal = np.all(p == 0, axis=1)
        keep = principal | (np.abs(geom) > drop * np.abs(geom[principal]).max())
        order_idx = np.argsort(~principal[keep], kind="stable")
        return q[keep][order_idx], geom[keep][order_idx]

    def describe(self) -> dict:
        return dict(N=list(self.N), L=list(self.L), dx=self.dx.tolist(), shape=self.shape,
                    poisson=self.poisson, gradient=self.gradient, scheme=self.scheme, fft=self.fft)
