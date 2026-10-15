"""Species and equilibria: drifting (bi-)Maxwellians, unmagnetised.

Each species is a drifting Maxwellian with independent thermal speeds along
the coordinate axes, ``v_th,i = sqrt(T_i / m)``.  For an electrostatic wave of
wavevector q only the projection of the distribution on q-hat matters; it is
a Maxwellian with mean ``q-hat . u`` and variance ``sum_i qhat_i^2 v_th,i^2``.
A neutralising immobile background is implied, so overall charge neutrality
of the listed species is not required.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np


@dataclass
class Species:
    """One drifting (bi-)Maxwellian species.

    Parameters
    ----------
    q, m, n
        Charge, mass and number density.
    drift
        Drift velocity (scalar for 1D or one value per axis).
    vth
        Thermal speed sqrt(T/m): scalar (isotropic) or one value per axis.
        Zero gives a cold species.
    ppc
        Optional markers per cell for this species (noise model only).
    mobile
        Immobile species do not respond to the field.
    """

    q: float
    m: float
    n: float
    drift: Sequence[float] | float = 0.0
    vth: Sequence[float] | float = 0.0
    name: str = ""
    ppc: float | None = None
    mobile: bool = True
    _u: np.ndarray = field(init=False, repr=False, default=None)
    _v: np.ndarray = field(init=False, repr=False, default=None)

    def resolve(self, dim: int) -> "Species":
        u = np.atleast_1d(np.asarray(self.drift, dtype=float))
        v = np.atleast_1d(np.asarray(self.vth, dtype=float))
        if u.size == 1 and dim > 1:
            u = np.concatenate([u, np.zeros(dim - 1)]) if np.any(u) else np.zeros(dim)
        if v.size == 1:
            v = np.full(dim, float(v[0]))
        if u.size != dim or v.size != dim:
            raise ValueError(f"species '{self.name}': drift and vth must have {dim} components")
        if np.any(v < 0):
            raise ValueError(f"species '{self.name}': vth must be non-negative")
        if self.m <= 0 or self.n < 0:
            raise ValueError(f"species '{self.name}': need m > 0 and n >= 0")
        self._u, self._v = u, v
        return self

    @property
    def u(self) -> np.ndarray:
        return self._u

    @property
    def v(self) -> np.ndarray:
        return self._v

    def omega_p2(self, eps0: float = 1.0) -> float:
        return self.n * self.q**2 / (eps0 * self.m)

    def describe(self) -> dict:
        return dict(name=self.name, q=self.q, m=self.m, n=self.n, drift=self.u.tolist(),
                    vth=self.v.tolist(), ppc=self.ppc, mobile=self.mobile)


@dataclass
class Plasma:
    species: list[Species]
    eps0: float = 1.0

    def resolve(self, dim: int) -> "Plasma":
        if not self.species:
            raise ValueError("at least one species is required")
        for i, s in enumerate(self.species):
            if not s.name:
                s.name = f"species{i}"
            s.resolve(dim)
        if not any(s.mobile and s.n > 0 for s in self.species):
            raise ValueError("no mobile species")
        return self

    @property
    def mobile(self) -> list[Species]:
        return [s for s in self.species if s.mobile and s.n > 0]

    @property
    def omega_p_total(self) -> float:
        return float(np.sqrt(sum(s.omega_p2(self.eps0) for s in self.mobile)))

    def charge_scale(self) -> float:
        """sum_s |q_s| n_s over mobile species (unit of noise/signal amplitudes)."""
        return float(sum(abs(s.q) * s.n for s in self.mobile))

    def debye_length(self) -> float:
        """Total Debye length from the mean thermal speed of each species."""
        inv = 0.0
        for s in self.mobile:
            vt2 = float(np.mean(s.v**2))
            if vt2 > 0:
                inv += s.omega_p2(self.eps0) / vt2
        return float(1.0 / np.sqrt(inv)) if inv > 0 else float("inf")

    def describe(self) -> dict:
        return dict(eps0=self.eps0, omega_p_total=self.omega_p_total,
                    species=[s.describe() for s in self.species])
