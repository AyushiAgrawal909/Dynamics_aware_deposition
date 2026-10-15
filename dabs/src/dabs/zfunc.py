"""Plasma dispersion function and the Maxwellian response function.

``Z(zeta) = i sqrt(pi) w(zeta)`` with the Faddeeva function ``w``.  Because
``w`` is entire, this is the Landau-continued Z function in the whole complex
plane, so damped roots (Im zeta < 0) need no special treatment.

The quantity that enters the electrostatic susceptibility of a Maxwellian is
``R(zeta) = 1 + zeta Z(zeta)``.  For large |zeta| it is evaluated from its
asymptotic series to avoid the cancellation in ``1 + zeta Z``.
"""

from __future__ import annotations

import numpy as np
from scipy.special import wofz

SQRT_PI = float(np.sqrt(np.pi))
_LARGE = 30.0
# (2n-1)!! / 2^n for n = 1..10
_ASYM = np.array([0.5, 0.75, 1.875, 6.5625, 29.53125, 162.421875, 1055.7421875,
                  7918.06640625, 67303.564453125, 639383.8623046875])


def Z(zeta):
    """Plasma dispersion function (Fried and Conte), Landau-continued."""
    return 1j * SQRT_PI * wofz(np.asarray(zeta, dtype=np.complex128))


def Zprime(zeta):
    """Derivative Z'(zeta) = -2 [1 + zeta Z(zeta)]."""
    return -2.0 * maxwell_response(zeta)


def maxwell_response(zeta):
    """Return 1 + zeta Z(zeta) with a stable large-argument branch."""
    out = _response(np.asarray(zeta, dtype=np.complex128))
    return out if out.shape else complex(out)


def _response(z: np.ndarray) -> np.ndarray:
    out = np.empty_like(z)
    big = np.abs(z) > _LARGE
    small = ~big
    if np.any(small):
        zs = z[small]
        out[small] = 1.0 + zs * Z(zs)
    if np.any(big):
        zb = z[big]
        inv_z2 = 1.0 / (zb * zb)
        series = np.zeros_like(zb)
        power = np.ones_like(zb)
        for c in _ASYM:
            power = power * inv_z2
            series = series + c * power
        # Landau pole contribution below the real axis (negligible above it).
        landau = np.where(zb.imag < 0.0, 2.0j * SQRT_PI * zb * np.exp(-zb * zb), 0.0)
        out[big] = -series + landau
    return out


def maxwell_response_derivative(zeta):
    """d/dzeta [1 + zeta Z(zeta)] = Z + zeta Z' = Z - 2 zeta (1 + zeta Z)."""
    z = np.asarray(zeta, dtype=np.complex128)
    r = _response(z)
    out = np.empty_like(z)
    big = np.abs(z) > _LARGE
    small = ~big
    if np.any(small):
        zs = z[small]
        out[small] = Z(zs) - 2.0 * zs * r[small]
    if np.any(big):
        zb = z[big]
        series = np.zeros_like(zb)
        power = 1.0 / zb
        for n, c in enumerate(_ASYM, start=1):
            power = power / (zb * zb)
            series = series + c * (2 * n) * power
        landau = np.where(
            zb.imag < 0.0,
            2.0j * SQRT_PI * (1.0 - 2.0 * zb * zb) * np.exp(-zb * zb),
            0.0,
        )
        out[big] = series + landau
    return out if out.shape else complex(out)
