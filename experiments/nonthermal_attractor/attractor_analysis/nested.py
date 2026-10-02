"""
attractor_analysis/nested.py
============================

Nested quiet loads for the regulator scan.

The load at nested level l uses the first ``N / 2**l`` markers of the level-0
load, with weight multiplied by ``2**l``.  Differences between levels on the
same seed are then correlated, which is what makes paired-difference
convergence tests sharp.

Each prefix must itself be a quiet (stratified) load.  A naive choice --
stratified markers in natural order, subset by every 2**l-th marker -- breaks
this when velocity quantiles come from a base-2 van der Corput sequence
(vdc(2**l m) = vdc(m) / 2**l collapses into one corner).  We therefore use the
2D Halton sequence with bases (2, 3): x from base 2 and the velocity quantile
from base 3.  Any prefix of length 2**k has x exactly on the lattice
{(m + c) / 2**k}, and a low-discrepancy velocity quantile set.  Independent
seeds use a Cranley-Patterson rotation (a random shift mod 1 per coordinate),
which preserves both properties.
"""

from __future__ import annotations

from typing import Callable, Tuple

import numpy as np


def radical_inverse(n: np.ndarray, base: int) -> np.ndarray:
    """Van der Corput radical inverse of integer array ``n`` in ``base``."""
    n = np.asarray(n, dtype=np.int64).copy()
    out = np.zeros(n.shape, dtype=np.float64)
    f = 1.0 / base
    while np.any(n > 0):
        out += f * (n % base)
        n //= base
        f /= base
    return out


def halton_2d(N: int, seed: int | None = None, offset: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """First N points of the (2, 3) Halton sequence in [0, 1)^2.

    ``offset`` = 0 keeps every power-of-two prefix an exact lattice in x
    (a block 0..2**k-1 of the base-2 sequence).  ``seed`` applies a
    Cranley-Patterson rotation; ``seed=None`` gives the unrotated sequence.
    """
    idx = np.arange(offset, offset + N, dtype=np.int64)
    u = radical_inverse(idx, 2)
    q = radical_inverse(idx, 3)
    if seed is not None:
        rng = np.random.default_rng(seed)
        su, sq = rng.random(2)
        u = (u + su) % 1.0
        q = (q + sq) % 1.0
    return u, q


def nested_load_indices(N: int, levels: int) -> list[slice]:
    """Index slices of the nested subsets: level l uses ``slice(0, N >> l)``.

    N must be divisible by ``2**(levels-1)`` so every level is a whole subset.
    """
    if N % (1 << (levels - 1)):
        raise ValueError("N must be divisible by 2**(levels-1)")
    return [slice(0, N >> l) for l in range(levels)]


def quiet_load_1d1v(N: int, L: float, inv_cdf: Callable[[np.ndarray], np.ndarray],
                    seed: int | None = None, n0: float = 1.0):
    """Quiet 1D1V load whose every power-of-two prefix is itself quiet.

    Returns ``x, v, w`` with uniform weights ``w = n0 L / N`` (total physical
    number ``n0 L``).  Velocity quantiles are mapped through ``inv_cdf``.
    """
    u, q = halton_2d(N, seed=seed)
    # keep quantiles strictly inside (0, 1) for unbounded inverse CDFs
    q = np.clip(q, 0.5 / (4 * N), 1.0 - 0.5 / (4 * N))
    x = u * L
    v = inv_cdf(q)
    w = np.full(N, n0 * L / N)
    return x, v, w


def subset(x, v, w, level: int):
    """Level-``level`` nested subset of a level-0 load (weights scaled by 2**level)."""
    m = len(x) >> level
    return x[:m].copy(), v[:m].copy(), w[:m] * (1 << level)


def ks_uniform(u: np.ndarray) -> float:
    """Kolmogorov-Smirnov distance of samples in [0,1) from the uniform law."""
    s = np.sort(np.asarray(u))
    n = len(s)
    i = np.arange(1, n + 1)
    return float(max(np.max(i / n - s), np.max(s - (i - 1) / n)))


# --------------------------------------------------------------------------
# inverse CDFs for the T2/T3 preparations
# --------------------------------------------------------------------------

def waterbag_beams_inv_cdf(v_b: float, dv_b: float) -> Callable[[np.ndarray], np.ndarray]:
    """Two equal waterbags of full width ``dv_b`` centred at -v_b and +v_b.

    The beams must not overlap (dv_b < 2 v_b).  Quantile q in [0, 1/2) maps to
    the left beam, [1/2, 1) to the right.
    """
    if dv_b >= 2 * v_b:
        raise ValueError("waterbags overlap")

    def inv(q):
        q = np.asarray(q, dtype=np.float64)
        left = q < 0.5
        out = np.empty_like(q)
        out[left] = -v_b - 0.5 * dv_b + dv_b * (2 * q[left])
        out[~left] = v_b - 0.5 * dv_b + dv_b * (2 * q[~left] - 1)
        return out

    return inv


def gaussian_beams_inv_cdf(v_b: float, width: float) -> Callable[[np.ndarray], np.ndarray]:
    """Equal-weight Gaussian beams at +-v_b with standard deviation ``width``."""
    from scipy import special

    def inv(q):
        q = np.asarray(q, dtype=np.float64)
        left = q < 0.5
        qq = np.where(left, 2 * q, 2 * q - 1)
        qq = np.clip(qq, 1e-15, 1 - 1e-15)
        z = np.sqrt(2.0) * special.erfinv(2 * qq - 1)
        return np.where(left, -v_b, v_b) + width * z

    return inv


def maxwellian_inv_cdf(v_th: float) -> Callable[[np.ndarray], np.ndarray]:
    from scipy import special

    def inv(q):
        q = np.clip(np.asarray(q, dtype=np.float64), 1e-15, 1 - 1e-15)
        return v_th * np.sqrt(2.0) * special.erfinv(2 * q - 1)

    return inv
