"""
attractor_analysis/energy.py
============================

Energy-distribution estimators for the 1D1V problem.

Definitions (frozen; see protocol/DATA_CONTRACT.md):

* Active energy ``eps = m v_x^2 / 2`` in the initial centre-of-momentum frame.
* ``N(eps, t) = dN_phys/d eps``.  For particles in bin b,
  ``N_b = sum_{p in b} w_p / Delta eps_b``.  The physical mass enters ``eps``;
  the macro weight enters only the sum.
* ``dN/d ln eps = eps N(eps)`` is a different quantity.  Every plotting helper
  must state which one it shows and never overlay the two.

Each histogram also stores the exact per-bin weight and energy sums so that
number and energy moments are exact, and the finite-bin (midpoint) moment
error can be measured separately.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy import special


# --------------------------------------------------------------------------
# bin edges
# --------------------------------------------------------------------------

def common_edges(eps_lo: float, eps_hi: float, n_log: int, n_lin: int = 8) -> np.ndarray:
    """Frozen common edges: ``n_lin`` linear bins on [0, eps_lo] then
    ``n_log`` logarithmic bins on [eps_lo, eps_hi].

    The low-energy linear part keeps moment checks complete; log-tail analysis
    uses only bins with strictly positive lower edge.
    """
    if not (0.0 < eps_lo < eps_hi):
        raise ValueError("need 0 < eps_lo < eps_hi")
    lin = np.linspace(0.0, eps_lo, n_lin + 1)
    log = np.geomspace(eps_lo, eps_hi, n_log + 1)
    return np.concatenate([lin, log[1:]])


# --------------------------------------------------------------------------
# histogram
# --------------------------------------------------------------------------

@dataclass
class EnergyHistogram:
    edges: np.ndarray
    N: np.ndarray            # dN/d eps (density)
    weight: np.ndarray       # sum of w per bin
    energy: np.ndarray       # sum of w * eps per bin (exact)
    neff: np.ndarray         # Kish effective count per bin
    overflow_weight: float
    overflow_energy: float

    @property
    def widths(self) -> np.ndarray:
        return np.diff(self.edges)

    @property
    def centers(self) -> np.ndarray:
        return 0.5 * (self.edges[1:] + self.edges[:-1])

    @property
    def geometric_centers(self) -> np.ndarray:
        lo = np.where(self.edges[:-1] > 0, self.edges[:-1], 0.5 * self.edges[1:])
        return np.sqrt(lo * self.edges[1:])

    def dN_dlneps(self) -> np.ndarray:
        """``eps N(eps)`` evaluated at geometric centres (log-bin density)."""
        return self.geometric_centers * self.N


def particle_energy(v: np.ndarray, mass: float = 1.0) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    return 0.5 * mass * v * v


def energy_histogram(eps: np.ndarray, w: np.ndarray, edges: np.ndarray) -> EnergyHistogram:
    """Weighted energy histogram on frozen edges.

    Energies below ``edges[0]`` are an error (energies are non-negative and the
    first edge is 0 for complete moments).  Energies at or above ``edges[-1]``
    are overflow and are reported, never clipped into the last bin.
    """
    eps = np.asarray(eps, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    edges = np.asarray(edges, dtype=np.float64)
    if np.any(eps < edges[0]):
        raise ValueError("energy below the first edge; edges must start at or below min energy")
    idx = np.searchsorted(edges, eps, side="right") - 1
    nb = len(edges) - 1
    over = idx >= nb
    inside = ~over
    weight = np.bincount(idx[inside], weights=w[inside], minlength=nb)
    energy = np.bincount(idx[inside], weights=(w * eps)[inside], minlength=nb)
    w2 = np.bincount(idx[inside], weights=(w * w)[inside], minlength=nb)
    with np.errstate(invalid="ignore", divide="ignore"):
        neff = np.where(w2 > 0, weight**2 / w2, 0.0)
    N = weight / np.diff(edges)
    return EnergyHistogram(
        edges=edges,
        N=N,
        weight=weight,
        energy=energy,
        neff=neff,
        overflow_weight=float(np.sum(w[over])),
        overflow_energy=float(np.sum((w * eps)[over])),
    )


def histogram_moments(h: EnergyHistogram) -> dict:
    """Number and energy moments from the histogram.

    ``number`` and ``energy_exact`` use the stored per-bin sums plus overflow
    and therefore equal direct particle sums to round-off.  ``energy_midpoint``
    is the estimate a reader would form from ``N`` alone; its difference from
    ``energy_exact`` is the finite-bin moment error.
    """
    number = float(np.sum(h.weight) + h.overflow_weight)
    energy_exact = float(np.sum(h.energy) + h.overflow_energy)
    energy_mid = float(np.sum(h.N * h.centers * h.widths) + h.overflow_energy)
    return {
        "number": number,
        "energy_exact": energy_exact,
        "energy_midpoint": energy_mid,
        "finite_bin_energy_error": energy_mid - energy_exact,
    }


# --------------------------------------------------------------------------
# analytic references
# --------------------------------------------------------------------------

def maxwellian_1v_N(eps: np.ndarray, T: float, n: float = 1.0) -> np.ndarray:
    """1V Maxwellian energy density: ``N(eps) = n eps^{-1/2} e^{-eps/T} / sqrt(pi T)``."""
    eps = np.asarray(eps, dtype=np.float64)
    with np.errstate(divide="ignore"):
        return n * np.exp(-eps / T) / np.sqrt(np.pi * T * eps)


def maxwellian_1v_bin_content(edges: np.ndarray, T: float, n: float = 1.0):
    """Exact per-bin number and energy for a 1V Maxwellian (eps/T ~ Gamma(1/2))."""
    x = np.asarray(edges, dtype=np.float64) / T
    cdf_n = special.gammainc(0.5, x)
    cdf_e = special.gammainc(1.5, x)
    number = n * np.diff(cdf_n)
    energy = n * 0.5 * T * np.diff(cdf_e)  # <eps> = T/2 for 1V
    return number, energy


# --------------------------------------------------------------------------
# tail moments and moving features
# --------------------------------------------------------------------------

def tail_moments(h: EnergyHistogram, eps_threshold: float, kinetic_total: Optional[float] = None) -> dict:
    """Fixed-threshold tail moments ``Q_N``, ``Q_E`` and ``f_tail = Q_E / K``.

    The threshold must coincide with a bin edge (it is frozen in the protocol
    on the common edges); overflow counts as tail.
    """
    j = np.flatnonzero(np.isclose(h.edges, eps_threshold, rtol=1e-12, atol=0.0))
    if j.size != 1:
        raise ValueError("eps_threshold must be one of the frozen bin edges")
    j = int(j[0])
    QN = float(np.sum(h.weight[j:]) + h.overflow_weight)
    QE = float(np.sum(h.energy[j:]) + h.overflow_energy)
    K = float(np.sum(h.energy) + h.overflow_energy) if kinetic_total is None else float(kinetic_total)
    return {"Q_N": QN, "Q_E": QE, "f_tail": QE / K if K > 0 else np.nan,
            "neff_tail": float(np.sum(h.neff[j:]))}


def quantile_feature(h: EnergyHistogram, q: float, min_counts: float = 100.0) -> float:
    """Energy ``eps_q`` at which the complementary cumulative number falls to
    the fraction ``q`` of the total.

    Interpolated linearly in (ln eps, ln C) between bin edges.  Returns NaN if
    fewer than ``min_counts`` effective particles lie above the estimate (the
    feature is then not supported by counts), or if it falls in overflow.
    """
    total = float(np.sum(h.weight) + h.overflow_weight)
    # complementary cumulative at each edge (inclusive of overflow)
    C = np.concatenate([np.cumsum(h.weight[::-1])[::-1], [0.0]]) + h.overflow_weight
    target = q * total
    if target >= C[0] or target <= C[-1]:
        return np.nan
    j = int(np.searchsorted(-C, -target, side="left"))  # first edge with C <= target
    e0, e1 = h.edges[j - 1], h.edges[j]
    c0, c1 = C[j - 1], C[j]
    if e0 > 0 and c1 > 0:
        frac = (np.log(target) - np.log(c0)) / (np.log(c1) - np.log(c0))
        eq = float(np.exp(np.log(e0) + frac * (np.log(e1) - np.log(e0))))
    else:
        frac = (target - c0) / (c1 - c0)
        eq = float(e0 + frac * (e1 - e0))
    neff_above = float(np.sum(h.neff[j:]))
    if neff_above < min_counts:
        return np.nan
    return eq


def local_log_slope(x: np.ndarray, y: np.ndarray, smooth: int = 1) -> np.ndarray:
    """Centered d ln y / d ln x with an optional running mean of half-width ``smooth``.

    Non-positive y gives NaN.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        ly = np.where(y > 0, np.log(y), np.nan)
    lx = np.log(x)
    if smooth > 0:
        k = 2 * smooth + 1
        pad = np.pad(ly, smooth, mode="edge")
        ly = np.array([np.nanmean(pad[i:i + k]) if np.any(np.isfinite(pad[i:i + k])) else np.nan
                       for i in range(len(ly))])
    return np.gradient(ly, lx)


def powerlaw_segment(x: np.ndarray, y: np.ndarray, support: np.ndarray, tol: float = 0.25,
                     min_bins: int = 4, smooth: int = 1) -> Optional[dict]:
    """Longest contiguous run of supported bins with near-constant log slope.

    The frozen curvature criterion is ``|s_b - median(run)| < tol``.  Returns
    the run's slope and its lower/upper edges (bin centres), or None.
    """
    s = local_log_slope(x, y, smooth=smooth)
    ok = support & np.isfinite(s)
    best = None
    i = 0
    n = len(x)
    while i < n:
        if not ok[i]:
            i += 1
            continue
        j = i
        # grow while all slopes stay within tol of the running median
        while j + 1 < n and ok[j + 1] and abs(s[j + 1] - np.median(s[i:j + 2])) < tol \
                and np.all(np.abs(s[i:j + 2] - np.median(s[i:j + 2])) < tol):
            j += 1
        if j - i + 1 >= min_bins and (best is None or (j - i) > (best[1] - best[0])):
            best = (i, j)
        i = j + 1
    if best is None:
        return None
    i, j = best
    return {"slope": float(np.median(s[i:j + 1])), "x_lo": float(x[i]), "x_hi": float(x[j]),
            "i_lo": i, "i_hi": j}


def upper_cutoff_feature(h: EnergyHistogram, min_counts: float = 25.0, tol: float = 0.25) -> float:
    """Upper end of the power-law segment of N(eps) on the log bins (NaN if none)."""
    pos = h.edges[:-1] > 0
    x = h.geometric_centers[pos]
    y = h.N[pos]
    seg = powerlaw_segment(x, y, h.neff[pos] >= min_counts, tol=tol)
    return np.nan if seg is None else seg["x_hi"]


def maxwellian_tail_excess(h: EnergyHistogram, eps_threshold: float) -> dict:
    """Tail number relative to a 1V Maxwellian with the same number and kinetic energy.

    ``excess = Q_N / Q_N^Maxwell``.  For a thermal plasma this is ~1; a
    nonthermal tail gives >> 1.  Uses the exact Gamma(1/2) tail.
    """
    mom = histogram_moments(h)
    n = mom["number"]
    T = 2.0 * mom["energy_exact"] / n
    tm = tail_moments(h, eps_threshold)
    QN_max = n * float(special.gammaincc(0.5, eps_threshold / T))
    return {"excess": tm["Q_N"] / QN_max if QN_max > 0 else np.inf, "T_equiv": T,
            "Q_N": tm["Q_N"], "Q_N_maxwell": QN_max, "neff_tail": tm["neff_tail"]}


def median_energy(h: EnergyHistogram) -> float:
    """Weighted median energy (linear interpolation within the bin)."""
    total = float(np.sum(h.weight) + h.overflow_weight)
    c = np.concatenate([[0.0], np.cumsum(h.weight)])
    j = int(np.searchsorted(c, 0.5 * total))
    j = min(max(j, 1), len(h.edges) - 1)
    frac = (0.5 * total - c[j - 1]) / max(h.weight[j - 1], 1e-300)
    return float(h.edges[j - 1] + frac * (h.edges[j] - h.edges[j - 1]))


# median of eps/T for a 1V Maxwellian (Gamma(1/2) median)
GAMMA_HALF_MEDIAN = float(special.gammaincinv(0.5, 0.5))


def core_tail_excess(h: EnergyHistogram, kappa: float = 10.0) -> dict:
    """Suprathermal excess relative to a Maxwellian fitted to the *core*.

    ``T_core`` comes from the median energy (insensitive to a tail carrying
    much of the energy, unlike an energy-matched Maxwellian).  The excess is
    the measured number above ``kappa T_core`` divided by the 1V Maxwellian
    expectation ``n Gamma(1/2, kappa)``.
    """
    n = float(np.sum(h.weight) + h.overflow_weight)
    T = median_energy(h) / GAMMA_HALF_MEDIAN
    e_th = kappa * T
    above = h.edges[:-1] >= e_th
    # partial bin containing e_th, assuming constant N within the bin
    j = int(np.searchsorted(h.edges, e_th, side="right") - 1)
    part = 0.0
    neff_part = 0.0
    if 0 <= j < len(h.N):
        frac = (h.edges[j + 1] - e_th) / (h.edges[j + 1] - h.edges[j])
        part = h.weight[j] * frac
        neff_part = h.neff[j] * frac
    QN = float(np.sum(h.weight[above]) + part + h.overflow_weight)
    neff = float(np.sum(h.neff[above]) + neff_part)
    QN_max = n * float(special.gammaincc(0.5, kappa))
    return {"excess": QN / QN_max if QN_max > 0 else np.inf, "T_core": T, "eps_threshold": e_th,
            "Q_N": QN, "Q_N_maxwell": QN_max, "neff_tail": neff}
