"""
attractor_analysis/waterbag.py
==============================

Waterbag content rho(eta, t): the phase-space volume per unit level eta
occupied by the (reconstructed) distribution, the primary 1D1V observable.

In the Lynden-Bell framework used by Ewart et al. (arXiv:2409.01742), the
energy distribution follows from rho(eta) and the total energy; the time
evolution of rho(eta) is the piece they name as missing.

Important caveat built into the API: a *reconstructed* (binned) rho is a
coarse-grained quantity.  Fine-grained Vlasov Casimirs are conserved;
coarse-grained estimates need not be, and binning-induced change is not
physical invariant breaking.  ``bin_insensitivity`` separates the two, and
``eta_floor`` reports the one-particle-per-cell level below which a PIC
reconstruction cannot resolve the distribution.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from .energy import local_log_slope


@dataclass
class WaterbagContent:
    eta_edges: np.ndarray
    rho: np.ndarray          # phase volume per unit eta
    volume: np.ndarray       # phase volume per eta bin
    mass: np.ndarray         # sum of f * cell volume per eta bin
    dx_b: float
    dv_b: float
    eta_floor: float         # smallest resolvable level (one particle per cell); 0 for grid f

    @property
    def eta_centers(self) -> np.ndarray:
        return np.sqrt(self.eta_edges[1:] * self.eta_edges[:-1])


def reconstruct_f_particles(x, v, w, L: float, v_lim: float, nx_b: int, nv_b: int) -> np.ndarray:
    """Nearest-grid-point phase-space density f_hat on a uniform (x, v) grid.

    ``f_hat = sum w / (dx_b dv_b)``.  Particles outside [-v_lim, v_lim) are
    dropped from f_hat (and must be accounted for by the caller's budget).
    """
    x = np.asarray(x, dtype=np.float64) % L
    v = np.asarray(v, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    dx_b = L / nx_b
    dv_b = 2.0 * v_lim / nv_b
    i = np.floor(x / dx_b).astype(np.int64)
    j = np.floor((v + v_lim) / dv_b).astype(np.int64)
    ok = (j >= 0) & (j < nv_b)
    i = np.minimum(i, nx_b - 1)
    H = np.bincount(i[ok] * nv_b + j[ok], weights=w[ok], minlength=nx_b * nv_b)
    return H.reshape(nx_b, nv_b) / (dx_b * dv_b)


def coarse_grain_grid(f: np.ndarray, bx: int, bv: int) -> np.ndarray:
    """Block-average a grid distribution f[nx, nv] by (bx, bv)."""
    nx, nv = f.shape
    if nx % bx or nv % bv:
        raise ValueError("block size must divide the grid")
    return f.reshape(nx // bx, bx, nv // bv, bv).mean(axis=(1, 3))


def waterbag_content(f_hat: np.ndarray, dx_b: float, dv_b: float, eta_edges: np.ndarray,
                     eta_floor: float = 0.0) -> WaterbagContent:
    """rho(eta) from a reconstructed distribution on frozen eta edges.

    Cells with f_hat == 0 carry no level and are excluded; cells outside the
    eta range are excluded from both volume and mass (callers check closure
    against their own totals with ``closure``).
    """
    f = np.asarray(f_hat, dtype=np.float64).ravel()
    cell = dx_b * dv_b
    edges = np.asarray(eta_edges, dtype=np.float64)
    nb = len(edges) - 1
    idx = np.searchsorted(edges, f, side="right") - 1
    ok = (f > 0) & (idx >= 0) & (idx < nb)
    volume = np.bincount(idx[ok], minlength=nb).astype(np.float64) * cell
    mass = np.bincount(idx[ok], weights=f[ok] * cell, minlength=nb)
    rho = volume / np.diff(edges)
    return WaterbagContent(edges, rho, volume, mass, dx_b, dv_b, eta_floor)


def closure(wc: WaterbagContent, f_hat: np.ndarray) -> dict:
    """Check int rho d eta = occupied phase volume and int eta rho d eta = N
    for the cells inside the eta range."""
    f = np.asarray(f_hat).ravel()
    cell = wc.dx_b * wc.dv_b
    inside = (f >= wc.eta_edges[0]) & (f < wc.eta_edges[-1]) & (f > 0)
    return {
        "volume_hist": float(np.sum(wc.rho * np.diff(wc.eta_edges))),
        "volume_direct": float(np.count_nonzero(inside) * cell),
        "mass_hist": float(np.sum(wc.mass)),
        "mass_direct": float(np.sum(f[inside]) * cell),
        "mass_outside_eta_range": float(np.sum(f[~inside & (f > 0)]) * cell),
    }


def particle_eta_floor(w_min: float, dx_b: float, dv_b: float) -> float:
    """One macroparticle per reconstruction cell."""
    return float(w_min / (dx_b * dv_b))


def eta_minus1_range(wc: WaterbagContent, tol: float = 0.35, min_bins: int = 3,
                     floor_factor: float = 10.0) -> dict:
    """Detect the eta^{-1} asymptotic range of rho(eta) and its lower edge.

    Uses the compensated content ``g(eta) = eta rho(eta)``, constant on an
    eta^{-1} range.  Frozen criterion:

    * plateau: the longest run of bins with ``|ln(g / median)| < tol``
      (median over the run), at least ``min_bins`` long;
    * lower edge ``eta_min``: scanning down from the plateau, the half-height
      point where g falls to half the plateau median, interpolated in ln eta.
      Half-height is unbiased for an edge smeared symmetrically in ln eta by
      counting noise or by a partially filled edge bin.

    ``resolution_limited`` is set when ``eta_min`` lies within
    ``floor_factor`` of the one-particle floor: there a cell holds fewer than
    ~``floor_factor`` particles, the reconstructed levels are discrete
    multiples of the floor, and a feature tracking the floor instead of time
    must not be read as physics.
    """
    eta = wc.eta_centers
    g = eta * wc.rho
    lg = np.where(g > 0, np.log(np.where(g > 0, g, 1.0)), np.nan)
    best = None
    n = len(eta)
    for i in range(n):
        if not np.isfinite(lg[i]):
            continue
        for j in range(i + min_bins - 1, n):
            seg = lg[i:j + 1]
            if not np.all(np.isfinite(seg)):
                break
            if np.all(np.abs(seg - np.median(seg)) < tol):
                if best is None or (j - i) > (best[1] - best[0]):
                    best = (i, j)
            else:
                break
    empty = {"eta_min": np.nan, "eta_max": np.nan, "plateau": np.nan, "resolution_limited": False,
             "eta_floor": wc.eta_floor, "found": False}
    if best is None:
        return empty
    i, j = best
    gm = float(np.exp(np.median(lg[i:j + 1])))
    k = i
    while k - 1 >= 0 and np.isfinite(lg[k - 1]) and g[k - 1] >= 0.5 * gm:
        k -= 1
    if k == 0:
        eta_min = float(wc.eta_edges[0])  # never drops within the eta range
    else:
        g_lo = g[k - 1] if np.isfinite(lg[k - 1]) else 0.0
        # interpolate in ln eta between centres k-1 and k at g = gm / 2
        frac = (0.5 * gm - g_lo) / max(g[k] - g_lo, 1e-300)
        eta_min = float(np.exp(np.log(eta[k - 1]) + frac * (np.log(eta[k]) - np.log(eta[k - 1]))))
    limited = bool(wc.eta_floor > 0 and eta_min <= floor_factor * wc.eta_floor)
    return {"eta_min": eta_min, "eta_max": float(wc.eta_edges[j + 1]), "plateau": gm,
            "resolution_limited": limited, "eta_floor": wc.eta_floor, "found": True}


def bin_insensitivity(x, v, w, L: float, v_lim: float, nx_b: int, nv_b: int,
                      eta_edges: np.ndarray, factors: Sequence[float] = (0.5, 1.0, 2.0),
                      rel_tol: float = 0.25, **detect_kw) -> dict:
    """Recompute eta_min with reconstruction cells scaled by ``factors`` (in
    both x and v).  A feature is *stable* when every non-resolution-limited
    estimate agrees with the reference (factor 1) within ``rel_tol``."""
    out = {}
    for fct in factors:
        nxb = max(1, int(round(nx_b / fct)))
        nvb = max(2, int(round(nv_b / fct)))
        fh = reconstruct_f_particles(x, v, w, L, v_lim, nxb, nvb)
        dx_b, dv_b = L / nxb, 2 * v_lim / nvb
        wc = waterbag_content(fh, dx_b, dv_b, eta_edges, particle_eta_floor(np.min(w), dx_b, dv_b))
        out[fct] = eta_minus1_range(wc, **detect_kw)
    ref = out.get(1.0, {}).get("eta_min", np.nan)
    vals = [r["eta_min"] for r in out.values() if r["found"] and not r["resolution_limited"]]
    stable = bool(np.isfinite(ref) and len(vals) >= 2
                  and all(abs(vv - ref) / ref < rel_tol for vv in vals))
    return {"by_factor": out, "stable": stable}
