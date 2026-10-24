"""
attractor_analysis/synthetic.py
===============================

Synthetic identifiability suite S1-S7 (T1 section 7).

Synthetic data validate the *analysis* only.  They are generated as particle
samples (not smooth curves) so the noise model is realistic, written through
the data contract, and kept apart from any plasma result
(``backend="synthetic"``).

Regulator levels mimic the T2 nested scan: level 0 is the finest
(``ppc_levels[0]`` particles per cell); level l has ``ppc_levels[0] / 2**l``.
All levels of one seed share common random numbers (the level-l sample uses
the first N/2**l uniforms of the level-0 stream), the synthetic analogue of
nested subsampling.

Case summary (expected verdicts are pre-stated in ``EXPECTED``):

S1  stationary 1V Maxwellian                         -> no tail; unbiased moments
S2  fixed core + self-similar tail sector with a   -> (alpha, beta) recovered; forecast wins
    moving cutoff and growing sector number
S3  stationary eps^-2 tail between far cutoffs      -> UNIDENTIFIABLE_DEGENERATE, alpha - 2 beta ~ 0
S4  core heats, halo fixed (not self-similar)       -> no nonthermal-scaling claim
S4b pure Maxwellian heating (self-similar, thermal) -> no nonthermal-scaling claim
S5  cutoff tracks ln N, fixed in time               -> gate FAIL_REGULATOR_DEPENDENT
S6  S2 dynamics with onset t0(N) = a + b ln N        -> gate passes after onset alignment; ln N preferred;
                                                       (alpha, beta) recovered after t0 profiling
S7  rho(eta) with eta^-1 range extending downward   -> eta_min recovered above the particle floor;
                                                       flagged RESOLUTION_LIMITED near it
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence

import numpy as np

from .energy import (EnergyHistogram, common_edges, energy_histogram, histogram_moments,
                     maxwellian_tail_excess, quantile_feature, tail_moments)
from .waterbag import (eta_minus1_range, particle_eta_floor, reconstruct_f_particles,
                       waterbag_content)


N_CELLS = 64
EDGES = common_edges(0.05, 1.0e4, n_log=56, n_lin=4)
TAIL_THRESHOLD = float(EDGES[np.argmin(np.abs(EDGES - 5.0))])


# --------------------------------------------------------------------------
# inverse-CDF sampling of energy shapes in a scaled variable
# --------------------------------------------------------------------------

class ShapeSampler:
    """Inverse-CDF sampler for an (unnormalised) density F(x) on x > 0."""

    def __init__(self, F: Callable[[np.ndarray], np.ndarray], x_lo=1e-10, x_hi=1e6, n=40000):
        lx = np.linspace(np.log(x_lo), np.log(x_hi), n)
        x = np.exp(lx)
        dens = F(x) * x  # density in ln x
        cdf = np.concatenate([[0.0], np.cumsum(0.5 * (dens[1:] + dens[:-1]) * np.diff(lx))])
        self.norm = cdf[-1]
        self.cdf = cdf / cdf[-1]
        self.lx = lx

    def __call__(self, u: np.ndarray) -> np.ndarray:
        return np.exp(np.interp(u, self.cdf, self.lx))


def F_maxwell(x):
    return x**-0.5 * np.exp(-x)


def F_curved_tail(x):
    """Thermal-like core, eps^-2 power law between x~1 and x~20, then a cutoff."""
    return x**-0.5 * (1.0 + x) ** -1.5 * np.exp(-x / 20.0)


def F_powerlaw_tail(x, lo=2.0):
    """Normalised eps^-2 on [lo, inf) (cutoffs far outside any fit window)."""
    return np.where(x >= lo, x**-2.0, 0.0)


_MAXWELL = ShapeSampler(F_maxwell)
_CURVED = ShapeSampler(F_curved_tail)
_POWER = ShapeSampler(F_powerlaw_tail, x_lo=1.0, x_hi=1e7)


def _mixture(u: np.ndarray, u2: np.ndarray, weight_tail: float, core: Callable, tail: Callable):
    """Two-component sample from two uniform streams (common random numbers)."""
    is_tail = u2 < weight_tail
    out = core(u)
    out[is_tail] = tail(u[is_tail])
    return out


# --------------------------------------------------------------------------
# case definitions: energy(u, u2, t, ppc) -> eps samples
# --------------------------------------------------------------------------

ALPHA_TRUE = -0.25
BETA_TRUE = -0.5
S2_TAIL_N1 = 0.08  # tail-sector number at s = 1


def F_sector_tail(x):
    """Tail sector: eps^-2 body, smooth low-energy edge, exponential cutoff."""
    return x**-2.0 * np.exp(-x / 50.0 - 2.0 / x)


_SECTOR = ShapeSampler(F_sector_tail, x_lo=1e-2, x_hi=1e6)


def _s2_energy(u, u2, t, ppc, t0=0.0):
    """Fixed T=1 core plus a self-similar tail sector

        N_tail(eps, t) = A s^alpha F(s^beta eps),  s = t - t0,

    whose number grows as s^(alpha - beta) (particles leave the core) and
    whose cutoff moves outward as s^(-beta).  The whole distribution is not
    self-similar; the sector is, inside the frozen raw-energy fit window.
    """
    s = t - t0
    n_tail = S2_TAIL_N1 * s ** (ALPHA_TRUE - BETA_TRUE)
    is_tail = u2 < n_tail
    out = _MAXWELL(u)
    out[is_tail] = _SECTOR(u[is_tail]) * s ** (-BETA_TRUE)
    return out


def s6_t0(ppc: float) -> float:
    return 0.5 + 0.25 * np.log(ppc / 512.0)


def _s5_cutoff(ppc: float) -> float:
    return 5.0 * np.log(ppc)


_S5_CACHE: Dict[float, ShapeSampler] = {}


def _s5_tail(ppc):
    ec = _s5_cutoff(ppc)
    if ec not in _S5_CACHE:
        _S5_CACHE[ec] = ShapeSampler(lambda x: np.where(x >= 2.0, x**-2.0 * np.exp(-x / ec), 0.0),
                                     x_lo=1.0, x_hi=1e5)
    return _S5_CACHE[ec]


CASES: Dict[str, Callable] = {
    "S1": lambda u, u2, t, ppc: _MAXWELL(u) * 1.0,
    "S2": lambda u, u2, t, ppc: _s2_energy(u, u2, t, ppc),
    "S3": lambda u, u2, t, ppc: _mixture(u, u2, 0.1, _MAXWELL, _POWER),
    "S4": lambda u, u2, t, ppc: _mixture(u, u2, 0.1, lambda q: _MAXWELL(q) * np.sqrt(t),
                                         lambda q: _MAXWELL(q) * 8.0),
    "S4b": lambda u, u2, t, ppc: _MAXWELL(u) * np.sqrt(t),
    "S5": lambda u, u2, t, ppc: _mixture(u, u2, 0.1, _MAXWELL, _s5_tail(ppc)),
    "S6": lambda u, u2, t, ppc: _s2_energy(u, u2, t, ppc, t0=s6_t0(ppc)),
}

TIMES = {
    "S1": np.arange(1.0, 6.5, 0.5),
    "S2": np.arange(1.0, 6.5, 0.5),
    "S3": np.arange(1.0, 6.5, 0.5),
    "S4": np.arange(1.0, 6.5, 0.5),
    "S4b": np.arange(1.0, 6.5, 0.5),
    "S5": np.arange(1.0, 6.5, 0.5),
    "S6": np.arange(1.5, 7.0, 0.5),
}

# frozen per-case protocol choices (raw-energy fit windows, training split)
FIT_WINDOW = {
    "S1": (0.1, 50.0), "S2": (8.0, 5000.0), "S3": (10.0, 1000.0), "S4": (0.1, 1000.0),
    "S4b": (0.1, 1000.0), "S5": (0.1, 1000.0), "S6": (8.0, 5000.0),
}
TRAIN_END = 3.5  # snapshots with t <= TRAIN_END are training, the rest held out

EXPECTED = {
    "S1": "NO_TAIL",
    "S2": "IDENTIFIED_AND_FORECAST_WINS",
    "S3": "UNIDENTIFIABLE_DEGENERATE",
    "S4": "NO_CLAIM",
    "S4b": "NO_CLAIM",
    "S5": "FAIL_REGULATOR_DEPENDENT",
    "S6": "PASS_AFTER_ONSET_ALIGNMENT+lnN+IDENTIFIED",
    "S7": "ETA_MIN_RECOVERED_OR_FLAGGED",
}


@dataclass
class LevelData:
    ppc: int
    times: np.ndarray
    hists: List[List[EnergyHistogram]]  # [seed][time]


def generate_case(case: str, ppc_levels: Sequence[int], n_seeds: int, times=None,
                  n_cells: int = N_CELLS, base_seed: int = 2026) -> Dict[int, LevelData]:
    """Generate per-level, per-seed energy histograms for a case.

    ``ppc_levels`` must be finest first, each half the previous (nested).
    Total physical number is normalised to 1, so macro weight = 1/N.
    """
    times = TIMES[case] if times is None else np.asarray(times)
    fn = CASES[case]
    N0 = ppc_levels[0] * n_cells
    out = {}
    for lvl, ppc in enumerate(ppc_levels):
        if ppc_levels[0] >> lvl != ppc:
            raise ValueError("ppc_levels must halve per level")
        out[lvl] = LevelData(ppc, times, [])
    for s in range(n_seeds):
        for ti, t in enumerate(times):
            rng = np.random.default_rng([base_seed, sum(map(ord, case)), s, ti])
            u = rng.random(N0)
            u2 = rng.random(N0)
            for lvl, ppc in enumerate(ppc_levels):
                m = N0 >> lvl
                eps = fn(u[:m], u2[:m], t, ppc)
                w = np.full(m, 1.0 / m)
                h = energy_histogram(eps, w, EDGES)
                if ti == 0:
                    out[lvl].hists.append([])
                out[lvl].hists[s].append(h)
    return out


# --------------------------------------------------------------------------
# S7: phase-space case for rho(eta)
# --------------------------------------------------------------------------

S7_L = 1.0
S7_VS = 1.0
S7_VLIM = 10.0


def s7_v_edge(t):
    return 2.0 + t


def s7_eta_min_true(t):
    return float(np.exp(-s7_v_edge(t) / S7_VS))


def s7_sample(N: int, t: float, rng: np.random.Generator):
    """f(x, v) = exp(-|v|/v_s) for |v| < v_e(t): rho(eta) = 2 L v_s / eta on [eta_min, 1]."""
    ve = s7_v_edge(t)
    u = rng.random(N)
    a = -S7_VS * np.log(1.0 - u * (1.0 - np.exp(-ve / S7_VS)))
    v = np.where(rng.random(N) < 0.5, -a, a)
    x = rng.random(N) * S7_L
    number = 2.0 * S7_L * S7_VS * (1.0 - np.exp(-ve / S7_VS))
    w = np.full(N, number / N)
    return x, v, w


S7_ETA_EDGES = np.exp(np.arange(np.log(1e-6), np.log(2.0) + 1e-9, 0.25))


def run_s7(ppc_levels: Sequence[int], times, n_cells: int = N_CELLS, nx_b: int = 16, nv_b: int = 200,
           seed: int = 7, floor_factor: float = 10.0) -> list:
    rows = []
    rng = np.random.default_rng(seed)
    for ppc in ppc_levels:
        N = ppc * n_cells
        for t in times:
            x, v, w = s7_sample(N, t, rng)
            fh = reconstruct_f_particles(x, v, w, S7_L, S7_VLIM, nx_b, nv_b)
            dx_b, dv_b = S7_L / nx_b, 2 * S7_VLIM / nv_b
            wc = waterbag_content(fh, dx_b, dv_b, S7_ETA_EDGES, particle_eta_floor(w[0], dx_b, dv_b))
            det = eta_minus1_range(wc, floor_factor=floor_factor)
            truth = s7_eta_min_true(t)
            rows.append({"ppc": ppc, "t": float(t), "eta_min_true": truth, **det})
    return rows


def s7_verdict(rows, ln_tol: float = 0.25) -> dict:
    """Recovered where not flagged; never a confident wrong answer.

    A *false resolved* is an unflagged estimate whose |ln(est/true)| exceeds
    ``ln_tol`` (one eta bin).  A *missed flag* is truth below the floor but
    estimate unflagged.
    """
    n_ok = n_flag = n_false = n_missing = 0
    for r in rows:
        if not r["found"]:
            n_missing += 1
            continue
        if r["resolution_limited"]:
            n_flag += 1
            continue
        if abs(np.log(r["eta_min"] / r["eta_min_true"])) <= ln_tol:
            n_ok += 1
        else:
            n_false += 1
    verdict = "ETA_MIN_RECOVERED_OR_FLAGGED" if (n_false == 0 and n_ok > 0 and n_flag > 0) else "S7_FAILED"
    return {"verdict": verdict, "recovered": n_ok, "flagged": n_flag, "false_resolved": n_false,
            "not_found": n_missing}
