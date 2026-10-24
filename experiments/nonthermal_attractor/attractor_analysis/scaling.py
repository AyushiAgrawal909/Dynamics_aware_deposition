"""
attractor_analysis/scaling.py
=============================

Identifiable self-similar scaling inference (T1 section 5).

Hypothesis::

    N(eps, t) = A* s^alpha F(s^beta eps / eps*),   s = (t - t0) / t* > 0.

For snapshot i the scaled data are ``x = s_i^beta eps`` and
``ln y = ln N_i - alpha ln s_i``.  A collapse means all ``ln y_i(x)`` lie on a
single curve.  There is **no** free per-snapshot amplitude: normalisation is
preserved, and ``A*``, ``eps*`` cancel from the collapse loss (they are
recorded only).

Identifiability: if F is a pure power law ``x^-p`` across the overlap, then
``ln y_i = const - (alpha - p beta) ln s_i - p ln x`` and only the combination
``alpha - p beta`` is constrained.  The loss landscape then has a valley along
the direction (p, 1) in (alpha, beta).  ``fit_scaling`` detects this from the
Hessian of the t0-profiled loss and returns ``UNIDENTIFIABLE_DEGENERATE`` with
the identified combination instead of separate exponents.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy import optimize


IDENTIFIED = "IDENTIFIED"
DEGENERATE = "UNIDENTIFIABLE_DEGENERATE"
NO_COLLAPSE = "NO_COLLAPSE"
STATIONARY = "STATIONARY"
BOUNDARY = "BOUNDARY_HIT"


@dataclass
class SnapshotSet:
    """Training snapshots on common energy bins.

    ``N`` and ``sigma_ln`` have shape (n_times, n_bins); ``support`` marks
    bins with enough effective counts; ``fit_bins`` (n_bins,) is the frozen
    raw-energy fit window.
    """
    times: np.ndarray
    eps: np.ndarray            # representative bin energies (geometric centres)
    N: np.ndarray
    sigma_ln: np.ndarray
    support: np.ndarray
    fit_bins: np.ndarray
    t_star: float = 1.0
    eps_star: float = 1.0

    def usable(self) -> np.ndarray:
        return self.support & self.fit_bins[None, :] & (self.N > 0) & np.isfinite(self.sigma_ln)


def sigma_ln_from_neff(neff: np.ndarray, floor: float = 1e-3) -> np.ndarray:
    """Counting approximation sigma(ln N_b) ~ 1/sqrt(neff_b).

    Optimistic for quiet loads and correlated snapshots; prefer
    ``sigma_ln_from_seeds`` when independent seeds exist.
    """
    with np.errstate(divide="ignore"):
        return np.sqrt(1.0 / np.maximum(neff, 1e-300) + floor**2)


def sigma_ln_from_seeds(N_seeds: np.ndarray, floor: float = 1e-3) -> np.ndarray:
    """Standard error of ln(mean over seeds) from the seed scatter.

    ``N_seeds`` has shape (n_seeds, n_times, n_bins).
    """
    m = N_seeds.mean(axis=0)
    sd = N_seeds.std(axis=0, ddof=1) / np.sqrt(N_seeds.shape[0])
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.sqrt((sd / m) ** 2 + floor**2)


def _largest_run(mask: np.ndarray) -> np.ndarray:
    out = np.zeros_like(mask)
    best, i, n = (0, -1), 0, len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j + 1 < n and mask[j + 1]:
                j += 1
            if j - i > best[1] - best[0]:
                best = (i, j)
            i = j + 1
        else:
            i += 1
    if best[1] >= 0:
        out[best[0]:best[1] + 1] = True
    return out


@dataclass
class _Prepared:
    lneps: list = field(default_factory=list)
    lnN: list = field(default_factory=list)
    sig: list = field(default_factory=list)


def _prepare(data: SnapshotSet) -> _Prepared:
    p = _Prepared()
    use = data.usable()
    for i in range(len(data.times)):
        m = _largest_run(use[i])
        p.lneps.append(np.log(data.eps[m] / data.eps_star))
        p.lnN.append(np.log(data.N[i, m]))
        p.sig.append(data.sigma_ln[i, m])
    return p


def collapse_loss(alpha: float, beta: float, t0: float, data: SnapshotSet, prep: Optional[_Prepared] = None,
                  min_overlap: float = 1.0) -> dict:
    """Chi-square collapse loss for given (alpha, beta, t0).

    Leave-one-out construction: every supported data point of snapshot i is
    compared with the inverse-variance mean of the *other* snapshots,
    interpolated in ln x only within their own supported ranges (no
    extrapolation).  ``overlap`` is the ln-x width spanned by points that have
    at least one comparison; widths below ``min_overlap`` (in e-folds) are
    penalised, so the fit cannot win by collapsing a vanishing overlap.
    """
    prep = prep or _prepare(data)
    s = (data.times - t0) / data.t_star
    bad = {"chi2": np.inf, "chi2_red": np.inf, "dof": 0, "overlap": 0.0, "loss": np.inf, "npts": 0}
    if np.any(s <= 0):
        return bad
    lns = np.log(s)
    lx = [le + beta * ls for le, ls in zip(prep.lneps, lns)]
    ly = [ln - alpha * ls for ln, ls in zip(prep.lnN, lns)]
    chi2 = 0.0
    npts = 0
    xs_cmp = []
    n = len(lx)
    for i in range(n):
        a = lx[i]
        if a.size == 0:
            continue
        wsum = np.zeros(a.size)
        wy = np.zeros(a.size)
        for j in range(n):
            if j == i or lx[j].size < 2:
                continue
            inside = (a >= lx[j][0]) & (a <= lx[j][-1])
            if not inside.any():
                continue
            yj = np.interp(a[inside], lx[j], ly[j])
            sj = np.interp(a[inside], lx[j], prep.sig[j])
            w = 1.0 / sj**2
            wsum[inside] += w
            wy[inside] += w * yj
        has = wsum > 0
        if not has.any():
            continue
        m = wy[has] / wsum[has]
        var = prep.sig[i][has] ** 2 + 1.0 / wsum[has]
        chi2 += float(np.sum((ly[i][has] - m) ** 2 / var))
        npts += int(has.sum())
        xs_cmp.append(a[has])
    if npts < 4:
        return bad
    allx = np.concatenate(xs_cmp)
    overlap = float(allx.max() - allx.min())
    dof = max(npts - 3, 1)
    chi2_red = chi2 / dof
    penalty = 0.0 if overlap >= min_overlap else 1e3 * (min_overlap - overlap) / min_overlap
    return {"chi2": chi2, "chi2_red": float(chi2_red), "dof": dof, "overlap": overlap,
            "loss": float(chi2_red + penalty), "npts": npts}


def _profile_t0(alpha, beta, data, prep, t0_bounds, **kw):
    if t0_bounds[0] == t0_bounds[1]:
        return t0_bounds[0], collapse_loss(alpha, beta, t0_bounds[0], data, prep, **kw)["loss"]
    grid = np.linspace(t0_bounds[0], t0_bounds[1], 7)
    vals = [collapse_loss(alpha, beta, t, data, prep, **kw)["loss"] for t in grid]
    j = int(np.argmin(vals))
    a = grid[max(j - 1, 0)]
    b = grid[min(j + 1, len(grid) - 1)]
    if b > a:
        res = optimize.minimize_scalar(lambda t: collapse_loss(alpha, beta, t, data, prep, **kw)["loss"],
                                       bounds=(a, b), method="bounded", options={"xatol": 1e-3 * (b - a)})
        if res.fun <= vals[j]:
            return float(res.x), float(res.fun)
    return float(grid[j]), float(vals[j])


def local_powerlaw_index(data: SnapshotSet) -> float:
    """Median -d ln N / d ln eps over usable fit bins (all snapshots)."""
    slopes = []
    use = data.usable()
    for i in range(len(data.times)):
        m = _largest_run(use[i])
        if m.sum() >= 3:
            le = np.log(data.eps[m])
            ln = np.log(data.N[i, m])
            slopes.append(-np.polyfit(le, ln, 1)[0])
    return float(np.median(slopes)) if slopes else np.nan


def fit_scaling(data: SnapshotSet, alpha_bounds=(-2.0, 2.0), beta_bounds=(-2.0, 2.0),
                t0_bounds=(0.0, 0.0), n_grid_ab: int = 41, cond_threshold: float = 50.0,
                angle_tol_deg: float = 15.0, chi2_red_max: float = 3.0, min_overlap: float = 1.0,
                valley_span: float = 0.4, valley_k: float = 5.0, stationary_tol: float = 0.03) -> dict:
    """Grid search + refinement of the t0-profiled collapse loss.

    Returns a dict with ``status`` in {IDENTIFIED, UNIDENTIFIABLE_DEGENERATE,
    STATIONARY, NO_COLLAPSE, BOUNDARY_HIT}, best (alpha, beta, t0), the profile-loss
    landscape, Hessian-based covariance, and for the degenerate case the
    identified combination ``alpha - p beta``.
    """
    prep = _prepare(data)
    kw = {"min_overlap": min_overlap}
    A = np.linspace(*alpha_bounds, n_grid_ab)
    B = np.linspace(*beta_bounds, n_grid_ab)
    land = np.full((len(A), len(B)), np.inf)
    t0map = np.zeros_like(land)
    for i, a in enumerate(A):
        for j, b in enumerate(B):
            # coarse landscape with t0 on a small grid only
            if t0_bounds[0] == t0_bounds[1]:
                t0s = [t0_bounds[0]]
            else:
                t0s = np.linspace(t0_bounds[0], t0_bounds[1], 5)
            vals = [collapse_loss(a, b, t, data, prep, **kw)["loss"] for t in t0s]
            k = int(np.argmin(vals))
            land[i, j] = vals[k]
            t0map[i, j] = t0s[k]
    i, j = np.unravel_index(np.argmin(land), land.shape)

    def prof(p):
        return _profile_t0(p[0], p[1], data, prep, t0_bounds, **kw)[1]

    res = optimize.minimize(prof, x0=[A[i], B[j]], method="Nelder-Mead",
                            options={"xatol": 1e-3, "fatol": 1e-4, "maxiter": 400})
    a_best, b_best = (float(res.x[0]), float(res.x[1]))
    a_best = float(np.clip(a_best, *alpha_bounds))
    b_best = float(np.clip(b_best, *beta_bounds))
    t0_best, _ = _profile_t0(a_best, b_best, data, prep, t0_bounds, **kw)
    best = collapse_loss(a_best, b_best, t0_best, data, prep, **kw)

    # Hessian of the profiled chi2 at the optimum (central differences)
    h = 0.05
    f0 = prof([a_best, b_best])
    H = np.zeros((2, 2))
    e = np.eye(2)
    for p in range(2):
        for q in range(2):
            if p == q:
                fp = prof(np.array([a_best, b_best]) + h * e[p])
                fm = prof(np.array([a_best, b_best]) - h * e[p])
                H[p, p] = (fp - 2 * f0 + fm) / h**2
            elif p < q:
                fpp = prof(np.array([a_best, b_best]) + h * (e[p] + e[q]))
                fpm = prof(np.array([a_best, b_best]) + h * (e[p] - e[q]))
                fmp = prof(np.array([a_best, b_best]) - h * (e[p] - e[q]))
                fmm = prof(np.array([a_best, b_best]) - h * (e[p] + e[q]))
                H[p, q] = H[q, p] = (fpp - fpm - fmp + fmm) / (4 * h**2)
    evals, evecs = np.linalg.eigh(H)
    lam_min, lam_max = float(max(evals[0], 1e-12)), float(max(evals[1], 1e-12))
    cond = lam_max / lam_min
    soft = evecs[:, 0] * np.sign(evecs[1, 0] if evecs[1, 0] != 0 else 1.0)
    p_idx = local_powerlaw_index(data)
    angle = np.nan
    if np.isfinite(p_idx):
        d = np.array([p_idx, 1.0]) / np.hypot(p_idx, 1.0)
        angle = float(np.degrees(np.arccos(min(1.0, abs(float(soft @ d))))))
    # chi2 covariance, inflated by the reduced chi2 when it exceeds 1
    infl = max(1.0, best["chi2_red"])
    try:
        # the loss is the reduced chi2, so the chi2 Hessian is H * dof
        cov = 2.0 * np.linalg.inv(H * best["dof"]) * infl
    except np.linalg.LinAlgError:
        cov = np.full((2, 2), np.inf)

    on_boundary = bool(np.isclose(a_best, alpha_bounds).any() or np.isclose(b_best, beta_bounds).any()
                       or (t0_bounds[1] > t0_bounds[0] and np.isclose(t0_best, t0_bounds).any()))
    out = {
        "alpha": a_best, "beta": b_best, "t0": t0_best,
        "chi2_red": best["chi2_red"], "overlap": best["overlap"],
        "hessian": H.tolist(), "cov": cov.tolist(),
        "alpha_err": float(np.sqrt(cov[0, 0])) if cov[0, 0] > 0 else np.inf,
        "beta_err": float(np.sqrt(cov[1, 1])) if cov[1, 1] > 0 else np.inf,
        "condition_number": cond, "soft_direction": soft.tolist(),
        "powerlaw_index_p": p_idx, "soft_vs_powerlaw_angle_deg": angle,
        "landscape": {"alpha": A.tolist(), "beta": B.tolist(), "loss": land.tolist()},
        "boundary_hit": on_boundary,
        "t_star": data.t_star, "eps_star": data.eps_star,
    }
    # Valley test along the power-law degeneracy direction (p, 1): profile the
    # total chi2 along it over +-``valley_span`` in beta.  The valley is flat
    # (only alpha - p beta measured) when the largest rise stays within
    # ``valley_k`` standard deviations of chi2 itself, sqrt(2 dof), inflated
    # by the reduced chi2.  Interpolated leave-one-out comparisons make chi2
    # fluctuate by O(sqrt(2 dof)) along a truly flat valley, so a fixed
    # absolute threshold would not be scale-free.
    valley = {"dchi2_max": np.nan}
    if np.isfinite(p_idx):
        u = np.array([p_idx, 1.0])
        ds = np.linspace(-valley_span, valley_span, 9)
        chi = []
        for d in ds:
            q = np.array([a_best, b_best]) + d * u
            if not (alpha_bounds[0] <= q[0] <= alpha_bounds[1] and beta_bounds[0] <= q[1] <= beta_bounds[1]):
                continue
            t0q, _ = _profile_t0(q[0], q[1], data, prep, t0_bounds, **kw)
            r = collapse_loss(q[0], q[1], t0q, data, prep, **kw)
            if r["overlap"] >= min_overlap:
                chi.append(r["chi2_red"] * best["dof"])
        if len(chi) >= 5:
            valley = {"dchi2_max": float(max(chi) - best["chi2_red"] * best["dof"]), "n_points": len(chi)}
    out["valley_test"] = valley
    valley["threshold"] = float(valley_k * np.sqrt(2.0 * best["dof"]) * max(1.0, best["chi2_red"]))
    degenerate = bool(np.isfinite(valley["dchi2_max"]) and valley["dchi2_max"] < valley["threshold"])
    stationary = bool(abs(a_best) < stationary_tol and abs(b_best) < stationary_tol)
    if best["chi2_red"] > chi2_red_max:
        out["status"] = NO_COLLAPSE
    elif degenerate:
        out["status"] = DEGENERATE
        comb = a_best - p_idx * b_best
        g = np.array([1.0, -p_idx])
        out["identified_combination"] = {"expr": f"alpha - {p_idx:.3f} beta", "value": float(comb),
                                         "err": float(np.sqrt(max(g @ cov @ g, 0.0)))}
        out["alpha_err"] = out["beta_err"] = np.inf
    elif stationary:
        # alpha = beta = 0: the data are stationary and t0 is undefined, so a
        # t0 boundary hit carries no information
        out["status"] = STATIONARY
    elif on_boundary:
        out["status"] = BOUNDARY
    else:
        out["status"] = IDENTIFIED
    return out


# --------------------------------------------------------------------------
# moving-sector budgets
# --------------------------------------------------------------------------

def sector_moments(edges: np.ndarray, N: np.ndarray, eps1: float, eps2: float) -> tuple[float, float]:
    """Number and energy between eps1 and eps2 from a piecewise-constant N(eps).

    Partial bins are included proportionally (exact for piecewise-constant
    N).  Used for sectors whose bounds move as ``eps* x s^-beta``.
    """
    lo = np.clip(edges[:-1], eps1, eps2)
    hi = np.clip(edges[1:], eps1, eps2)
    w = np.maximum(hi - lo, 0.0)
    num = float(np.sum(N * w))
    en = float(np.sum(N * 0.5 * (hi**2 - lo**2)))
    return num, en


def sector_scaling_check(times, edges, N_t, x1, x2, alpha, beta, t0, t_star=1.0, eps_star=1.0) -> dict:
    """Compare measured moving-sector moments with s^(alpha-beta) and s^(alpha-2beta).

    For bounds moving with the scaled variable the self-similar form predicts
    these powers exactly; no flux term enters because the sector is defined in
    scaled coordinates.  (Fixed-threshold moments differ: differentiating
    them requires the boundary flux N(eps_th) d eps_th/dt, which vanishes
    only for fixed thresholds.)
    """
    s = (np.asarray(times) - t0) / t_star
    nums, ens = [], []
    for si, Ni in zip(s, N_t):
        e1 = eps_star * x1 * si ** (-beta)
        e2 = eps_star * x2 * si ** (-beta)
        n, e = sector_moments(edges, Ni, e1, e2)
        nums.append(n)
        ens.append(e)
    nums, ens = np.array(nums), np.array(ens)
    ls = np.log(s)
    out = {"pred_number_power": alpha - beta, "pred_energy_power": alpha - 2 * beta}
    ok = (nums > 0) & (ens > 0)
    if ok.sum() >= 3:
        out["meas_number_power"] = float(np.polyfit(ls[ok], np.log(nums[ok]), 1)[0])
        out["meas_energy_power"] = float(np.polyfit(ls[ok], np.log(ens[ok]), 1)[0])
    return out
