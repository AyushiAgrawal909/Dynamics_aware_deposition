"""Roots of the discrete dispersion relation, branch continuation in the
filter coupling, marginal couplings and admissible coupling intervals.

Conventions: perturbations ~ exp(i k.x - i omega t), omega = omega_r + i gamma;
gamma > 0 grows, gamma < 0 is damped.  ``a`` is the round-trip filter transfer
at the mode (a = 1 is the unfiltered baseline).
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

from .dispersion import ModeDispersion


class RootError(RuntimeError):
    pass


# ---------------------------------------------------------------- Newton
def newton(disp, omega0: complex, a: float = 1.0, *, tol: float = 1e-12,
           max_iter: int = 60, max_step: float | None = None, ftol: float = 1e-8) -> complex:
    """Damped complex Newton iteration on eps(omega; a) = 1 + a G(omega)."""
    w = complex(omega0)
    if max_step is None:
        max_step = 0.25 * disp.plasma.omega_p_total + 0.25 * abs(w)
    for _ in range(max_iter):
        g = disp.G(w)
        f = 1.0 + a * g
        dg = a * disp.dG(w)
        if not (np.isfinite(f) and np.isfinite(dg)) or dg == 0:
            raise RootError("non-finite residual or derivative")
        step = f / dg
        if abs(step) > max_step:
            step *= max_step / abs(step)
        w -= step
        if abs(step) <= tol * (abs(w) + disp.plasma.omega_p_total):
            break
    else:
        raise RootError(f"Newton did not converge (omega={w})")
    g = disp.G(w)
    f = 1.0 + a * g
    if not np.isfinite(f) or abs(f) > ftol * (1.0 + abs(a * g)):
        raise RootError(f"Newton stalled (omega={w}, |eps|={abs(f):.2e})")
    return w


def _wrap(disp, w: complex) -> complex:
    """Map a leapfrog root into the principal frequency strip."""
    og = getattr(disp, "omega_grid", None)
    if og:
        re = (w.real + 0.5 * og) % og - 0.5 * og
        return complex(re, w.imag)
    return w


# ---------------------------------------------------------- initial guesses
def _fluid_guesses(disp) -> list[complex]:
    """Roots of cold and warm-fluid dispersion polynomials of the principal mode."""
    k = disp.k
    grid = disp.grid if hasattr(disp, "grid") else disp.d0.grid
    order = disp.order if hasattr(disp, "order") else disp.d0.order
    S2 = float(grid.shape_transform(k, order) ** 2)
    out: list[complex] = []
    species = disp.plasma.mobile
    for gamma_ad in (None, 1.0, 3.0):
        polys = []
        for s in species:
            u = float(k @ s.u)
            c2 = 0.0 if gamma_ad is None else gamma_ad * float((k**2) @ s.v**2)
            # (omega - u)^2 - c2
            polys.append((np.poly1d([1.0, -2.0 * u, u * u - c2]), s.omega_p2(disp.plasma.eps0)))
        total = np.poly1d([1.0])
        for p, _ in polys:
            total = total * p
        rhs = np.poly1d([0.0])
        for i, (_, wp2) in enumerate(polys):
            term = np.poly1d([S2 * wp2])
            for j, (pj, _) in enumerate(polys):
                if j != i:
                    term = term * pj
            rhs = rhs + term
        try:
            roots = np.roots((total - rhs).coeffs)
        except np.linalg.LinAlgError:
            continue
        out.extend(complex(r) for r in roots if np.isfinite(r))
    return out


def _grid_guesses(disp, n_re: int = 24, n_im: int = 8) -> list[complex]:
    lo, hi, im_max = disp.frequency_window()
    re = np.linspace(lo, hi, n_re)
    im = np.linspace(-0.5 * im_max, im_max, n_im)
    return [complex(r, i) for r in re for i in im]


def _resonance_guesses(disp, rel: float = 1e-6, max_n: int = 200) -> list[complex]:
    """Starts just above the Doppler-shifted resonances of the aliases."""
    sh, sc = disp.resonances(rel)
    if sh.size == 0:
        return []
    wp = disp.plasma.omega_p_total
    sh = np.unique(np.round([_wrap(disp, complex(x, 0.0)).real for x in sh], 9))
    if sh.size > max_n:
        sh = sh[np.linspace(0, sh.size - 1, max_n).astype(int)]
    return [complex(x, y * wp) for x in sh for y in (0.02, 0.2)]


def find_roots(disp, a: float = 1.0, *, guesses: Sequence[complex] = (),
               grid_density: int = 1, tol_merge: float = 1e-7) -> list[complex]:
    """Return distinct roots of eps(omega; a) found from many starting points.

    Starts: user guesses, roots of fluid dispersion polynomials, and (for
    ``grid_density > 0``) a grid over the principal frequency window plus
    points above every alias resonance q.u.
    """
    starts = list(guesses) + _fluid_guesses(disp)
    if grid_density > 0:
        starts += _grid_guesses(disp, 24 * grid_density, 8 * grid_density)
        starts += _resonance_guesses(disp)
    scale = disp.plasma.omega_p_total
    roots: list[complex] = []
    for w0 in starts:
        try:
            w = _wrap(disp, newton(disp, w0, a))
        except RootError:
            continue
        if not np.isfinite(w):
            continue
        if all(abs(w - r) > tol_merge * (abs(r) + scale) for r in roots):
            roots.append(w)
    roots.sort(key=lambda z: -z.imag)
    return roots


def _eval_eps(disp, a: float, w: np.ndarray, chunk: int = 4096) -> np.ndarray:
    out = np.empty(w.shape, dtype=np.complex128)
    with np.errstate(all="ignore"):
        for i in range(0, w.size, chunk):
            out[i:i + chunk] = 1.0 + a * np.asarray(disp.G(w[i:i + chunk]))
    return out


def _phase_increments(disp, a, pts, vals, poles):
    """Phase increments of eps * prod_j g_j along the contour.

    g_j = (omega - s_j)^2 (continuous) or sin^2(dt (omega - s_j) / 2)
    (leapfrog) removes the double poles of cold species on the real axis,
    so the function is smooth near the bottom edge.  The product of the
    per-segment ratios is formed directly (no overflow, no branch ambiguity).
    """
    ratio = vals[1:] / vals[:-1]
    if poles.size:
        og = getattr(disp, "omega_grid", None)
        w0, w1 = pts[:-1, None], pts[1:, None]
        if og:
            h = math.pi / og
            r = np.sin(h * (w1 - poles)) / np.sin(h * (w0 - poles))
        else:
            r = (w1 - poles) / (w0 - poles)
        ratio = ratio * np.prod(r * r, axis=1)
    return np.angle(ratio)


def count_unstable(disp, a: float = 1.0, *, im_min: float | None = None, n: int = 512,
                   max_refine: int = 40) -> int:
    """Number of roots with Im(omega) > im_min, from the argument principle.

    The contour is a rectangle in the upper half-plane: its far edges lie
    where eps -> 1 (continuous time), or it spans one period of the
    frequency strip (leapfrog).  It covers the Doppler shifts of all retained
    aliases.  The bottom edge is sampled densely around every warm
    resonance; cold double poles are divided out.  Segments are bisected
    until every phase increment is below pi/6.  Returns -1 if the contour
    cannot be resolved (a root lies on it) or the winding is not a
    non-negative integer.
    """
    wp = disp.plasma.omega_p_total
    if im_min is None:
        im_min = 1e-6 * wp
    lo, hi, im_max = disp.frequency_window(all_terms=True)
    og = getattr(disp, "omega_grid", None)
    if og:
        lo, hi = -0.5 * og, 0.5 * og
    else:
        span = hi - lo
        lo, hi = lo - 0.5 * span - 2.0 * wp, hi + 0.5 * span + 2.0 * wp
    top = max(4.0 * im_max, 2.0 * im_min)
    sh, sc = disp.strong_warm_resonances()
    bottom = [np.linspace(lo, hi, n, endpoint=False)]
    if sh.size:
        if og:
            sh = np.array([_wrap(disp, complex(x, 0.0)).real for x in sh])
        width = np.maximum(sc, im_min)
        cluster = (sh[:, None] + width[:, None] * np.linspace(-6.0, 6.0, 49)[None, :]).ravel()
        bottom.append(cluster[(cluster > lo) & (cluster < hi)])
    xb = np.unique(np.concatenate(bottom))
    bottom_pts = xb + 1j * im_min
    side = np.linspace(im_min, top, n, endpoint=False)
    top_x = np.linspace(hi, lo, n, endpoint=False)
    pts = np.concatenate([bottom_pts, hi + 1j * side, top_x + 1j * top, lo + 1j * side[::-1],
                          np.array([complex(lo, im_min)])])
    poles = disp.cold_poles()
    if og and poles.size:
        poles = np.unique(np.round([_wrap(disp, complex(x, 0.0)).real for x in poles], 12))
    vals = _eval_eps(disp, a, pts)
    for _ in range(max_refine):
        dphi = _phase_increments(disp, a, pts, vals, poles)
        bad = np.flatnonzero(~(np.abs(dphi) < math.pi / 6))
        if bad.size == 0:
            w = np.sum(dphi) / (2.0 * math.pi)
            if abs(w - round(w)) > 0.1 or round(w) < 0:
                return -1
            return int(round(w))
        if pts.size > 2_000_000:
            break
        mids = 0.5 * (pts[bad] + pts[bad + 1])
        mvals = _eval_eps(disp, a, mids)
        pts = np.insert(pts, bad + 1, mids)
        vals = np.insert(vals, bad + 1, mvals)
    return -1


def verify_most_unstable(disp, a: float, gamma: float, margin: float = 0.02) -> bool | None:
    """True if no root has Im(omega) > (1 + margin) * gamma (argument principle).

    Returns None if the contour could not be resolved (a root lies on it).
    """
    for m in (margin, 2 * margin, 5 * margin):
        n = count_unstable(disp, a, im_min=(1.0 + m) * gamma)
        if n >= 0:
            return n == 0
    return None


# --------------------------------------------------------------- branches
BRANCH_RULES = ("most_unstable", "least_damped", "nearest")


def select_branch(roots: Sequence[complex], rule: str = "most_unstable",
                  guess: complex | None = None, sign: int = 0) -> complex:
    cand = list(roots)
    if sign:
        scale = max(abs(r) for r in cand)
        signed = [r for r in cand if np.sign(r.real) == np.sign(sign) or abs(r.real) < 1e-12 * scale]
        cand = signed or cand
    if not cand:
        raise RootError("no roots found for this mode")
    if rule == "nearest":
        if guess is None:
            raise ValueError("branch rule 'nearest' needs omega_guess")
        return min(cand, key=lambda r: abs(r - guess))
    if rule in ("most_unstable", "least_damped"):
        best = max(r.imag for r in cand)
        scale = max(abs(r) for r in cand)
        ties = [r for r in cand if abs(r.imag - best) <= 1e-9 * scale]
        # Prefer the forward (positive-frequency) member of a +-omega_r pair.
        return max(ties, key=lambda r: (r.real >= 0, -abs(r.real)))
    raise ValueError(f"unknown branch rule '{rule}'; use one of {BRANCH_RULES}")


# ----------------------------------------------------------- continuation
@dataclass
class Branch:
    """Root of one branch tabulated against the coupling a."""

    a: list[float] = field(default_factory=list)
    omega: list[complex] = field(default_factory=list)
    status: str = "resolved"        # 'resolved', 'marginal' or 'unresolved'
    a_end: float | None = None       # coupling where tracking stopped
    a_marginal: float | None = None  # a_c where Im(omega) reaches 0 (growing branch)
    omega_marginal: float | None = None

    def sorted(self):
        idx = np.argsort(self.a)
        return np.asarray(self.a)[idx], np.asarray(self.omega)[idx]


def _marginal_point(disp, omega_r: float, a: float) -> tuple[float, float]:
    """Solve eps(omega_r + i0, a) = 0 for real (omega_r, a).

    Least-squares Newton in (omega_r / omega_p, a); for symmetric equilibria
    Im eps vanishes identically on the axis and the system is degenerate, which
    lstsq handles.  The solution must stay near the tracked branch.
    """
    wp = disp.plasma.omega_p_total
    x = np.array([omega_r / wp, a], dtype=float)
    for _ in range(80):
        w = complex(x[0] * wp, 0.0)
        g = disp.G(w)
        dg = disp.dG(w) * wp
        f = 1.0 + x[1] * g
        J = np.array([[(x[1] * dg).real, g.real], [(x[1] * dg).imag, g.imag]])
        F = np.array([f.real, f.imag])
        step, *_ = np.linalg.lstsq(J, F, rcond=None)
        lim = 0.05 * max(1.0, abs(x[1]))
        if abs(step[1]) > lim:
            step *= lim / abs(step[1])
        x -= step
        if np.max(np.abs(step)) < 1e-14 * max(1.0, abs(x[0])) and abs(f) < 1e-10:
            break
    w = complex(x[0] * wp, 0.0)
    f = 1.0 + x[1] * disp.G(w)
    if abs(f) > 1e-8:
        raise RootError("marginal point did not converge")
    if abs(w.real - omega_r) > 0.25 * (abs(omega_r) + wp):
        raise RootError("marginal point is far from the tracked branch")
    return float(w.real), float(x[1])


def continue_branch(disp, omega0: complex, a_start: float, a_stop: float, *,
                    step0: float = 0.005, min_step: float = 1e-7, max_step: float = 0.01,
                    stop_on_marginal: bool = True) -> Branch:
    """Follow one root from a_start to a_stop with a tangent predictor.

    The tangent is d omega / d a = -G / (a G') (Eq. 9 of the paper).  A step
    is accepted if Newton converges close to the prediction; otherwise it is
    halved.  For a growing root the tracking stops where Im(omega) reaches 0
    and the marginal coupling a_c is then solved directly on the real axis.
    """
    br = Branch(a=[a_start], omega=[complex(omega0)])
    direction = 1.0 if a_stop > a_start else -1.0
    a, w = a_start, complex(omega0)
    growing = w.imag > 0
    h = step0
    while (a_stop - a) * direction > 1e-15:
        h = min(h, abs(a_stop - a), max_step)
        a_new = a + direction * h
        try:
            tangent = -disp.G(w) / (a * disp.dG(w))
            pred = w + tangent * (a_new - a)
            dpred = abs(tangent) * h
            w_new = _wrap(disp, newton(disp, pred, a_new, max_step=dpred + 1e-6 * (
                abs(w) + disp.plasma.omega_p_total)))
            ok = abs(w_new - pred) <= 0.2 * dpred + 1e-9 * (abs(w) + disp.plasma.omega_p_total)
        except (RootError, ZeroDivisionError, FloatingPointError):
            ok = False
        if ok and growing and stop_on_marginal and w_new.imag <= 0.0:
            ok = False
            if h <= min_step * 10:
                break
        if not ok:
            h *= 0.5
            if h < min_step:
                break
            continue
        a, w = a_new, w_new
        br.a.append(a)
        br.omega.append(w)
        h = min(h * 1.5, max_step)
        if growing and stop_on_marginal and w.imag < 1e-4 * abs(br.omega[0].imag):
            break
    br.a_end = a
    if (a_stop - a) * direction <= 1e-15:
        br.status = "resolved"
        return br
    if growing and stop_on_marginal:
        try:
            wr, ac = _marginal_point(disp, w.real, a)
            if (ac - a) * direction >= -1e-12 and (a_stop - ac) * direction >= 0 \
                    and abs(ac - a) < 0.1:
                br.status = "marginal"
                br.a_marginal = ac
                br.omega_marginal = wr
                return br
        except RootError:
            pass
    br.status = "unresolved"
    return br


def solve_at(disp, branch: Branch, a: float) -> complex:
    """Exact root of the tracked branch at coupling a.

    Inside the tracked range the root is continued from the nearest tabulated
    point.  Between the last tracked point and a marginal coupling a_c the
    square-root law gamma ~ sqrt(a - a_c) supplies the Newton start.
    """
    av, wv = branch.sorted()
    if branch.a_marginal is not None and branch.status == "marginal":
        a_last, w_last = branch.a[-1], branch.omega[-1]
        between = (a - branch.a_marginal) * (a_last - branch.a_marginal) > 0 and \
            abs(a - branch.a_marginal) < abs(a_last - branch.a_marginal)
        if between:
            frac = math.sqrt(abs(a - branch.a_marginal) / abs(a_last - branch.a_marginal))
            guess = complex(w_last.real, w_last.imag * frac)
            w = newton(disp, guess, a, max_step=0.5 * abs(w_last.imag) + 1e-12)
            if w.imag <= 0:
                raise RootError("square-root start converged to a non-growing root")
            return w
    if not (av[0] - 1e-12 <= a <= av[-1] + 1e-12):
        raise RootError(f"a={a} outside the tracked range [{av[0]}, {av[-1]}]")
    i = int(np.argmin(np.abs(av - a)))
    if abs(av[i] - a) < 1e-15:
        return complex(wv[i])
    sub = continue_branch(disp, wv[i], float(av[i]), float(a), step0=abs(a - av[i]),
                          stop_on_marginal=False)
    if sub.status != "resolved":
        raise RootError(f"could not reach a={a} from the tracked branch")
    return complex(sub.omega[-1])


def bisect_coupling(fun: Callable[[float], float], a_in: float, a_out: float,
                    tol: float = 1e-12, max_iter: int = 200) -> float:
    """Boundary between a_in (fun <= 0) and a_out (fun > 0)."""
    f_in = fun(a_in)
    if f_in > 0:
        raise ValueError("a_in is not inside")
    for _ in range(max_iter):
        mid = 0.5 * (a_in + a_out)
        if fun(mid) <= 0:
            a_in = mid
        else:
            a_out = mid
        if abs(a_out - a_in) < tol:
            break
    return a_in
