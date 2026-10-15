"""Dynamics-aware kernel selection.

1. For every protected mode, solve the dispersion relation of the
   implemented coupling at a = 1 (unfiltered baseline kernel), select the
   branch, and continue it as the round-trip transfer a decreases.  This
   gives R_gamma(a) = gamma(a) / gamma_0, the admissible coupling interval
   |R_gamma - 1| <= eps and, for growing branches, the marginal coupling a_c.
2. A candidate kernel is admissible only if, for every protected mode, its
   transfer gives a resolved root of the same branch with |R_gamma - 1| <= eps
   (and, optionally, the real frequency within its own tolerance).  A failed
   root solve is never read as stability.
3. Among admissible candidates, minimise the statistical objective.  The
   Gaussian family is optimised continuously: its admissible set is the
   polytope sum_i (k_i dx_i)^2 h_i^2 <= -ln a_lo(k) over the protected modes.
"""

from __future__ import annotations

import math
import time
import warnings
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from scipy.optimize import minimize

from .dispersion import BlendedDispersion, ModeDispersion
from .filters import Binomial, Filter, Gaussian, Kernel, NoFilter
from .grid import Grid, shape_name, shape_order
from .objectives import Objective
from .plasma import Plasma
from . import roots as rt


# ====================================================================== modes
@dataclass
class ProtectedMode:
    mode: tuple
    branch: str = "most_unstable"
    omega_guess: complex | None = None
    sign: int = 0
    tolerance: float | None = None
    frequency_tolerance: float | None = None
    source: str = "user"
    members: tuple = ()


@dataclass
class ModeAnalysis:
    """Branch of one protected mode for one particle shape."""

    spec: ProtectedMode
    order: int
    disp: object
    omega0: complex                  # baseline root (baseline shape, a = 1)
    omega1: complex                  # this shape's root at a = 1
    branch: rt.Branch
    eps: float
    eps_acc: float
    eps_freq: float | None
    verified: bool | None = None     # argument-principle check of most-unstable
    n_roots_found: int = 0
    _cache: dict = field(default_factory=dict, repr=False)

    @property
    def gamma0(self) -> float:
        return self.omega0.imag

    def at(self, a: float) -> tuple[str, complex | None, float | None]:
        """(status, omega, R_gamma) of the tracked branch at coupling a."""
        key = round(float(a), 15)
        if key in self._cache:
            return self._cache[key]
        br = self.branch
        av = np.asarray(br.a)
        lo, hi = float(av.min()), float(av.max())
        if lo - 1e-13 <= a <= hi + 1e-13:
            try:
                w = rt.solve_at(self.disp, br, a)
                out = (self._status_name(w), w, w.imag / self.gamma0)
            except rt.RootError:
                out = ("unresolved", None, None)
        elif a < lo and br.status == "marginal" and br.a_marginal is not None:
            if a <= br.a_marginal:
                out = ("marginal", None, None)
            else:
                try:
                    w = rt.solve_at(self.disp, br, a)
                    out = (self._status_name(w), w, w.imag / self.gamma0)
                except rt.RootError:
                    out = ("unresolved", None, None)
        else:
            out = ("unresolved", None, None)
        self._cache[key] = out
        return out

    @staticmethod
    def _status_name(w: complex) -> str:
        """Status of a found root: growing (gamma > 0) or damped."""
        return "growing" if w.imag > 0 else "damped"

    def admissible_value(self, w: complex | None, R: float | None, eps: float) -> bool:
        if w is None or R is None:
            return False
        if abs(R - 1.0) > eps + 1e-12:
            return False
        if self.eps_freq is not None and abs(self.omega0.real) > 1e-8 * abs(self.omega0):
            if abs(w.real / self.omega0.real - 1.0) > self.eps_freq + 1e-12:
                return False
        return True

    def lower_threshold(self, eps: float) -> float | None:
        """Smallest a of the interval [a_lo, 1] on which the mode stays within eps.

        Returns None if a = 1 itself is outside the tolerance (possible only
        for a shape different from the baseline).
        """
        key = ("a_lo", eps)
        if key in self._cache:
            return self._cache[key]
        st, w, R = self.at(1.0)
        if not self.admissible_value(w, R, eps):
            self._cache[key] = None
            return None
        av, wv = self.branch.sorted()
        order = np.argsort(-av)
        prev = 1.0
        found = None
        for i in order:
            a = float(av[i])
            if a > 1.0:
                continue
            w = complex(wv[i])
            if not self.admissible_value(w, w.imag / self.gamma0, eps):
                found = (a, prev)
                break
            prev = a
        if found is None:
            # tolerance never exceeded on the tracked range
            a_out = self.branch.a_marginal if self.branch.status == "marginal" else None
            if a_out is None:
                res = float(av.min())
                self._cache[key] = res
                return res
            found = (a_out, prev)

        def bad(a):
            st, w, R = self.at(a)
            return 1.0 if not self.admissible_value(w, R, eps) else -1.0

        res = rt.bisect_coupling(bad, found[1], found[0], tol=1e-12)
        self._cache[key] = res
        return res

    def report(self) -> dict:
        br = self.branch
        av, wv = br.sorted()
        return dict(
            mode=list(self.spec.mode), branch=self.spec.branch, source=self.spec.source,
            mirror_modes=[list(m) for m in self.spec.members],
            shape=shape_name(self.order), k=list(map(float, self.disp.k)),
            omega0=[self.omega0.real, self.omega0.imag],
            omega_shape_a1=[self.omega1.real, self.omega1.imag],
            tolerance=self.eps, acceptance=self.eps_acc, frequency_tolerance=self.eps_freq,
            a_lo_tolerance=self.lower_threshold(self.eps),
            a_lo_acceptance=self.lower_threshold(self.eps_acc),
            branch_status=br.status, a_marginal=br.a_marginal, a_tracked_min=float(av.min()),
            most_unstable_verified=self.verified, n_roots_found=self.n_roots_found,
            table=dict(a=av.tolist(), omega_r=wv.real.tolist(), gamma=wv.imag.tolist()),
        )


# ================================================================ candidates
@dataclass
class CandidateResult:
    kernel: Kernel
    a: list[float]
    W: list[float]
    status: list[str]
    omega: list[complex | None]
    R: list[float | None]
    admissible: bool
    admissible_acc: bool
    objective: float = float("nan")
    objective_field: float = float("nan")
    reduction: float = float("nan")        # 1 - MSE/MSE_baseline (selection data)
    reduction_field: float = float("nan")
    evaluation: dict = field(default_factory=dict)
    note: str = ""

    @property
    def label(self) -> str:
        return self.kernel.label

    @property
    def family(self) -> str:
        return self.kernel.family

    def to_dict(self, grid: Grid) -> dict:
        d = dict(label=self.label, family=self.family, shape=shape_name(self.kernel.order),
                 filter=self.kernel.filter.describe(grid), coupling_power=self.kernel.coupling_power,
                 W_protected=self.W, a_protected=self.a, status=self.status,
                 omega=[None if w is None else [w.real, w.imag] for w in self.omega],
                 R_gamma=self.R, admissible=self.admissible, admissible_acceptance=self.admissible_acc,
                 objective=self.objective, objective_field=self.objective_field,
                 reduction=self.reduction, reduction_field=self.reduction_field,
                 evaluation=self.evaluation, note=self.note)
        return d


@dataclass
class SelectionResult:
    grid: Grid
    plasma: Plasma
    modes: list[ModeAnalysis]
    candidates: list[CandidateResult]
    baseline: CandidateResult
    best: CandidateResult | None
    best_acceptance: CandidateResult | None
    best_by_family: dict
    unconstrained: CandidateResult
    objective: Objective
    metric: str
    settings: dict
    warnings: list[str]
    timing: dict

    def summary(self) -> str:
        from .report import text_summary
        return text_summary(self)

    def to_dict(self) -> dict:
        g = self.grid
        pick = lambda c: None if c is None else c.to_dict(g)
        return dict(
            settings=self.settings, grid=g.describe(), plasma=self.plasma.describe(),
            objective=self.objective.info(), metric=self.metric,
            protected_modes=[m.report() for m in self.modes],
            best=pick(self.best), best_acceptance=pick(self.best_acceptance),
            best_by_family={k: pick(v) for k, v in self.best_by_family.items()},
            unconstrained=pick(self.unconstrained), baseline=pick(self.baseline),
            candidates=[c.to_dict(g) for c in self.candidates],
            warnings=self.warnings, timing=self.timing,
        )


# ==================================================================== selector
class Selector:
    """Build the protected-mode analyses and select the best admissible kernel."""

    def __init__(self, grid: Grid, plasma: Plasma, protected: Sequence[ProtectedMode],
                 objective: Objective, *, tolerance: float = 0.05, acceptance: float = 0.10,
                 frequency_tolerance: float | None = None, metric: str = "density",
                 n_alias: int = 4, time: str = "continuous", dt: float | None = None,
                 n_time_alias: int = 8, coupling_power: int = 2,
                 shapes: Sequence[str | int] | None = None,
                 gaussian: dict | None = None, binomial: dict | None = None,
                 extra_candidates: Sequence[Filter] = (), a_min: float = 0.02,
                 verify_roots: bool = True, log=None):
        self.grid = grid
        self.plasma = plasma
        self.protected = list(protected)
        if not self.protected:
            raise ValueError("at least one protected mode is required")
        self.objective = objective
        self.eps = float(tolerance)
        self.eps_acc = max(float(acceptance), self.eps)
        self.eps_freq = frequency_tolerance
        self.metric = metric
        self.n_alias = int(n_alias)
        self.time = time
        self.dt = dt
        self.n_time_alias = n_time_alias
        self.power = int(coupling_power)
        if self.power not in (1, 2):
            raise ValueError("coupling_power must be 1 (density only) or 2 (matched)")
        self.orders = [grid.order] + [o for o in (shape_order(s) for s in (shapes or []))
                                      if o != grid.order]
        self.gaussian_cfg = gaussian
        self.binomial_cfg = binomial
        self.extra = list(extra_candidates)
        self.a_min = float(a_min)
        self.verify = verify_roots
        self.log = log or (lambda *a, **k: None)
        self.warnings: list[str] = []
        self.analyses: dict[int, list[ModeAnalysis]] = {}

    # ------------------------------------------------------------- dispersion
    def _disp(self, mode, order):
        return ModeDispersion(self.grid, self.plasma, mode, n_alias=self.n_alias, time=self.time,
                              dt=self.dt, n_time_alias=self.n_time_alias, order=order)

    def analyse_modes(self) -> list[ModeAnalysis]:
        base = self.grid.order
        out_base = []
        for spec in self.protected:
            d = self._disp(spec.mode, base)
            roots = rt.find_roots(d, 1.0, guesses=[spec.omega_guess]
                                  if spec.omega_guess is not None else ())
            if not roots:
                raise rt.RootError(f"no dispersion root found for mode {spec.mode}")
            w0 = rt.select_branch(roots, spec.branch, spec.omega_guess, spec.sign)
            if abs(w0.imag) < 1e-10 * (abs(w0) + self.plasma.omega_p_total):
                raise rt.RootError(f"mode {spec.mode}: baseline rate is zero; R_gamma is undefined")
            verified = None
            if self.verify and spec.branch == "most_unstable" and w0.imag > 0:
                verified = rt.verify_most_unstable(d, 1.0, w0.imag)
                if verified is False:
                    self.warnings.append(
                        f"mode {spec.mode}: the argument principle finds a root more unstable "
                        f"than the selected one (gamma={w0.imag:.4g}); check the branch")
            eps = spec.tolerance if spec.tolerance is not None else self.eps
            eps_acc = max(self.eps_acc, eps)
            eps_f = spec.frequency_tolerance if spec.frequency_tolerance is not None else self.eps_freq
            br = rt.continue_branch(d, w0, 1.0, self.a_min)
            ma = ModeAnalysis(spec, base, d, w0, w0, br, eps, eps_acc, eps_f, verified, len(roots))
            out_base.append(ma)
            self.log(f"  mode {tuple(spec.mode)}: omega0 = {w0.real:+.6g}{w0.imag:+.6g}i, "
                     f"branch {br.status}" + (f", a_c = {br.a_marginal:.6f}" if br.a_marginal else ""))
        self.analyses[base] = out_base
        for order in self.orders[1:]:
            lst = []
            for ma0 in out_base:
                d1 = self._disp(ma0.spec.mode, order)
                w = ma0.omega0
                ok = True
                wp = self.plasma.omega_p_total
                for t in np.linspace(0.0, 1.0, 41)[1:]:
                    try:
                        bd = BlendedDispersion(ma0.disp, d1, float(t))
                        w_new = rt._wrap(bd, rt.newton(bd, w, 1.0))
                    except rt.RootError:
                        ok = False
                        break
                    if abs(w_new - w) > 0.1 * (abs(w) + wp):     # jumped to another root
                        ok = False
                        break
                    w = w_new
                if not ok:
                    self.warnings.append(f"shape {shape_name(order)}: branch of mode "
                                         f"{ma0.spec.mode} lost in the shape homotopy")
                    br = rt.Branch(a=[1.0], omega=[w], status="unresolved", a_end=1.0)
                else:
                    br = rt.continue_branch(d1, w, 1.0, self.a_min) if (w.imag > 0) == (ma0.gamma0 > 0) \
                        else rt.Branch(a=[1.0], omega=[w], status="unresolved", a_end=1.0)
                lst.append(ModeAnalysis(ma0.spec, order, d1, ma0.omega0, w, br, ma0.eps,
                                        ma0.eps_acc, ma0.eps_freq, None, 0))
            self.analyses[order] = lst
        return out_base

    # -------------------------------------------------------------- evaluate
    def evaluate(self, kernel: Kernel, note: str = "") -> CandidateResult:
        an = self.analyses[kernel.order]
        a_list, W_list, st_list, w_list, R_list = [], [], [], [], []
        adm = adm_acc = True
        for ma in an:
            W = kernel.filter.at_mode(self.grid, ma.spec.mode)
            a = W ** kernel.coupling_power
            st, w, R = ma.at(a) if a <= 1.0 + 1e-12 else self._above_one(ma, a)
            a_list.append(a); W_list.append(W); st_list.append(st); w_list.append(w); R_list.append(R)
            adm &= ma.admissible_value(w, R, ma.eps)
            adm_acc &= ma.admissible_value(w, R, ma.eps_acc)
            # mirror images merged by auto_protect share the branch; a filter that is
            # not reflection-symmetric must satisfy the constraint at each of them
            for mm in ma.spec.members:
                if tuple(mm) == tuple(ma.spec.mode):
                    continue
                am = kernel.filter.at_mode(self.grid, mm) ** kernel.coupling_power
                if abs(am - a) > 1e-14:
                    stm, wm, Rm = ma.at(am) if am <= 1.0 + 1e-12 else self._above_one(ma, am)
                    adm &= ma.admissible_value(wm, Rm, ma.eps)
                    adm_acc &= ma.admissible_value(wm, Rm, ma.eps_acc)
        res = CandidateResult(kernel, a_list, W_list, st_list, w_list, R_list, bool(adm),
                              bool(adm_acc), note=note)
        if self.objective.supports_order(kernel.order):
            res.objective = self.objective.mse(kernel, self.metric, "select")
            res.objective_field = self.objective.mse(kernel, "field" if self.metric == "density"
                                                     else "density", "select")
        return res

    def _above_one(self, ma: ModeAnalysis, a: float):
        key = ("up", round(a, 15))
        if key in ma._cache:
            return ma._cache[key]
        br = rt.continue_branch(ma.disp, ma.omega1, 1.0, a, stop_on_marginal=False)
        if br.status == "resolved":
            w = br.omega[-1]
            out = (ma._status_name(w), w, w.imag / ma.gamma0)
        else:
            out = ("unresolved", None, None)
        ma._cache[key] = out
        return out

    # ------------------------------------------------------------- families
    def _gaussian_candidates(self, order: int) -> list[CandidateResult]:
        cfg = dict(self.gaussian_cfg or {})
        if cfg.get("enabled", True) is False:
            return []
        dim = self.grid.dim
        h_max = np.broadcast_to(np.asarray(cfg.get("h_max", 3.0), dtype=float), (dim,)).copy()
        axes = cfg.get("axes")
        if axes is not None:
            mask = np.zeros(dim, bool)
            mask[list(axes)] = True
            h_max[~mask] = 0.0
        isotropic = bool(cfg.get("isotropic", False))
        n_grid = int(cfg.get("grid", {1: 400, 2: 61, 3: 21}[dim]))
        an = self.analyses[order]
        # linear constraints  sum_i c_i u_i <= qmax  (u_i = h_i^2)
        rows, qmax = [], []
        feasible_shape = True
        for ma in an:
            a_lo = ma.lower_threshold(ma.eps)
            if a_lo is None:
                feasible_shape = False
                break
            # a = W^power = exp(-(power/2) sum_i (k_i dx_i)^2 u_i) >= a_lo
            rows.append((ma.disp.k * self.grid.dx) ** 2 * (self.power / 2.0))
            qmax.append(-math.log(max(a_lo, 1e-300)))
        Acon = np.array(rows)
        bcon = np.array(qmax)
        results = []

        def kernel_of(u):
            h = np.sqrt(np.maximum(u, 0.0))
            return Kernel(order, Gaussian(tuple(h)), self.power)

        def f(u):
            return self.objective.mse(kernel_of(u), self.metric, "select", cache=False)

        def feasible(u):
            return feasible_shape and np.all(Acon @ u <= bcon * (1 + 1e-12) + 1e-15)

        # parametrisation
        if isotropic:
            # same physical sigma: u_i = (sigma/dx_i)^2 ; variable t = sigma^2
            scale = 1.0 / self.grid.dx**2
            t_max = float(np.min(h_max[h_max > 0] ** 2 / scale[h_max > 0])) if np.any(h_max > 0) else 0.0
            to_u = lambda t: np.where(h_max > 0, t[0] * scale, 0.0)
            box = [(0.0, t_max)]
        else:
            free = np.flatnonzero(h_max > 0)
            to_u = lambda t: _embed(t, free, dim)
            box = [(0.0, float(h_max[i] ** 2)) for i in free]
        nv = len(box)
        if nv == 0:
            return []
        A_t = np.array([Acon @ to_u(np.eye(nv)[j]) for j in range(nv)]).T if feasible_shape \
            else None

        def lattice(upper):
            # lattice in h = sqrt(t) for resolution near zero
            axes_h = [np.linspace(0.0, math.sqrt(max(ub, 0.0)), n_grid) for ub in upper]
            mesh = np.meshgrid(*axes_h, indexing="ij")
            pts = np.stack([m.ravel() ** 2 for m in mesh], axis=-1)
            vals = np.array([f(to_u(t)) for t in pts])
            feas = np.array([feasible(to_u(t)) for t in pts])
            return axes_h, pts, vals, feas

        # unconstrained optimum over the whole box
        _, pts_u, vals_u, _ = lattice([b[1] for b in box])
        best_unc = self._refine(f, to_u, pts_u[np.argsort(vals_u)[:3]], box, None, None)
        results.append(("gaussian_unconstrained", kernel_of(to_u(best_unc))))
        lat_axes, lat_vals, lat_feas = None, None, None
        if feasible_shape:
            # lattice over the bounding box of the admissible polytope (x 1.3)
            upper = []
            for j in range(nv):
                col = A_t[:, j]
                lim = np.min(bcon[col > 0] / col[col > 0]) if np.any(col > 0) else box[j][1]
                upper.append(min(box[j][1], 1.69 * lim))
            lat_axes, pts_c, lat_vals, lat_feas = lattice(upper)
            if np.any(lat_feas):
                starts = pts_c[lat_feas][np.argsort(lat_vals[lat_feas])[:4]]
                best_c = self._refine(f, to_u, starts, box, A_t, bcon)
                if not feasible(to_u(best_c)):
                    best_c = starts[0]
                results.append(("gaussian_constrained", kernel_of(to_u(best_c))))
        if lat_axes is None:
            lat_axes, _, lat_vals, lat_feas = lattice([b[1] for b in box])
        self._gauss_lattice = dict(order=order, h=[a.tolist() for a in lat_axes],
                                   values=lat_vals.reshape([n_grid] * nv).tolist(),
                                   feasible=lat_feas.reshape([n_grid] * nv).tolist(),
                                   isotropic=isotropic,
                                   variable="sigma" if isotropic else "h_cells",
                                   free_axes=None if isotropic else
                                   np.flatnonzero(h_max > 0).tolist(),
                                   constraints=dict(A=Acon.tolist(), b=bcon.tolist()))
        out = []
        for note, ker in results:
            c = self.evaluate(ker, note)
            if note == "gaussian_constrained" and not c.admissible:
                c = self._shrink_to_admissible(ker, c)
            out.append(c)
        return out

    def _refine(self, f, to_u, starts, box, A_t, b):
        best_t, best_v = None, np.inf
        cons = []
        if A_t is not None:
            cons = [dict(type="ineq", fun=lambda t: b - A_t @ t, jac=lambda t: -A_t)]
        for t0 in starts:
            try:
                r = minimize(lambda t: f(to_u(t)), t0, method="SLSQP", bounds=box,
                             constraints=cons, options=dict(ftol=1e-14, maxiter=200))
                t = np.clip(r.x, [lo for lo, _ in box], [hi for _, hi in box])
                t = np.where(t < 1e-14 * np.array([hi for _, hi in box]), 0.0, t)
                if A_t is not None and np.any(A_t @ t > b * (1 + 1e-10) + 1e-15):
                    # pull back onto the feasible set along the ray to the origin
                    s = np.min(b / np.maximum(A_t @ t, 1e-300))
                    t = t * min(1.0, s * (1 - 1e-12))
                v = f(to_u(t))
            except Exception:  # pragma: no cover - fallback to the start point
                t, v = t0, f(to_u(t0))
            if v < best_v:
                best_t, best_v = t, v
            v0 = f(to_u(t0))
            if v0 < best_v:
                best_t, best_v = t0, v0
        return np.asarray(best_t)

    def _shrink_to_admissible(self, ker: Kernel, c: CandidateResult) -> CandidateResult:
        h = np.array(ker.filter.h)
        for _ in range(60):
            h *= 1.0 - 1e-6
            ker = Kernel(ker.order, Gaussian(tuple(h)), ker.coupling_power)
            c = self.evaluate(ker, c.note)
            if c.admissible:
                return c
        self.warnings.append("constrained Gaussian optimum could not be verified as admissible")
        return c

    def _binomial_candidates(self, order: int) -> list[CandidateResult]:
        cfg = dict(self.binomial_cfg or {})
        if cfg.get("enabled", True) is False:
            return []
        ps = [int(x) for x in cfg.get("p", [0, 1, 2, 3, 4])]
        ns = [int(x) for x in cfg.get("n", [1, 2, 3, 4, 6, 8])]
        per_axis = bool(cfg.get("per_axis", False))
        out = []
        if per_axis:
            from itertools import product
            ns0 = sorted(set([0] + ns))
            combos = [c for c in product(ns0, repeat=self.grid.dim) if any(c)]
        else:
            combos = [(n,) for n in ns]
        for p in ps:
            for n in combos:
                out.append(self.evaluate(Kernel(order, Binomial(p, n), self.power)))
        return out

    # ----------------------------------------------------------------- run
    def run(self) -> SelectionResult:
        t0 = time.time()
        self.log("protected modes:")
        self.analyse_modes()
        t1 = time.time()
        base = self.evaluate(Kernel(self.grid.order, NoFilter(), self.power), "baseline")
        cands: list[CandidateResult] = [base]
        for order in self.orders:
            if not self.objective.supports_order(order):
                self.warnings.append(f"objective has no data for shape {shape_name(order)}; skipped")
                continue
            if order != self.grid.order:
                cands.append(self.evaluate(Kernel(order, NoFilter(), self.power), "shape"))
            cands += self._gaussian_candidates(order)
            cands += self._binomial_candidates(order)
            for flt in self.extra:
                cands.append(self.evaluate(Kernel(order, flt, self.power), "user"))
        t2 = time.time()
        b_obj, b_fld = base.objective, base.objective_field
        for c in cands:
            c.reduction = 1.0 - c.objective / b_obj if b_obj > 0 else float("nan")
            c.reduction_field = 1.0 - c.objective_field / b_fld if b_fld > 0 else float("nan")

        def argmin(pool):
            pool = [c for c in pool if np.isfinite(c.objective)]
            return min(pool, key=lambda c: c.objective) if pool else None

        best = argmin([c for c in cands if c.admissible])
        best_acc = argmin([c for c in cands if c.admissible_acc])
        unconstrained = argmin(cands)
        fams = {}
        for fam in sorted({c.family for c in cands}):
            fams[fam] = argmin([c for c in cands if c.family == fam and c.admissible])
        if best is None:
            self.warnings.append("no admissible candidate: even the unfiltered kernel violates "
                                 "the tolerance (only possible for alternative shapes)")
        # independent evaluation (snapshot objective)
        if hasattr(self.objective, "reduction_interval") and not self.objective.held_out:
            self.warnings.append("the evaluation realisations overlap the selection realisations; "
                                 "the reported 'held-out' reductions are not independent")
        if hasattr(self.objective, "reduction_interval"):
            for c in {id(x): x for x in [best, best_acc, unconstrained, *fams.values()]
                      if x is not None}.values():
                for metric in ("density", "field"):
                    c.evaluation[metric] = list(self.objective.reduction_interval(
                        c.kernel, base.kernel, metric, "evaluate"))
        for c in [best, unconstrained]:
            if c is not None and any(s == "unresolved" for s in c.status):
                self.warnings.append(f"{c.label}: unresolved root status for a protected mode")
        self._bound_warnings([best, fams.get("gaussian")])
        self._time_warning()
        timing = dict(modes_s=t1 - t0, candidates_s=t2 - t1, total_s=time.time() - t0)
        settings = dict(tolerance=self.eps, acceptance=self.eps_acc,
                        frequency_tolerance=self.eps_freq, metric=self.metric,
                        n_alias=self.n_alias, time=self.time, dt=self.dt,
                        coupling_power=self.power, shapes=[shape_name(o) for o in self.orders],
                        a_min=self.a_min)
        res = SelectionResult(self.grid, self.plasma, self.analyses[self.grid.order], cands, base,
                              best, best_acc, fams, unconstrained, self.objective, self.metric,
                              settings, self.warnings, timing)
        res.gaussian_lattice = getattr(self, "_gauss_lattice", None)
        res.all_analyses = self.analyses
        return res

    def _bound_warnings(self, picks):
        cfg = dict(self.gaussian_cfg or {})
        h_max = np.broadcast_to(np.asarray(cfg.get("h_max", 3.0), dtype=float), (self.grid.dim,))
        kabs = np.array([np.abs(ma.disp.k) for ma in self.analyses[self.grid.order]])
        done = set()
        for c in picks:
            if c is None or c.family != "gaussian" or c.label in done:
                continue
            done.add(c.label)
            h = np.broadcast_to(np.array(c.kernel.filter.h), (self.grid.dim,))
            for i in range(self.grid.dim):
                if h_max[i] > 0 and h[i] >= h_max[i] * (1 - 1e-6):
                    why = ("no protected mode has k_%d != 0, so the rate constraint does not "
                           "limit this axis" % i) if np.all(kabs[:, i] == 0) else \
                        "the objective prefers wider smoothing than allowed"
                    self.warnings.append(
                        f"{c.label}: width on axis {i} sits at the bound h_max = {h_max[i]:g}; {why}. "
                        "Protect the modes that matter on this axis (or use auto_protect), or set "
                        "candidates.gaussian.h_max deliberately.")

    def _time_warning(self):
        if self.dt is None:
            return
        for ma in self.analyses[self.grid.order]:
            x = abs(ma.omega0) * self.dt
            if self.time == "continuous" and x > 0.1:
                self.warnings.append(f"mode {ma.spec.mode}: |omega| dt = {x:.2f}; consider "
                                     "time_integrator: leapfrog")
        wp_dt = self.plasma.omega_p_total * self.dt
        if wp_dt > 1.0:
            self.warnings.append(f"omega_p dt = {wp_dt:.2f} is large for an explicit scheme")


def _embed(t, free, dim):
    u = np.zeros(dim)
    u[free] = t
    return u


# ============================================================ auto protection
def _canonical(m):
    """Representative of {m, -m} whose first non-zero index is positive."""
    for x in m:
        if x != 0:
            return tuple(m) if x > 0 else tuple(-y for y in m)
    return tuple(m)


def unstable_modes(grid: Grid, plasma: Plasma, *, min_growth: float = 0.0,
                   max_modes: int = 64, n_alias: int = 0, time: str = "continuous",
                   dt: float | None = None, mode_limit: Sequence[int] | None = None,
                   log=None) -> list[tuple[tuple, float]]:
    """Scan the grid modes for growing roots of the baseline kernel.

    Each mode is screened with the argument principle (number of roots with
    Im(omega) above a small threshold), so kinetic instabilities that fluid
    estimates miss are not skipped; the growth rate of every flagged mode is
    then found by Newton iteration.  The principal-harmonic coupling is used
    by default (``n_alias = 0``).  Returns ``[(mode, gamma), ...]`` sorted by
    decreasing growth.
    """
    from itertools import product
    lims = mode_limit or [n // 2 for n in grid.N]
    im_min = max(float(min_growth), 1e-4 * plasma.omega_p_total)
    modes = sorted({_canonical(m) for m in product(*[range(-l, l + 1) for l in lims]) if any(m)})
    out = []
    for m in modes:
        try:
            d = ModeDispersion(grid, plasma, m, n_alias=n_alias, time=time, dt=dt)
        except ValueError:
            continue
        n = rt.count_unstable(d, 1.0, im_min=im_min, max_refine=16)
        if n == 0:
            continue
        roots = [r for r in rt.find_roots(d, 1.0, grid_density=0) if r.imag > im_min]
        if not roots:
            roots = [r for r in rt.find_roots(d, 1.0, grid_density=1) if r.imag > im_min]
        if roots:
            w = max(roots, key=lambda r: r.imag)
            out.append((tuple(m), w))
    # Mirror images (same |m_i| on every axis and the same root) give identical
    # constraints for the symmetric filters used here; keep one representative.
    groups: dict = {}
    for m, w in out:
        key = (tuple(abs(x) for x in m), round(w.imag, 9), round(abs(w.real), 9))
        groups.setdefault(key, []).append(m)
    merged = []
    for key, members in groups.items():
        rep = max(members, key=lambda mm: (sum(x >= 0 for x in mm), mm))
        merged.append((rep, key[1], members))
    merged.sort(key=lambda x: (-x[1], x[0]))
    if log:
        log(f"  auto-protect: {len(out)} growing modes among {len(modes)} "
            f"({len(merged)} distinct up to mirror symmetry)")
    return merged[:max_modes]
