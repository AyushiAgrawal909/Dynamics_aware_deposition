"""Text, JSON and figure reports of a selection."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from .grid import shape_name

COLORS = dict(none="#000000", gaussian="#1f3b73", binomial="#c47f00", custom="#6a3d9a",
              unconstrained="#00897b")


def _fmt_mode(m):
    return "(" + ",".join(str(int(x)) for x in m) + ")"


def _fmt_omega(w):
    return f"{w.real:+.5f}{w.imag:+.5f}i"


def describe_kernel(c, grid) -> list[str]:
    """Human-readable lines for one candidate."""
    f = c.kernel.filter
    lines = []
    shape = shape_name(c.kernel.order).upper()
    coupling = "matched (density and field)" if c.kernel.coupling_power == 2 else "density only"
    if f.family == "none":
        lines.append(f"{shape} deposit/gather, no additional filter")
    elif f.family == "gaussian":
        h = np.broadcast_to(np.array(f.h), (grid.dim,))
        sig = h * grid.dx
        lines.append(f"{shape} + Gaussian filter, {coupling}")
        lines.append("  widths h = sigma/dx = (" + ", ".join(f"{x:.5f}" for x in h) + ") cells")
        lines.append("  sigma         = (" + ", ".join(f"{x:.5g}" for x in sig) + ")")
        lines.append("  transfer W(k) = exp(-1/2 sum_i k_i^2 sigma_i^2)")
    elif f.family == "binomial":
        n = f.n * grid.dim if len(f.n) == 1 else f.n
        kind = "raw binomial" if f.p == 0 else f"compensated binomial, degree p = {f.p}"
        lines.append(f"{shape} + {kind}, passes n = {'x'.join(map(str, n))}, {coupling}")
        from .filters import binomial_stencil
        groups = {}
        for ax, ni in enumerate(n):
            if ni:
                groups.setdefault(ni, []).append(ax)
        for ni, axes in groups.items():
            taps = binomial_stencil(ni, f.p)
            where = "every axis" if len(axes) == grid.dim and grid.dim > 1 else \
                "axis " + ",".join(map(str, axes))
            lines.append(f"  1D stencil on {where} ({taps.size} taps, offsets "
                         f"{-(taps.size // 2)}..{taps.size // 2}, applied separably):")
            lines.append("    " + " ".join(f"{t:.8g}" for t in taps))
    else:
        lines.append(f"{shape} + {f.label}, {coupling}")
    hp = [f.half_power_wavenumber(grid, i) for i in range(grid.dim)]
    if f.family != "none":
        lines.append("  half-power wavenumber / Nyquist per axis: "
                     + ", ".join("none (W > 1/2)" if x is None else f"{x:.3f}" for x in hp))
    return lines


def text_summary(res) -> str:
    g = res.grid
    s = res.settings
    out = []
    out.append("Dynamics-aware deposition/gather kernel selection")
    out.append("=" * 49)
    out.append(f"grid      : {'x'.join(map(str, g.N))} cells, L = {'x'.join(f'{x:g}' for x in g.L)}, "
               f"shape {g.shape.upper()}, {g.poisson} Poisson, {g.gradient} gradient, "
               f"{g.scheme} scheme")
    t = s["time"] + (f", dt = {s['dt']:g}" if s.get("dt") else "")
    out.append(f"time      : {t}; spatial aliases per axis: {s['n_alias']}")
    for sp in res.plasma.species:
        out.append(f"species   : {sp.name}: q={sp.q:g} m={sp.m:g} n={sp.n:g} "
                   f"drift=({', '.join(f'{x:g}' for x in sp.u)}) vth=({', '.join(f'{x:g}' for x in sp.v)})"
                   + ("" if sp.mobile else " [immobile]"))
    info = res.objective.info()
    if info["kind"] == "model":
        out.append(f"objective : model {res.metric} MSE, reference-free (uncorrelated markers, "
                   f"PPC {info['ppc']}, {len(info['signal'])} signal mode(s))")
    else:
        out.append(f"objective : {res.metric} MSE against paired reference snapshots "
                   f"({info['realisations']} realisations x {info['times']} times; "
                   f"select {info['select']}, evaluate {info['evaluate']})")
    out.append(f"tolerance : |R_gamma - 1| <= {s['tolerance']:g} (acceptance {s['acceptance']:g})"
               + (f", frequency {s['frequency_tolerance']:g}" if s.get("frequency_tolerance") else ""))
    out.append("")
    out.append("Protected modes (baseline kernel, a = W^2 = 1)")
    out.append(f"  {'mode':<10}{'omega_0':<26}{'a_lo(tol)':>11}{'a_lo(acc)':>11}{'a_c':>11}  check")
    for ma in res.modes:
        r = ma.report()
        ac = "-" if r["a_marginal"] is None else f"{r['a_marginal']:.6f}"
        lo = lambda x: "none" if x is None else f"{x:.6f}"
        chk = {True: "most unstable", False: "NOT most unstable", None: "-"}[ma.verified]
        out.append(f"  {_fmt_mode(ma.spec.mode):<10}{_fmt_omega(ma.omega0):<26}"
                   f"{lo(r['a_lo_tolerance']):>11}{lo(r['a_lo_acceptance']):>11}{ac:>11}  {chk}"
                   + ("  [auto" + (f", x{len(ma.spec.members)} mirror" if len(ma.spec.members) > 1
                                   else "") + "]" if ma.spec.source == "auto" else ""))
    out.append("")

    def cand_block(title, c):
        if c is None:
            out.append(f"{title}: none")
            return
        out.append(f"{title}: {c.label}")
        for line in describe_kernel(c, g):
            out.append("  " + line)
        for ma, a, W, st, R in zip(res.modes, c.a, c.W, c.status, c.R):
            Rtxt = "undefined" if R is None else f"{R:.5f}"
            out.append(f"    mode {_fmt_mode(ma.spec.mode)}: W = {W:.6f}, W^2 = {a:.6f}, "
                       f"status {st}, R_gamma = {Rtxt}")
        out.append(f"    {res.metric} MSE reduction (selection data): {100 * c.reduction:.2f}%"
                   + (f"; {'field' if res.metric == 'density' else 'density'}: "
                      f"{100 * c.reduction_field:.2f}%" if np.isfinite(c.reduction_field) else ""))
        for metric, v in c.evaluation.items():
            if v:
                N, lo_, hi_ = v
                pct = lambda x: 100 * (1 - 1 / x) if x and np.isfinite(x) else float("nan")
                tag = "held-out" if info.get("held_out", True) else "evaluation (not held out)"
                out.append(f"    {tag} {metric} MSE reduction: {pct(N):.2f}% "
                           f"[{pct(lo_):.2f}, {pct(hi_):.2f}]  (N = {N:.4f})")

    cand_block("RECOMMENDED KERNEL", res.best)
    out.append("")
    out.append("Best admissible kernel per family")
    for fam, c in res.best_by_family.items():
        if fam == "none":
            continue
        if c is None:
            out.append(f"  {fam}: no admissible candidate")
            continue
        k = int(np.argmax([abs(R - 1) for R in c.R]))
        mtxt = f" at {_fmt_mode(res.modes[k].spec.mode)}" if len(c.R) > 1 else ""
        out.append(f"  {fam:<9}: {c.label:<36} reduction {100 * c.reduction:6.2f}%   "
                   f"worst R_gamma {c.R[k]:.4f}{mtxt}")
    out.append("")
    u = res.unconstrained
    if u is not None:
        sts = ", ".join(u.status)
        out.append(f"Unconstrained {res.metric}-error optimum (for comparison): {u.label}")
        out.append(f"    reduction {100 * u.reduction:.2f}%, protected-mode status: {sts}"
                   + ("  -> violates the rate tolerance" if not u.admissible else ""))
    if res.best_acceptance is not None and res.best is not None and \
            res.best_acceptance.label != res.best.label:
        b = res.best_acceptance
        out.append(f"At the acceptance tolerance {s['acceptance']:g} the choice would be "
                   f"{b.label} (reduction {100 * b.reduction:.2f}%)")
    n_adm = sum(c.admissible for c in res.candidates)
    out.append("")
    out.append(f"{len(res.candidates)} candidates evaluated, {n_adm} admissible; "
               f"time {res.timing['total_s']:.1f} s")
    if info["kind"] == "model":
        out.append("Note: the model objective is reference-free and ignores marker correlations "
                   "(quiet starts, shielding); confirm with snapshot data where possible.")
    for w in res.warnings:
        out.append(f"WARNING: {w}")
    return "\n".join(out)


def write_json(res, path):
    from .config import to_plain
    d = to_plain(res.to_dict())
    lat = getattr(res, "gaussian_lattice", None)
    if lat is not None:
        d["gaussian_lattice"] = to_plain(lat)
    Path(path).write_text(json.dumps(d, indent=1))


# ======================================================================= plots
def _style():
    import matplotlib as mpl
    mpl.rcParams.update({
        "font.family": "serif", "font.serif": ["DejaVu Serif"], "mathtext.fontset": "dejavuserif",
        "font.size": 8.5, "axes.labelsize": 9, "axes.linewidth": 0.7,
        "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True, "ytick.right": True,
        "xtick.major.size": 3.5, "ytick.major.size": 3.5, "xtick.minor.size": 2,
        "ytick.minor.size": 2, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "legend.frameon": False, "legend.fontsize": 7.5, "lines.linewidth": 1.1,
        "savefig.dpi": 300,
    })


def _dense_branch(ma, n=240):
    """R_gamma on a fine grid of a for plotting (exact solves along the branch)."""
    from . import roots as rt
    av, wv = ma.branch.sorted()
    lo = float(av.min())
    if ma.branch.status == "marginal" and ma.branch.a_marginal is not None:
        lo = ma.branch.a_marginal
    grid = np.unique(np.concatenate([np.linspace(lo, 1.0, n), av]))
    R = []
    for a in grid:
        st, w, r = ma.at(float(a))
        R.append(np.nan if r is None else r)
    R = np.array(R)
    if ma.branch.a_marginal is not None:
        R[np.isclose(grid, ma.branch.a_marginal)] = 0.0
    del rt
    return grid, R


def plot(res, path, *, show_mode: int = 0):
    """Four-panel summary: rate ratio vs coupling, error vs rate ratio,
    filter transfer, and the Gaussian error map with its admissible region."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    _style()
    g = res.grid
    tol, acc = res.settings["tolerance"], res.settings["acceptance"]
    fig, axs = plt.subplots(2, 2, figsize=(7.0, 5.6))
    j = min(show_mode, len(res.modes) - 1)
    line_styles = ["-", "--", "-.", ":", (0, (5, 1)), (0, (3, 1, 1, 1))]

    # (a) R_gamma against the round-trip transfer
    ax = axs[0, 0]
    lows = []
    for i, ma in enumerate(res.modes[:6]):
        a_d, R_d = _dense_branch(ma)
        ax.plot(a_d, R_d, color="k", ls=line_styles[i % 6], lw=1.0,
                label=r"$\mathbf{m}=" + _fmt_mode(ma.spec.mode) + "$")
        if ma.branch.a_marginal is not None:
            ax.plot([ma.branch.a_marginal], [0.0], marker="o", ms=3.5, mfc="white", mec="k",
                    mew=0.8, zorder=5)
        lows.append(np.nanmin(a_d))
    lo_x = min(lows)
    span = 1.0 - lo_x
    ax.axhspan(1 - tol, 1 + tol, color="0.88", lw=0, zorder=0)
    for y in (1 - acc, 1 + acc):
        ax.axhline(y, color="0.4", lw=0.7, ls=":")
    for c in res.candidates:
        if c.R[j] is not None and c.family in ("binomial", "gaussian", "custom") and c.note in ("", "user"):
            ax.plot(c.a[j], c.R[j], "o", ms=2.6, color=COLORS.get(c.family, "0.5"), mew=0, zorder=4)
    if res.best is not None and res.best.R[j] is not None:
        ax.plot(res.best.a[j], res.best.R[j], marker="o", ms=7, mfc="none", mec="k", mew=1.0, zorder=6)
    gb = res.best_by_family.get("gaussian")
    if gb is not None and gb.R[j] is not None:
        ax.plot(gb.a[j], gb.R[j], marker="D", ms=4, color=COLORS["gaussian"], zorder=6)
    ax.set_xlim(max(0.0, lo_x - 0.06 * span), 1.0 + 0.02 * span)
    rmax = max([1.0 + 2 * acc] + [np.nanmax(_dense_branch(ma)[1]) + 0.05 for ma in res.modes[:6]])
    rmin = min([1.0 - 2 * acc] + [np.nanmin(_dense_branch(ma)[1]) - 0.05 for ma in res.modes[:6]])
    ax.set_ylim(min(rmin, -0.05) if res.modes[0].gamma0 > 0 else rmin, rmax)
    ax.set_xlabel(r"round-trip transfer $a=W^2(\mathbf{k})$")
    ax.set_ylabel(r"$R_\gamma=\gamma(a)/\gamma_0$")
    if len(res.modes) > 1:
        ax.legend(loc="best", fontsize=6.5)

    # (b) objective against the least favourable rate ratio
    ax = axs[0, 1]
    base = res.baseline.objective

    def worst_R(c):
        if any(R is None for R in c.R):
            return None
        return c.R[int(np.argmax([abs(R - 1) for R in c.R]))]

    strip_x = -0.12
    xs = [worst_R(c) for c in res.candidates if worst_R(c) is not None]
    x_hi = max([1.0 + 1.5 * acc] + [x + 0.05 for x in xs])
    for c in res.candidates:
        if not np.isfinite(c.objective):
            continue
        y = c.objective / base
        R = worst_R(c)
        col = COLORS.get(c.family, "0.5")
        marker = {"gaussian": "D", "binomial": "o", "none": "s"}.get(c.family, "^")
        ax.plot(strip_x if R is None else R, y, marker=marker, ls="none",
                ms=2.8 if marker == "D" else 3.2, mfc=col if c.admissible else "white", mec=col, mew=0.7)
    ax.axvspan(1 - tol, 1 + tol, color="0.88", lw=0, zorder=0)
    ax.axvline(1 - acc, color="0.4", lw=0.7, ls=":")
    if x_hi > 1 + acc:
        ax.axvline(1 + acc, color="0.4", lw=0.7, ls=":")
    ax.axvline(strip_x + 0.06, color="0.6", lw=0.5)
    for c, mk in ((res.best, "o"), (res.unconstrained, "s")):
        if c is not None and np.isfinite(c.objective):
            R = worst_R(c)
            ax.plot(strip_x if R is None else R, c.objective / base, marker=mk, ms=8, mfc="none",
                    mec="k", mew=1.0)
    ax.set_yscale("log")
    ax.set_xlim(strip_x - 0.06, x_hi)
    ticks = [t for t in ax.get_xticks() if 0 <= t <= x_hi]
    ax.set_xticks([strip_x] + ticks)
    ax.set_xticklabels(["none"] + [f"{t:g}" for t in ticks])
    ax.set_xlabel(r"$R_\gamma$ (least favourable protected mode)")
    ax.set_ylabel(r"$\widehat{\mathcal{E}}(\theta)\,/\,\widehat{\mathcal{E}}(\mathrm{baseline})$")
    handles = [Line2D([], [], marker="D", ls="none", color=COLORS["gaussian"], ms=3, label="Gaussian"),
               Line2D([], [], marker="o", ls="none", color=COLORS["binomial"], ms=3.5, label="binomial"),
               Line2D([], [], marker="o", ls="none", mfc="white", mec="0.3", ms=3.5, label="inadmissible"),
               Line2D([], [], marker="o", ls="none", mfc="none", mec="k", ms=7, label="selected"),
               Line2D([], [], marker="s", ls="none", mfc="none", mec="k", ms=7,
                      label="unconstrained optimum")]
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.2, 0.0), handletextpad=0.3)

    # (c) transfer along the axes
    ax = axs[1, 0]
    frac = np.linspace(0.0, 1.0, 400)
    seen = []
    entries = [(res.best, "selected", COLORS.get(res.best.family, "k") if res.best else "k")]
    if gb is not None:
        entries.append((gb, "best Gaussian", COLORS["gaussian"]))
    entries.append((res.unconstrained, "unconstrained", COLORS["unconstrained"]))
    for c, name, col in entries:
        if c is None or c.label in seen:
            continue
        seen.append(c.label)
        curves = []
        for ax_i in range(g.dim):
            k = np.zeros((frac.size, g.dim))
            k[:, ax_i] = frac * math.pi / g.dx[ax_i]
            curves.append(np.asarray(c.kernel.filter.transfer(k, g)))
        same = all(np.allclose(cv, curves[0], atol=1e-6) for cv in curves)
        for ax_i, cv in enumerate(curves[:1] if same else curves):
            lab = name if (same or g.dim == 1) else f"{name}, axis {ax_i}"
            ax.plot(frac, cv, color=col, ls=line_styles[ax_i % 3], label=lab)
    for ma in res.modes[:6]:
        for ax_i in range(g.dim):
            x = abs(ma.disp.k[ax_i]) * g.dx[ax_i] / math.pi
            if x > 0:
                ax.axvline(x, color="0.6", lw=0.6, ls=line_styles[ax_i % 3], zorder=0)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlabel(r"$k_i\Delta_i/\pi$")
    ax.set_ylabel(r"filter transfer $W$ along axis $i$")
    ax.legend(loc="center right", fontsize=6.5)

    # (d) Gaussian error map
    ax = axs[1, 1]
    lat = getattr(res, "gaussian_lattice", None)
    if lat is not None and not lat["isotropic"] and len(lat["h"]) == 2:
        fa = lat["free_axes"]
        hx, hy = np.array(lat["h"][0]), np.array(lat["h"][1])
        V = np.array(lat["values"]) / base
        A = np.array(lat["constraints"]["A"])
        b = np.array(lat["constraints"]["b"])
        fx, fy = np.linspace(hx[0], hx[-1], 500), np.linspace(hy[0], hy[-1], 500)
        FX, FY = np.meshgrid(fx, fy, indexing="ij")
        U = np.zeros(FX.shape + (g.dim,))
        U[..., fa[0]], U[..., fa[1]] = FX**2, FY**2
        F = np.all(U @ A.T <= b, axis=-1).astype(float) if A.size else np.ones(FX.shape)
        ax.contourf(fx, fy, F.T, levels=[0.5, 1.5], colors=["0.88"])
        ax.contour(fx, fy, F.T, levels=[0.5], colors="k", linewidths=0.9)
        levels = np.unique(np.round(np.linspace(V.min(), V.max(), 9), 4))
        cs = ax.contour(hx, hy, V.T, levels=levels, colors="0.35", linewidths=0.5)
        ax.clabel(cs, fontsize=5.5, fmt="%.3g")
        if gb is not None:
            h = np.array(gb.kernel.filter.h)
            ax.plot(h[fa[0]], h[fa[1]], marker="D", ms=5, color=COLORS["gaussian"], clip_on=False,
                    zorder=6)
        ax.set_xlabel(rf"$h_{fa[0]}=\sigma_{fa[0]}/\Delta_{fa[0]}$")
        ax.set_ylabel(rf"$h_{fa[1]}=\sigma_{fa[1]}/\Delta_{fa[1]}$")
        ax.set_xlim(hx[0], hx[-1])
        ax.set_ylim(hy[0], hy[-1])
    elif lat is not None and len(lat["h"]) == 1:
        h = np.array(lat["h"][0])
        V = np.array(lat["values"]) / base
        F = np.array(lat["feasible"], dtype=bool)
        ax.plot(h, V, color=COLORS["gaussian"])
        if F.any():
            ax.axvspan(h[F].min(), h[F].max(), color="0.88", lw=0, zorder=0)
        if gb is not None:
            hv = gb.kernel.filter.h
            x = hv[0] if not lat["isotropic"] else hv[0] * g.dx[0]
            ax.plot(x, gb.objective / base, marker="D", ms=5, color=COLORS["gaussian"])
        ax.set_xlabel(r"isotropic $\sigma$" if lat["isotropic"] else r"$h=\sigma/\Delta$")
        ax.set_ylabel(r"$\widehat{\mathcal{E}}/\widehat{\mathcal{E}}(\mathrm{baseline})$")
        ax.set_xlim(h[0], h[-1])
    else:
        ax.axis("off")
    for a_, lab in zip(axs.ravel(), "abcd"):
        if a_.axison:
            a_.text(-0.17, 1.03, f"({lab})", transform=a_.transAxes, va="bottom", ha="left", fontsize=9)
    fig.tight_layout(h_pad=1.6)
    fig.savefig(path)
    plt.close(fig)
    return path
