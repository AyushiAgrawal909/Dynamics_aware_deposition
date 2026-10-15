"""Command-line interface: ``dabs <command> ...``."""

from __future__ import annotations

import argparse
import json
import sys
from importlib import resources
from pathlib import Path

import numpy as np

from . import __version__

EXAMPLES = {
    "two_stream": "Oblique two-stream benchmark of the paper (v_b = 0.80, mode (12,1)).",
    "landau": "Weak Landau damping, k lambda_D = 0.4 (damped protected mode).",
    "beam_plasma": "Warm electron beam in a Maxwellian plasma with mobile ions, 1D.",
    "ion_acoustic": "Ion-acoustic wave, T_e/T_i = 10, protected damped ion mode.",
    "two_stream_3d": "Symmetric two-stream instability on a 3D mesh, auto-protected modes.",
    "two_stream_snapshots": "Paper case with the reference-assisted snapshot objective.",
}


def _log(quiet):
    return (lambda *a, **k: None) if quiet else (lambda *a, **k: print(*a, **k, file=sys.stderr))


def cmd_select(args):
    from .config import build_selector, load
    from .report import plot, text_summary, write_json
    cfg = load(args.config)
    if args.tolerance is not None:
        cfg["tolerance"] = args.tolerance
    if args.acceptance is not None:
        cfg["acceptance"] = args.acceptance
    log = _log(args.quiet)
    log(f"dabs {__version__}: {args.config}")
    res = build_selector(cfg, log=log).run()
    text = text_summary(res)
    print(text)
    out = Path(args.out) if args.out else None
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)
        (out / "summary.txt").write_text(text + "\n")
        write_json(res, out / "selection.json")
        if not args.no_plot:
            try:
                plot(res, out / "selection.pdf")
                plot(res, out / "selection.png")
            except Exception as exc:  # pragma: no cover
                print(f"WARNING: plotting failed: {exc}", file=sys.stderr)
        if res.best is not None:
            W = res.best.kernel.filter.transfer_grid(res.grid)
            np.save(out / "best_transfer_fftn.npy", W)
        log(f"wrote {out}/summary.txt, selection.json"
            + ("" if args.no_plot else ", selection.pdf/.png") + ", best_transfer_fftn.npy")
    return 0 if res.best is not None else 2


def cmd_roots(args):
    from .config import build_grid, build_plasma, load
    from .dispersion import ModeDispersion
    from . import roots as rt
    cfg = load(args.config)
    grid = build_grid(cfg)
    plasma = build_plasma(cfg, grid.dim)
    t = cfg.get("time") or {}
    modes = [tuple(args.mode)] if args.mode else [tuple(p["mode"]) for p in cfg.get("protected_modes", [])]
    n_alias = int((cfg.get("dispersion") or {}).get("aliases", 4))
    for m in modes:
        d = ModeDispersion(grid, plasma, m, n_alias=n_alias, time=t.get("integrator", "continuous"),
                           dt=t.get("dt"), n_time_alias=int(t.get("temporal_aliases", 8)))
        roots = rt.find_roots(d, args.a)
        print(f"mode {m}  k = {np.round(d.k, 6).tolist()}  a = W^2 = {args.a:g}")
        for r in roots[: args.max]:
            print(f"   omega = {r.real:+.8f} {r.imag:+.8f}i")
        if roots and roots[0].imag > 0:
            n = rt.count_unstable(d, args.a, im_min=1e-6 * abs(roots[0]))
            print(f"   argument principle: {n} root(s) with Im(omega) > 0")
    return 0


def cmd_scan(args):
    from .config import build_selector, load
    cfg = load(args.config)
    sel = build_selector(cfg, log=_log(args.quiet))
    sel.analyse_modes()
    for ma in sel.analyses[sel.grid.order]:
        av, wv = ma.branch.sorted()
        print(f"# mode {ma.spec.mode}: omega0 = {ma.omega0}, branch status {ma.branch.status}, "
              f"a_c = {ma.branch.a_marginal}")
        print("# a            omega_r          gamma            R_gamma")
        for a, w in zip(av[::-1], wv[::-1]):
            print(f"{a:.8f}  {w.real:+.8e}  {w.imag:+.8e}  {w.imag / ma.gamma0:.8f}")
        for eps in sorted({ma.eps, ma.eps_acc}):
            print(f"# a_lo(eps={eps:g}) = {ma.lower_threshold(eps)}")
    return 0


def cmd_example(args):
    if args.name is None:
        for k, v in EXAMPLES.items():
            print(f"{k:<22} {v}")
        return 0
    if args.name not in EXAMPLES:
        print(f"unknown example '{args.name}'; choose from {', '.join(EXAMPLES)}", file=sys.stderr)
        return 1
    text = resources.files("dabs").joinpath("configs", f"{args.name}.yaml").read_text()
    if args.out:
        Path(args.out).write_text(text)
        print(f"wrote {args.out}", file=sys.stderr)
    else:
        print(text, end="")
    return 0


def cmd_stencil(args):
    from .filters import binomial_axis_transfer, binomial_stencil
    taps = binomial_stencil(args.n, args.p)
    half = taps.size // 2
    print(f"compensated binomial p={args.p}, n={args.n}: {taps.size} taps")
    for off, t in zip(range(-half, half + 1), taps):
        print(f"  {off:+3d}  {t:+.12f}")
    print(f"  sum = {taps.sum():.12f}")
    for kd in (0.25, 0.5, 1.0):
        s = np.sin(0.5 * kd * np.pi) ** 2
        print(f"  W(k dx = {kd:g} pi) = {float(binomial_axis_transfer(s, args.n, args.p)):.8f}")
    return 0


def cmd_stats(args):
    from .grid import Grid
    from .objectives import SnapshotObjective
    data = np.load(args.snapshots)
    rho, ref = data[args.rho_key], data[args.ref_key]
    dim = len(args.L)
    if rho.ndim < dim:
        print(f"arrays of shape {rho.shape} cannot hold {dim}D meshes", file=sys.stderr)
        return 1
    N = rho.shape[-dim:]
    extra = {}
    for item in args.extra or []:
        name, key = item.split("=")
        extra[name] = data[key]
    grid = Grid(list(N), args.L, shape=args.shape)
    obj = SnapshotObjective(grid, rho, ref, extra_shapes=extra)
    obj.save_statistics(args.out)
    print(f"wrote {args.out}: {obj.n_real} realisations x {obj.n_times} times, mesh {N}",
          file=sys.stderr)
    return 0


def cmd_validate(args):
    from .validation import run_validation
    ok = run_validation(verbose=True)
    return 0 if ok else 1


def main(argv=None):
    p = argparse.ArgumentParser(prog="dabs", description="Dynamics-aware selection of PIC "
                                "deposition/gather filters and bandwidths.")
    p.add_argument("--version", action="version", version=f"dabs {__version__}")
    p.add_argument("--debug", action="store_true", help="show full tracebacks on errors")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("select", help="select the best admissible kernel for a configuration")
    s.add_argument("config")
    s.add_argument("-o", "--out", help="output directory (summary, JSON, plots)")
    s.add_argument("--tolerance", type=float, help="override the rate tolerance epsilon")
    s.add_argument("--acceptance", type=float, help="override the acceptance tolerance")
    s.add_argument("--no-plot", action="store_true")
    s.add_argument("-q", "--quiet", action="store_true")
    s.set_defaults(func=cmd_select)

    r = sub.add_parser("roots", help="list dispersion roots of the configured modes")
    r.add_argument("config")
    r.add_argument("--mode", type=int, nargs="+")
    r.add_argument("--a", type=float, default=1.0, help="round-trip transfer W^2 (default 1)")
    r.add_argument("--max", type=int, default=8)
    r.set_defaults(func=cmd_roots)

    c = sub.add_parser("scan", help="tabulate R_gamma against the coupling a for protected modes")
    c.add_argument("config")
    c.add_argument("-q", "--quiet", action="store_true")
    c.set_defaults(func=cmd_scan)

    e = sub.add_parser("example", help="list or write example configurations")
    e.add_argument("name", nargs="?")
    e.add_argument("-o", "--out")
    e.set_defaults(func=cmd_example)

    t = sub.add_parser("stencil", help="real-space taps of a compensated binomial filter")
    t.add_argument("--p", type=int, required=True, help="compensation degree")
    t.add_argument("--n", type=int, required=True, help="number of passes")
    t.set_defaults(func=cmd_stencil)

    st = sub.add_parser("stats", help="reduce paired snapshots to compact spectral statistics")
    st.add_argument("snapshots", help=".npz with rho and rho_ref of shape (realisations, times, *N)")
    st.add_argument("--L", type=float, nargs="+", required=True, help="domain lengths")
    st.add_argument("--shape", default="cic", help="shape used to deposit rho and rho_ref")
    st.add_argument("--rho-key", default="rho")
    st.add_argument("--ref-key", default="rho_ref")
    st.add_argument("--extra", nargs="*", help="other deposit shapes as name=key, e.g. tsc=rho_tsc")
    st.add_argument("-o", "--out", required=True)
    st.set_defaults(func=cmd_stats)

    v = sub.add_parser("validate", help="reproduce reference values from the paper and theory")
    v.set_defaults(func=cmd_validate)

    args = p.parse_args(argv)
    from .config import ConfigError
    from .roots import RootError
    try:
        return args.func(args)
    except BrokenPipeError:  # pragma: no cover - output piped into head etc.
        return 0
    except (ConfigError, RootError, ValueError, FileNotFoundError, KeyError) as exc:
        if getattr(args, "debug", False):
            raise
        print(f"dabs: error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
