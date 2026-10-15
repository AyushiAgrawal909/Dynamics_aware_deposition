"""Configuration files (YAML or JSON) for the command-line interface.

See ``docs/configuration.md`` for every key; ``dabs example`` writes
commented examples.
"""

from __future__ import annotations

import difflib
import json
from pathlib import Path
from typing import Any

import numpy as np

from .filters import Binomial, Gaussian
from .grid import Grid
from .objectives import ModelObjective, SnapshotObjective
from .plasma import Plasma, Species
from .selection import ProtectedMode, Selector, unstable_modes

TOP_KEYS = {"title", "grid", "time", "species", "eps0", "protected_modes", "auto_protect",
            "tolerance", "acceptance", "frequency_tolerance", "dispersion", "filter",
            "candidates", "objective", "output"}
SECTION_KEYS = {
    "grid": {"N", "L", "dx", "shape", "poisson", "gradient", "scheme", "fft"},
    "time": {"integrator", "dt", "temporal_aliases"},
    "dispersion": {"aliases", "a_min", "verify_roots"},
    "filter": {"coupling"},
    "candidates": {"shapes", "gaussian", "binomial", "extra"},
    "gaussian": {"enabled", "h_max", "axes", "isotropic", "grid"},
    "binomial": {"enabled", "p", "n", "per_axis"},
    "objective": {"mode", "metric", "ppc", "signal", "snapshots", "statistics", "rho_key",
                  "ref_key", "select", "evaluate", "extra_shapes"},
    "auto_protect": {"enabled", "min_growth", "min_growth_fraction", "max_modes", "aliases",
                     "branch"},
    "species": {"name", "q", "m", "n", "drift", "vth", "T", "ppc", "mobile"},
    "protected_modes": {"mode", "branch", "omega_guess", "sign", "tolerance",
                        "frequency_tolerance"},
    "output": {"directory", "plots", "json"},
}


class ConfigError(ValueError):
    pass


def _check_keys(section: str, given: dict, allowed: set):
    for key in given:
        if key not in allowed:
            hint = difflib.get_close_matches(key, sorted(allowed), n=1)
            raise ConfigError(f"unknown key '{key}' in {section}"
                              + (f" (did you mean '{hint[0]}'?)" if hint else ""))


def load(path: str | Path) -> dict:
    path = Path(path)
    text = path.read_text()
    if path.suffix.lower() in (".yaml", ".yml"):
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover
            raise ConfigError("reading YAML needs PyYAML (pip install pyyaml); "
                              "or use a JSON config") from exc
        cfg = yaml.safe_load(text)
    else:
        cfg = json.loads(text)
    if not isinstance(cfg, dict):
        raise ConfigError("the configuration must be a mapping")
    cfg.setdefault("_base_dir", str(path.parent.resolve()))
    return cfg


def _complex(x):
    if x is None:
        return None
    if isinstance(x, (list, tuple)):
        return complex(float(x[0]), float(x[1]))
    return complex(x)


def build_grid(cfg: dict) -> Grid:
    g = dict(cfg.get("grid") or {})
    _check_keys("grid", g, SECTION_KEYS["grid"])
    if "N" not in g:
        raise ConfigError("grid.N is required")
    N = np.atleast_1d(g["N"]).astype(int)
    if "L" in g:
        L = np.atleast_1d(g["L"]).astype(float)
    elif "dx" in g:
        L = N * np.broadcast_to(np.atleast_1d(g["dx"]).astype(float), N.shape)
    else:
        raise ConfigError("grid needs L (domain lengths) or dx (cell sizes)")
    return Grid(N.tolist(), L.tolist(), shape=g.get("shape", "cic"),
                poisson=g.get("poisson", "spectral"), gradient=g.get("gradient", "spectral"),
                scheme=g.get("scheme", "momentum"), fft=g.get("fft", "rfft"))


def build_plasma(cfg: dict, dim: int) -> Plasma:
    lst = cfg.get("species")
    if not lst:
        raise ConfigError("at least one species is required")
    species = []
    for i, s in enumerate(lst):
        _check_keys(f"species[{i}]", s, SECTION_KEYS["species"])
        for key in ("q", "m", "n"):
            if key not in s:
                raise ConfigError(f"species[{i}] needs '{key}'")
        if "vth" in s and "T" in s:
            raise ConfigError(f"species[{i}]: give vth or T, not both")
        vth = s.get("vth", 0.0)
        if "T" in s:
            vth = np.sqrt(np.atleast_1d(np.asarray(s["T"], dtype=float)) / float(s["m"])).tolist()
        species.append(Species(q=float(s["q"]), m=float(s["m"]), n=float(s["n"]),
                               drift=s.get("drift", 0.0), vth=vth, name=s.get("name", ""),
                               ppc=s.get("ppc"), mobile=bool(s.get("mobile", True))))
    return Plasma(species, eps0=float(cfg.get("eps0", 1.0))).resolve(dim)


def build_objective(cfg: dict, grid: Grid, plasma: Plasma):
    o = dict(cfg.get("objective") or {})
    _check_keys("objective", o, SECTION_KEYS["objective"])
    mode = o.get("mode", "model")
    metric = o.get("metric", "density")
    if mode == "model":
        return ModelObjective(grid, plasma, ppc=o.get("ppc"), signal=o.get("signal", [])), metric
    if mode == "snapshots":
        if "statistics" in o:
            path = Path(o["statistics"])
            if not path.is_absolute():
                path = Path(cfg.get("_base_dir", ".")) / path
            return SnapshotObjective.from_statistics(grid, path, select=o.get("select"),
                                                     evaluate=o.get("evaluate")), metric
        if "snapshots" not in o:
            raise ConfigError("snapshot mode needs objective.snapshots (paired meshes, .npz) or "
                              "objective.statistics (a file written by 'dabs stats')")
        path = Path(o["snapshots"])
        if not path.is_absolute():
            path = Path(cfg.get("_base_dir", ".")) / path
        data = np.load(path)
        rk, fk = o.get("rho_key", "rho"), o.get("ref_key", "rho_ref")
        for key in (rk, fk):
            if key not in data:
                raise ConfigError(f"'{key}' not found in {path}; keys: {list(data.keys())}")
        extra = {name: data[key] for name, key in (o.get("extra_shapes") or {}).items()}
        obj = SnapshotObjective(grid, data[rk], data[fk], select=o.get("select"),
                                evaluate=o.get("evaluate"), extra_shapes=extra)
        return obj, metric
    raise ConfigError("objective.mode must be 'model' or 'snapshots'")


def build_protected(cfg: dict, grid: Grid, plasma: Plasma, time_kw: dict, log=None):
    specs = []
    for i, p in enumerate(cfg.get("protected_modes") or []):
        _check_keys(f"protected_modes[{i}]", p, SECTION_KEYS["protected_modes"])
        if "mode" not in p:
            raise ConfigError(f"protected_modes[{i}] needs 'mode'")
        specs.append(ProtectedMode(mode=tuple(int(x) for x in np.atleast_1d(p["mode"])),
                                   branch=p.get("branch", "most_unstable"),
                                   omega_guess=_complex(p.get("omega_guess")),
                                   sign=int(p.get("sign", 0)), tolerance=p.get("tolerance"),
                                   frequency_tolerance=p.get("frequency_tolerance")))
    ap = cfg.get("auto_protect")
    if ap and ap.get("enabled", True):
        _check_keys("auto_protect", ap, SECTION_KEYS["auto_protect"])
        found = unstable_modes(grid, plasma, min_growth=float(ap.get("min_growth", 0.0)),
                               max_modes=int(ap.get("max_modes", 64)),
                               n_alias=int(ap.get("aliases", 0)), log=log, **time_kw)
        if found and "min_growth_fraction" in ap:
            gmax = found[0][1]
            found = [f for f in found if f[1] >= float(ap["min_growth_fraction"]) * gmax]
        have = {tuple(s.mode) for s in specs} | {tuple(-x for x in s.mode) for s in specs}
        for m, g, members in found:
            if not any(tuple(mm) in have for mm in members):
                specs.append(ProtectedMode(mode=m, branch=ap.get("branch", "most_unstable"),
                                           source="auto", members=tuple(members)))
    if not specs:
        if ap and ap.get("enabled", True):
            raise ConfigError("auto_protect found no growing mode for the unfiltered kernel; list "
                              "the modes to protect (e.g. damped ones) under protected_modes")
        raise ConfigError("no protected modes: list protected_modes or enable auto_protect")
    return specs


def build_selector(cfg: dict, log=None) -> Selector:
    _check_keys("the configuration", {k: v for k, v in cfg.items() if not k.startswith("_")},
                TOP_KEYS)
    for sec in ("time", "dispersion", "filter", "candidates"):
        if cfg.get(sec) is not None:
            _check_keys(sec, cfg[sec], SECTION_KEYS[sec])
    grid = build_grid(cfg)
    plasma = build_plasma(cfg, grid.dim)
    t = dict(cfg.get("time") or {})
    integ = t.get("integrator", "continuous")
    dt = t.get("dt")
    time_kw = dict(time=integ, dt=dt)
    dsp = dict(cfg.get("dispersion") or {})
    cand = dict(cfg.get("candidates") or {})
    for fam in ("gaussian", "binomial"):
        if cand.get(fam) is not None:
            _check_keys(f"candidates.{fam}", cand[fam], SECTION_KEYS[fam])
    extra = []
    for i, e in enumerate(cand.get("extra") or []):
        fam = e.get("family")
        if fam == "gaussian":
            extra.append(Gaussian(e["h"]))
        elif fam == "binomial":
            extra.append(Binomial(e.get("p", 0), e["n"]))
        else:
            raise ConfigError(f"candidates.extra[{i}]: family must be gaussian or binomial")
    coupling = (cfg.get("filter") or {}).get("coupling", "matched")
    if coupling not in ("matched", "density_only"):
        raise ConfigError("filter.coupling must be 'matched' or 'density_only'")
    objective, metric = build_objective(cfg, grid, plasma)
    protected = build_protected(cfg, grid, plasma, time_kw, log)
    return Selector(grid, plasma, protected, objective,
                    tolerance=float(cfg.get("tolerance", 0.05)),
                    acceptance=float(cfg.get("acceptance", 0.10)),
                    frequency_tolerance=cfg.get("frequency_tolerance"), metric=metric,
                    n_alias=int(dsp.get("aliases", 4)), time=integ, dt=dt,
                    n_time_alias=int(t.get("temporal_aliases", 8)),
                    coupling_power=2 if coupling == "matched" else 1,
                    shapes=cand.get("shapes"), gaussian=cand.get("gaussian"),
                    binomial=cand.get("binomial"), extra_candidates=extra,
                    a_min=float(dsp.get("a_min", 0.02)),
                    verify_roots=bool(dsp.get("verify_roots", True)), log=log)


def run(cfg: dict | str | Path, log=None):
    if not isinstance(cfg, dict):
        cfg = load(cfg)
    return build_selector(cfg, log=log).run()


def to_plain(obj: Any):
    """JSON-safe conversion (numpy scalars, complex numbers, NaN -> None)."""
    if isinstance(obj, dict):
        return {str(k): to_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_plain(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return to_plain(obj.tolist())
    if isinstance(obj, (np.floating, float)):
        x = float(obj)
        return None if not np.isfinite(x) else x
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, complex):
        return [obj.real, obj.imag]
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj
