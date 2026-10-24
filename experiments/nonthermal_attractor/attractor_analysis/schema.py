"""
attractor_analysis/schema.py
============================

Data contract between run backends (T2: ``pic1d1v``, ``vlasov1d1v``; later
``hexapic``) and the analysis library.

One run is one ``.npz`` bundle. Array diagnostics are stored as named arrays;
metadata is stored as a JSON string under the key ``"metadata_json"``. The
validator below rejects any bundle that is missing a required field, so no
analysis runs on an incomplete record. The human-readable contract is
``protocol/DATA_CONTRACT.md``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping

import numpy as np


SCHEMA_VERSION = "nta-1"

BACKENDS = ("pic1d1v", "vlasov1d1v", "hexapic", "synthetic")

REQUIRED_METADATA = (
    "schema_version",
    "backend",
    "git_revision",
    "config",
    "normalization",
    "seed",
    "nested_level",
    "dt",
    "dx",
    "staggering",
    "wall_time_s",
)

# Backend-specific metadata (regulator descriptors).
REQUIRED_METADATA_BY_BACKEND = {
    "pic1d1v": ("particles_per_cell", "n_particles", "macro_weight"),
    "vlasov1d1v": ("dv", "v_max"),
    "hexapic": ("particles_per_cell", "n_particles", "macro_weight"),
    "synthetic": ("synthetic_case", "n_particles"),
}

# name -> required number of dimensions (time is always the leading axis).
REQUIRED_ARRAYS = {
    "times": 1,
    # scalar budgets per snapshot
    "number": 1,
    "total_charge": 1,
    "momentum": 1,
    "kinetic_energy": 1,
    "field_energy": 1,
    "total_energy": 1,
    # N(eps, t) on frozen common edges
    "eps_edges": 1,
    "N_eps": 2,
    "eps_bin_weight": 2,
    "eps_bin_energy": 2,
    "eps_bin_neff": 2,
    "eps_overflow": 2,  # columns: weight, energy
    # rho(eta, t) on frozen eta edges
    "eta_edges": 1,
    "rho_eta": 2,
    "rho_bin_volume": 2,
    "rho_bin_mass": 2,
    "rho_reconstruction": 1,  # [dx_b, dv_b, eta_floor]
    # field spectrum
    "k": 1,
    "Ek2": 2,
}

OPTIONAL_ARRAYS = ("phase_space_x", "phase_space_v", "phase_space_w", "phase_space_times",
                   "mode_amplitudes", "mode_indices")


class BundleError(ValueError):
    """Raised when a bundle violates the data contract."""


def write_bundle(path: str | Path, metadata: Mapping[str, Any], arrays: Mapping[str, np.ndarray]) -> Path:
    """Validate and write a bundle. Returns the written path."""
    md = dict(metadata)
    md.setdefault("schema_version", SCHEMA_VERSION)
    validate(md, arrays)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {k: np.asarray(v) for k, v in arrays.items()}
    payload["metadata_json"] = np.asarray(json.dumps(md, sort_keys=True, default=_json_default))
    np.savez_compressed(path, **payload)
    return path if path.suffix == ".npz" else path.with_suffix(path.suffix + ".npz")


def read_bundle(path: str | Path) -> tuple[Dict[str, Any], Dict[str, np.ndarray]]:
    """Read and validate a bundle."""
    with np.load(path, allow_pickle=False) as data:
        if "metadata_json" not in data:
            raise BundleError(f"{path}: missing metadata_json")
        md = json.loads(str(data["metadata_json"]))
        arrays = {k: data[k] for k in data.files if k != "metadata_json"}
    validate(md, arrays)
    return md, arrays


def validate(metadata: Mapping[str, Any], arrays: Mapping[str, np.ndarray]) -> None:
    """Raise BundleError listing every contract violation."""
    problems = []
    for key in REQUIRED_METADATA:
        if key not in metadata:
            problems.append(f"metadata.{key} missing")
    backend = metadata.get("backend")
    if backend not in BACKENDS:
        problems.append(f"metadata.backend={backend!r} not in {BACKENDS}")
    else:
        for key in REQUIRED_METADATA_BY_BACKEND[backend]:
            if key not in metadata:
                problems.append(f"metadata.{key} missing (required for backend {backend})")
    if metadata.get("schema_version", SCHEMA_VERSION) != SCHEMA_VERSION:
        problems.append(f"schema_version {metadata.get('schema_version')} != {SCHEMA_VERSION}")

    for name, ndim in REQUIRED_ARRAYS.items():
        if name not in arrays:
            problems.append(f"array {name} missing")
            continue
        arr = np.asarray(arrays[name])
        if arr.ndim != ndim:
            problems.append(f"array {name} has ndim {arr.ndim}, expected {ndim}")

    if not problems:
        nt = len(arrays["times"])
        for name, ndim in REQUIRED_ARRAYS.items():
            if name in ("times", "eps_edges", "eta_edges", "k", "rho_reconstruction"):
                continue
            if np.asarray(arrays[name]).shape[0] != nt:
                problems.append(f"array {name} leading axis {np.asarray(arrays[name]).shape[0]} != n_times {nt}")
        neps = len(arrays["eps_edges"]) - 1
        for name in ("N_eps", "eps_bin_weight", "eps_bin_energy", "eps_bin_neff"):
            if np.asarray(arrays[name]).shape[1] != neps:
                problems.append(f"array {name} has {np.asarray(arrays[name]).shape[1]} bins, edges imply {neps}")
        neta = len(arrays["eta_edges"]) - 1
        for name in ("rho_eta", "rho_bin_volume", "rho_bin_mass"):
            if np.asarray(arrays[name]).shape[1] != neta:
                problems.append(f"array {name} has {np.asarray(arrays[name]).shape[1]} bins, edges imply {neta}")
        if np.asarray(arrays["eps_overflow"]).shape[1] != 2:
            problems.append("array eps_overflow must have 2 columns (weight, energy)")
        if np.any(np.diff(arrays["times"]) <= 0):
            problems.append("times must be strictly increasing")
        if np.any(np.diff(arrays["eps_edges"]) <= 0) or arrays["eps_edges"][0] < 0:
            problems.append("eps_edges must be non-negative and strictly increasing")
        if np.any(np.diff(arrays["eta_edges"]) <= 0) or arrays["eta_edges"][0] <= 0:
            problems.append("eta_edges must be positive and strictly increasing")
    if problems:
        raise BundleError("; ".join(problems))


def _json_default(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"not JSON serializable: {type(obj)}")
