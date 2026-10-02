#!/usr/bin/env python3
"""Reconstruct deterministic E2 initial states without time integration.

This diagnostic is intentionally independent of saved history amplitudes.  It
prints hashes of the arrays produced by the actual initializer under the active
Python/NumPy runtime.  Historical jobs did not save these array hashes, so the
output is reconstruction evidence, not a recovered historical hash.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys

import numpy as np


ARRAY_NAMES = ("x", "y", "vx", "vy", "w_full", "w_delta")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def hash_initial_state(particles: object) -> tuple[str, dict[str, str]]:
    aggregate = hashlib.sha256()
    per_array: dict[str, str] = {}
    for name in ARRAY_NAMES:
        array = np.ascontiguousarray(getattr(particles, name))
        digest = hashlib.sha256()
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes(order="C"))
        value = digest.hexdigest()
        per_array[name] = value
        aggregate.update(name.encode("ascii"))
        aggregate.update(value.encode("ascii"))
    for name in (
        "Lx",
        "Ly",
        "f0_v_thermal",
        "f0_density",
        "background_kind",
        "beam_velocity",
    ):
        aggregate.update(name.encode("ascii"))
        aggregate.update(repr(getattr(particles, name)).encode("utf-8"))
    return aggregate.hexdigest(), per_array


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--seeds", default="200,201,202,203,204,205,206,207,208,209")
    args = parser.parse_args()
    project = args.project.resolve()
    for candidate in (project, project / "scripts"):
        if str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
    import run_linearized_delta_f_mode as production

    source_paths = (
        "scripts/run_linearized_delta_f_mode.py",
        "pic_kde_2d/conventions.py",
        "pic_kde_2d/core/grid2d.py",
        "pic_kde_2d/core/particles2d.py",
        "pic_kde_2d/methods/cic.py",
        "pic_kde_2d/methods/delta_f.py",
    )
    states = {}
    for seed_text in args.seeds.split(","):
        seed = int(seed_text)
        production_args = production.build_parser().parse_args(
            [
                "--output", "unused-compatibility-output",
                "--benchmark", "two_stream",
                "--mode", "12",
                "--ky-mode", "1",
                "--seed", str(seed),
                "--ppc", "512",
                "--alpha", "1e-8",
                "--filter-kind", "gaussian",
                "--sigma-x-dx", "0",
                "--sigma-y-dy", "0",
                "--Nx", "128",
                "--Ny", "16",
                "--Lx", "64",
                "--Ly", "16",
                "--v-beam", "0.8",
                "--v-thermal", "0.05",
                "--dt", "0.02",
                "--t-end", "71.16",
                "--diag-stride", "2",
                "--project-target-mode",
            ]
        )
        grid, particles = production.build_markers(production_args)
        aggregate, arrays = hash_initial_state(particles)
        states[str(seed)] = {
            "reconstructed_initial_state_sha256": aggregate,
            "array_sha256": arrays,
            "array_shapes": {
                name: list(np.asarray(getattr(particles, name)).shape)
                for name in ARRAY_NAMES
            },
            "grid": {
                "Nx": int(grid.Nx),
                "Ny": int(grid.Ny),
                "Lx": float(grid.Lx),
                "Ly": float(grid.Ly),
            },
        }
    result = {
        "evidence_class": "deterministic reconstruction; not saved historical array hashes",
        "no_time_integration": True,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "project": str(project),
        "source_sha256": {
            relative: sha256_file(project / relative) for relative in source_paths
        },
        "states": states,
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
