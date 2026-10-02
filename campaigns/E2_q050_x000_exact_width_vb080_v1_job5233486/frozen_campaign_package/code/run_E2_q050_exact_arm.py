#!/usr/bin/env python3
"""Run one frozen CIC or exact-q050 E2 history with auditable pairing."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys

import numpy as np


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def hash_initial_state(particles: object) -> tuple[str, dict[str, str]]:
    """Hash the actual arrays that will be evolved, including dtype and shape."""
    aggregate = hashlib.sha256()
    per_array: dict[str, str] = {}
    for name in ("x", "y", "vx", "vy", "w_full", "w_delta"):
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


def locate_project(explicit: Path | None) -> Path:
    candidates = [] if explicit is None else [explicit]
    here = Path(__file__).resolve()
    candidates.extend(
        [
            here.parents[2] / "pic_kde_2d_landau_cv_stage",
            Path("/home/users/u103576/hexapic-main/pic_kde_2d"),
        ]
    )
    for candidate in candidates:
        candidate = candidate.resolve()
        if (candidate / "scripts/run_linearized_delta_f_mode.py").is_file():
            return candidate
    raise FileNotFoundError("could not locate the frozen audited research source")


def verify_source(project: Path, config: dict) -> dict[str, str]:
    observed: dict[str, str] = {}
    for relative, expected in config["source_identity"]["sha256"].items():
        path = project / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        observed[relative] = sha256_file(path)
        if observed[relative] != expected:
            raise AssertionError(
                f"source hash mismatch for {relative}: "
                f"expected {expected}, observed {observed[relative]}"
            )
    return observed


def make_args(production, config: dict, arm: str, seed: int, output: Path):
    arm_config = config["arms"][arm]
    return production.build_parser().parse_args(
        [
            "--output",
            str(output),
            "--benchmark",
            "two_stream",
            "--mode",
            str(config["mode"][0]),
            "--ky-mode",
            str(config["mode"][1]),
            "--seed",
            str(seed),
            "--ppc",
            str(config["loading"]["ppc"]),
            "--alpha",
            str(config["initial_perturbation_alpha"]),
            "--filter-kind",
            "gaussian",
            "--sigma-x-dx",
            str(arm_config["sigma_x_dx"]),
            "--sigma-y-dy",
            str(arm_config["sigma_y_dy"]),
            "--Nx",
            str(config["grid"]["Nx"]),
            "--Ny",
            str(config["grid"]["Ny"]),
            "--Lx",
            str(config["grid"]["Lx"]),
            "--Ly",
            str(config["grid"]["Ly"]),
            "--v-beam",
            str(config["equilibrium"]["v_beam_plus"]),
            "--v-thermal",
            str(config["equilibrium"]["v_thermal_x"]),
            "--dt",
            str(config["dt"]),
            "--t-end",
            str(config["t_end"]),
            "--diag-stride",
            str(config["diagnostic_stride"]),
            "--project-target-mode",
        ]
    )


def run_with_built_markers(production, args, grid, particles) -> dict:
    """Production calculation with the already-hashed marker object."""
    sigma_x_dx, sigma_y_dy = production.resolved_sigma_cells(args)
    if sigma_x_dx > 0.0 or sigma_y_dy > 0.0:
        W = production.make_W_gaussian(
            grid, sigma_x_dx * grid.dx, sigma_y_dy * grid.dy
        )
    else:
        W = None
    deposit_unprojected, gather = production.make_delta_f_methods(W=W)

    def deposit(particles_here, grid_here):
        return production.project_density_to_mode(
            deposit_unprojected(particles_here, grid_here),
            grid_here,
            args.mode,
            args.ky_mode,
        )

    n_steps = int(round(args.t_end / args.dt))
    history = production.run_simulation_linearized_delta_f(
        particles,
        grid,
        args.dt,
        n_steps,
        deposit,
        gather,
        diag_stride=args.diag_stride,
        tracked_modes=[(args.mode, args.ky_mode)],
        validity_threshold=args.validity_threshold,
    )
    times = np.asarray(history.times, dtype=np.float64)
    mode = np.asarray(
        history.mode_amplitudes_Ex[(args.mode, args.ky_mode)],
        dtype=np.complex128,
    )
    validity = np.asarray(history.delta_f_ratio, dtype=np.float64)
    args.output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(
        args.output / "mode_history.npz",
        times=times,
        mode=mode,
        delta_f_ratio=validity,
        seed=np.asarray([args.seed], dtype=np.int64),
    )
    return {
        "benchmark": args.benchmark,
        "kx_mode": args.mode,
        "ky_mode": args.ky_mode,
        "seed": args.seed,
        "ppc": args.ppc,
        "alpha": args.alpha,
        "filter_kind": args.filter_kind,
        "sigma_x_dx": sigma_x_dx,
        "sigma_y_dy": sigma_y_dy,
        "dt": args.dt,
        "t_end": args.t_end,
        "diagnostic_stride": args.diag_stride,
        "max_delta_f_ratio": float(np.max(validity)),
        "validity_threshold": args.validity_threshold,
        "valid_through_run": bool(np.max(validity) < args.validity_threshold),
        "background": particles.background_kind,
        "marker_dynamics": "free_streaming_frozen_equilibrium",
        "target_mode_projection": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--config-sha256", required=True)
    parser.add_argument("--project", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arm", choices=("cic", "q050_x000"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()

    if sha256_file(args.config) != args.config_sha256:
        raise AssertionError("frozen config hash mismatch")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config["status"] != "FROZEN_NOT_SUBMITTED":
        raise AssertionError("unexpected campaign status")
    if args.seed not in config["seeds"]:
        raise ValueError("seed is outside the frozen evaluation set")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")

    project = locate_project(args.project)
    source_hashes = verify_source(project, config)
    for candidate in (project, project / "scripts"):
        if str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
    import run_linearized_delta_f_mode as production

    production_args = make_args(production, config, args.arm, args.seed, args.output)
    grid, particles = production.build_markers(production_args)
    initial_hash, array_hashes = hash_initial_state(particles)
    expected_hash = config["existing_cic_reuse"][
        "production_runtime_reconstructed_initial_state_sha256"
    ][str(args.seed)]
    if initial_hash != expected_hash:
        raise AssertionError(
            "initialized state does not match the frozen production-runtime "
            f"reconstruction for seed {args.seed}: expected {expected_hash}, "
            f"observed {initial_hash}; no time integration was started"
        )
    result = run_with_built_markers(production, production_args, grid, particles)

    result.update(
        {
            "campaign_id": config["campaign_id"],
            "arm": args.arm,
            "config_sha256": args.config_sha256,
            "initial_state_sha256": initial_hash,
            "initial_state_matches_production_runtime_reconstruction": True,
            "initial_array_sha256": array_hashes,
            "runtime": {
                "python": platform.python_version(),
                "numpy": np.__version__,
            },
            "project": str(project),
            "source_sha256": source_hashes,
        }
    )
    (args.output / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
