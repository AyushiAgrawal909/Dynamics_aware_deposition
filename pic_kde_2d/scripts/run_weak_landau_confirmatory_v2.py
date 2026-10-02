#!/usr/bin/env python3
"""Run the v2 weak-Landau confirmation with corrected global velocities.

The failed v1 runner is intentionally left byte-for-byte unchanged.  This
wrapper reuses its audited particle-field integration and transfer checks but
changes the initializer's velocity assignment to the global inverse-CDF quiet
start verified by pilot job 5200354.  Spatial loading remains exactly 1024
particles per field cell before the perturbation displacement.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np


PROJECT = Path(__file__).resolve().parents[1]
WORKSPACE = PROJECT.parent
SCRIPTS = PROJECT / "scripts"
for item in (PROJECT, SCRIPTS):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from pic_kde_2d.conventions import marker_grid_shape  # noqa: E402
from pic_kde_2d.core.particles2d import (  # noqa: E402
    _inverse_normal_cdf,
    init_landau_2d_eigenmode,
)
import run_weak_landau_confirmatory as base  # noqa: E402


DEFAULT_CONFIG = (
    WORKSPACE
    / "paper1_source_package/reproducibility/weak_landau_confirmatory_v2_global_velocity.json"
)
EXPECTED_CAMPAIGN = "weak_landau_confirmatory_v2_global_velocity"


def runtime_source_paths(config_path: Path) -> list[Path]:
    return [
        Path(__file__).resolve(),
        config_path.resolve(),
        WORKSPACE / "paper1_source_package/reproducibility/analyze_weak_landau_confirmatory_v2.py",
        WORKSPACE / "paper1_source_package/reproducibility/analyze_weak_landau_confirmatory.py",
        PROJECT / "scripts/run_weak_landau_confirmatory.py",
        PROJECT / "pic_kde_2d/core/particles2d.py",
        PROJECT / "pic_kde_2d/core/pic_loop.py",
        PROJECT / "pic_kde_2d/core/grid2d.py",
        PROJECT / "pic_kde_2d/methods/spectral_W.py",
        PROJECT / "pic_kde_2d/methods/spectral_W_wrappers.py",
        PROJECT / "pic_kde_2d/methods/cic.py",
    ]


def runtime_source_hashes(config_path: Path) -> dict[str, str]:
    return {
        str(path.relative_to(WORKSPACE)): base.file_sha256(path)
        for path in runtime_source_paths(config_path)
    }


def load_config(path: Path) -> tuple[dict, Path]:
    config, resolved = base.load_config(path)
    if config["campaign_id"] != EXPECTED_CAMPAIGN:
        raise AssertionError("unexpected v2 campaign")
    benchmark = config["benchmark"]
    if not benchmark.get("velocity_assignment_scope", "").startswith("global"):
        raise AssertionError("v2 requires global velocity assignment")
    confirmation = config["confirmation"]
    development = set()
    for item in config["development_data"].values():
        development.update(int(seed) for seed in item.get("seeds", []))
    current = set(range(confirmation["seed_start"], confirmation["seed_stop_inclusive"] + 1))
    if current.intersection(development):
        raise AssertionError("v2 reuses development seeds")
    return config, resolved


def validate_launch_preflight(config: dict, config_path: Path, preflight_path: Path | None) -> dict:
    if preflight_path is None:
        raise ValueError("v2 production simulation requires --preflight")
    resolved = preflight_path.resolve()
    record = json.loads(resolved.read_text(encoding="utf-8"))
    if record.get("campaign_id") != EXPECTED_CAMPAIGN or record.get("status") != "PASS":
        raise AssertionError("the supplied v2 preflight is not a passing record")
    if record.get("production_submitted") is not False:
        raise AssertionError("v2 preflight has invalid production_submitted value")
    current = runtime_source_hashes(config_path)
    if record.get("source_sha256") != current:
        raise AssertionError("v2 runtime source differs from the supplied preflight")
    support = record.get("velocity_support_check", {})
    if not support.get("resonance_in_support"):
        raise AssertionError("v2 preflight lacks resonant velocity support")
    return {
        "preflight_path_at_launch": str(resolved),
        "preflight_sha256": base.file_sha256(resolved),
        "source_sha256": current,
    }


def make_initial_particles(config: dict, grid, seed: int, ppc: int):
    benchmark = config["benchmark"]
    nx_marker, ny_marker = marker_grid_shape(grid.Nx, grid.Ny, ppc)
    return init_landau_2d_eigenmode(
        Nx_marker=nx_marker,
        Ny_marker=ny_marker,
        Lx=grid.Lx,
        Ly=grid.Ly,
        v_thermal=benchmark["velocity_thermal"],
        alpha=benchmark["alpha"],
        kx_mode=benchmark["kx_mode"],
        ky_mode=benchmark["ky_mode"],
        omega_r=2.0 * math.pi / config["fit_rule"]["envelope_period"],
        rng=np.random.default_rng(seed),
        grid_shape=None,
    )


def velocity_support_check(config: dict, grid, ppc: int) -> dict:
    nx_marker, ny_marker = marker_grid_shape(grid.Nx, grid.Ny, ppc)
    n_particles = nx_marker * ny_marker
    maximum_quantile = (n_particles - 0.5) / n_particles
    # Reuse the same inverse-CDF routine that constructs the particles, while
    # evaluating only the extreme node needed for the launch guard.
    maximum = float(_inverse_normal_cdf(np.asarray([maximum_quantile]))[0])
    maximum *= float(config["benchmark"]["velocity_thermal"])
    k = 2.0 * math.pi * int(config["benchmark"]["kx_mode"]) / grid.Lx
    omega_r = 2.0 * math.pi / float(config["fit_rule"]["envelope_period"])
    phase_velocity = omega_r / k
    velocity_thermal = float(config["benchmark"]["velocity_thermal"])
    normalized_phase_velocity = phase_velocity / velocity_thermal
    cdf_at_resonance = 0.5 * (
        1.0 + math.erf(normalized_phase_velocity / math.sqrt(2.0))
    )
    # The initializer uses midpoint quantiles u_i=(i+1/2)/N.  Count those
    # nodes exactly rather than rounding the continuous Gaussian tail mass.
    first_resonant_index = math.ceil(n_particles * cdf_at_resonance - 0.5)
    first_resonant_index = min(max(first_resonant_index, 0), n_particles)
    nodes_at_or_above = n_particles - first_resonant_index
    return {
        "ppc": ppc,
        "n_particles": n_particles,
        "distinct_unperturbed_longitudinal_nodes": n_particles,
        "longitudinal_velocity_max": maximum,
        "resonant_phase_velocity": phase_velocity,
        "nodes_at_or_above_positive_resonance": nodes_at_or_above,
        "resonance_in_support": bool(maximum >= phase_velocity and nodes_at_or_above > 0),
    }


def simulate_low_seed(config: dict, config_path: Path, preflight_path: Path | None, output: Path, seed: int) -> None:
    base.check_seed(config, seed)
    launch_record = validate_launch_preflight(config, config_path, preflight_path)
    partial, target = base.atomic_seed_directory(output, seed)
    try:
        grid = base.frozen_grid(config)
        ppc = int(config["benchmark"]["ppc"])
        initial = make_initial_particles(config, grid, seed, ppc)
        initial_digest = base.particle_sha256(initial)
        observed = []
        for arm in config["arms"]:
            particles = base.clone_particles(initial)
            digest = base.particle_sha256(particles)
            observed.append(digest)
            if digest != initial_digest:
                raise AssertionError("v2 arm particle clone mismatch")
            history = base.run_history(particles, grid, config, float(arm["sigma_cells"]))
            metadata = {
                **base.common_metadata(config, config_path, launch_record, grid, seed, ppc),
                "arm": arm,
                "initial_particle_sha256": initial_digest,
                "paired_low_ppc_arms": [item["id"] for item in config["arms"]],
                "velocity_assignment_scope": "global",
                "spatial_particles_per_cell_before_perturbation": ppc,
            }
            base.write_run(partial / arm["id"], history, metadata)
        if len(set(observed)) != 1:
            raise AssertionError("v2 arms did not start byte-identically")
        manifest = {
            "seed": seed,
            "initial_particle_sha256": initial_digest,
            "arm_initial_hashes": dict(zip((arm["id"] for arm in config["arms"]), observed)),
            "all_three_arm_hashes_match": True,
            "velocity_assignment_scope": "global",
        }
        (partial / "pairing_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        partial.rename(target)
    except Exception:
        raise
    print(json.dumps({"completed": str(target), "seed": seed, "particle_sha256": initial_digest}))


def simulate_reference_seed(config: dict, config_path: Path, preflight_path: Path | None, output: Path, seed: int) -> None:
    base.check_seed(config, seed)
    launch_record = validate_launch_preflight(config, config_path, preflight_path)
    partial, target = base.atomic_seed_directory(output, seed)
    try:
        grid = base.frozen_grid(config)
        ppc = int(config["same_campaign_noise_metrics"]["reference_ppc"])
        particles = make_initial_particles(config, grid, seed, ppc)
        initial_digest = base.particle_sha256(particles)
        history = base.run_history(particles, grid, config, 0.0)
        arm = {"id": "cic_reference", "name": "PPC-4096 CIC reference", "sigma_cells": 0.0}
        metadata = {
            **base.common_metadata(config, config_path, launch_record, grid, seed, ppc),
            "arm": arm,
            "initial_particle_sha256": initial_digest,
            "reference_role": "finite-PPC reference for density and grid-field squared-error power only",
            "velocity_assignment_scope": "global",
            "spatial_particles_per_cell_before_perturbation": ppc,
        }
        base.write_run(partial / arm["id"], history, metadata)
        partial.rename(target)
    except Exception:
        raise
    print(json.dumps({"completed": str(target), "seed": seed, "particle_sha256": initial_digest}))


def run_base_check_with_v2_hooks(function, config: dict, config_path: Path, output: Path) -> None:
    original_particles = base.make_initial_particles
    original_hashes = base.runtime_source_hashes
    try:
        base.make_initial_particles = make_initial_particles
        base.runtime_source_hashes = runtime_source_hashes
        function(config, config_path, output)
    finally:
        base.make_initial_particles = original_particles
        base.runtime_source_hashes = original_hashes


def preflight(config: dict, config_path: Path, output: Path) -> None:
    run_base_check_with_v2_hooks(base.preflight, config, config_path, output)
    path = output.resolve() / "preflight.json"
    record = json.loads(path.read_text(encoding="utf-8"))
    grid = base.frozen_grid(config)
    support = velocity_support_check(config, grid, int(config["benchmark"]["ppc"]))
    reference_support = velocity_support_check(
        config, grid, int(config["same_campaign_noise_metrics"]["reference_ppc"])
    )
    if not support["resonance_in_support"] or not reference_support["resonance_in_support"]:
        raise AssertionError("v2 velocity support preflight failed")
    record["velocity_support_check"] = support
    record["reference_velocity_support_check"] = reference_support
    record["spatial_loading_check"] = {
        "regular_lattice": True,
        "low_particles_per_cell_before_perturbation": config["benchmark"]["ppc"],
        "reference_particles_per_cell_before_perturbation": config["same_campaign_noise_metrics"]["reference_ppc"],
    }
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--preflight", type=Path)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("preflight", "smoke-test"):
        item = sub.add_parser(name)
        item.add_argument("--output", type=Path, required=True)
    low = sub.add_parser("simulate-low-seed")
    low.add_argument("--output", type=Path, required=True)
    low.add_argument("--seed", type=int, required=True)
    reference = sub.add_parser("simulate-reference-seed")
    reference.add_argument("--output", type=Path, required=True)
    reference.add_argument("--seed", type=int, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config, config_path = load_config(args.config)
    if args.command == "preflight":
        preflight(config, config_path, args.output)
    elif args.command == "smoke-test":
        run_base_check_with_v2_hooks(base.smoke_test, config, config_path, args.output)
    elif args.command == "simulate-low-seed":
        simulate_low_seed(config, config_path, args.preflight, args.output, args.seed)
    else:
        simulate_reference_seed(config, config_path, args.preflight, args.output, args.seed)


if __name__ == "__main__":
    main()
