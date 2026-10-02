#!/usr/bin/env python3
"""Fresh-seed diagnostic for the weak-Landau velocity-loading failure.

The v1 confirmation remains immutable.  This pilot changes only the velocity
assignment from a 32x32 tensor repeated in every field cell to the historical
global inverse-CDF quiet start.  Spatial marker positions still place exactly
1024 particles in every grid cell.  The pilot is diagnostic data and cannot be
used as the preregistered 40-seed confirmation.
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
for path in (PROJECT, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from pic_kde_2d.conventions import marker_grid_shape  # noqa: E402
from pic_kde_2d.core.particles2d import (  # noqa: E402
    _inverse_normal_cdf,
    init_landau_2d_eigenmode,
)
from run_weak_landau_confirmatory import (  # noqa: E402
    atomic_seed_directory,
    clone_particles,
    common_metadata,
    file_sha256,
    frozen_grid,
    load_config,
    particle_sha256,
    run_history,
    validate_launch_preflight,
    write_run,
)


DEFAULT_CONFIG = WORKSPACE / "paper1_source_package/reproducibility/weak_landau_confirmatory_v1.json"
PILOT_SEEDS = (202609180, 202609181, 202609182)
PILOT_ARMS = ("cic", "c4_sigma1")


def pilot_source_sha256() -> str:
    return file_sha256(Path(__file__).resolve())


def global_particles(config: dict, grid, seed: int):
    benchmark = config["benchmark"]
    ppc = int(benchmark["ppc"])
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


def velocity_support(config: dict, grid) -> dict:
    ppc = int(config["benchmark"]["ppc"])
    nx_marker, ny_marker = marker_grid_shape(grid.Nx, grid.Ny, ppc)
    n_particles = nx_marker * ny_marker
    quantiles = (np.arange(n_particles, dtype=np.float64) + 0.5) / n_particles
    nodes = _inverse_normal_cdf(quantiles) * float(config["benchmark"]["velocity_thermal"])
    k = 2.0 * math.pi * int(config["benchmark"]["kx_mode"]) / grid.Lx
    omega_r = 2.0 * math.pi / float(config["fit_rule"]["envelope_period"])
    phase_velocity = omega_r / k
    above = int(np.count_nonzero(nodes >= phase_velocity))
    return {
        "n_particles": n_particles,
        "distinct_longitudinal_unperturbed_velocity_nodes": n_particles,
        "longitudinal_velocity_min": float(nodes[0]),
        "longitudinal_velocity_max": float(nodes[-1]),
        "resonant_phase_velocity": phase_velocity,
        "nodes_at_or_above_positive_resonance": above,
        "resonance_in_support": bool(nodes[-1] > phase_velocity and above > 0),
    }


def spatial_cell_counts(particles, grid) -> dict:
    # The alpha displacement crosses some field-cell boundaries, so verify the
    # unperturbed regular lattice analytically rather than mislabeling the
    # perturbed density modulation as a loading-count defect.
    ppc = particles.n_particles // (grid.Nx * grid.Ny)
    return {
        "unperturbed_construction": "regular marker lattice",
        "field_cells": grid.Nx * grid.Ny,
        "particles_per_cell_before_density_perturbation": ppc,
        "exact_requested_ppc": bool(ppc == 1024),
    }


def preflight(config: dict, config_path: Path, base_preflight: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    launch = validate_launch_preflight(config, config_path, base_preflight)
    grid = frozen_grid(config)
    particles = global_particles(config, grid, PILOT_SEEDS[0])
    support = velocity_support(config, grid)
    cells = spatial_cell_counts(particles, grid)
    if not support["resonance_in_support"]:
        raise AssertionError("global velocity template still misses the Landau resonance")
    if not cells["exact_requested_ppc"]:
        raise AssertionError("global velocity correction changed spatial PPC")
    output.mkdir(parents=True)
    result = {
        "status": "PASS",
        "scope": "loading correction and inherited transfer preflight only; no production validation",
        "pilot_seeds": list(PILOT_SEEDS),
        "pilot_arms": list(PILOT_ARMS),
        "config_sha256": file_sha256(config_path),
        "pilot_source_sha256": pilot_source_sha256(),
        "base_preflight": launch,
        "velocity_support": support,
        "spatial_loading": cells,
        "changed_setting": "velocity quadrature assignment only: per-cell tensor -> global inverse-CDF shuffle",
        "unchanged_settings": "grid, spatial lattice/PPC, perturbation, solver, filters, dt, duration and cadence",
    }
    path = output / "preflight.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


def validate_pilot_preflight(config_path: Path, path: Path) -> dict:
    record = json.loads(path.read_text(encoding="utf-8"))
    if record.get("status") != "PASS":
        raise AssertionError("pilot preflight did not pass")
    if record.get("pilot_seeds") != list(PILOT_SEEDS) or record.get("pilot_arms") != list(PILOT_ARMS):
        raise AssertionError("pilot seed/arm set differs from preflight")
    if record.get("config_sha256") != file_sha256(config_path):
        raise AssertionError("base configuration changed after pilot preflight")
    if record.get("pilot_source_sha256") != pilot_source_sha256():
        raise AssertionError("pilot source changed after pilot preflight")
    return {"pilot_preflight_sha256": file_sha256(path), "pilot_source_sha256": pilot_source_sha256()}


def simulate_seed(
    config: dict,
    config_path: Path,
    base_preflight: Path,
    pilot_preflight: Path,
    output: Path,
    seed: int,
) -> None:
    if seed not in PILOT_SEEDS:
        raise ValueError(f"pilot seed must be one of {PILOT_SEEDS}")
    launch = validate_launch_preflight(config, config_path, base_preflight)
    pilot_launch = validate_pilot_preflight(config_path, pilot_preflight)
    partial, target = atomic_seed_directory(output, seed)
    grid = frozen_grid(config)
    ppc = int(config["benchmark"]["ppc"])
    initial = global_particles(config, grid, seed)
    initial_digest = particle_sha256(initial)
    arms = [arm for arm in config["arms"] if arm["id"] in PILOT_ARMS]
    observed = []
    try:
        for arm in arms:
            particles = clone_particles(initial)
            digest = particle_sha256(particles)
            if digest != initial_digest:
                raise AssertionError("pilot arm particle clone mismatch")
            observed.append(digest)
            history = run_history(particles, grid, config, float(arm["sigma_cells"]))
            metadata = {
                **common_metadata(config, config_path, launch, grid, seed, ppc),
                **pilot_launch,
                "campaign_id": "weak_landau_global_loading_pilot_v1",
                "base_failed_campaign_id": config["campaign_id"],
                "arm": arm,
                "velocity_loading": "global inverse-CDF quiet-start shuffle over all particles",
                "spatial_loading": "regular marker lattice with exactly 1024 particles per cell before perturbation",
                "initial_particle_sha256": initial_digest,
                "paired_pilot_arms": list(PILOT_ARMS),
                "validation_role": "diagnostic pilot only; excluded from replacement confirmation",
            }
            write_run(partial / arm["id"], history, metadata)
        if len(set(observed)) != 1:
            raise AssertionError("pilot arms did not start byte-identically")
        manifest = {
            "seed": seed,
            "initial_particle_sha256": initial_digest,
            "arm_initial_hashes": dict(zip(PILOT_ARMS, observed)),
            "all_arm_hashes_match": True,
        }
        (partial / "pairing_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        partial.rename(target)
    except Exception:
        raise
    print(json.dumps({"completed": str(target), "seed": seed, "particle_sha256": initial_digest}))


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    result.add_argument("--base-preflight", type=Path, required=True)
    sub = result.add_subparsers(dest="command", required=True)
    check = sub.add_parser("preflight")
    check.add_argument("--output", type=Path, required=True)
    run = sub.add_parser("simulate-seed")
    run.add_argument("--pilot-preflight", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--seed", type=int, required=True)
    return result


def main() -> None:
    args = parser().parse_args()
    config, config_path = load_config(args.config)
    if args.command == "preflight":
        preflight(config, config_path, args.base_preflight, args.output)
    else:
        simulate_seed(
            config,
            config_path,
            args.base_preflight,
            args.pilot_preflight,
            args.output,
            args.seed,
        )


if __name__ == "__main__":
    main()
