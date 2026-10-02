#!/usr/bin/env python3
"""Run the frozen weak-Landau confirmatory campaign and its preflight checks.

Production commands consume the immutable JSON specification rather than
duplicating settings in a scheduler script.  A low-PPC seed runs all three
arms from byte-identical particle arrays.  The high-PPC CIC reference is a
separate, consistently configured run used only for error-power metrics.
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
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from pic_kde_2d.conventions import marker_grid_shape  # noqa: E402
from pic_kde_2d.core.grid2d import Grid2D  # noqa: E402
from pic_kde_2d.core.particles2d import Particles2D, init_landau_2d_eigenmode  # noqa: E402
from pic_kde_2d.core.pic_loop import pic_step  # noqa: E402
from pic_kde_2d.methods.spectral_W import apply_W_to_grid, make_W_gaussian  # noqa: E402
from pic_kde_2d.methods.spectral_W_wrappers import make_spectral_W_methods  # noqa: E402


DEFAULT_CONFIG = (
    WORKSPACE
    / "paper1_source_package/reproducibility/weak_landau_confirmatory_v1.json"
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def runtime_source_paths(config_path: Path) -> list[Path]:
    return [
        Path(__file__).resolve(),
        config_path.resolve(),
        WORKSPACE / "paper1_source_package/reproducibility/analyze_weak_landau_confirmatory.py",
        PROJECT / "pic_kde_2d/core/particles2d.py",
        PROJECT / "pic_kde_2d/core/pic_loop.py",
        PROJECT / "pic_kde_2d/core/grid2d.py",
        PROJECT / "pic_kde_2d/methods/spectral_W.py",
        PROJECT / "pic_kde_2d/methods/spectral_W_wrappers.py",
        PROJECT / "pic_kde_2d/methods/cic.py",
    ]


def runtime_source_hashes(config_path: Path) -> dict[str, str]:
    return {
        str(path.relative_to(WORKSPACE)): file_sha256(path)
        for path in runtime_source_paths(config_path)
    }


def validate_launch_preflight(config: dict, config_path: Path, preflight_path: Path | None) -> dict:
    if preflight_path is None:
        raise ValueError("production simulation requires --preflight")
    resolved = preflight_path.resolve()
    preflight = json.loads(resolved.read_text(encoding="utf-8"))
    if preflight.get("campaign_id") != config["campaign_id"] or preflight.get("status") != "PASS":
        raise AssertionError("the supplied preflight is not a passing record for this campaign")
    if preflight.get("production_submitted") is not False:
        raise AssertionError("preflight record has an invalid production_submitted value")
    current = runtime_source_hashes(config_path)
    if preflight.get("source_sha256") != current:
        raise AssertionError("runtime source differs from the supplied passing preflight")
    return {
        "preflight_path_at_launch": str(resolved),
        "preflight_sha256": file_sha256(resolved),
        "source_sha256": current,
    }


def particle_sha256(particles: Particles2D) -> str:
    digest = hashlib.sha256()
    for name in ("x", "y", "vx", "vy", "w"):
        array = np.ascontiguousarray(getattr(particles, name), dtype=np.float64)
        digest.update(name.encode("ascii"))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(array.tobytes())
    digest.update(np.asarray([particles.Lx, particles.Ly], dtype=np.float64).tobytes())
    return digest.hexdigest()


def clone_particles(particles: Particles2D) -> Particles2D:
    return Particles2D(
        x=particles.x.copy(),
        y=particles.y.copy(),
        vx=particles.vx.copy(),
        vy=particles.vy.copy(),
        w=particles.w.copy(),
        Lx=particles.Lx,
        Ly=particles.Ly,
    )


def load_config(path: Path) -> tuple[dict, Path]:
    resolved = path.resolve()
    config = json.loads(resolved.read_text(encoding="utf-8"))
    required = {
        "campaign_id",
        "status",
        "confirmation",
        "benchmark",
        "grid",
        "time_integration",
        "field_solver",
        "arms",
        "fit_rule",
        "same_campaign_noise_metrics",
    }
    missing = required.difference(config)
    if missing:
        raise KeyError(f"configuration lacks {sorted(missing)}")
    if config["status"] != "frozen_not_submitted":
        raise AssertionError("production code accepts only the frozen_not_submitted specification")
    confirmation = config["confirmation"]
    expected_count = confirmation["seed_stop_inclusive"] - confirmation["seed_start"] + 1
    if confirmation["n_seeds"] != expected_count or expected_count != 40:
        raise AssertionError("the confirmatory sample must be exactly 40 fixed seeds")
    if config["benchmark"]["ppc"] != 1024:
        raise AssertionError("PPC-1024 is frozen")
    time = config["time_integration"]
    if not math.isclose(time["dt"] * time["n_steps"], time["simulation_t_stop"]):
        raise AssertionError("dt*n_steps does not match the frozen simulation stop")
    fit = config["fit_rule"]
    reference = abs(config["numerical_reference"]["continuum_gamma_for_context_only"])
    if fit["fit_t_stop_max"] < fit["fit_t_start"] + 2.0 / reference:
        raise AssertionError("fit cap is too early for two reference amplitude e-foldings")
    arm_ids = [arm["id"] for arm in config["arms"]]
    if arm_ids != ["cic", "c4_sigma1", "lscv_job5008392"]:
        raise AssertionError(f"unexpected or reordered arms: {arm_ids}")
    if config["arms"][2]["sigma_cells"] != 7.244293147028428:
        raise AssertionError("the independently selected LOO-CV width changed")
    return config, resolved


def frozen_grid(config: dict) -> Grid2D:
    item = config["grid"]
    grid = Grid2D(Nx=item["Nx"], Ny=item["Ny"], Lx=item["Lx"], Ly=item["Ly"])
    if not math.isclose(grid.dx, item["dx"], rel_tol=0.0, abs_tol=1.0e-15):
        raise AssertionError("configured dx is inconsistent")
    if not math.isclose(grid.dy, item["dy"], rel_tol=0.0, abs_tol=1.0e-15):
        raise AssertionError("configured dy is inconsistent")
    return grid


def make_initial_particles(config: dict, grid: Grid2D, seed: int, ppc: int) -> Particles2D:
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
        grid_shape=(grid.Nx, grid.Ny),
    )


def arm_methods(grid: Grid2D, sigma_cells: float):
    W = make_W_gaussian(grid, sigma_cells * grid.dx, sigma_cells * grid.dy)
    deposit, gather = make_spectral_W_methods(W)
    return W, deposit, gather


def run_history(
    particles: Particles2D,
    grid: Grid2D,
    config: dict,
    sigma_cells: float,
) -> dict[str, np.ndarray]:
    time = config["time_integration"]
    benchmark = config["benchmark"]
    _, deposit, gather = arm_methods(grid, sigma_cells)
    mode_times: list[float] = []
    mode: list[complex] = []
    field_energy: list[float] = []
    kinetic_energy: list[float] = []
    total_charge: list[float] = []
    snapshot_times: list[float] = []
    rho_snapshots: list[np.ndarray] = []
    Ex_snapshots: list[np.ndarray] = []
    Ey_snapshots: list[np.ndarray] = []
    mode_stride = int(time["mode_diagnostic_stride_steps"])
    noise_stride = int(time["noise_snapshot_stride_steps"])
    first_noise_step = int(round(time["noise_snapshot_t_start"] / time["dt"]))
    last_noise_step = int(round(time["noise_snapshot_t_stop"] / time["dt"]))

    for step in range(int(time["n_steps"]) + 1):
        needs_mode = step % mode_stride == 0
        needs_noise = first_noise_step <= step <= last_noise_step and step % noise_stride == 0
        if needs_mode or needs_noise:
            rho = deposit(particles, grid)
            Ex, Ey = grid.poisson_solve(rho)
            current_time = step * time["dt"]
            if needs_mode:
                mode_times.append(current_time)
                mode.append(grid.mode_amplitude(Ex, benchmark["kx_mode"], benchmark["ky_mode"]))
                field_energy.append(grid.field_energy(Ex, Ey))
                kinetic_energy.append(particles.kinetic_energy())
                total_charge.append(particles.total_charge())
            if needs_noise:
                snapshot_times.append(current_time)
                rho_snapshots.append(rho.copy())
                Ex_snapshots.append(Ex.copy())
                Ey_snapshots.append(Ey.copy())
        if step == int(time["n_steps"]):
            break
        pic_step(particles, grid, time["dt"], deposit, gather)

    return {
        "times": np.asarray(mode_times, dtype=np.float64),
        "mode": np.asarray(mode, dtype=np.complex128),
        "field_energy": np.asarray(field_energy, dtype=np.float64),
        "kinetic_energy": np.asarray(kinetic_energy, dtype=np.float64),
        "total_charge": np.asarray(total_charge, dtype=np.float64),
        "snapshot_times": np.asarray(snapshot_times, dtype=np.float64),
        "rho_snapshots": np.asarray(rho_snapshots, dtype=np.float64),
        "Ex_snapshots": np.asarray(Ex_snapshots, dtype=np.float64),
        "Ey_snapshots": np.asarray(Ey_snapshots, dtype=np.float64),
    }


def write_run(
    directory: Path,
    history: dict[str, np.ndarray],
    metadata: dict,
) -> None:
    directory.mkdir(parents=True)
    np.savez_compressed(directory / "history_and_noise_snapshots.npz", **history)
    (directory / "run_parameters.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def atomic_seed_directory(output: Path, seed: int) -> tuple[Path, Path]:
    target = output.resolve() / f"seed{seed}"
    partial = target.with_name(target.name + ".partial")
    if target.exists() or partial.exists():
        raise FileExistsError(f"refusing to overwrite {target} or {partial}")
    partial.parent.mkdir(parents=True, exist_ok=True)
    partial.mkdir()
    return partial, target


def common_metadata(
    config: dict,
    config_path: Path,
    launch_record: dict,
    grid: Grid2D,
    seed: int,
    ppc: int,
) -> dict:
    return {
        "campaign_id": config["campaign_id"],
        "config_path_at_launch": str(config_path),
        "config_sha256": file_sha256(config_path),
        **launch_record,
        "seed": seed,
        "ppc": ppc,
        "Nx": grid.Nx,
        "Ny": grid.Ny,
        "Lx": grid.Lx,
        "Ly": grid.Ly,
        "dx": grid.dx,
        "dy": grid.dy,
        "alpha": config["benchmark"]["alpha"],
        "k_lambda_D": config["benchmark"]["k_lambda_D"],
        "kx_mode": config["benchmark"]["kx_mode"],
        "ky_mode": config["benchmark"]["ky_mode"],
        "velocity_loading": config["benchmark"]["velocity_loading"],
        "dt": config["time_integration"]["dt"],
        "n_steps": config["time_integration"]["n_steps"],
        "simulation_t_stop": config["time_integration"]["simulation_t_stop"],
        "solver": config["field_solver"],
        "rho_snapshot_stage": "after CIC deposit and arm W, before Poisson",
        "field_snapshot_stage": "after arm density W and spectral Poisson, before return-path W and particle gather",
    }


def check_seed(config: dict, seed: int) -> None:
    confirmation = config["confirmation"]
    if not confirmation["seed_start"] <= seed <= confirmation["seed_stop_inclusive"]:
        raise ValueError(f"seed {seed} is outside the frozen 40-seed set")


def simulate_low_seed(
    config: dict,
    config_path: Path,
    preflight_path: Path | None,
    output: Path,
    seed: int,
) -> None:
    check_seed(config, seed)
    launch_record = validate_launch_preflight(config, config_path, preflight_path)
    partial, target = atomic_seed_directory(output, seed)
    try:
        grid = frozen_grid(config)
        ppc = int(config["benchmark"]["ppc"])
        initial = make_initial_particles(config, grid, seed, ppc)
        initial_digest = particle_sha256(initial)
        observed_digests = []
        for arm in config["arms"]:
            particles = clone_particles(initial)
            observed_digest = particle_sha256(particles)
            observed_digests.append(observed_digest)
            if observed_digest != initial_digest:
                raise AssertionError("particle clone differs before its arm starts")
            history = run_history(particles, grid, config, float(arm["sigma_cells"]))
            metadata = {
                **common_metadata(config, config_path, launch_record, grid, seed, ppc),
                "arm": arm,
                "initial_particle_sha256": initial_digest,
                "paired_low_ppc_arms": [item["id"] for item in config["arms"]],
            }
            write_run(partial / arm["id"], history, metadata)
        if len(set(observed_digests)) != 1:
            raise AssertionError("the three low-PPC arms did not start byte-identically")
        manifest = {
            "seed": seed,
            "initial_particle_sha256": initial_digest,
            "arm_initial_hashes": dict(zip((arm["id"] for arm in config["arms"]), observed_digests)),
            "all_three_arm_hashes_match": True,
        }
        (partial / "pairing_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        partial.rename(target)
    except Exception:
        # Preserve partial output for diagnosis; never silently replace it.
        raise
    print(json.dumps({"completed": str(target), "seed": seed, "particle_sha256": initial_digest}))


def simulate_reference_seed(
    config: dict,
    config_path: Path,
    preflight_path: Path | None,
    output: Path,
    seed: int,
) -> None:
    check_seed(config, seed)
    launch_record = validate_launch_preflight(config, config_path, preflight_path)
    partial, target = atomic_seed_directory(output, seed)
    try:
        grid = frozen_grid(config)
        ppc = int(config["same_campaign_noise_metrics"]["reference_ppc"])
        particles = make_initial_particles(config, grid, seed, ppc)
        initial_digest = particle_sha256(particles)
        history = run_history(particles, grid, config, 0.0)
        arm = {"id": "cic_reference", "name": "PPC-4096 CIC reference", "sigma_cells": 0.0}
        metadata = {
            **common_metadata(config, config_path, launch_record, grid, seed, ppc),
            "arm": arm,
            "initial_particle_sha256": initial_digest,
            "reference_role": "finite-PPC reference for density and grid-field squared-error power only",
        }
        write_run(partial / arm["id"], history, metadata)
        partial.rename(target)
    except Exception:
        raise
    print(json.dumps({"completed": str(target), "seed": seed, "particle_sha256": initial_digest}))


def complex_mode_ratio(after: np.ndarray, before: np.ndarray, m: int, n: int) -> complex:
    after_hat = np.fft.rfft2(after)
    before_hat = np.fft.rfft2(before)
    return complex(after_hat[m % before.shape[0], n] / before_hat[m % before.shape[0], n])


def preflight(config: dict, config_path: Path, output: Path) -> None:
    target = output.resolve()
    if target.exists():
        raise FileExistsError(f"refusing to overwrite preflight directory: {target}")
    target.mkdir(parents=True)
    grid = frozen_grid(config)
    benchmark = config["benchmark"]
    m, n = int(benchmark["kx_mode"]), int(benchmark["ky_mode"])
    X, Y = np.meshgrid(grid.x, grid.y, indexing="ij")
    rho_mode = np.cos(2.0 * np.pi * m * X / grid.Lx + 2.0 * np.pi * n * Y / grid.Ly)
    Ex_raw, Ey_raw = grid.poisson_solve(rho_mode)
    probe = Particles2D(
        x=((np.arange(257) + 0.371) * grid.Lx / 257.0) % grid.Lx,
        y=((np.arange(257) * 73 + 0.619) * grid.Ly / 257.0) % grid.Ly,
        vx=np.zeros(257),
        vy=np.zeros(257),
        w=np.zeros(257),
        Lx=grid.Lx,
        Ly=grid.Ly,
    )
    _, baseline_deposit, baseline_gather = arm_methods(grid, 0.0)
    del baseline_deposit
    raw_force_x, raw_force_y = baseline_gather(Ex_raw, Ey_raw, probe, grid)
    raw_force = np.concatenate((raw_force_x, raw_force_y))
    rows = []
    for arm in config["arms"]:
        sigma = float(arm["sigma_cells"])
        W, _, gather = arm_methods(grid, sigma)
        rho_after = apply_W_to_grid(rho_mode, W, grid)
        Ex_after, Ey_after = grid.poisson_solve(rho_after)
        force_x, force_y = gather(Ex_after, Ey_after, probe, grid)
        force = np.concatenate((force_x, force_y))
        measured_deposition = complex_mode_ratio(rho_after, rho_mode, m, n).real
        measured_field = complex_mode_ratio(Ex_after, Ex_raw, m, n).real
        measured_force = float(np.dot(raw_force, force) / np.dot(raw_force, raw_force))
        expected_W = math.exp(
            -0.5
            * (
                (2.0 * math.pi * m * sigma / grid.Nx) ** 2
                + (2.0 * math.pi * n * sigma / grid.Ny) ** 2
            )
        )
        expected_force = expected_W**2
        errors = {
            "deposition": abs(measured_deposition - expected_W),
            "field": abs(measured_field - expected_W),
            "particle_force": abs(measured_force - expected_force),
        }
        if max(errors.values()) > 2.0e-12:
            raise AssertionError(f"transfer mismatch for {arm['id']}: {errors}")
        rows.append(
            {
                "arm": arm["id"],
                "sigma_cells": sigma,
                "mode": [m, n],
                "expected_W_after_deposition": expected_W,
                "measured_W_after_deposition": measured_deposition,
                "expected_W_after_field_calculation": expected_W,
                "measured_W_after_field_calculation": measured_field,
                "expected_W2_particle_force": expected_force,
                "measured_W2_particle_force": measured_force,
                "maximum_absolute_error": max(errors.values()),
            }
        )

    seed = int(config["confirmation"]["seed_start"])
    initial = make_initial_particles(config, grid, seed, int(config["benchmark"]["ppc"]))
    clone_hashes = [particle_sha256(clone_particles(initial)) for _ in config["arms"]]
    if len(set(clone_hashes)) != 1 or clone_hashes[0] != particle_sha256(initial):
        raise AssertionError("paired clone preflight failed")
    result = {
        "campaign_id": config["campaign_id"],
        "status": "PASS",
        "production_submitted": False,
        "single_mode_transfer_tolerance": 2.0e-12,
        "transfers": rows,
        "paired_clone_check": {
            "seed": seed,
            "n_arms": len(clone_hashes),
            "all_hashes_match": True,
            "particle_sha256": clone_hashes[0],
        },
        "source_sha256": runtime_source_hashes(config_path),
    }
    (target / "preflight.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


def smoke_test(config: dict, config_path: Path, output: Path) -> None:
    """Exercise the production integration/capture path on a tiny in-memory case."""
    target = output.resolve()
    if target.exists():
        raise FileExistsError(f"refusing to overwrite smoke-test directory: {target}")
    target.mkdir(parents=True)
    tiny = json.loads(json.dumps(config))
    tiny["grid"].update(
        {
            "Nx": 8,
            "Ny": 4,
            "dx": tiny["grid"]["Lx"] / 8.0,
            "dy": tiny["grid"]["Ly"] / 4.0,
        }
    )
    tiny["benchmark"]["ppc"] = 4
    tiny["benchmark"]["per_cell_velocity_quadrature"] = "2 by 2 smoke-test quadrature"
    tiny["benchmark"]["particle_loading"] = "16 by 8 smoke-test marker lattice"
    tiny["time_integration"].update(
        {
            "n_steps": 4,
            "simulation_t_stop": 0.2,
            "mode_diagnostic_stride_steps": 1,
            "mode_diagnostic_cadence": 0.05,
            "noise_snapshot_t_start": 0.0,
            "noise_snapshot_t_stop": 0.2,
            "noise_snapshot_stride_steps": 2,
            "noise_snapshot_cadence": 0.1,
        }
    )
    grid = frozen_grid(tiny)
    seed = int(config["confirmation"]["seed_start"])
    initial = make_initial_particles(tiny, grid, seed, 4)
    initial_hash = particle_sha256(initial)
    arms = []
    for arm in tiny["arms"]:
        particles = clone_particles(initial)
        if particle_sha256(particles) != initial_hash:
            raise AssertionError("smoke-test clone mismatch")
        history = run_history(particles, grid, tiny, float(arm["sigma_cells"]))
        expected_shapes = {
            "times": (5,),
            "mode": (5,),
            "field_energy": (5,),
            "kinetic_energy": (5,),
            "total_charge": (5,),
            "snapshot_times": (3,),
            "rho_snapshots": (3, 8, 4),
            "Ex_snapshots": (3, 8, 4),
            "Ey_snapshots": (3, 8, 4),
        }
        observed_shapes = {key: value.shape for key, value in history.items()}
        if observed_shapes != expected_shapes:
            raise AssertionError(
                f"smoke-test capture mismatch for {arm['id']}: {observed_shapes}"
            )
        if not np.all(np.isfinite(history["field_energy"])):
            raise AssertionError(f"nonfinite smoke-test field energy for {arm['id']}")
        arms.append(
            {
                "arm": arm["id"],
                "shapes": {key: list(value) for key, value in observed_shapes.items()},
                "final_field_energy": float(history["field_energy"][-1]),
                "charge_range": [
                    float(np.min(history["total_charge"])),
                    float(np.max(history["total_charge"])),
                ],
            }
        )
    result = {
        "campaign_id": config["campaign_id"],
        "status": "PASS",
        "production_submitted": False,
        "purpose": "miniature code-path test only; not physical evidence",
        "frozen_config_sha256": file_sha256(config_path),
        "miniature_overrides": {
            "Nx": 8,
            "Ny": 4,
            "ppc": 4,
            "n_steps": 4,
            "dt": 0.05,
        },
        "initial_particle_sha256": initial_hash,
        "arms": arms,
    }
    (target / "smoke_test.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--preflight",
        type=Path,
        help="passing preflight.json required by production simulation commands",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    preflight_parser = subparsers.add_parser("preflight")
    preflight_parser.add_argument("--output", type=Path, required=True)
    smoke_parser = subparsers.add_parser("smoke-test")
    smoke_parser.add_argument("--output", type=Path, required=True)
    low_parser = subparsers.add_parser("simulate-low-seed")
    low_parser.add_argument("--output", type=Path, required=True)
    low_parser.add_argument("--seed", type=int, required=True)
    reference_parser = subparsers.add_parser("simulate-reference-seed")
    reference_parser.add_argument("--output", type=Path, required=True)
    reference_parser.add_argument("--seed", type=int, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config, config_path = load_config(args.config)
    if args.command == "preflight":
        preflight(config, config_path, args.output)
    elif args.command == "smoke-test":
        smoke_test(config, config_path, args.output)
    elif args.command == "simulate-low-seed":
        simulate_low_seed(config, config_path, args.preflight, args.output, args.seed)
    else:
        simulate_reference_seed(config, config_path, args.preflight, args.output, args.seed)


if __name__ == "__main__":
    main()
