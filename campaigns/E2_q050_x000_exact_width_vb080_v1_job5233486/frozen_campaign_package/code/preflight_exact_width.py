#!/usr/bin/env python3
"""Read-only preflight for the frozen q050 exact-width campaign."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path
import platform
import sys

import numpy as np


def load_runner(path: Path):
    spec = importlib.util.spec_from_file_location("q050_runner", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--config-sha256", required=True)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--reuse-audit", type=Path)
    args = parser.parse_args()
    runner_path = Path(__file__).with_name("run_E2_q050_exact_arm.py")
    runner = load_runner(runner_path)
    observed_config_hash = runner.sha256_file(args.config)
    if observed_config_hash != args.config_sha256:
        raise AssertionError("config hash mismatch")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    runner.verify_source(args.project, config)
    sigma = float(config["arms"]["q050_x000"]["sigma_y_dy"])
    q_value = (2.0 * math.pi * config["mode"][1] * sigma / config["grid"]["Ny"]) ** 2
    W = math.exp(-0.5 * q_value)
    if abs(q_value - config["filter_chain"]["q_at_target_mode"]) > 1.0e-15:
        raise AssertionError("q value does not match frozen config")
    if abs(W - config["filter_chain"]["q050_W_one_application"]) > 1.0e-15:
        raise AssertionError("one-way transfer does not match frozen config")

    for candidate in (args.project, args.project / "scripts"):
        if str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
    import run_linearized_delta_f_mode as production

    seeds = config["seeds"] if args.seed is None else [args.seed]
    pairing = {}
    pairing_arrays = {}
    production_expected = config["existing_cic_reuse"][
        "production_runtime_reconstructed_initial_state_sha256"
    ]
    production_hash_match = {}
    audit = (
        None
        if args.reuse_audit is None
        else json.loads(args.reuse_audit.read_text(encoding="utf-8"))
    )
    production_array_match = {}
    for seed in seeds:
        hashes = {}
        arrays = {}
        for arm in ("cic", "q050_x000"):
            production_args = runner.make_args(
                production, config, arm, int(seed), Path("unused-preflight-output")
            )
            _, particles = production.build_markers(production_args)
            hashes[arm], arrays[arm] = runner.hash_initial_state(particles)
        if hashes["cic"] != hashes["q050_x000"]:
            raise AssertionError(f"pairing mismatch for seed {seed}")
        if arrays["cic"] != arrays["q050_x000"]:
            raise AssertionError(f"per-array pairing mismatch for seed {seed}")
        pairing[str(seed)] = hashes["cic"]
        pairing_arrays[str(seed)] = arrays["cic"]
        production_hash_match[str(seed)] = hashes["cic"] == production_expected[str(seed)]
        if audit is not None:
            production_array_match[str(seed)] = {
                name: value == audit["seeds"][str(seed)][f"{name}_sha256"]
                for name, value in arrays["cic"].items()
            }
    result = {
        "status": "PASS",
        "no_simulation_run": True,
        "runtime": {"python": platform.python_version(), "numpy": np.__version__},
        "config_sha256": observed_config_hash,
        "source_hashes_verified": True,
        "q_at_target_mode": q_value,
        "W_one_application": W,
        "W_squared_particle_force": W**2,
        "same_runtime_cic_q050_reconstructed_state_sha256": pairing,
        "same_runtime_cic_q050_reconstructed_array_sha256": pairing_arrays,
        "matches_meluxina_production_runtime_hash": production_hash_match,
        "matches_meluxina_production_runtime_array_hash": production_array_match,
        "interpretation": (
            "CIC and q050 initialize identically within this runtime. Exact hashes "
            "can differ across NumPy runtimes in w_delta; only the MeluXina "
            "production-runtime hashes are used to admit historical CIC reuse."
        ),
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
