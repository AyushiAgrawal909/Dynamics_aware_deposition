#!/usr/bin/env python3
"""Fail if final exact-width campaign artifacts are not synchronized."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    manifest = json.loads((ROOT / "FINAL_CAMPAIGN_MANIFEST.json").read_text())
    config = json.loads(
        (ROOT / "config/q050_x000_vb080_exact_width.json").read_text()
    )
    status = json.loads((ROOT / "PACKAGE_STATUS.json").read_text())
    audit = json.loads(
        (
            ROOT
            / "evidence/existing_cic_vb0800_5201231/reuse_audit.json"
        ).read_text()
    )
    seeds = list(range(200, 210))
    assert config["seeds"] == seeds
    assert manifest["seeds"]["final_reuse_design"] == seeds
    assert status["final_seed_design"] == seeds
    assert config["cost"]["new_histories"] == 10
    assert config["cost"]["expected_cpu_node_hours"] == 0.6
    assert config["cost"]["hard_cap_cpu_node_hours"] == 1.0
    assert config["arms"]["q050_x000"]["sigma_x_dx"] == 0.0
    assert config["arms"]["q050_x000"]["sigma_y_dy"] == 0.1716392815470568
    assert config["analysis"]["fit_t_start"] == 24.32
    assert config["analysis"]["fit_t_stop"] == 69.16
    assert config["acceptance"]["rate_preservation_interval"] == [0.9, 1.1]
    config_hash = sha256(ROOT / "config/q050_x000_vb080_exact_width.json")
    assert audit["config_sha256"] == config_hash
    assert audit["all_registered_cic_quality_gates_pass"] is True
    assert audit["historical_saved_particle_array_hash_available"] is False
    for relative, expected in manifest["sha256"].items():
        assert sha256(ROOT / relative) == expected, relative
    sbatch = (ROOT / "meluxina/submit_q050_exact_width_cpu.sbatch").read_text()
    assert "SEEDS=(200 201 202 203 204 205 206 207 208 209)" in sbatch
    assert "#SBATCH --cpus-per-task=10" in sbatch
    assert "#SBATCH --time=01:00:00" in sbatch
    assert "--arm q050_x000" in sbatch
    assert "ARMS=(cic q050_x000)" not in sbatch
    print("PASS: final exact-width package is synchronized")


if __name__ == "__main__":
    main()
