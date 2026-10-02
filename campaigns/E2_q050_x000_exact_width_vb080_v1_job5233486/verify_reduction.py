#!/usr/bin/env python3
"""Verify a replay of the frozen job-5233486 exact-width reducer."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
ARCHIVED = ROOT / "raw_remote_output" / "analysis"
EXPECTED_CONFIG_SHA256 = (
    "3b763155abe28c4b105a1824969ff411a6d0b1b6a5f3d9319b23ff3c39ec4800"
)


def load_csv(path: Path) -> dict[tuple[int, str], dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return {(int(row["seed"]), row["arm"]): row for row in rows}


def assert_close(actual: float, expected: float, *, atol: float = 1.0e-11) -> None:
    if not np.isclose(actual, expected, rtol=0.0, atol=atol):
        raise AssertionError(f"{actual!r} != {expected!r} within {atol}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    args = parser.parse_args()

    archived = json.loads((ARCHIVED / "summary.json").read_text(encoding="utf-8"))
    candidate = json.loads((args.candidate / "summary.json").read_text(encoding="utf-8"))
    for summary in (archived, candidate):
        if summary["campaign_id"] != "E2_q050_x000_exact_width_vb080_v1":
            raise AssertionError("campaign identity mismatch")
        if summary["config_sha256"] != EXPECTED_CONFIG_SHA256:
            raise AssertionError("frozen config hash mismatch")
        if summary["status"] != "COMPLETE":
            raise AssertionError(f"reducer status is {summary['status']}")
        if summary["fit_window"] != [24.32, 69.16]:
            raise AssertionError("fit window changed")
        if summary["n_paired_seeds"] != 10:
            raise AssertionError("paired sample size changed")
        if not summary["primary_preservation_established"]:
            raise AssertionError("primary archived verdict not reproduced")
        if not summary["secondary_preservation_established"]:
            raise AssertionError("secondary archived verdict not reproduced")
        if not all(
            all(gates.values()) for gates in summary["quality_gates"].values()
        ):
            raise AssertionError("one or more frozen quality gates failed")

    scalar_keys = (
        "primary_individual_rate_mean_ratio",
        "secondary_ensemble_amplitude_ratio",
        "model_operating_ratio",
    )
    interval_keys = (
        "primary_paired_95pct_interval",
        "secondary_paired_95pct_interval",
    )
    for key in scalar_keys:
        assert_close(float(candidate[key]), float(archived[key]))
    for key in interval_keys:
        np.testing.assert_allclose(
            np.asarray(candidate[key], dtype=float),
            np.asarray(archived[key], dtype=float),
            rtol=0.0,
            atol=1.0e-11,
        )
    if candidate["input_sha256"] != archived["input_sha256"]:
        raise AssertionError("reducer input hashes changed")

    archived_rows = load_csv(ARCHIVED / "individual_seed_rates.csv")
    candidate_rows = load_csv(args.candidate / "individual_seed_rates.csv")
    if set(archived_rows) != set(candidate_rows) or len(candidate_rows) != 20:
        raise AssertionError("individual-rate row set changed")
    for key in sorted(archived_rows):
        old = archived_rows[key]
        new = candidate_rows[key]
        if old["initial_state_sha256"] != new["initial_state_sha256"]:
            raise AssertionError(f"initial-state hash changed for {key}")
        for column in (
            "individual_gamma",
            "individual_r_squared",
            "max_delta_f_ratio",
        ):
            assert_close(float(new[column]), float(old[column]))

    print("PASS_EXACT_WIDTH_REDUCTION_REPRODUCED")
    print(
        json.dumps(
            {
                "primary_ratio": candidate["primary_individual_rate_mean_ratio"],
                "primary_95pct_interval": candidate["primary_paired_95pct_interval"],
                "secondary_ratio": candidate["secondary_ensemble_amplitude_ratio"],
                "secondary_95pct_interval": candidate["secondary_paired_95pct_interval"],
                "model_ratio": candidate["model_operating_ratio"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
