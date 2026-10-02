#!/usr/bin/env python3
"""Paired common-window sensitivity reduction for the frozen v2 campaign.

This reducer is deliberately secondary.  It verifies the v2 raw-history and
Stage-1 locks, takes the intersection of the registered CIC and C4 windows,
and resamples paired seed indices.  It never changes the registered primary
arm-specific result or its promotion decision.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import analyze_weak_landau_confirmatory as base  # noqa: E402
import analyze_weak_landau_confirmatory_v2 as v2  # noqa: E402


DEFAULT_CONFIG = HERE / "weak_landau_confirmatory_v2_global_velocity.json"


def analyse(
    config_path: Path,
    low_root: Path,
    decisions_path: Path,
    primary_path: Path,
    output: Path,
) -> dict:
    target = output.resolve()
    if target.exists():
        raise FileExistsError(f"refusing to overwrite sensitivity directory: {target}")

    config, resolved_config, config_hash = v2.load_config(config_path)
    data = base.verify_low_tree(config, config_hash, low_root.resolve())
    decisions = base.verify_decision_lock(
        decisions_path.resolve(), config, config_hash, low_root.resolve(), data
    )
    primary = json.loads(primary_path.read_text(encoding="utf-8"))
    if primary["campaign_id"] != config["campaign_id"]:
        raise AssertionError("primary result belongs to another campaign")
    if primary["window_decisions_sha256"] != base.file_sha256(decisions_path.resolve()):
        raise AssertionError("primary result does not reference the supplied Stage-1 lock")
    if primary["n_fresh_seeds"] != len(data["seeds"]) or primary["development_seeds_pooled"]:
        raise AssertionError("primary result does not use exactly the frozen fresh sample")

    selected = decisions["decisions"]
    start = max(selected["cic"]["accepted_t_start"], selected["c4_sigma1"]["accepted_t_start"])
    stop = min(selected["cic"]["accepted_t_stop"], selected["c4_sigma1"]["accepted_t_stop"])
    if not math.isclose(start, 5.0, abs_tol=1.0e-12) or not math.isclose(
        stop, 30.1, abs_tol=1.0e-12
    ):
        raise AssertionError(f"unexpected common window [{start}, {stop}]")

    arms = ("cic", "c4_sigma1")
    central = {
        arm: base.fit_fixed_window(data["histories"][arm], data["times"], start, stop, config)
        for arm in arms
    }
    np.testing.assert_allclose(central["cic"]["gamma"], -0.0685811, rtol=0.0, atol=5.0e-8)
    np.testing.assert_allclose(central["c4_sigma1"]["gamma"], -0.0698686, rtol=0.0, atol=5.0e-8)

    n_draws = int(config["fit_rule"]["bootstrap_draws"])
    rng = np.random.default_rng(int(config["fit_rule"]["bootstrap_seed"]))
    ratios = np.full(n_draws, np.nan, dtype=np.float64)
    cic_rates = np.full(n_draws, np.nan, dtype=np.float64)
    c4_rates = np.full(n_draws, np.nan, dtype=np.float64)
    paired_quality = 0
    for draw in range(n_draws):
        indices = rng.integers(0, len(data["seeds"]), size=len(data["seeds"]))
        cic = base.fit_fixed_window(
            data["histories"]["cic"][indices], data["times"], start, stop, config
        )
        c4 = base.fit_fixed_window(
            data["histories"]["c4_sigma1"][indices], data["times"], start, stop, config
        )
        cic_rates[draw] = cic["gamma"]
        c4_rates[draw] = c4["gamma"]
        if np.isfinite(cic["gamma"]) and np.isfinite(c4["gamma"]) and cic["gamma"] != 0.0:
            ratios[draw] = c4["gamma"] / cic["gamma"]
        paired_quality += int(cic["quality_pass"] and c4["quality_pass"])

    valid = ratios[np.isfinite(ratios)]
    if len(valid) != n_draws:
        raise AssertionError(f"only {len(valid)} of {n_draws} paired draws are valid")
    common_ratio = central["c4_sigma1"]["gamma"] / central["cic"]["gamma"]
    common_ci = np.percentile(valid, [2.5, 97.5])
    registered = primary["rates"]["bootstrap"]["comparisons"]["c4_sigma1"]

    result = {
        "campaign_id": config["campaign_id"],
        "analysis_role": "secondary common-window sensitivity analysis",
        "primary_result_preserved": True,
        "primary_result_path": str(primary_path.resolve()),
        "primary_result_sha256": base.file_sha256(primary_path.resolve()),
        "registered_primary_arm_specific": {
            "cic_window": [
                selected["cic"]["accepted_t_start"], selected["cic"]["accepted_t_stop"]
            ],
            "c4_window": [
                selected["c4_sigma1"]["accepted_t_start"],
                selected["c4_sigma1"]["accepted_t_stop"],
            ],
            "c4_to_cic_ratio": registered["central_ratio_to_cic"],
            "ratio_ci_lo": registered["ratio_ci_lo"],
            "ratio_ci_hi": registered["ratio_ci_hi"],
            "promotion_status": primary["promotion_status"],
        },
        "common_window": [start, stop],
        "window_construction": "intersection of the locked CIC and C4 accepted windows",
        "observable": config["fit_rule"]["observable"],
        "operation_order": config["fit_rule"]["operation_order"],
        "central": central,
        "paired_bootstrap": {
            "seed": int(config["fit_rule"]["bootstrap_seed"]),
            "n_draws": n_draws,
            "valid_draws": int(len(valid)),
            "valid_fraction": float(len(valid) / n_draws),
            "paired_quality_fraction": float(paired_quality / n_draws),
            "c4_to_cic_ratio": float(common_ratio),
            "ratio_ci_lo": float(common_ci[0]),
            "ratio_ci_hi": float(common_ci[1]),
        },
        "window_sensitivity": {
            "ratio_shift_common_minus_primary": float(
                common_ratio - registered["central_ratio_to_cic"]
            ),
            "percentage_point_shift": float(
                100.0 * (common_ratio - registered["central_ratio_to_cic"])
            ),
            "point_estimate_direction_changes_across_one": bool(
                registered["central_ratio_to_cic"] < 1.0 < common_ratio
            ),
        },
        "interpretation": (
            "Both point estimates and both paired intervals lie inside [0.90,1.10], but "
            "the common-window point estimate is above one whereas the registered "
            "arm-specific point estimate is below one.  Both common-window fits fail "
            "the preregistered two-amplitude-e-folding gate, so this sensitivity result "
            "does not change the primary FAIL decision."
        ),
        "n_fresh_seeds": len(data["seeds"]),
        "development_seeds_pooled": False,
        "config_path": str(resolved_config),
        "config_sha256": config_hash,
        "window_decisions_path": str(decisions_path.resolve()),
        "window_decisions_sha256": base.file_sha256(decisions_path.resolve()),
        "input_file_sha256": data["file_hashes"],
    }

    target.mkdir(parents=True)
    np.savez_compressed(
        target / "paired_bootstrap_draws.npz",
        cic_gamma=cic_rates,
        c4_gamma=c4_rates,
        c4_to_cic_ratio=ratios,
    )
    (target / "common_window_sensitivity.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--input-low", type=Path, required=True)
    parser.add_argument("--window-decisions", type=Path, required=True)
    parser.add_argument("--primary-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    analyse(
        args.config,
        args.input_low,
        args.window_decisions,
        args.primary_result,
        args.output,
    )


if __name__ == "__main__":
    main()
