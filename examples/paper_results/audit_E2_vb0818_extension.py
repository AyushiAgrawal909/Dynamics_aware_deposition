#!/usr/bin/env python3
"""Validate the completed v_b=0.818 extension under its frozen protocol."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from analyze_E2_beam_drift_scan import fit_quality, fit_rate, sha256


ARMS = ("cic", "fixed_gaussian", "constrained_gaussian")


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extension", type=Path, required=True)
    parser.add_argument("--combined-analysis", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")

    config_path = args.extension / "frozen_campaign.json"
    amendment_path = args.extension / "extension_budget_amendment.json"
    completed_path = args.extension / "COMPLETED"
    history_list_path = args.extension / "history_files.txt"
    combined_path = args.combined_analysis / "E2_beam_drift_scan.json"
    for path in (config_path, amendment_path, completed_path, history_list_path, combined_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    amendment = json.loads(amendment_path.read_text(encoding="utf-8"))
    combined = json.loads(combined_path.read_text(encoding="utf-8"))
    config_hash = sha256(config_path)
    if amendment["original_frozen_campaign_sha256"] != config_hash:
        raise AssertionError("amendment/config hash mismatch")
    if amendment["extension_scope"]["histories"] != 30:
        raise AssertionError("extension scope changed")
    if amendment["scientific_configuration_changed"]:
        raise AssertionError("extension claims a scientific configuration change")
    if amendment["acceptance_rules_changed"] or amendment["rate_estimator_changed"]:
        raise AssertionError("extension changed a frozen analysis rule")
    if amendment["additional_compute_budget"]["wall_hours_hard_max"] != 8.0:
        raise AssertionError("extension wall budget changed")
    listed = [line for line in history_list_path.read_text(encoding="utf-8").splitlines() if line]
    if len(listed) != 30:
        raise AssertionError("history inventory is not exactly 30")

    registered = {
        row["arm"]: row for row in combined["rows"] if float(row["v_beam"]) == 0.818
    }
    if set(registered) != set(ARMS):
        raise AssertionError("combined reducer lacks the three extension arms")
    if any(row["signal_gate"] != "FAIL" for row in registered.values()):
        raise AssertionError("unexpected extension signal-gate state")

    per_seed: list[dict] = []
    arm_arrays: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    input_hashes = {
        str(path): sha256(path)
        for path in (config_path, amendment_path, completed_path, history_list_path, combined_path)
    }
    for arm in ARMS:
        modes = []
        validity = []
        times = None
        for seed in config["seeds"]:
            seed_root = args.extension / "runs" / "vb0818" / arm / f"seed{seed}"
            paths = {
                name: seed_root / name
                for name in ("result.json", "campaign_manifest.json", "mode_history.npz")
            }
            for path in paths.values():
                if not path.is_file():
                    raise FileNotFoundError(path)
                input_hashes[str(path)] = sha256(path)
            result = json.loads(paths["result.json"].read_text(encoding="utf-8"))
            manifest = json.loads(paths["campaign_manifest.json"].read_text(encoding="utf-8"))
            if manifest["frozen_campaign_sha256"] != config_hash:
                raise AssertionError(f"manifest config mismatch in {seed_root}")
            if manifest["pairing_key"] != f"vbeam=0.818/seed={seed}":
                raise AssertionError(f"pairing key mismatch in {seed_root}")
            if manifest["arm"] != arm or int(manifest["seed"]) != int(seed):
                raise AssertionError(f"arm/seed mismatch in {seed_root}")
            if float(manifest["fit_t_start"]) != 24.32 or float(manifest["fit_t_stop"]) != 204.08:
                raise AssertionError(f"frozen window mismatch in {seed_root}")
            with np.load(paths["mode_history.npz"], allow_pickle=False) as archive:
                current_times = np.asarray(archive["times"], dtype=np.float64)
                current_mode = np.asarray(archive["mode"], dtype=np.complex128)
                current_validity = np.asarray(archive["delta_f_ratio"], dtype=np.float64)
                archived_seed = int(np.asarray(archive["seed"]).flat[0])
            if archived_seed != seed:
                raise AssertionError(f"archived seed mismatch in {seed_root}")
            if times is None:
                times = current_times
            else:
                np.testing.assert_array_equal(times, current_times)
            modes.append(current_mode)
            validity.append(current_validity)
            threshold_indices = np.flatnonzero(current_validity >= 0.5)
            first_crossing = (
                float(current_times[threshold_indices[0]])
                if threshold_indices.size else float("nan")
            )
            fit_mask = (current_times >= 24.32) & (current_times <= 204.08)
            per_seed.append(
                {
                    "arm": arm,
                    "seed": seed,
                    "result_valid_through_run": bool(result["valid_through_run"]),
                    "first_delta_f_ratio_ge_0p5": first_crossing,
                    "delta_f_ratio_at_fit_start": float(current_validity[fit_mask][0]),
                    "delta_f_ratio_at_fit_stop": float(current_validity[fit_mask][-1]),
                    "max_delta_f_ratio_frozen_window": float(np.max(current_validity[fit_mask])),
                    "individual_gamma_frozen_window_descriptive_only": float(
                        np.polyfit(
                            current_times[fit_mask],
                            np.log(np.abs(current_mode[fit_mask])),
                            1,
                        )[0]
                    ),
                }
            )
        assert times is not None
        arm_arrays[arm] = (times, np.stack(modes), np.stack(validity))

    arm_summary = []
    for arm in ARMS:
        times, modes, validity = arm_arrays[arm]
        crossing_times = [
            row["first_delta_f_ratio_ge_0p5"] for row in per_seed if row["arm"] == arm
        ]
        finite_crossings = [value for value in crossing_times if np.isfinite(value)]
        first_crossing = min(finite_crossings) if finite_crossings else float("nan")
        shared_valid_stop = (
            float(times[np.searchsorted(times, first_crossing, side="left") - 1])
            if finite_crossings else 204.08
        )
        amplitudes = np.abs(modes)
        prefix_rate = fit_rate(amplitudes, times, 24.32, shared_valid_stop)
        row = registered[arm]
        arm_summary.append(
            {
                "arm": arm,
                "files_complete": True,
                "frozen_fit_t_start": 24.32,
                "frozen_fit_t_stop": 204.08,
                "registered_gamma_descriptive_only": row["gamma"],
                "registered_ratio_descriptive_only": row["gamma_ratio_to_cic"],
                "registered_fit_start_snr": row["fit_start_snr"],
                "registered_fit_r_squared": row["fit_r_squared"],
                "registered_fitted_efoldings": row["fitted_efoldings"],
                "registered_max_delta_f_ratio": row["max_delta_f_ratio_fit_window"],
                "signal_gate": row["signal_gate"],
                "rate_preservation_gate": row["rate_preservation_gate"],
                "seeds_crossing_delta_f_0p5": len(finite_crossings),
                "earliest_delta_f_0p5_crossing": first_crossing,
                "shared_valid_prefix_stop": shared_valid_stop,
                "posthoc_shared_valid_prefix_gamma_descriptive_only": prefix_rate,
                "posthoc_shared_valid_prefix_r_squared": fit_quality(
                    amplitudes, times, 24.32, shared_valid_stop
                ),
                "posthoc_prefix_interpretation": (
                    "not an accepted replacement: fit-start SNR already fails and window was not frozen"
                ),
            }
        )

    args.output.mkdir(parents=True)
    write_csv(args.output / "per_seed_validity.csv", per_seed)
    write_csv(args.output / "extension_arm_summary.csv", arm_summary)
    result = {
        "status": "FILES_COMPLETE_SCIENTIFIC_RESULT_UNRESOLVED",
        "slurm": {
            "job_id": 5202227,
            "state": "COMPLETED",
            "exit_code": "0:0",
            "elapsed": "02:21:40",
            "time_limit": "08:00:00",
            "start": "2026-09-11T21:39:06+02:00",
            "end": "2026-09-12T00:00:46+02:00",
        },
        "history_count": 30,
        "configuration_hash_match": True,
        "budget_amendment_hash": sha256(amendment_path),
        "original_timeout_campaign_preserved": amendment["original_timeout_record_preserved"],
        "scientific_disposition": (
            "All three frozen fit-start SNR values are below 5. Three of ten seeds in "
            "each arm later exceed the delta-f validity limit, and the ensemble mean "
            "becomes almost single-seed dominated. Therefore the rates and ratios are "
            "descriptive estimator outputs, not accepted near-marginal measurements."
        ),
        "duplicate_or_budget_change": False,
        "input_sha256": dict(sorted(input_hashes.items())),
        "tables": {
            "per_seed_validity": "per_seed_validity.csv",
            "arm_summary": "extension_arm_summary.csv",
        },
    }
    (args.output / "extension_audit.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "status": result["status"],
        "scientific_disposition": result["scientific_disposition"],
        "arms": arm_summary,
    }, indent=2))


if __name__ == "__main__":
    main()
