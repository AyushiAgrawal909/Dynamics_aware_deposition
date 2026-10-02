#!/usr/bin/env python3
"""Locked two-stage reduction for the weak-Landau confirmatory campaign.

Stage 1 records accepted windows and end reasons without calculating any
between-arm rate ratio.  Stage 2 verifies the stage-1 and raw-file hashes,
keeps those windows fixed, and only then computes comparative intervals and
same-campaign squared-error power metrics.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np


WORKSPACE = Path(__file__).resolve().parents[2]
PROJECT = WORKSPACE / "pic_kde_2d"
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from pic_kde_2d.validation.landau_censored_fit import (  # noqa: E402
    coherent_amplitude,
    fit_censored_mode,
    geometric_envelope,
    quiet_tail_floor,
)


DEFAULT_CONFIG = Path(__file__).resolve().parent / "weak_landau_confirmatory_v1.json"
HISTORY_NAME = "history_and_noise_snapshots.npz"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_config(path: Path) -> tuple[dict, Path, str]:
    resolved = path.resolve()
    config = json.loads(resolved.read_text(encoding="utf-8"))
    if config["campaign_id"] != "weak_landau_confirmatory_v1":
        raise AssertionError("unexpected campaign")
    confirmation = config["confirmation"]
    if confirmation["n_seeds"] != 40 or not confirmation["fixed_sample"]:
        raise AssertionError("confirmatory sample is not frozen at 40")
    if not confirmation["sequential_seed_addition_forbidden"]:
        raise AssertionError("sequential seed addition must be forbidden")
    return config, resolved, file_sha256(resolved)


def expected_seeds(config: dict) -> list[int]:
    confirmation = config["confirmation"]
    return list(range(confirmation["seed_start"], confirmation["seed_stop_inclusive"] + 1))


def arm_ids(config: dict) -> list[str]:
    return [arm["id"] for arm in config["arms"]]


def verify_exact_seed_tree(root: Path, seeds: list[int]) -> None:
    expected = {f"seed{seed}" for seed in seeds}
    observed = {path.name for path in root.glob("seed*")}
    if observed != expected:
        missing = sorted(expected.difference(observed))
        extra = sorted(observed.difference(expected))
        raise AssertionError(f"fixed seed tree mismatch below {root}: missing={missing}, extra={extra}")


def verify_low_tree(config: dict, config_hash: str, root: Path) -> dict[str, Any]:
    seeds = expected_seeds(config)
    verify_exact_seed_tree(root, seeds)
    histories: dict[str, list[np.ndarray]] = {arm: [] for arm in arm_ids(config)}
    times: np.ndarray | None = None
    file_hashes: dict[str, str] = {}
    particle_hashes: dict[int, str] = {}
    runtime_source_sha256: dict[str, str] | None = None
    preflight_sha256: str | None = None
    for seed in seeds:
        seed_root = root / f"seed{seed}"
        pairing_path = seed_root / "pairing_manifest.json"
        pairing = json.loads(pairing_path.read_text(encoding="utf-8"))
        if pairing["seed"] != seed or not pairing["all_three_arm_hashes_match"]:
            raise AssertionError(f"pairing failure for seed {seed}")
        if set(pairing["arm_initial_hashes"]) != set(arm_ids(config)):
            raise AssertionError(f"pairing arm set mismatch for seed {seed}")
        if len(set(pairing["arm_initial_hashes"].values())) != 1:
            raise AssertionError(f"arm particle hashes differ for seed {seed}")
        particle_hashes[seed] = pairing["initial_particle_sha256"]
        file_hashes[str(pairing_path.relative_to(root))] = file_sha256(pairing_path)
        for arm in arm_ids(config):
            arm_root = seed_root / arm
            metadata_path = arm_root / "run_parameters.json"
            history_path = arm_root / HISTORY_NAME
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata["config_sha256"] != config_hash:
                raise AssertionError(f"config hash mismatch in {metadata_path}")
            if metadata["seed"] != seed or metadata["ppc"] != 1024:
                raise AssertionError(f"seed/PPC mismatch in {metadata_path}")
            if metadata["arm"]["id"] != arm:
                raise AssertionError(f"arm mismatch in {metadata_path}")
            if metadata["initial_particle_sha256"] != particle_hashes[seed]:
                raise AssertionError(f"metadata pairing hash mismatch in {metadata_path}")
            if runtime_source_sha256 is None:
                runtime_source_sha256 = metadata["source_sha256"]
                preflight_sha256 = metadata["preflight_sha256"]
            elif (
                metadata["source_sha256"] != runtime_source_sha256
                or metadata["preflight_sha256"] != preflight_sha256
            ):
                raise AssertionError(f"runtime source/preflight mismatch in {metadata_path}")
            with np.load(history_path, allow_pickle=False) as archive:
                required = {
                    "times",
                    "mode",
                    "field_energy",
                    "kinetic_energy",
                    "total_charge",
                    "snapshot_times",
                    "rho_snapshots",
                    "Ex_snapshots",
                    "Ey_snapshots",
                }
                missing = required.difference(archive.files)
                if missing:
                    raise KeyError(f"{history_path} lacks {sorted(missing)}")
                current_times = np.asarray(archive["times"], dtype=np.float64)
                mode = np.asarray(archive["mode"], dtype=np.complex128)
            if times is None:
                times = current_times
            else:
                np.testing.assert_array_equal(times, current_times)
            if mode.shape != current_times.shape:
                raise AssertionError(f"mode/time mismatch in {history_path}")
            histories[arm].append(mode)
            file_hashes[str(history_path.relative_to(root))] = file_sha256(history_path)
            file_hashes[str(metadata_path.relative_to(root))] = file_sha256(metadata_path)
    assert times is not None
    assert runtime_source_sha256 is not None and preflight_sha256 is not None
    return {
        "times": times,
        "histories": {arm: np.stack(values) for arm, values in histories.items()},
        "file_hashes": file_hashes,
        "particle_hashes": particle_hashes,
        "seeds": seeds,
        "runtime_source_sha256": runtime_source_sha256,
        "preflight_sha256": preflight_sha256,
    }


def fit_kwargs(config: dict) -> dict[str, Any]:
    rule = config["fit_rule"]
    return {
        "fit_t_start": rule["fit_t_start"],
        "fit_t_stop_max": rule["fit_t_stop_max"],
        "noise_floor_start": rule["noise_floor_t_start"],
        "noise_floor_stop": rule["noise_floor_t_stop"],
        "noise_factor": rule["noise_threshold_factor"],
        "envelope_period": rule["envelope_period"],
        "persistence_points": rule["persistence_points"],
        "min_points": rule["min_points"],
        "min_duration": rule["min_duration"],
        "min_model_efoldings": rule["min_fitted_amplitude_efoldings"],
        "min_r_squared": rule["min_fit_r_squared"],
    }


def first_persistent_true(mask: np.ndarray, count: int) -> int | None:
    if len(mask) < count:
        return None
    runs = np.convolve(mask.astype(np.int64), np.ones(count, dtype=np.int64), mode="valid")
    matches = np.flatnonzero(runs == count)
    return None if matches.size == 0 else int(matches[0])


def postfit_recurrence(envelope: np.ndarray, times: np.ndarray, fit_stop: float, config: dict) -> dict:
    rule = config["fit_rule"]
    period_points = int(rule["envelope_points_at_frozen_cadence"])
    indices = np.flatnonzero(
        (times > fit_stop)
        & (times < rule["noise_floor_t_start"])
        & (np.arange(len(times)) >= period_points)
    )
    if indices.size == 0:
        return {"flag": False, "first_t": None}
    delta = np.log(np.maximum(envelope[indices], 1.0e-300)) - np.log(
        np.maximum(envelope[indices - period_points], 1.0e-300)
    )
    start = first_persistent_true(delta >= 0.0, int(rule["persistence_points"]))
    return {
        "flag": start is not None,
        "first_t": None if start is None else float(times[indices[start]]),
    }


def ending_reason(fit: dict, times: np.ndarray, config: dict) -> tuple[str, str]:
    rule = config["fit_rule"]
    if fit["floor_censored"]:
        exclusive = float(fit["censor_t"] - rule["lookahead_guard_points"] * np.median(np.diff(times)))
        return (
            "noise_threshold",
            f"first persistent 5x-floor crossing at t={fit['censor_t']:.1f}; "
            f"centered-envelope guard excludes t>={exclusive:.1f}",
        )
    if rule["fit_t_stop_max"] < float(times[-1]):
        return "analysis_cap", f"no persistent crossing before frozen t={rule['fit_t_stop_max']:.1f} cap"
    if fit["n_points"] >= 2:
        return "simulation_end", f"no persistent crossing before final sample t={times[-1]:.1f}"
    return "insufficient_or_nonfinite_data", "fewer than two accepted finite samples"


def select_windows(config: dict, config_path: Path, config_hash: str, low_root: Path, output: Path) -> None:
    target = output.resolve()
    if target.exists():
        raise FileExistsError(f"refusing to overwrite window-decision directory: {target}")
    data = verify_low_tree(config, config_hash, low_root.resolve())
    times = data["times"]
    decisions = {}
    for arm in arm_ids(config):
        histories = data["histories"][arm]
        fit = fit_censored_mode(histories, times, **fit_kwargs(config))
        amplitude = coherent_amplitude(histories)
        envelope, envelope_points = geometric_envelope(
            amplitude, times, config["fit_rule"]["envelope_period"]
        )
        if envelope_points != config["fit_rule"]["envelope_points_at_frozen_cadence"]:
            raise AssertionError("envelope point count differs from the preregistration")
        reason, detail = ending_reason(fit, times, config)
        recurrence = postfit_recurrence(envelope, times, float(fit["fit_t_stop"]), config)
        decisions[arm] = {
            "accepted_t_start": fit["fit_t_start"],
            "accepted_t_stop": fit["fit_t_stop"],
            "fit_duration": fit["fit_duration"],
            "gamma": fit["gamma"],
            "fitted_amplitude_efoldings": fit["model_efoldings"],
            "observed_log_drop": fit["observed_log_drop"],
            "fit_r_squared": fit["r_squared"],
            "fit_start_signal_to_floor": fit["fit_start_snr"],
            "noise_floor": fit["noise_floor"],
            "floor_window": [fit["floor_t_start"], fit["floor_t_stop"]],
            "censor_t": None if not fit["floor_censored"] else fit["censor_t"],
            "ending_reason": reason,
            "ending_detail": detail,
            "recurrence_diagnostic": recurrence,
            "envelope_distortion_flag": bool(fit["r_squared"] < config["fit_rule"]["min_fit_r_squared"]),
            "central_fit_quality_pass": fit["quality_pass"],
            "signal_to_threshold_capacity_from_fit_start": float(
                math.log(fit["fit_start_snr"] / config["fit_rule"]["noise_threshold_factor"])
            ),
        }
    result = {
        "campaign_id": config["campaign_id"],
        "stage": 1,
        "rate_ratios_computed": False,
        "production_data_root": str(low_root.resolve()),
        "config_path": str(config_path),
        "config_sha256": config_hash,
        "n_seeds": len(data["seeds"]),
        "seeds": data["seeds"],
        "all_pairing_hashes_verified": True,
        "runtime_source_sha256": data["runtime_source_sha256"],
        "preflight_sha256": data["preflight_sha256"],
        "input_file_sha256": data["file_hashes"],
        "decisions": decisions,
        "instruction": "Freeze and retain this file before running stage 2; stage 2 refuses changed histories or windows.",
    }
    target.mkdir(parents=True)
    decision_path = target / "window_decisions.json"
    decision_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


def fit_fixed_window(histories: np.ndarray, times: np.ndarray, start: float, stop: float, config: dict) -> dict:
    rule = config["fit_rule"]
    raw = coherent_amplitude(histories)
    envelope, _ = geometric_envelope(raw, times, rule["envelope_period"])
    floor = quiet_tail_floor(
        envelope,
        times,
        tail_start=rule["noise_floor_t_start"],
        tail_stop=rule["noise_floor_t_stop"],
    )["noise_floor"]
    indices = np.flatnonzero(
        (times >= start) & (times <= stop) & np.isfinite(envelope) & (envelope > 0.0)
    )
    if len(indices) < 2:
        return {
            "gamma": float("nan"),
            "r_squared": float("nan"),
            "duration": 0.0,
            "efoldings": 0.0,
            "fit_start_snr": float("nan"),
            "quality_pass": False,
        }
    x = times[indices]
    y = np.log(envelope[indices])
    coefficients = np.polyfit(x, y, 1)
    predicted = np.polyval(coefficients, x)
    residual = float(np.sum((y - predicted) ** 2))
    total = float(np.sum((y - np.mean(y)) ** 2))
    gamma = float(coefficients[0])
    r_squared = float(1.0 - residual / total) if total > 0 else float("nan")
    duration = float(x[-1] - x[0])
    efoldings = float(max(0.0, -gamma * duration))
    fit_start_snr = float(envelope[indices[0]] / max(floor, 1.0e-300))
    quality = bool(
        gamma < 0.0
        and len(indices) >= rule["min_points"]
        and duration >= rule["min_duration"]
        and efoldings >= rule["min_fitted_amplitude_efoldings"]
        and np.isfinite(r_squared)
        and r_squared >= rule["min_fit_r_squared"]
        and fit_start_snr >= rule["min_fit_start_signal_to_floor"]
    )
    return {
        "gamma": gamma,
        "r_squared": r_squared,
        "duration": duration,
        "efoldings": efoldings,
        "fit_start_snr": fit_start_snr,
        "n_points": int(len(indices)),
        "quality_pass": quality,
    }


def verify_decision_lock(
    decisions_path: Path,
    config: dict,
    config_hash: str,
    low_root: Path,
    data: dict,
) -> dict:
    decisions = json.loads(decisions_path.read_text(encoding="utf-8"))
    if decisions["stage"] != 1 or decisions["rate_ratios_computed"]:
        raise AssertionError("invalid stage-1 decision file")
    if decisions["campaign_id"] != config["campaign_id"]:
        raise AssertionError("decision campaign mismatch")
    if decisions["config_sha256"] != config_hash:
        raise AssertionError("configuration changed after window selection")
    if Path(decisions["production_data_root"]).resolve() != low_root.resolve():
        raise AssertionError("stage-2 low-data root differs from stage 1")
    if decisions["input_file_sha256"] != data["file_hashes"]:
        raise AssertionError("one or more stage-1 inputs changed before comparative analysis")
    if decisions["seeds"] != data["seeds"]:
        raise AssertionError("seed list changed after window selection")
    if (
        decisions["runtime_source_sha256"] != data["runtime_source_sha256"]
        or decisions["preflight_sha256"] != data["preflight_sha256"]
    ):
        raise AssertionError("runtime source or preflight record changed after window selection")
    if set(decisions["decisions"]) != set(arm_ids(config)):
        raise AssertionError("window arm set mismatch")
    return decisions


def bootstrap_rates(data: dict, decisions: dict, config: dict) -> tuple[dict, list[dict]]:
    histories = data["histories"]
    times = data["times"]
    rule = config["fit_rule"]
    central = {}
    for arm in arm_ids(config):
        decision = decisions["decisions"][arm]
        fit = fit_fixed_window(
            histories[arm], times, decision["accepted_t_start"], decision["accepted_t_stop"], config
        )
        if not math.isclose(fit["gamma"], decision["gamma"], rel_tol=0.0, abs_tol=5.0e-14):
            raise AssertionError(f"locked central rate mismatch for {arm}")
        central[arm] = fit

    rng = np.random.default_rng(int(rule["bootstrap_seed"]))
    draws: dict[str, list[float]] = {arm: [] for arm in arm_ids(config)}
    ratios: dict[str, list[float]] = {arm: [] for arm in arm_ids(config) if arm != "cic"}
    quality_pairs: dict[str, int] = {arm: 0 for arm in arm_ids(config) if arm != "cic"}
    quality_cic = 0
    n_boot = int(rule["bootstrap_draws"])
    n_seeds = len(data["seeds"])
    for _ in range(n_boot):
        indices = rng.integers(0, n_seeds, size=n_seeds)
        fits = {}
        for arm in arm_ids(config):
            decision = decisions["decisions"][arm]
            fits[arm] = fit_fixed_window(
                histories[arm][indices],
                times,
                decision["accepted_t_start"],
                decision["accepted_t_stop"],
                config,
            )
            draws[arm].append(fits[arm]["gamma"])
        quality_cic += int(fits["cic"]["quality_pass"])
        for arm in ratios:
            if np.isfinite(fits[arm]["gamma"]) and np.isfinite(fits["cic"]["gamma"]) and fits["cic"]["gamma"] != 0.0:
                ratios[arm].append(fits[arm]["gamma"] / fits["cic"]["gamma"])
            quality_pairs[arm] += int(fits["cic"]["quality_pass"] and fits[arm]["quality_pass"])

    reference = abs(config["numerical_reference"]["field_amplitude_gamma"])
    cic_reference_draws = np.abs(np.asarray(draws["cic"])) / reference
    summary = {
        "central": central,
        "bootstrap": {
            "n_draws": n_boot,
            "cic_quality_fraction": quality_cic / n_boot,
            "cic_absolute_ratio_to_discrete_reference": {
                "central": abs(central["cic"]["gamma"]) / reference,
                "ci_lo": float(np.percentile(cic_reference_draws, 2.5)),
                "ci_hi": float(np.percentile(cic_reference_draws, 97.5)),
            },
            "comparisons": {},
        },
    }
    for arm, values in ratios.items():
        array = np.asarray(values, dtype=np.float64)
        summary["bootstrap"]["comparisons"][arm] = {
            "central_ratio_to_cic": central[arm]["gamma"] / central["cic"]["gamma"],
            "ratio_ci_lo": float(np.percentile(array, 2.5)),
            "ratio_ci_hi": float(np.percentile(array, 97.5)),
            "valid_fraction": len(array) / n_boot,
            "paired_quality_fraction": quality_pairs[arm] / n_boot,
        }

    single_rows = []
    for index, seed in enumerate(data["seeds"]):
        for arm in arm_ids(config):
            decision = decisions["decisions"][arm]
            fit = fit_fixed_window(
                histories[arm][index : index + 1],
                times,
                decision["accepted_t_start"],
                decision["accepted_t_stop"],
                config,
            )
            single_rows.append({"seed": seed, "arm": arm, **fit})
    return summary, single_rows


def load_reference_snapshot(
    config: dict,
    config_hash: str,
    expected_source_sha256: dict[str, str],
    expected_preflight_sha256: str,
    root: Path,
    seed: int,
) -> dict[str, np.ndarray]:
    arm_root = root / f"seed{seed}" / "cic_reference"
    metadata_path = arm_root / "run_parameters.json"
    history_path = arm_root / HISTORY_NAME
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata["config_sha256"] != config_hash:
        raise AssertionError(f"reference config mismatch in {metadata_path}")
    if metadata["seed"] != seed or metadata["ppc"] != config["same_campaign_noise_metrics"]["reference_ppc"]:
        raise AssertionError(f"reference seed/PPC mismatch in {metadata_path}")
    if (
        metadata["source_sha256"] != expected_source_sha256
        or metadata["preflight_sha256"] != expected_preflight_sha256
    ):
        raise AssertionError(f"reference runtime source/preflight mismatch in {metadata_path}")
    with np.load(history_path, allow_pickle=False) as archive:
        return {
            key: np.asarray(archive[key], dtype=np.float64)
            for key in ("snapshot_times", "rho_snapshots", "Ex_snapshots", "Ey_snapshots")
        }


def load_low_snapshot(root: Path, seed: int, arm: str) -> dict[str, np.ndarray]:
    path = root / f"seed{seed}" / arm / HISTORY_NAME
    with np.load(path, allow_pickle=False) as archive:
        return {
            key: np.asarray(archive[key], dtype=np.float64)
            for key in ("snapshot_times", "rho_snapshots", "Ex_snapshots", "Ey_snapshots")
        }


def bootstrap_metric_ratio(baseline: np.ndarray, current: np.ndarray, n_boot: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n_boot):
        indices = rng.integers(0, len(baseline), size=len(baseline))
        values.append(float(np.mean(baseline[indices]) / np.mean(current[indices])))
    array = np.asarray(values)
    return {
        "central": float(np.mean(baseline) / np.mean(current)),
        "ci_lo": float(np.percentile(array, 2.5)),
        "ci_hi": float(np.percentile(array, 97.5)),
    }


def noise_metrics(
    config: dict,
    config_hash: str,
    expected_source_sha256: dict[str, str],
    expected_preflight_sha256: str,
    low_root: Path,
    reference_root: Path,
) -> tuple[dict, list[dict]]:
    seeds = expected_seeds(config)
    verify_exact_seed_tree(reference_root, seeds)
    arms = arm_ids(config)
    per_seed_density = {arm: [] for arm in arms}
    per_seed_field = {arm: [] for arm in arms}
    rows = []
    for seed in seeds:
        reference = load_reference_snapshot(
            config,
            config_hash,
            expected_source_sha256,
            expected_preflight_sha256,
            reference_root,
            seed,
        )
        for arm in arms:
            current = load_low_snapshot(low_root, seed, arm)
            np.testing.assert_array_equal(current["snapshot_times"], reference["snapshot_times"])
            density = np.mean(
                (current["rho_snapshots"] - reference["rho_snapshots"]) ** 2,
                axis=(1, 2),
            )
            field = np.mean(
                (current["Ex_snapshots"] - reference["Ex_snapshots"]) ** 2
                + (current["Ey_snapshots"] - reference["Ey_snapshots"]) ** 2,
                axis=(1, 2),
            )
            per_seed_density[arm].append(float(np.mean(density)))
            per_seed_field[arm].append(float(np.mean(field)))
            rows.extend(
                {
                    "seed": seed,
                    "arm": arm,
                    "time": float(time),
                    "density_error_power": float(density[index]),
                    "grid_field_error_power": float(field[index]),
                }
                for index, time in enumerate(current["snapshot_times"])
            )
    n_boot = int(config["fit_rule"]["bootstrap_draws"])
    summary = {"reference": config["same_campaign_noise_metrics"]["reference"], "arms": {}}
    baseline_density = np.asarray(per_seed_density["cic"])
    baseline_field = np.asarray(per_seed_field["cic"])
    for offset, arm in enumerate(arms):
        density = np.asarray(per_seed_density[arm])
        field = np.asarray(per_seed_field[arm])
        summary["arms"][arm] = {
            "mean_density_error_power": float(np.mean(density)),
            "mean_grid_field_error_power": float(np.mean(field)),
            "density_error_power_reduction_vs_cic": bootstrap_metric_ratio(
                baseline_density, density, n_boot, 20261010 + offset
            ),
            "grid_field_error_power_reduction_vs_cic": bootstrap_metric_ratio(
                baseline_field, field, n_boot, 20261110 + offset
            ),
        }
    return summary, rows


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def comparative_analysis(
    config: dict,
    config_path: Path,
    config_hash: str,
    low_root: Path,
    reference_root: Path,
    decisions_path: Path,
    output: Path,
) -> None:
    target = output.resolve()
    if target.exists():
        raise FileExistsError(f"refusing to overwrite comparative-analysis directory: {target}")
    data = verify_low_tree(config, config_hash, low_root.resolve())
    decisions = verify_decision_lock(decisions_path.resolve(), config, config_hash, low_root, data)
    rates, single_rows = bootstrap_rates(data, decisions, config)
    noise, noise_rows = noise_metrics(
        config,
        config_hash,
        data["runtime_source_sha256"],
        data["preflight_sha256"],
        low_root.resolve(),
        reference_root.resolve(),
    )
    comparison = rates["bootstrap"]["comparisons"]["c4_sigma1"]
    absolute = rates["bootstrap"]["cic_absolute_ratio_to_discrete_reference"]
    rule = config["fit_rule"]
    central = rates["central"]
    checks = {
        "cic_two_efoldings": central["cic"]["efoldings"] >= rule["min_fitted_amplitude_efoldings"],
        "c4_two_efoldings": central["c4_sigma1"]["efoldings"] >= rule["min_fitted_amplitude_efoldings"],
        "cic_signal_and_fit_quality": central["cic"]["quality_pass"],
        "c4_signal_and_fit_quality": central["c4_sigma1"]["quality_pass"],
        "paired_bootstrap_quality_fraction": comparison["paired_quality_fraction"] >= 0.80,
        "cic_absolute_rate_reference_interval": absolute["ci_lo"] >= 0.90 and absolute["ci_hi"] <= 1.10,
        "c4_to_cic_paired_interval": comparison["ratio_ci_lo"] >= 0.90 and comparison["ratio_ci_hi"] <= 1.10,
        "coverage_and_traceability": True,
    }
    target.mkdir(parents=True)
    write_csv(target / "single_run_fit_diagnostics.csv", single_rows)
    write_csv(target / "noise_error_power_by_seed_time.csv", noise_rows)
    np.savez_compressed(
        target / "coherent_ensemble_histories.npz",
        times=data["times"],
        **{
            f"{arm}_coherent_complex_mode": np.mean(data["histories"][arm], axis=0)
            for arm in arm_ids(config)
        },
    )
    result = {
        "campaign_id": config["campaign_id"],
        "stage": 2,
        "window_decisions_sha256": file_sha256(decisions_path.resolve()),
        "config_path": str(config_path),
        "config_sha256": config_hash,
        "runtime_source_sha256": data["runtime_source_sha256"],
        "preflight_sha256": data["preflight_sha256"],
        "n_fresh_seeds": len(data["seeds"]),
        "development_seeds_pooled": False,
        "primary_observable": "coherently averaged complex field, then magnitude and envelope",
        "single_run_diagnostics_are_primary": False,
        "rates": rates,
        "same_campaign_noise_metrics": noise,
        "promotion_checks": checks,
        "promotion_status": "PASS" if all(checks.values()) else "FAIL",
    }
    (target / "confirmatory_result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    subparsers = parser.add_subparsers(dest="command", required=True)
    windows = subparsers.add_parser("select-windows")
    windows.add_argument("--input-low", type=Path, required=True)
    windows.add_argument("--output", type=Path, required=True)
    analyze = subparsers.add_parser("analyze")
    analyze.add_argument("--input-low", type=Path, required=True)
    analyze.add_argument("--input-reference", type=Path, required=True)
    analyze.add_argument("--window-decisions", type=Path, required=True)
    analyze.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config, config_path, config_hash = load_config(args.config)
    if args.command == "select-windows":
        select_windows(config, config_path, config_hash, args.input_low, args.output)
    else:
        comparative_analysis(
            config,
            config_path,
            config_hash,
            args.input_low,
            args.input_reference,
            args.window_decisions,
            args.output,
        )


if __name__ == "__main__":
    main()
