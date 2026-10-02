#!/usr/bin/env python3
"""Reduce the preregistered paired E2 beam-drift scan and make its main figure."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[1]
FIGURE_CAPTION_NOTE = "No growing model branch;\nmeasurement unavailable."
for candidate in (WORKSPACE / "pic_kde_2d/paper1_selector_comparison/figures",):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fit_rate(
    amplitudes: np.ndarray,
    times: np.ndarray,
    start: float,
    stop: float,
) -> float:
    """Audited E2 estimator: slope of log(mean of seed magnitudes)."""
    ensemble = np.mean(np.asarray(amplitudes, dtype=np.float64), axis=0)
    mask = (
        (times >= start)
        & (times <= stop)
        & np.isfinite(ensemble)
        & (ensemble > 1.0e-30)
    )
    if np.count_nonzero(mask) < 5:
        return float("nan")
    return float(np.polyfit(times[mask], np.log(ensemble[mask]), 1)[0])


def estimate_growing_floor(
    times: np.ndarray,
    amplitudes: np.ndarray,
    fit_start: float,
    window_duration: float = 2.0,
) -> dict:
    """Existing E2 quietest rolling-median pre-fit floor estimator."""
    ensemble = np.mean(amplitudes, axis=0)
    dt = float(np.median(np.diff(times)))
    width = max(5, int(round(window_duration / dt)))
    stop_limit = int(np.searchsorted(times, fit_start, side="left"))
    if stop_limit < width:
        raise RuntimeError("insufficient pre-fit samples for the growing-mode floor")
    best = None
    for start in range(0, stop_limit - width + 1):
        stop = start + width
        floor = float(np.median(ensemble[start:stop]))
        candidate = (floor, start, stop)
        if best is None or candidate[0] < best[0]:
            best = candidate
    assert best is not None
    floor, start, stop = best
    fit_index = int(np.searchsorted(times, fit_start, side="left"))
    return {
        "noise_floor": floor,
        "floor_t_start": float(times[start]),
        "floor_t_stop": float(times[stop - 1]),
        "fit_start_amplitude": float(ensemble[fit_index]),
        "fit_start_snr": float(ensemble[fit_index] / max(floor, 1.0e-300)),
        "floor_definition": "quietest rolling-median pre-emergence window",
    }


def drift_label(value: float) -> str:
    return f"vb{int(round(1000.0 * value)):04d}"


def fit_quality(
    amplitudes: np.ndarray,
    times: np.ndarray,
    start: float,
    stop: float,
) -> float:
    mask = (times >= start) & (times <= stop)
    x = times[mask]
    y = np.log(np.maximum(np.mean(amplitudes, axis=0)[mask], 1.0e-300))
    slope, intercept = np.polyfit(x, y, 1)
    residual = float(np.sum((y - (slope * x + intercept)) ** 2))
    total = float(np.sum((y - np.mean(y)) ** 2))
    return 1.0 - residual / total if total > 0.0 else 0.0


def load_group(
    root: Path,
    config: dict,
    config_sha256: str,
    scan: dict,
    arm: str,
) -> dict:
    histories = []
    validity = []
    seed_ids = []
    manifests = []
    input_hashes = {}
    times = None
    directory = root / drift_label(float(scan["v_beam"])) / arm
    for seed in config["seeds"]:
        seed_root = directory / f"seed{seed}"
        result_path = seed_root / "result.json"
        history_path = seed_root / "mode_history.npz"
        manifest_path = seed_root / "campaign_manifest.json"
        for path in (result_path, history_path, manifest_path):
            if not path.is_file():
                raise FileNotFoundError(path)
            input_hashes[str(path.relative_to(root))] = sha256(path)
        result = json.loads(result_path.read_text(encoding="utf-8"))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["frozen_campaign_sha256"] != config_sha256:
            raise AssertionError(f"config hash mismatch in {manifest_path}")
        if manifest["pairing_key"] != f"vbeam={scan['v_beam']:.3f}/seed={seed}":
            raise AssertionError(f"pairing key mismatch in {manifest_path}")
        if manifest["arm"] != arm or int(manifest["seed"]) != int(seed):
            raise AssertionError(f"arm/seed mismatch in {manifest_path}")
        if (int(result["kx_mode"]), int(result["ky_mode"])) != tuple(config["mode"]):
            raise AssertionError(f"mode mismatch in {result_path}")
        if int(result["ppc"]) != int(config["ppc"]):
            raise AssertionError(f"PPC mismatch in {result_path}")
        if not bool(result["target_mode_projection"]):
            raise AssertionError(f"mode projection disabled in {result_path}")
        with np.load(history_path, allow_pickle=False) as archive:
            current_times = np.asarray(archive["times"], dtype=np.float64)
            histories.append(np.abs(np.asarray(archive["mode"], dtype=np.complex128)))
            validity.append(np.asarray(archive["delta_f_ratio"], dtype=np.float64))
            archived_seed = int(np.asarray(archive["seed"]).flat[0])
        if archived_seed != seed:
            raise AssertionError(f"NPZ seed mismatch in {history_path}")
        if times is None:
            times = current_times
        else:
            np.testing.assert_array_equal(times, current_times)
        seed_ids.append(seed)
        manifests.append(manifest)
    if seed_ids != config["seeds"]:
        raise AssertionError("seed order or membership differs from frozen config")
    assert times is not None
    return {
        "arm": arm,
        "times": times,
        "amplitudes": np.stack(histories),
        "validity": np.stack(validity),
        "manifests": manifests,
        "input_sha256": input_hashes,
    }


def bootstrap_drift(
    groups: list[dict],
    start: float,
    stop: float,
    nboot: int,
    seed: int,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    n_seeds = groups[0]["amplitudes"].shape[0]
    rng = np.random.default_rng(seed)
    rate_samples = [[] for _ in groups]
    ratio_samples = [[] for _ in groups]
    for _ in range(nboot):
        indices = rng.integers(0, n_seeds, size=n_seeds)
        rates = [
            fit_rate(group["amplitudes"][indices], group["times"], start, stop)
            for group in groups
        ]
        if not np.all(np.isfinite(rates)) or rates[0] <= 0.0:
            continue
        for index, rate in enumerate(rates):
            rate_samples[index].append(float(rate))
            ratio_samples[index].append(float(rate / rates[0]))
    return (
        [np.asarray(values, dtype=np.float64) for values in rate_samples],
        [np.asarray(values, dtype=np.float64) for values in ratio_samples],
    )


def make_figure(rows: list[dict], config: dict, output: Path) -> None:
    import matplotlib.pyplot as plt

    try:
        from paperstyle import COLORS, DOUBLE_COL, apply_style

        apply_style()
        width = DOUBLE_COL
        navy = COLORS["CIC"]
        green = COLORS["C4"]
    except ImportError:
        plt.rcParams.update(
            {
                "font.family": "serif",
                "font.size": 9,
                "axes.labelsize": 9,
                "xtick.labelsize": 8,
                "ytick.labelsize": 8,
                "axes.spines.top": False,
                "axes.spines.right": False,
                "axes.grid": True,
                "grid.alpha": 0.25,
            }
        )
        width, navy, green = 7.0, "#14213D", "#2A7F62"
    red = "#8B1E1E"
    grey = "#737373"
    arms = ("cic", "fixed_gaussian", "constrained_gaussian")
    by_arm = {
        arm: sorted(
            (
                row
                for row in rows
                if (
                    row["arm"] == arm
                    and row["measurement_status"] == "COMPLETE"
                    and row["signal_gate"] == "PASS"
                )
            ),
            key=lambda row: row["v_beam"],
        )
        for arm in arms
    }
    scans = sorted(config["scan_rows"], key=lambda row: row["v_beam"])
    x_model = np.asarray([row["v_beam"] for row in scans])
    fig, axes = plt.subplots(1, 3, figsize=(width, 2.8))

    baseline = by_arm["cic"]
    x_measured = np.asarray([row["v_beam"] for row in baseline])
    theory = np.asarray([row["grid_time_discrete_cic_gamma"] for row in scans])
    measured = np.asarray([row["gamma"] for row in baseline])
    lower = measured - np.asarray([row["gamma_ci_lo"] for row in baseline])
    upper = np.asarray([row["gamma_ci_hi"] for row in baseline]) - measured
    axes[0].plot(x_model, theory, "--", color=grey, linewidth=1.2, label="numerical theory")
    axes[0].errorbar(x_measured, measured, yerr=np.vstack((lower, upper)), color=navy, marker="o", linewidth=1.1, capsize=2, label="CIC measured")
    missing_drifts = sorted(
        {row["v_beam"] for row in rows if row["measurement_status"] != "COMPLETE"}
    )
    unresolved_drifts = sorted(
        {
            row["v_beam"]
            for row in rows
            if row["measurement_status"] == "COMPLETE" and row["signal_gate"] != "PASS"
        }
    )
    for value in missing_drifts:
        axes[0].axvline(value, color=grey, linestyle=":", linewidth=0.8)
    for value in unresolved_drifts:
        axes[0].axvline(value, color=red, linestyle=":", linewidth=0.8)
    axes[0].set_xlabel(r"beam drift $v_b$")
    axes[0].set_ylabel(r"growth rate $\gamma/\omega_p$")
    axes[0].set_title("(a) Baseline branch")
    axes[0].legend(frameon=False, loc="best")

    axes[1].axhspan(0.90, 1.10, color="#DDEBDD", alpha=0.55, linewidth=0)
    for arm, color, marker, label in (
        ("fixed_gaussian", red, "s", "fixed Gaussian"),
        ("constrained_gaussian", green, "o", "constrained"),
    ):
        selected = by_arm[arm]
        central = np.asarray([row["gamma_ratio_to_cic"] for row in selected])
        lo = central - np.asarray([row["ratio_ci_lo"] for row in selected])
        hi = np.asarray([row["ratio_ci_hi"] for row in selected]) - central
        predicted_key = {
            "fixed_gaussian": "fixed_predicted_rate_ratio",
            "constrained_gaussian": "constrained_predicted_rate_ratio",
        }[arm]
        predicted = np.asarray([row[predicted_key] for row in scans])
        arm_x = np.asarray([row["v_beam"] for row in selected])
        plotted_prediction = predicted.copy()
        plotted_prediction[plotted_prediction <= 0.0] = np.nan
        axes[1].plot(x_model, plotted_prediction, ":", color=color, linewidth=1.0)
        axes[1].errorbar(arm_x, central, yerr=np.vstack((lo, hi)), color=color, marker=marker, linewidth=1.1, capsize=2, label=label)
    axes[1].axhline(1.0, color=grey, linewidth=0.8)
    for value in sorted(set(missing_drifts + unresolved_drifts)):
        axes[1].axvline(value, color=grey, linestyle=":", linewidth=0.8)
    if missing_drifts:
        axes[1].text(
            missing_drifts[-1] - 0.003,
            1.047,
            "timeout",
            rotation=90,
            ha="center",
            va="center",
            fontsize=6.7,
            color=grey,
        )
    axes[1].set_xlabel(r"beam drift $v_b$")
    axes[1].set_ylabel(r"$\gamma_{\rm filtered}/\gamma_{\rm CIC}$")
    axes[1].set_title("(b) Rate preservation")
    axes[1].set_ylim(0.885, 1.105)
    axes[1].legend(frameon=False, loc="upper left", fontsize=7.2)

    boundary = [row["sigma_y_10pct_boundary_dy"] for row in scans]
    constrained = [row["constrained_sigma_y_dy"] for row in scans]
    fixed = [row["fixed_sigma_y_dy"] for row in scans]
    axes[2].plot(x_model, boundary, color=red, linewidth=1.2, label="10% boundary")
    axes[2].plot(x_model, constrained, color=green, marker="o", linewidth=1.1, label="frozen 5% setting")
    axes[2].plot(x_model, fixed, "--", color=navy, linewidth=1.0, label="fixed control")
    axes[2].set_xlabel(r"beam drift $v_b$")
    axes[2].set_ylabel(r"transverse width $\sigma_y/\Delta y$")
    axes[2].set_title("(c) Predicted admissible bandwidth")
    axes[2].legend(frameon=False, loc="best")
    axes[2].text(
        0.03,
        0.03,
        rf"fixed $W^2={config['fixed_gaussian_particle_force_W2']:.6f}$",
        transform=axes[2].transAxes,
        fontsize=7.5,
    )
    fig.tight_layout(w_pad=1.0)
    fig.savefig(output / "fig_E2_beam_drift_scan.pdf")
    fig.savefig(output / "fig_E2_beam_drift_scan.png", dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--extension-input",
        type=Path,
        help=(
            "separate, completed extension tree used only for a whole drift block "
            "that is absent from --input; the original input tree is never modified"
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--timeout-record",
        type=Path,
        help=(
            "explicit Slurm-timeout evidence permitting whole missing drift blocks "
            "to be recorded as unresolved; strict completeness remains the default"
        ),
    )
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    config_sha256 = sha256(args.config)
    if config["status"] != "FROZEN_BEFORE_SIMULATION_RESULTS":
        raise AssertionError("campaign config was not frozen before results")
    timeout_record = None
    if args.timeout_record is not None:
        timeout_record = json.loads(args.timeout_record.read_text(encoding="utf-8"))
        if timeout_record["job_state"] != "TIMEOUT":
            raise AssertionError("timeout record does not document a Slurm TIMEOUT")
        if timeout_record["frozen_campaign_sha256"] != config_sha256:
            raise AssertionError("timeout record campaign hash mismatch")
        if float(timeout_record["wall_hours_consumed"]) < float(
            config["budget"]["requested_wall_hours_hard_max"]
        ):
            raise AssertionError("timeout record does not exhaust the frozen budget")

    extension_record = None
    if args.extension_input is not None:
        extension_config = args.extension_input / "frozen_campaign.json"
        extension_amendment = args.extension_input / "extension_budget_amendment.json"
        extension_completed = args.extension_input / "COMPLETED"
        extension_history_list = args.extension_input / "history_files.txt"
        for path in (
            extension_config,
            extension_amendment,
            extension_completed,
            extension_history_list,
        ):
            if not path.is_file():
                raise FileNotFoundError(path)
        if sha256(extension_config) != config_sha256:
            raise AssertionError("extension frozen campaign hash mismatch")
        amendment = json.loads(extension_amendment.read_text(encoding="utf-8"))
        expected_scope = {
            "v_beam": 0.818,
            "drift_index": 4,
            "arms": ["cic", "fixed_gaussian", "constrained_gaussian"],
            "seeds": config["seeds"],
            "histories": 30,
            "fit_window": [24.32, 204.08],
            "run_stop": 206.08,
        }
        if amendment["status"] != "FROZEN_EXTENSION_BUDGET_BEFORE_EXTENSION_RESULTS":
            raise AssertionError("extension amendment was not frozen before results")
        if amendment["original_campaign_id"] != config["campaign_id"]:
            raise AssertionError("extension campaign identity mismatch")
        if amendment["original_job"] != 5201231:
            raise AssertionError("extension does not identify the original timeout job")
        if amendment["original_frozen_campaign_sha256"] != config_sha256:
            raise AssertionError("extension amendment campaign hash mismatch")
        if amendment["extension_scope"] != expected_scope:
            raise AssertionError("extension scope differs from the frozen missing block")
        if amendment["scientific_configuration_changed"]:
            raise AssertionError("extension changed the scientific configuration")
        if amendment["acceptance_rules_changed"] or amendment["rate_estimator_changed"]:
            raise AssertionError("extension changed a frozen analysis rule")
        budget = amendment["additional_compute_budget"]
        if float(budget["wall_hours_hard_max"]) != 8.0 or not budget["seed_count_fixed"]:
            raise AssertionError("extension budget or fixed sample size changed")
        if not budget["no_sequential_seed_addition"]:
            raise AssertionError("extension permits sequential seed addition")
        listed_histories = [
            line for line in extension_history_list.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if len(listed_histories) != 30:
            raise AssertionError("extension history inventory is not exactly 30")
        extension_record = {
            "path": str(args.extension_input),
            "frozen_campaign_sha256": sha256(extension_config),
            "budget_amendment_sha256": sha256(extension_amendment),
            "completion_marker": extension_completed.read_text(encoding="utf-8").strip(),
            "history_inventory_count": len(listed_histories),
            "amendment": amendment,
        }

    rows = []
    all_hashes = {}
    draw_archive = {}
    arm_order = ("cic", "fixed_gaussian", "constrained_gaussian")
    for drift_index, scan in enumerate(config["scan_rows"]):
        scan_root = args.input
        expected = [
            scan_root / drift_label(float(scan["v_beam"])) / arm / f"seed{seed}" / name
            for arm in arm_order
            for seed in config["seeds"]
            for name in ("result.json", "mode_history.npz", "campaign_manifest.json")
        ]
        missing = [path for path in expected if not path.is_file()]
        if missing and args.extension_input is not None:
            extension_expected = [
                args.extension_input
                / "runs"
                / drift_label(float(scan["v_beam"]))
                / arm
                / f"seed{seed}"
                / name
                for arm in arm_order
                for seed in config["seeds"]
                for name in ("result.json", "mode_history.npz", "campaign_manifest.json")
            ]
            extension_missing = [path for path in extension_expected if not path.is_file()]
            if not extension_missing:
                scan_root = args.extension_input / "runs"
                expected = extension_expected
                missing = []
        if missing:
            if timeout_record is None:
                raise FileNotFoundError(missing[0])
            declared = [float(value) for value in timeout_record["missing_v_beam"]]
            if float(scan["v_beam"]) not in declared:
                raise AssertionError("missing drift is not declared in timeout record")
            if len(missing) != len(expected):
                raise AssertionError("partial drift block cannot be timeout-censored")
            for arm in arm_order:
                prediction_key = {
                    "cic": ("grid_time_discrete_cic_gamma", None),
                    "fixed_gaussian": ("fixed_predicted_gamma", "fixed_predicted_rate_ratio"),
                    "constrained_gaussian": (
                        "constrained_predicted_gamma",
                        "constrained_predicted_rate_ratio",
                    ),
                }[arm]
                sigma_x_key = "fixed_sigma_x_dx" if arm == "fixed_gaussian" else "constrained_sigma_x_dx"
                sigma_y_key = "fixed_sigma_y_dy" if arm == "fixed_gaussian" else "constrained_sigma_y_dy"
                rows.append(
                    {
                        "v_beam": scan["v_beam"],
                        "arm": arm,
                        "measurement_status": "UNRESOLVED_BUDGET_TIMEOUT",
                        "missing_reason": timeout_record["unresolved_reason"],
                        "seed_count": 0,
                        "sigma_x_dx": 0.0 if arm == "cic" else scan[sigma_x_key],
                        "sigma_y_dy": 0.0 if arm == "cic" else scan[sigma_y_key],
                        "predicted_gamma": float(scan[prediction_key[0]]),
                        "gamma": None,
                        "gamma_ci_lo": None,
                        "gamma_ci_hi": None,
                        "prediction_inside_measured_95pct_interval": None,
                        "relative_prediction_error": None,
                        "predicted_rate_ratio": 1.0 if prediction_key[1] is None else float(scan[prediction_key[1]]),
                        "gamma_ratio_to_cic": None,
                        "ratio_ci_lo": None,
                        "ratio_ci_hi": None,
                        "fit_t_start": scan["fit_t_start"],
                        "fit_t_stop": scan["fit_t_stop"],
                        "fit_r_squared": None,
                        "fitted_efoldings": None,
                        "noise_floor": None,
                        "floor_t_start": None,
                        "floor_t_stop": None,
                        "fit_start_amplitude": None,
                        "fit_start_snr": None,
                        "floor_definition": None,
                        "max_delta_f_ratio_fit_window": None,
                        "positive_growth_gate": "UNRESOLVED",
                        "duration_gate": "UNRESOLVED_BUDGET_TIMEOUT",
                        "signal_gate": "UNRESOLVED",
                        "rate_preservation_gate": "UNRESOLVED_BUDGET_TIMEOUT",
                        "bootstrap_valid": 0,
                    }
                )
            continue
        groups = [
            load_group(scan_root, config, config_sha256, scan, arm)
            for arm in arm_order
        ]
        times = groups[0]["times"]
        for group in groups[1:]:
            np.testing.assert_array_equal(times, group["times"])
        start = float(scan["fit_t_start"])
        stop = float(scan["fit_t_stop"])
        if start < times[0] or stop > times[-1]:
            raise AssertionError("frozen fit window is outside a saved history")
        central = [fit_rate(group["amplitudes"], times, start, stop) for group in groups]
        rate_draws, ratio_draws = bootstrap_drift(
            groups,
            start,
            stop,
            int(config["uncertainty"]["bootstrap_samples"]),
            int(config["uncertainty"]["bootstrap_seed"]) + drift_index,
        )
        if any(values.size != int(config["uncertainty"]["bootstrap_samples"]) for values in rate_draws):
            raise AssertionError("not all preregistered paired bootstrap draws were valid")
        for index, (arm, group, gamma) in enumerate(zip(arm_order, groups, central)):
            prediction_key = {
                "cic": ("grid_time_discrete_cic_gamma", None),
                "fixed_gaussian": ("fixed_predicted_gamma", "fixed_predicted_rate_ratio"),
                "constrained_gaussian": (
                    "constrained_predicted_gamma",
                    "constrained_predicted_rate_ratio",
                ),
            }[arm]
            predicted_gamma = float(scan[prediction_key[0]])
            predicted_ratio = 1.0 if prediction_key[1] is None else float(scan[prediction_key[1]])
            rates = rate_draws[index]
            ratios = ratio_draws[index]
            gamma_lo, gamma_hi = np.percentile(rates, (2.5, 97.5))
            ratio_lo, ratio_hi = np.percentile(ratios, (2.5, 97.5))
            duration = stop - start
            r_squared = fit_quality(group["amplitudes"], times, start, stop)
            floor = estimate_growing_floor(times, group["amplitudes"], start)
            fit_mask = (times >= start) & (times <= stop)
            max_validity = float(np.max(group["validity"][:, fit_mask]))
            fitted_efoldings = float(max(0.0, gamma * duration))
            positive_growth = bool(gamma > 0.0 and gamma_lo > 0.0)
            duration_pass = fitted_efoldings >= float(config["quality_rules"]["fitted_efoldings_min"])
            signal_pass = bool(
                positive_growth
                and r_squared >= float(config["quality_rules"]["fit_r_squared_min"])
                and floor["fit_start_snr"] >= float(config["quality_rules"]["fit_start_snr_min"])
                and max_validity < float(config["quality_rules"]["delta_f_ratio_max"])
            )
            if arm == "cic":
                rate_gate = "NOT_APPLICABLE"
            elif not signal_pass or not duration_pass:
                rate_gate = "UNRESOLVED"
            else:
                rate_gate = (
                    "PASS"
                    if ratio_lo >= 0.90 and ratio_hi <= 1.10
                    else "FAIL"
                )
            row = {
                "v_beam": scan["v_beam"],
                "arm": arm,
                "measurement_status": "COMPLETE",
                "missing_reason": None,
                "seed_count": len(config["seeds"]),
                "sigma_x_dx": group["manifests"][0]["sigma_x_dx"],
                "sigma_y_dy": group["manifests"][0]["sigma_y_dy"],
                "predicted_gamma": predicted_gamma,
                "gamma": gamma,
                "gamma_ci_lo": float(gamma_lo),
                "gamma_ci_hi": float(gamma_hi),
                "prediction_inside_measured_95pct_interval": bool(
                    gamma_lo <= predicted_gamma <= gamma_hi
                ),
                "relative_prediction_error": (
                    abs(gamma - predicted_gamma) / abs(gamma)
                    if gamma != 0.0
                    else None
                ),
                "predicted_rate_ratio": predicted_ratio,
                "gamma_ratio_to_cic": gamma / central[0],
                "ratio_ci_lo": float(ratio_lo),
                "ratio_ci_hi": float(ratio_hi),
                "fit_t_start": start,
                "fit_t_stop": stop,
                "fit_r_squared": r_squared,
                "fitted_efoldings": fitted_efoldings,
                **floor,
                "max_delta_f_ratio_fit_window": max_validity,
                "positive_growth_gate": "PASS" if positive_growth else "FAIL",
                "duration_gate": "PASS" if duration_pass else "FAIL",
                "signal_gate": "PASS" if signal_pass else "FAIL",
                "rate_preservation_gate": rate_gate,
                "bootstrap_valid": int(rates.size),
            }
            rows.append(row)
            key = f"{drift_label(float(scan['v_beam']))}_{arm}"
            draw_archive[f"{key}_gamma"] = rates
            draw_archive[f"{key}_ratio"] = ratios
            all_hashes.update(group["input_sha256"])

    args.output.mkdir(parents=True)
    with (args.output / "E2_beam_drift_scan.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(args.output / "paired_bootstrap_draws.npz", **draw_archive)
    result = {
        "campaign_id": config["campaign_id"],
        "scope": config["scope"],
        "frozen_campaign_sha256": config_sha256,
        "rate_definition": config["fit"]["observable"],
        "bootstrap_unit": config["uncertainty"]["unit"],
        "selection_scope": config["selection_scope"],
        "figure_caption_note": FIGURE_CAPTION_NOTE,
        "fixed_filter_control": {
            "sigma_y_dy": config["fixed_gaussian_sigma_y_dy"],
            "W_one_application": config["fixed_gaussian_W_one_application"],
            "particle_force_W2": config["fixed_gaussian_particle_force_W2"],
        },
        "measurement_completeness": {
            "expected_histories": int(config["budget"]["histories"]),
            "complete_histories": int(sum(row["seed_count"] for row in rows)),
            "complete_drift_values": sorted(
                {
                    row["v_beam"]
                    for row in rows
                    if row["measurement_status"] == "COMPLETE"
                }
            ),
            "unresolved_drift_values": sorted(
                {
                    row["v_beam"]
                    for row in rows
                    if row["measurement_status"] != "COMPLETE"
                }
            ),
            "timeout_record": None
            if args.timeout_record is None
            else {
                "path": str(args.timeout_record),
                "sha256": sha256(args.timeout_record),
                **timeout_record,
            },
            "extension_record": extension_record,
        },
        "input_sha256": all_hashes,
        "rows": rows,
    }
    (args.output / "E2_beam_drift_scan.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    make_figure(rows, config, args.output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
