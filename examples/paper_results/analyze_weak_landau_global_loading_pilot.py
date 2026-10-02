#!/usr/bin/env python3
"""Reduce the preregistered three-seed global-velocity loading pilot."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[1]
PROJECT = WORKSPACE / "pic_kde_2d"
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from pic_kde_2d.validation.landau_censored_fit import (  # noqa: E402
    coherent_amplitude,
    geometric_envelope,
)


DEFAULT_RULE = HERE / "weak_landau_global_loading_pilot_v1_analysis.json"
DEFAULT_FAILED_V1 = HERE / "results/weak_landau_confirmatory_production_20260911/low"
HISTORY = "history_and_noise_snapshots.npz"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_arm(root: Path, seeds: list[int], arm: str) -> tuple[np.ndarray, np.ndarray, list[dict], dict[str, str]]:
    times = None
    histories = []
    metadata = []
    hashes = {}
    for seed in seeds:
        directory = root / f"seed{seed}" / arm
        archive_path = directory / HISTORY
        metadata_path = directory / "run_parameters.json"
        if not archive_path.exists() or not metadata_path.exists():
            raise FileNotFoundError(f"missing pilot input in {directory}")
        with np.load(archive_path) as archive:
            if "times" not in archive.files or "mode" not in archive.files:
                raise KeyError(f"{archive_path} lacks times/mode; found {archive.files}")
            current_times = np.asarray(archive["times"], dtype=np.float64)
            mode = np.asarray(archive["mode"], dtype=np.complex128)
        if times is None:
            times = current_times
        else:
            np.testing.assert_array_equal(times, current_times)
        item = json.loads(metadata_path.read_text(encoding="utf-8"))
        if item["seed"] != seed or item["arm"]["id"] != arm or item["ppc"] != 1024:
            raise AssertionError(f"metadata mismatch in {metadata_path}")
        if item["campaign_id"] != "weak_landau_global_loading_pilot_v1":
            raise AssertionError(f"wrong pilot campaign in {metadata_path}")
        if "global inverse-CDF" not in item["velocity_loading"]:
            raise AssertionError(f"wrong velocity layout in {metadata_path}")
        histories.append(mode)
        metadata.append(item)
        hashes[str(archive_path.relative_to(root))] = sha256(archive_path)
        hashes[str(metadata_path.relative_to(root))] = sha256(metadata_path)
    assert times is not None
    return times, np.stack(histories), metadata, hashes


def verify_pairing(root: Path, seeds: list[int], arms: list[str]) -> dict[str, str]:
    hashes = {}
    for seed in seeds:
        path = root / f"seed{seed}" / "pairing_manifest.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest["seed"] != seed or not manifest["all_arm_hashes_match"]:
            raise AssertionError(f"pairing failed in {path}")
        values = [manifest["arm_initial_hashes"][arm] for arm in arms]
        if len(set(values)) != 1 or values[0] != manifest["initial_particle_sha256"]:
            raise AssertionError(f"arm hashes differ in {path}")
        hashes[str(path.relative_to(root))] = sha256(path)
    return hashes


def fit_fixed(times: np.ndarray, envelope: np.ndarray, window: list[float]) -> dict:
    mask = (times >= window[0]) & (times <= window[1]) & np.isfinite(envelope) & (envelope > 0.0)
    x = times[mask]
    y = np.log(envelope[mask])
    if x.size < 3:
        raise AssertionError("fixed pilot fit has fewer than three samples")
    slope, intercept = np.polyfit(x, y, 1)
    predicted = slope * x + intercept
    residual = float(np.sum((y - predicted) ** 2))
    total = float(np.sum((y - np.mean(y)) ** 2))
    return {
        "t_start": float(x[0]),
        "t_stop": float(x[-1]),
        "n_points": int(x.size),
        "gamma": float(slope),
        "intercept": float(intercept),
        "r_squared": float(1.0 - residual / total),
        "fitted_amplitude_efoldings": float(-slope * (x[-1] - x[0])),
    }


def at_time(times: np.ndarray, values: np.ndarray, target: float) -> float:
    return float(values[int(np.argmin(np.abs(times - target)))])


def failed_v1_envelope(root: Path, arm: str, times: np.ndarray, period: float) -> np.ndarray:
    paths = sorted(root.glob(f"seed*/{arm}/{HISTORY}"))
    if len(paths) != 40:
        raise AssertionError(f"expected 40 failed-v1 {arm} histories, found {len(paths)}")
    histories = []
    for path in paths:
        with np.load(path) as archive:
            np.testing.assert_array_equal(times, archive["times"])
            histories.append(np.asarray(archive["mode"], dtype=np.complex128))
    amplitude = coherent_amplitude(np.stack(histories))
    envelope, _ = geometric_envelope(amplitude, times, period)
    return envelope


def make_figure(output: Path, times: np.ndarray, data: dict, window: list[float]) -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 8,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    colors = {"cic": "#102a43", "c4_sigma1": "#8b1e2d"}
    labels = {"cic": "CIC", "c4_sigma1": "C4, sigma=1 dx"}
    figure, axes = plt.subplots(1, 2, figsize=(7.0, 2.8))
    for arm in ("cic", "c4_sigma1"):
        color = colors[arm]
        histories = data[arm]["histories"]
        for history in histories:
            axes[0].plot(times, history.real, color=color, alpha=0.13, linewidth=0.55)
        axes[0].plot(times, data[arm]["coherent_mode"].real, color=color, linewidth=1.4, label=labels[arm])
    axes[0].set_xlim(0.0, 30.0)
    axes[0].set_xlabel(r"$t\,\omega_p$")
    axes[0].set_ylabel(r"$\Re\,\widehat E_x(1,0)$")
    axes[0].set_title("(a) fresh complex-mode histories")
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False)

    axes[1].axvspan(window[0], window[1], color="0.7", alpha=0.15)
    for arm in ("cic", "c4_sigma1"):
        color = colors[arm]
        envelope = data[arm]["envelope"]
        fit = data[arm]["fit"]
        axes[1].semilogy(times, envelope, color=color, linewidth=1.4, label=labels[arm])
        fit_times = np.asarray(window)
        axes[1].semilogy(
            fit_times,
            np.exp(fit["intercept"] + fit["gamma"] * fit_times),
            color=color,
            linestyle="--",
            linewidth=1.0,
        )
    axes[1].axvline(35.2444, color="0.45", linestyle=":", linewidth=0.9, label="two-e-fold target")
    axes[1].set_xlim(0.0, 50.0)
    axes[1].set_xlabel(r"$t\,\omega_p$")
    axes[1].set_ylabel("coherent geometric envelope")
    axes[1].set_title("(b) fixed pilot fit [5,20]")
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False, fontsize=7)
    figure.tight_layout(w_pad=1.6)
    figure.savefig(output / "fig_weak_landau_global_loading_pilot.pdf")
    figure.savefig(output / "fig_weak_landau_global_loading_pilot.png", dpi=300)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--failed-v1", type=Path, default=DEFAULT_FAILED_V1)
    parser.add_argument("--rule", type=Path, default=DEFAULT_RULE)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    args.output.mkdir(parents=True)

    rule = json.loads(args.rule.read_text(encoding="utf-8"))
    seeds = [int(value) for value in rule["seeds"]]
    arms = list(rule["arms"])
    period = float(rule["envelope"]["period"])
    window = [float(value) for value in rule["fixed_diagnostic_fit_window"]]
    limits = rule["checks"]
    file_hashes = verify_pairing(args.input, seeds, arms)
    data = {}
    shared_times = None
    source_records = []
    rows = []
    for arm in arms:
        times, histories, metadata, hashes = load_arm(args.input, seeds, arm)
        file_hashes.update(hashes)
        if shared_times is None:
            shared_times = times
        else:
            np.testing.assert_array_equal(shared_times, times)
        source_records.extend((item["pilot_source_sha256"], item["pilot_preflight_sha256"]) for item in metadata)
        amplitude = coherent_amplitude(histories)
        envelope, points = geometric_envelope(amplitude, times, period)
        if points != int(rule["envelope"]["expected_points_at_cadence"]):
            raise AssertionError("pilot envelope point count differs from frozen rule")
        fit = fit_fixed(times, envelope, window)
        metrics = {
            "envelope_t20_over_t5": at_time(times, envelope, 20.0) / at_time(times, envelope, 5.0),
            "envelope_t30_over_t5": at_time(times, envelope, 30.0) / at_time(times, envelope, 5.0),
        }
        failed = failed_v1_envelope(args.failed_v1, arm, times, period)
        metrics["pilot_to_failed_v1_envelope_at_t30"] = at_time(times, envelope, 30.0) / at_time(times, failed, 30.0)
        checks = {
            "gamma_in_expected_negative_band": limits["absolute_gamma_min"] <= fit["gamma"] <= limits["absolute_gamma_max"],
            "r_squared": fit["r_squared"] >= limits["minimum_r_squared"],
            "decay_to_t20": metrics["envelope_t20_over_t5"] <= limits["maximum_envelope_t20_over_t5"],
            "decay_to_t30": metrics["envelope_t30_over_t5"] <= limits["maximum_envelope_t30_over_t5"],
            "contrast_to_failed_v1": metrics["pilot_to_failed_v1_envelope_at_t30"] <= limits["maximum_pilot_to_failed_v1_envelope_ratio_at_t30"],
        }
        data[arm] = {
            "histories": histories,
            "coherent_mode": np.mean(histories, axis=0),
            "envelope": envelope,
            "fit": fit,
            "metrics": metrics,
            "checks": checks,
            "pass": bool(all(checks.values())),
        }
        rows.append({"arm": arm, **fit, **metrics, "pass": data[arm]["pass"]})
    assert shared_times is not None
    if len(set(source_records)) != 1:
        raise AssertionError("pilot source/preflight records differ across seeds or arms")

    result = {
        "campaign_id": rule["campaign_id"],
        "status": "PASS" if all(data[arm]["pass"] for arm in arms) else "FAIL",
        "scope": "velocity-loading diagnostic only; not confirmatory",
        "rate_ratios_computed": False,
        "analysis_rule_sha256": sha256(args.rule),
        "input_file_sha256": file_hashes,
        "seeds": seeds,
        "arms": {
            arm: {
                "fit": data[arm]["fit"],
                "metrics": data[arm]["metrics"],
                "checks": data[arm]["checks"],
                "pass": data[arm]["pass"],
            }
            for arm in arms
        },
        "prohibitions": rule["prohibitions"],
        "next_step": rule["next_step_if_pass"] if all(data[arm]["pass"] for arm in arms) else rule["next_step_if_fail"],
    }
    (args.output / "pilot_result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    with (args.output / "pilot_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(
        args.output / "complex_histories_and_envelopes.npz",
        times=shared_times,
        **{f"{arm}_complex_histories": data[arm]["histories"] for arm in arms},
        **{f"{arm}_coherent_complex_mode": data[arm]["coherent_mode"] for arm in arms},
        **{f"{arm}_geometric_envelope": data[arm]["envelope"] for arm in arms},
    )
    make_figure(args.output, shared_times, data, window)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
