#!/usr/bin/env python3
"""Recover bootstrap acceptance counts omitted by some historical rate CSVs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys

import numpy as np


WORKSPACE = Path(__file__).resolve().parents[2]
RAW = WORKSPACE / "pic_kde_2d" / "results_raw_rate_audit"
STAGE = WORKSPACE / "pic_kde_2d_landau_cv_stage"
TOP_SCRIPTS = WORKSPACE / "pic_kde_2d" / "scripts"
sys.path.insert(0, str(TOP_SCRIPTS))
sys.path.insert(0, str(STAGE))

from pic_kde_2d.benchmarks.theory import langmuir_omega_r  # noqa: E402
from pic_kde_2d.validation.landau_censored_fit import (  # noqa: E402
    fit_censored_mode,
    paired_bootstrap_ratio,
)
from analyze_E2_mode_growth import select_growth_window  # noqa: E402
from analyze_linearized_delta_f_m12_sigma import fit_rate  # noqa: E402


def weak_kwargs() -> dict:
    return {
        "fit_t_start": 5.0,
        "fit_t_stop_max": 30.0,
        "noise_floor_start": 70.0,
        "noise_floor_stop": 90.0,
        "noise_factor": 5.0,
        "envelope_period": 2.0 * math.pi / langmuir_omega_r(0.4),
        "persistence_points": 5,
        "min_points": 30,
        "min_duration": 5.0,
        "min_model_efoldings": 0.5,
        "min_r_squared": 0.75,
    }


def load_weak(root: Path, passes: int) -> tuple[np.ndarray, np.ndarray]:
    histories = []
    times = None
    for seed in range(10):
        path = root / f"passes{passes}" / f"seed{seed}" / "history.npz"
        with np.load(path, allow_pickle=False) as archive:
            current_times = np.asarray(archive["times"], dtype=float)
            histories.append(np.asarray(archive["Ex_mode_1_0"], dtype=complex))
            assert int(archive["seed"][0]) == seed
            assert int(archive["passes"][0]) == passes
        if times is None:
            times = current_times
        else:
            np.testing.assert_array_equal(times, current_times)
    assert times is not None
    return times, np.stack(histories)


def weak_rows() -> list[dict]:
    specifications = [
        ("weak_raw_binomial", RAW / "landau_binomial_weak_5048412/shards",
         [0, 1, 2, 4, 8, 12, 13, 16]),
        ("weak_compensated_p1", RAW / "compensated_binomial_20260811/rates/landau/shards",
         [0, 1, 2, 4, 8, 16]),
        ("weak_compensated_p2", RAW / "higher_order_compensation_20260811/rates/p2/landau/shards",
         [0, 1, 2, 4]),
        ("weak_compensated_p3", RAW / "higher_order_compensation_20260811/rates/p3/landau/shards",
         [0, 1, 2, 4]),
    ]
    rows = []
    for campaign, root, passes_values in specifications:
        print(campaign, flush=True)
        times, baseline = load_weak(root, 0)
        base = fit_censored_mode(baseline, times, **weak_kwargs())
        rows.append({
            "campaign": campaign, "setting": "passes0", "n_boot": 0,
            "valid_draws": 0, "invalid_draws": 0,
            "quality_draws": 0, "valid_fraction": "not bootstrapped",
            "quality_fraction": "not bootstrapped",
            "central_quality": base["quality_pass"],
            "rule": "CIC baseline is assigned ratio=1 without bootstrap",
        })
        for passes in passes_values[1:]:
            _, current = load_weak(root, passes)
            result = paired_bootstrap_ratio(
                baseline, current, times, n_boot=2000,
                seed=82100 + passes, **weak_kwargs(),
            )
            valid = int(round(2000 * result["bootstrap_valid_fraction"]))
            quality = int(round(2000 * result["bootstrap_quality_fraction"]))
            central = fit_censored_mode(current, times, **weak_kwargs())
            rows.append({
                "campaign": campaign, "setting": f"passes{passes}", "n_boot": 2000,
                "valid_draws": valid, "invalid_draws": 2000 - valid,
                "quality_draws": quality,
                "valid_fraction": result["bootstrap_valid_fraction"],
                "quality_fraction": result["bootstrap_quality_fraction"],
                "central_quality": central["quality_pass"],
                "rule": "paired seed resample; each arm independently floor-censored",
            })
    return rows


def load_e2(root: Path, passes: int) -> tuple[np.ndarray, np.ndarray]:
    histories = []
    times = None
    for seed in range(10):
        path = root / f"passes{passes}" / f"seed{seed}"
        metadata = json.loads((path / "result.json").read_text(encoding="utf-8"))
        assert (metadata["kx_mode"], metadata["ky_mode"]) == (12, 0)
        with np.load(path / "mode_history.npz", allow_pickle=False) as archive:
            current_times = np.asarray(archive["times"], dtype=float)
            histories.append(np.abs(np.asarray(archive["mode"], dtype=complex)))
        if times is None:
            times = current_times
        else:
            np.testing.assert_array_equal(times, current_times)
    assert times is not None
    return times, np.stack(histories)


def e2_rows() -> list[dict]:
    specifications = [
        ("e2_longitudinal_compensated_p1", RAW / "compensated_binomial_20260811/rates/e2/shards",
         [0, 1, 2, 4, 8, 16]),
        ("e2_longitudinal_compensated_p2", RAW / "higher_order_compensation_20260811/rates/p2/e2/shards",
         [0, 1, 2, 4]),
        ("e2_longitudinal_compensated_p3", RAW / "higher_order_compensation_20260811/rates/p3/e2/shards",
         [0, 1, 2, 4]),
    ]
    rows = []
    for campaign, root, passes_values in specifications:
        print(campaign, flush=True)
        times, baseline = load_e2(root, 0)
        groups = {0: baseline}
        for passes in passes_values[1:]:
            _, groups[passes] = load_e2(root, passes)
        window = select_growth_window(
            times, np.mean(baseline, axis=0), t_min=0.3,
            t_start_max=min(30.0, float(times[-1]) - 5.0),
            t_max=float(times[-1]), min_duration=5.0, min_points=30,
            target_r2=0.98, saturation_fraction=0.95,
        )
        rng = np.random.default_rng(84120)
        valid = 0
        for _ in range(2000):
            indices = rng.integers(0, 10, size=10)
            rates = [
                fit_rate(groups[passes][indices], times,
                         float(window["t_start"]), float(window["t_stop"]))
                for passes in passes_values
            ]
            valid += int(rates[0] > 0.0 and np.all(np.isfinite(rates)))
        for passes in passes_values:
            rows.append({
                "campaign": campaign, "setting": f"passes{passes}", "n_boot": 2000,
                "valid_draws": valid, "invalid_draws": 2000 - valid,
                "quality_draws": "not evaluated",
                "valid_fraction": valid / 2000.0,
                "quality_fraction": "not evaluated",
                "central_quality": "window selected from CIC only",
                "rule": "same seed resample and CIC-selected fixed window for every setting",
            })
    return rows


def copy_recorded_rows(rebuild_root: Path) -> list[dict]:
    rows = []
    for campaign_dir in sorted(rebuild_root.iterdir()):
        csvs = list(campaign_dir.glob("*.csv")) if campaign_dir.is_dir() else []
        for path in csvs:
            records = list(csv.DictReader(path.open(encoding="utf-8")))
            if not records or "bootstrap_valid_fraction" not in records[0]:
                continue
            for record in records:
                nboot = 0 if record.get("sigma_dx") == "0.0" else (
                    1000 if campaign_dir.name == "weak_gaussian_grid" else 2000
                )
                valid_fraction = float(record["bootstrap_valid_fraction"])
                quality_fraction = float(record["bootstrap_quality_fraction"])
                rows.append({
                    "campaign": campaign_dir.name,
                    "setting": (
                        ("ppc" + record["ppc"] + ":") if record.get("ppc") else ""
                    ) + "sigma" + record.get("sigma_dx", "unknown"),
                    "n_boot": nboot,
                    "valid_draws": int(round(nboot * valid_fraction)),
                    "invalid_draws": int(round(nboot * (1.0 - valid_fraction))),
                    "quality_draws": int(round(nboot * quality_fraction)),
                    "valid_fraction": valid_fraction if nboot else "not bootstrapped",
                    "quality_fraction": quality_fraction if nboot else "not bootstrapped",
                    "central_quality": record.get("quality_pass", "unknown"),
                    "rule": (
                        "paired seed resample; each arm independently floor-censored"
                        if campaign_dir.name.startswith("weak_gaussian")
                        else "paired seed resample on a common pre-bounce window"
                    ),
                })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rebuild-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    rows = copy_recorded_rows(args.rebuild_root.resolve()) + weak_rows() + e2_rows()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
