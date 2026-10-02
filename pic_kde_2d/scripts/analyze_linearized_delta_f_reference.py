#!/usr/bin/env python3
"""Analyze projected linearized delta-f Landau and two-stream references."""

from __future__ import annotations

import argparse
import csv
import glob
import json
import pathlib
import sys

import numpy as np

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from analyze_E2_mode_growth import bootstrap_ensemble_rate, select_growth_window
from pic_kde_2d.benchmarks.theory import landau_gamma_theory
from pic_kde_2d.validation.landau_censored_fit import fit_censored_mode


def load_group(pattern: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    paths = [pathlib.Path(value) for value in sorted(glob.glob(pattern))]
    if not paths:
        raise FileNotFoundError(f"no histories match {pattern}")
    histories = []
    ratios = []
    times = None
    for path in paths:
        with np.load(path) as data:
            this_times = np.asarray(data["times"], dtype=np.float64)
            histories.append(np.asarray(data["mode"], dtype=np.complex128))
            ratios.append(float(np.max(data["delta_f_ratio"])))
        if times is None:
            times = this_times
        else:
            np.testing.assert_allclose(times, this_times, rtol=0.0, atol=0.0)
    assert times is not None
    return times, np.stack(histories), np.asarray(ratios)


def bootstrap_landau(
    histories: np.ndarray,
    times: np.ndarray,
    nboot: int,
    seed: int,
) -> dict:
    rng = np.random.default_rng(seed)
    gammas = []
    quality = 0
    for _ in range(nboot):
        indices = rng.integers(0, histories.shape[0], size=histories.shape[0])
        fit = fit_censored_mode(histories[indices], times)
        gamma = float(fit["gamma"])
        if np.isfinite(gamma) and gamma < 0.0:
            gammas.append(gamma)
            quality += int(fit["quality_pass"])
    values = np.asarray(gammas, dtype=np.float64)
    if values.size == 0:
        return {
            "gamma_ci_lo": float("nan"),
            "gamma_ci_hi": float("nan"),
            "gamma_sem": float("nan"),
            "bootstrap_valid_fraction": 0.0,
            "bootstrap_quality_fraction": 0.0,
        }
    return {
        "gamma_ci_lo": float(np.percentile(values, 2.5)),
        "gamma_ci_hi": float(np.percentile(values, 97.5)),
        "gamma_sem": float(np.std(values)),
        "bootstrap_valid_fraction": float(values.size / nboot),
        "bootstrap_quality_fraction": float(quality / nboot),
    }


def analyze_landau(root: pathlib.Path, seeds: int, nboot: int) -> dict:
    times, histories, ratios = load_group(
        str(root / "landau" / "seed*" / "mode_history.npz")
    )
    if histories.shape[0] != seeds:
        raise RuntimeError(f"expected {seeds} Landau seeds, found {histories.shape[0]}")
    fit = fit_censored_mode(histories, times)
    boot = bootstrap_landau(histories, times, nboot, seed=51001)
    theory = float(landau_gamma_theory(0.4))
    gamma = float(fit["gamma"])
    relative_error = abs(gamma - theory) / abs(theory)
    return {
        "benchmark": "landau",
        "mode": 1,
        "gamma": gamma,
        "gamma_theory": theory,
        "gamma_relative_error": relative_error,
        **boot,
        "fit_t_start": fit["fit_t_start"],
        "fit_t_stop": fit["fit_t_stop"],
        "fit_r2": fit["r_squared"],
        "fit_efoldings": fit["model_efoldings"],
        "fit_status": fit["status"],
        "fit_quality_pass": bool(fit["quality_pass"]),
        "noise_floor": fit["noise_floor"],
        "max_delta_f_ratio": float(np.max(ratios)),
        "n_seeds": seeds,
        "reference_pass": bool(
            fit["quality_pass"]
            and relative_error <= 0.15
            and boot["bootstrap_quality_fraction"] >= 0.8
            and np.max(ratios) < 0.5
        ),
    }


def analyze_growing(
    root: pathlib.Path,
    mode: int,
    seeds: int,
    nboot: int,
) -> dict:
    times, histories, ratios = load_group(
        str(root / "two_stream" / f"m{mode}" / "seed*" / "mode_history.npz")
    )
    if histories.shape[0] != seeds:
        raise RuntimeError(
            f"expected {seeds} two-stream m={mode} seeds, found {histories.shape[0]}"
        )
    amplitudes = np.abs(histories)
    mean_amplitude = np.mean(amplitudes, axis=0)
    window = select_growth_window(
        times,
        mean_amplitude,
        t_min=0.3,
        t_start_max=min(30.0, float(times[-1]) - 5.0),
        t_max=float(times[-1]),
        min_duration=5.0,
        min_points=30,
        target_r2=0.98,
        saturation_fraction=0.95,
    )
    boot = bootstrap_ensemble_rate(
        amplitudes,
        times,
        float(window["t_start"]),
        float(window["t_stop"]),
        nboot,
        seed=52000 + mode,
    )
    clean = bool(
        np.isfinite(window["gamma"])
        and window["r2"] >= 0.98
        and window["efoldings"] >= 2.0
        and np.max(ratios) < 0.5
    )
    return {
        "benchmark": "two_stream",
        "mode": mode,
        "gamma": float(window["gamma"]),
        "gamma_theory": float("nan"),
        "gamma_relative_error": float("nan"),
        "gamma_ci_lo": boot["ci_lo"],
        "gamma_ci_hi": boot["ci_hi"],
        "gamma_sem": boot["sem"],
        "bootstrap_valid_fraction": 1.0,
        "bootstrap_quality_fraction": float(clean),
        "fit_t_start": window["t_start"],
        "fit_t_stop": window["t_stop"],
        "fit_r2": window["r2"],
        "fit_efoldings": window["efoldings"],
        "fit_status": "clean" if clean else "short-window",
        "fit_quality_pass": clean,
        "noise_floor": float("nan"),
        "max_delta_f_ratio": float(np.max(ratios)),
        "n_seeds": seeds,
        "reference_pass": clean,
    }


def write_csv(path: pathlib.Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--nboot", type=int, default=2000)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = [
        analyze_landau(args.input, args.seeds, args.nboot),
        analyze_growing(args.input, 8, args.seeds, args.nboot),
        analyze_growing(args.input, 12, args.seeds, args.nboot),
    ]
    write_csv(args.output / "linearized_delta_f_reference.csv", rows)
    result = {
        "n_seeds": args.seeds,
        "rows": rows,
        "all_references_pass": all(row["reference_pass"] for row in rows),
    }
    (args.output / "linearized_delta_f_reference.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
