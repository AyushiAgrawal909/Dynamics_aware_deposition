#!/usr/bin/env python3
"""Analyze the isolated PPC-2048 E2 m=8 matched-filter sigma sweep."""

from __future__ import annotations

import argparse
import csv
import json
import math
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from analyze_E2_mode_growth import (
    bootstrap_ensemble_rate,
    fixed_window_seed_rates,
    select_growth_window,
)


def parse_floats(value: str) -> list[float]:
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def sigma_tag(sigma_dx: float) -> str:
    return f"{sigma_dx:.6g}".replace(".", "p").replace("-", "m")


def fit_ensemble_fixed(amplitudes: np.ndarray, times: np.ndarray, start: float, stop: float) -> float:
    mask = (times >= start) & (times <= stop)
    if np.count_nonzero(mask) < 5:
        return float("nan")
    mean_amp = np.mean(amplitudes, axis=0)
    return float(np.polyfit(times[mask], np.log(np.maximum(mean_amp[mask], 1.0e-300)), 1)[0])


def paired_ratio_bootstrap(
    baseline: np.ndarray,
    baseline_times: np.ndarray,
    baseline_window: tuple[float, float],
    current: np.ndarray,
    current_times: np.ndarray,
    current_window: tuple[float, float],
    nboot: int,
    seed: int,
) -> dict:
    if baseline.shape[0] != current.shape[0]:
        raise ValueError("paired bootstrap requires the same number of seeds")
    rng = np.random.default_rng(seed)
    ratios = []
    current_gammas = []
    for _ in range(nboot):
        indices = rng.integers(0, baseline.shape[0], size=baseline.shape[0])
        gamma0 = fit_ensemble_fixed(
            baseline[indices], baseline_times, baseline_window[0], baseline_window[1]
        )
        gamma = fit_ensemble_fixed(
            current[indices], current_times, current_window[0], current_window[1]
        )
        if np.isfinite(gamma0) and gamma0 > 0.0 and np.isfinite(gamma):
            ratios.append(gamma / gamma0)
            current_gammas.append(gamma)
    ratio_array = np.asarray(ratios, dtype=np.float64)
    gamma_array = np.asarray(current_gammas, dtype=np.float64)
    if ratio_array.size == 0:
        return {
            "ratio_sem": float("nan"),
            "ratio_ci_lo": float("nan"),
            "ratio_ci_hi": float("nan"),
            "gamma_ci_lo": float("nan"),
            "gamma_ci_hi": float("nan"),
        }
    return {
        "ratio_sem": float(np.std(ratio_array)),
        "ratio_ci_lo": float(np.percentile(ratio_array, 2.5)),
        "ratio_ci_hi": float(np.percentile(ratio_array, 97.5)),
        "gamma_ci_lo": float(np.percentile(gamma_array, 2.5)),
        "gamma_ci_hi": float(np.percentile(gamma_array, 97.5)),
    }


def write_csv(path: pathlib.Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument(
        "--baseline-npz",
        type=pathlib.Path,
        default=pathlib.Path(
            "/mnt/tier2/users/u103576/logs/E2_isolated_highppc_4860474/raw/mode_08_isolated.npz"
        ),
    )
    parser.add_argument("--sigma-dx", default="0.1,0.2,0.35,0.5,1,2,3,5,8")
    parser.add_argument("--target-mode", type=int, default=8)
    parser.add_argument("--target-r2", type=float, default=0.98)
    parser.add_argument("--min-efoldings", type=float, default=2.0)
    parser.add_argument("--competitor-ratio-max", type=float, default=0.5)
    parser.add_argument("--nboot", type=int, default=2000)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    with (args.input / "run_parameters.json").open(encoding="utf-8") as handle:
        metadata = json.load(handle)
    with np.load(args.baseline_npz) as baseline_data:
        baseline_times = np.asarray(baseline_data["times"], dtype=np.float64)
        baseline = np.asarray(baseline_data["target_amplitudes"], dtype=np.float64)
        baseline_seeds = np.asarray(baseline_data["seeds"], dtype=np.int64)

    baseline_ensemble = np.mean(baseline, axis=0)
    baseline_window = select_growth_window(
        baseline_times,
        baseline_ensemble,
        t_min=0.3,
        t_start_max=15.0,
        t_max=float(baseline_times[-1]),
        min_duration=2.0,
        min_points=20,
        target_r2=args.target_r2,
        saturation_fraction=0.90,
    )
    if not np.isfinite(baseline_window["gamma"]):
        raise RuntimeError("sigma=0 baseline did not produce a growth fit")
    gamma0 = float(baseline_window["gamma"])
    baseline_seed_rates = fixed_window_seed_rates(
        baseline,
        baseline_times,
        float(baseline_window["t_start"]),
        float(baseline_window["t_stop"]),
    )
    baseline_std = float(np.std(baseline_seed_rates, ddof=1))
    baseline_boot = bootstrap_ensemble_rate(
        baseline,
        baseline_times,
        float(baseline_window["t_start"]),
        float(baseline_window["t_stop"]),
        args.nboot,
        seed=93000,
    )

    rows = [
        {
            "sigma_dx": 0.0,
            "gamma_sigma": gamma0,
            "gamma_seed_std": baseline_std,
            "gamma_ci_lo": baseline_boot["ci_lo"],
            "gamma_ci_hi": baseline_boot["ci_hi"],
            "gamma_ratio": 1.0,
            "ratio_ci_lo": 1.0,
            "ratio_ci_hi": 1.0,
            "fit_t_start": baseline_window["t_start"],
            "fit_t_stop": baseline_window["t_stop"],
            "fit_r2": baseline_window["r2"],
            "fit_efoldings": baseline_window["efoldings"],
            "max_competitor_to_target": 0.0,
            "competitor_crossing_t": float("nan"),
            "effective_W2_target": 1.0,
            "status": "clean-baseline",
            "n_seeds": int(baseline.shape[0]),
        }
    ]
    trace_data = {0.0: (baseline_times, baseline_ensemble, None, baseline_window)}
    dx = float(metadata["Lx"]) / int(metadata["Nx"])
    k = 2.0 * math.pi * args.target_mode / float(metadata["Lx"])

    for sigma in parse_floats(args.sigma_dx):
        with np.load(args.input / f"sigma_{sigma_tag(sigma)}.npz") as data:
            times = np.asarray(data["times"], dtype=np.float64)
            amplitudes = np.asarray(data["amplitudes"], dtype=np.float64)
            modes = np.asarray(data["modes"], dtype=np.int64)
            seeds = np.asarray(data["seeds"], dtype=np.int64)
        if not np.array_equal(seeds, baseline_seeds):
            raise RuntimeError(f"seed mismatch at sigma={sigma}")
        target_index = int(np.flatnonzero(modes == args.target_mode)[0])
        target = amplitudes[:, target_index, :]
        competitors = np.delete(amplitudes, target_index, axis=1)
        target_ensemble = np.mean(target, axis=0)
        competitor_ensemble = np.max(np.mean(competitors, axis=0), axis=0)
        window = select_growth_window(
            times,
            target_ensemble,
            t_min=0.3,
            t_start_max=15.0,
            t_max=float(times[-1]),
            min_duration=2.0,
            min_points=20,
            target_r2=args.target_r2,
            saturation_fraction=0.90,
        )
        competitor_crossing_t = float("nan")
        if np.isfinite(window["t_start"]):
            ratio_all = competitor_ensemble / np.maximum(target_ensemble, 1.0e-300)
            dominant = ratio_all <= args.competitor_ratio_max
            edges = np.diff(np.r_[False, dominant, False].astype(np.int8))
            starts = np.flatnonzero(edges == 1)
            stops = np.flatnonzero(edges == -1) - 1

            # Projection transients can make competitors briefly larger before
            # the seeded m=8 branch dominates. Keep the longest dominant segment
            # that overlaps the preliminary growth fit, then refit inside it.
            segments = []
            for start, stop in zip(starts, stops):
                overlap_start = max(float(times[start]), float(window["t_start"]))
                overlap_stop = min(float(times[stop]), float(window["t_stop"]))
                overlap_duration = overlap_stop - overlap_start
                if overlap_duration >= 2.0:
                    segments.append((overlap_duration, int(start), int(stop)))
            if segments:
                _, guarded_start_index, guarded_stop_index = max(segments)
                guarded_start = float(times[guarded_start_index])
                guarded_stop = float(times[guarded_stop_index])
                if guarded_stop_index + 1 < len(times):
                    competitor_crossing_t = float(times[guarded_stop_index + 1])
                window = select_growth_window(
                    times,
                    target_ensemble,
                    t_min=max(0.3, guarded_start),
                    t_start_max=min(15.0, guarded_stop),
                    t_max=guarded_stop,
                    min_duration=2.0,
                    min_points=20,
                    target_r2=args.target_r2,
                    saturation_fraction=0.90,
                )
            else:
                window = {
                    "gamma": float("nan"),
                    "r2": float("nan"),
                    "efoldings": 0.0,
                    "n_points": 0,
                    "t_start": float("nan"),
                    "t_stop": float("nan"),
                }
        if np.isfinite(window["t_start"]):
            fit_mask = (times >= window["t_start"]) & (times <= window["t_stop"])
            competitor_ratio = float(
                np.max(competitor_ensemble[fit_mask] / np.maximum(target_ensemble[fit_mask], 1.0e-300))
            )
            seed_rates = fixed_window_seed_rates(
                target, times, float(window["t_start"]), float(window["t_stop"])
            )
            gamma_std = float(np.std(seed_rates, ddof=1))
            paired = paired_ratio_bootstrap(
                baseline,
                baseline_times,
                (float(baseline_window["t_start"]), float(baseline_window["t_stop"])),
                target,
                times,
                (float(window["t_start"]), float(window["t_stop"])),
                args.nboot,
                seed=93000 + int(round(100.0 * sigma)),
            )
            gamma = float(window["gamma"])
            ratio = gamma / gamma0
            positive_ci = bool(np.isfinite(paired["gamma_ci_lo"]) and paired["gamma_ci_lo"] > 0.0)
            monotone_guard = bool(
                paired["ratio_ci_lo"] <= 1.0
                or abs(ratio - 1.0) <= 2.0 * paired["ratio_sem"]
            )
            competitor_guard = competitor_ratio <= args.competitor_ratio_max
            if not competitor_guard or not monotone_guard:
                status = "contaminated"
            elif positive_ci and window["r2"] >= args.target_r2 and window["efoldings"] >= args.min_efoldings:
                status = "clean"
            elif positive_ci and window["r2"] >= 0.90:
                status = "short-window"
            else:
                status = "unresolved"
        else:
            competitor_ratio = float("nan")
            gamma_std = float("nan")
            gamma = float("nan")
            ratio = float("nan")
            paired = {
                "ratio_ci_lo": float("nan"),
                "ratio_ci_hi": float("nan"),
                "gamma_ci_lo": float("nan"),
                "gamma_ci_hi": float("nan"),
            }
            status = "no-growing-window"

        W2 = math.exp(-((k * sigma * dx) ** 2))
        rows.append(
            {
                "sigma_dx": sigma,
                "gamma_sigma": gamma,
                "gamma_seed_std": gamma_std,
                "gamma_ci_lo": paired["gamma_ci_lo"],
                "gamma_ci_hi": paired["gamma_ci_hi"],
                "gamma_ratio": ratio,
                "ratio_ci_lo": paired["ratio_ci_lo"],
                "ratio_ci_hi": paired["ratio_ci_hi"],
                "fit_t_start": window["t_start"],
                "fit_t_stop": window["t_stop"],
                "fit_r2": window["r2"],
                "fit_efoldings": window["efoldings"],
                "max_competitor_to_target": competitor_ratio,
                "competitor_crossing_t": competitor_crossing_t,
                "effective_W2_target": W2,
                "status": status,
                "n_seeds": int(target.shape[0]),
            }
        )
        trace_data[sigma] = (times, target_ensemble, competitor_ensemble, window)

    write_csv(args.output / "E2_m8_sigma_sweep.csv", rows)
    admissible = [row for row in rows if row["status"] in {"clean-baseline", "clean", "short-window"}]
    clean = [row for row in rows if row["status"] in {"clean-baseline", "clean"}]
    monotone = True
    for left, right in zip(admissible, admissible[1:]):
        if float(right["ratio_ci_lo"]) > float(left["ratio_ci_hi"]):
            monotone = False
    summary = {
        "gamma0": gamma0,
        "gamma0_seed_std": baseline_std,
        "baseline_fit_window": [baseline_window["t_start"], baseline_window["t_stop"]],
        "baseline_r2": baseline_window["r2"],
        "baseline_efoldings": baseline_window["efoldings"],
        "clean_sigma_dx": [float(row["sigma_dx"]) for row in clean],
        "short_window_sigma_dx": [float(row["sigma_dx"]) for row in rows if row["status"] == "short-window"],
        "contaminated_sigma_dx": [float(row["sigma_dx"]) for row in rows if row["status"] == "contaminated"],
        "no_growth_sigma_dx": [float(row["sigma_dx"]) for row in rows if row["status"] == "no-growing-window"],
        "monotone_within_paired_ci": monotone,
        "R_calibration_usable": bool(len(clean) >= 3 and monotone),
        "competitor_guard": f"max other-mode amplitude / m8 amplitude <= {args.competitor_ratio_max}",
        "physics_chain": metadata["deposition"],
    }
    with (args.output / "E2_m8_sigma_sweep.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)

    figure, axis = plt.subplots(figsize=(7.4, 4.8))
    for row in rows:
        if not np.isfinite(row["gamma_ratio"]):
            continue
        color = "#24733d" if row["status"] in {"clean-baseline", "clean"} else "#d17b0f"
        if row["status"] == "contaminated":
            color = "#9b2f2f"
        yerr = np.asarray(
            [
                [max(0.0, row["gamma_ratio"] - row["ratio_ci_lo"])],
                [max(0.0, row["ratio_ci_hi"] - row["gamma_ratio"])],
            ]
        )
        axis.errorbar(row["sigma_dx"], row["gamma_ratio"], yerr=yerr, fmt="o", color=color, capsize=3)
    axis.axhline(1.0, color="black", linewidth=0.9)
    axis.axhline(0.90, color="#a33b20", linestyle="--", label="10% damage")
    axis.set_xscale("symlog", linthresh=0.1)
    axis.set_xlabel("sigma / dx")
    axis.set_ylabel("gamma_sigma / gamma_0")
    axis.set_title("Isolated PPC-2048 m=8 smoothing response")
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(args.output / "E2_m8_sigma_sweep.png", dpi=220)
    plt.close(figure)

    ncols = 2
    nrows = int(math.ceil(len(trace_data) / ncols))
    figure, axes = plt.subplots(nrows, ncols, figsize=(11, 3.0 * nrows), squeeze=False)
    row_by_sigma = {float(row["sigma_dx"]): row for row in rows}
    for (sigma, values), axis in zip(sorted(trace_data.items()), axes.flat):
        times, target_amp, competitor_amp, window = values
        axis.plot(times, np.log(np.maximum(target_amp, 1.0e-300)), label="m=8")
        if competitor_amp is not None:
            axis.plot(times, np.log(np.maximum(competitor_amp, 1.0e-300)), alpha=0.65, label="max m!=8")
        if np.isfinite(window["t_start"]):
            axis.axvspan(window["t_start"], window["t_stop"], color="#77a96b", alpha=0.2)
        row = row_by_sigma[sigma]
        axis.set_title(f"sigma={sigma:g}: {row['status']}, R2={row['fit_r2']:.3f}, Ne={row['fit_efoldings']:.2f}")
        axis.set_xlabel("time")
        axis.set_ylabel("log |Ex_hat|")
        axis.legend(frameon=False, fontsize=8)
    for axis in axes.flat[len(trace_data) :]:
        axis.set_visible(False)
    figure.tight_layout()
    figure.savefig(args.output / "E2_m8_sigma_fit_windows.png", dpi=190)
    plt.close(figure)

    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    print("M8 R-CALIBRATION PASS" if summary["R_calibration_usable"] else "M8 R-CALIBRATION PARTIAL", flush=True)


if __name__ == "__main__":
    main()
