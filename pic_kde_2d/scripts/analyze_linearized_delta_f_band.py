#!/usr/bin/env python3
"""Analyze projected linearized-delta-f two-stream modes m=1..14."""

from __future__ import annotations

import argparse
import csv
import json
import math
import pathlib
import sys

import numpy as np

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from analyze_E2_mode_growth import bootstrap_ensemble_rate, select_growth_window
from pic_kde_2d.benchmarks.theory import (
    matched_cic_force_coupling,
    rate_tolerance_crossing,
    warm_two_stream_growth,
)


def parse_modes(spec: str) -> list[int]:
    """Parse comma-separated modes and inclusive ranges such as ``1-3,8``."""
    modes: set[int] = set()
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            start_text, stop_text = token.split("-", maxsplit=1)
            modes.update(range(int(start_text), int(stop_text) + 1))
        else:
            modes.add(int(token))
    parsed = sorted(modes)
    if not parsed or parsed[0] < 1 or parsed[-1] > 14:
        raise argparse.ArgumentTypeError("modes must be a nonempty subset of 1..14")
    return parsed


def load_mode(
    root: pathlib.Path,
    mode: int,
    seeds: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    histories = []
    validity_histories = []
    seed_ids = []
    times = None
    for path in sorted((root / f"m{mode:02d}").glob("seed*/mode_history.npz")):
        with np.load(path) as data:
            this_times = np.asarray(data["times"], dtype=np.float64)
            histories.append(np.abs(np.asarray(data["mode"], dtype=np.complex128)))
            validity_histories.append(
                np.asarray(data["delta_f_ratio"], dtype=np.float64)
            )
            seed_ids.append(int(np.asarray(data["seed"]).reshape(-1)[0]))
        if times is None:
            times = this_times
        else:
            np.testing.assert_allclose(times, this_times, rtol=0.0, atol=0.0)
    if len(histories) != seeds or sorted(seed_ids) != list(range(seeds)):
        raise RuntimeError(f"m={mode}: incomplete or mismatched seed ensemble")
    assert times is not None
    order = np.argsort(seed_ids)
    return (
        times,
        np.stack(histories)[order],
        np.stack(validity_histories)[order],
    )


def analytic_sigma_ceiling(mode: int, tolerance: float) -> float:
    k = 2.0 * math.pi * mode / 64.0

    def rate(sigma_dx: float) -> float:
        coupling = matched_cic_force_coupling(mode, 128, sigma_dx)
        return warm_two_stream_growth(k, 0.8, 0.05, coupling)

    if rate(0.0) <= 0.0:
        return float("nan")
    return rate_tolerance_crossing(rate, tolerance=tolerance, sigma_upper=64.0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--nboot", type=int, default=2000)
    parser.add_argument("--rate-tolerance", type=float, default=0.10)
    parser.add_argument("--modes", type=parse_modes, default=parse_modes("1-14"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    rows = []
    for mode in args.modes:
        times, amplitudes, validity = load_mode(args.input, mode, args.seeds)
        validity_envelope = np.max(validity, axis=0)
        validity_crossings = np.flatnonzero(validity_envelope >= 0.5)
        if validity_crossings.size:
            crossing_index = int(validity_crossings[0])
            valid_stop_index = max(crossing_index - 1, 0)
            first_validity_crossing = float(times[crossing_index])
        else:
            valid_stop_index = len(times) - 1
            first_validity_crossing = float("nan")
        valid_t_stop = float(times[valid_stop_index])
        ensemble = np.mean(amplitudes, axis=0)
        window = select_growth_window(
            times,
            ensemble,
            t_min=0.3,
            t_start_max=min(45.0, valid_t_stop - 5.0),
            t_max=valid_t_stop,
            min_duration=5.0,
            min_points=30,
            target_r2=0.98,
            saturation_fraction=0.98,
        )
        gamma = float(window["gamma"])
        if np.isfinite(gamma):
            fit_mask = (
                (times >= float(window["t_start"]))
                & (times <= float(window["t_stop"]))
            )
            fit_validity_max = float(np.max(validity[:, fit_mask]))
        else:
            fit_validity_max = float("nan")
        if np.isfinite(gamma):
            boot = bootstrap_ensemble_rate(
                amplitudes,
                times,
                float(window["t_start"]),
                float(window["t_stop"]),
                args.nboot,
                seed=54000 + mode,
            )
        else:
            boot = {"sem": float("nan"), "ci_lo": float("nan"), "ci_hi": float("nan")}
        strict = bool(
            np.isfinite(gamma)
            and gamma > 0.0
            and window["r2"] >= 0.98
            and window["efoldings"] >= 2.0
            and boot["ci_lo"] > 0.0
            and fit_validity_max < 0.5
        )
        net_amplitude_ratio = float(ensemble[-1] / max(ensemble[0], 1.0e-300))
        if strict:
            classification = "clean_unstable"
        elif net_amplitude_ratio >= 10.0:
            classification = "unresolved_positive_growth"
        else:
            classification = "no_resolved_growth"
        k = 2.0 * math.pi * mode / 64.0
        theory = warm_two_stream_growth(
            k,
            0.8,
            0.05,
            matched_cic_force_coupling(mode, 128, 0.0),
        )
        rows.append(
            {
                "mode": mode,
                "k": k,
                "gamma": gamma,
                "gamma_sem": boot["sem"],
                "gamma_ci_lo": boot["ci_lo"],
                "gamma_ci_hi": boot["ci_hi"],
                "gamma_warm": theory,
                "fit_t_start": window["t_start"],
                "fit_t_stop": window["t_stop"],
                "fit_r2": window["r2"],
                "fit_efoldings": window["efoldings"],
                "strict_clean_unstable": strict,
                "classification": classification,
                "net_amplitude_ratio": net_amplitude_ratio,
                "max_delta_f_ratio_full_run": float(np.max(validity)),
                "max_delta_f_ratio_fit_window": fit_validity_max,
                "first_validity_crossing": first_validity_crossing,
                "valid_fit_t_stop_cap": valid_t_stop,
                "analytic_sigma_R_dx": (
                    analytic_sigma_ceiling(mode, args.rate_tolerance)
                    if theory > 0.0
                    else float("nan")
                ),
                "n_seeds": args.seeds,
            }
        )

    with (args.output / "linearized_delta_f_band.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    unstable_modes = [row["mode"] for row in rows if row["strict_clean_unstable"]]
    unresolved_positive_modes = [
        row["mode"] for row in rows
        if row["classification"] == "unresolved_positive_growth"
    ]
    no_resolved_growth_modes = [
        row["mode"] for row in rows
        if row["classification"] == "no_resolved_growth"
    ]
    highest_unstable = max(unstable_modes) if unstable_modes else None
    complete_scan = args.modes == list(range(1, 15))
    edge_resolved = bool(
        complete_scan
        and
        highest_unstable is not None
        and highest_unstable < 14
        and all(
            row["classification"] == "no_resolved_growth"
            for row in rows
            if row["mode"] > highest_unstable
        )
    )
    alive_rows = [row for row in rows if row["strict_clean_unstable"]]
    binding = (
        min(alive_rows, key=lambda row: row["analytic_sigma_R_dx"])
        if alive_rows
        else None
    )
    result = {
        "n_seeds": args.seeds,
        "requested_modes": args.modes,
        "complete_m1_m14_scan": complete_scan,
        "measured_clean_unstable_modes": unstable_modes,
        "unresolved_positive_growth_modes": unresolved_positive_modes,
        "no_resolved_growth_modes": no_resolved_growth_modes,
        "highest_clean_unstable_mode": highest_unstable,
        "upper_edge_resolved": edge_resolved,
        "binding_mode": binding["mode"] if binding else None,
        "binding_analytic_sigma_R_dx": binding["analytic_sigma_R_dx"] if binding else None,
        "production_sigma_dx": 0.18445488487247555,
        "production_width_stands": bool(
            edge_resolved
            and binding is not None
            and binding["analytic_sigma_R_dx"] >= 0.18445488487247555
        ),
        "rows": rows,
    }
    (args.output / "linearized_delta_f_band.json").write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
