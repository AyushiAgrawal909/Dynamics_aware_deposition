#!/usr/bin/env python3
"""Analyze matched-binomial pass budgets for weak or strong Landau."""

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

from pic_kde_2d.benchmarks.theory import langmuir_omega_r
from pic_kde_2d.validation.landau_censored_fit import (
    fit_censored_mode,
    paired_bootstrap_ratio,
)
from pic_kde_2d.validation.strong_landau_fit import (
    detect_first_bounce,
    fit_initial_decay,
    paired_bootstrap_ratio_fixed_window,
)


PASSES = (0, 1, 2, 4, 8, 12, 13, 16)


def load_group(
    root: pathlib.Path,
    passes: int,
    seeds: int,
    filter_family: str,
    compensation_order: int,
) -> tuple[np.ndarray, np.ndarray]:
    histories = []
    times = None
    for seed in range(seeds):
        path = root / f"passes{passes}" / f"seed{seed}" / "history.npz"
        metadata = json.loads(
            (path.parent / "run_parameters.json").read_text(encoding="utf-8")
        )
        actual_family = metadata.get("filter_family", "binomial")
        if actual_family != filter_family:
            raise ValueError(
                f"filter family {actual_family!r} != {filter_family!r} in {path.parent}"
            )
        if (
            filter_family == "compensated_binomial"
            and int(metadata.get("compensation_order", 1)) != compensation_order
        ):
            raise ValueError(f"compensation order mismatch in {path.parent}")
        with np.load(path) as data:
            this_times = np.asarray(data["times"], dtype=np.float64)
            histories.append(np.asarray(data["Ex_mode_1_0"], dtype=np.complex128))
            if int(data["passes"][0]) != passes or int(data["seed"][0]) != seed:
                raise ValueError(f"metadata mismatch in {path}")
        if times is None:
            times = this_times
        else:
            np.testing.assert_array_equal(times, this_times)
    assert times is not None
    return times, np.stack(histories)


def weak_fit_kwargs() -> dict:
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


def analyze_weak(groups, times, nboot, eps, passes_values):
    baseline = groups[0]
    kwargs = weak_fit_kwargs()
    base_fit = fit_censored_mode(baseline, times, **kwargs)
    rows = []
    for passes in passes_values:
        fit = fit_censored_mode(groups[passes], times, **kwargs)
        if passes == 0:
            boot = {"ratio_ci_lo": 1.0, "ratio_ci_hi": 1.0, "ratio_sem": 0.0}
        else:
            boot = paired_bootstrap_ratio(
                baseline,
                groups[passes],
                times,
                n_boot=nboot,
                seed=82100 + passes,
                **kwargs,
            )
        ratio = float(fit["gamma"] / base_fit["gamma"])
        rows.append(_row(passes, fit, base_fit, boot, ratio, eps, None))
    return rows


def analyze_strong(groups, times, nboot, eps, fit_start, passes_values):
    envelope_period = 2.0 * math.pi / langmuir_omega_r(0.4)
    baseline = groups[0]
    base_bounce = detect_first_bounce(
        baseline,
        times,
        envelope_period=envelope_period,
        search_start=8.0,
        search_stop=35.0,
    )
    rows = []
    for passes in passes_values:
        current = groups[passes]
        bounce = detect_first_bounce(
            current,
            times,
            envelope_period=envelope_period,
            search_start=8.0,
            search_stop=35.0,
        )
        common_stop = min(30.0, base_bounce["safe_fit_stop"], bounce["safe_fit_stop"])
        base_fit = fit_initial_decay(
            baseline,
            times,
            fit_start=fit_start,
            fit_stop=common_stop,
            envelope_period=envelope_period,
        )
        fit = fit_initial_decay(
            current,
            times,
            fit_start=fit_start,
            fit_stop=common_stop,
            envelope_period=envelope_period,
        )
        if passes == 0:
            boot = {"ratio_ci_lo": 1.0, "ratio_ci_hi": 1.0, "ratio_sem": 0.0}
        else:
            boot = paired_bootstrap_ratio_fixed_window(
                baseline,
                current,
                times,
                fit_start=fit_start,
                fit_stop=common_stop,
                envelope_period=envelope_period,
                n_boot=nboot,
                seed=83100 + passes,
            )
        ratio = float(fit["gamma"] / base_fit["gamma"])
        rows.append(_row(passes, fit, base_fit, boot, ratio, eps, bounce["bounce_time"]))
    return rows


def _row(passes, fit, base_fit, boot, ratio, eps, bounce_time):
    ci_pass = bool(boot["ratio_ci_lo"] >= 1.0 - eps and boot["ratio_ci_hi"] <= 1.0 + eps)
    quality = bool(fit["quality_pass"] and base_fit["quality_pass"])
    return {
        "passes": passes,
        "sigma_eq_dx": float(np.sqrt(passes / 2.0)),
        "gamma": fit["gamma"],
        "gamma_cic_common_window": base_fit["gamma"],
        "gamma_ratio_to_cic": ratio,
        "ratio_sem": boot["ratio_sem"],
        "ratio_ci_lo": boot["ratio_ci_lo"],
        "ratio_ci_hi": boot["ratio_ci_hi"],
        "fit_t_start": fit["fit_t_start"],
        "fit_t_stop": fit["fit_t_stop"],
        "fit_r_squared": fit["r_squared"],
        "model_efoldings": fit["model_efoldings"],
        "bounce_time": bounce_time,
        "quality_pass": quality,
        "ci_supported_rate_gate": ci_pass,
        "gate_status": "PASS" if quality and ci_pass else "FAIL",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--nboot", type=int, default=2000)
    parser.add_argument("--eps-gamma", type=float, default=0.10)
    parser.add_argument(
        "--filter-family",
        choices=("binomial", "compensated_binomial"),
        default="binomial",
    )
    parser.add_argument("--passes", type=int, nargs="+", default=list(PASSES))
    parser.add_argument("--compensation-order", type=int, default=1)
    parser.add_argument(
        "--strong-fit-start",
        type=float,
        default=3.0,
        help="validated post-projection start for the short pre-bounce strong-Landau window",
    )
    args = parser.parse_args()
    passes_values = tuple(args.passes)
    if len(set(passes_values)) != len(passes_values) or 0 not in passes_values:
        raise ValueError("--passes must be unique and include the CIC baseline 0")
    args.output.mkdir(parents=True, exist_ok=True)
    groups = {}
    times = None
    for passes in passes_values:
        this_times, histories = load_group(
            args.input,
            passes,
            args.seeds,
            args.filter_family,
            args.compensation_order,
        )
        if times is None:
            times = this_times
        else:
            np.testing.assert_array_equal(times, this_times)
        groups[passes] = histories
    assert times is not None
    rows = (
        analyze_strong(
            groups,
            times,
            args.nboot,
            args.eps_gamma,
            args.strong_fit_start,
            passes_values,
        )
        if args.alpha >= 0.1
        else analyze_weak(groups, times, args.nboot, args.eps_gamma, passes_values)
    )
    for row in rows:
        row["filter_family"] = args.filter_family
        row["compensation_order"] = (
            args.compensation_order
            if args.filter_family == "compensated_binomial"
            else None
        )
        if args.filter_family == "compensated_binomial":
            row["sigma_eq_dx"] = None
    passing = [row["passes"] for row in rows if row["gate_status"] == "PASS"]
    result = {
        "alpha": args.alpha,
        "filter_family": args.filter_family,
        "compensation_order": (
            args.compensation_order
            if args.filter_family == "compensated_binomial"
            else None
        ),
        "passes": list(passes_values),
        "largest_measured_passing_n": max(passing) if passing else None,
        "analytic_n_max": 12 if args.filter_family == "binomial" else None,
        "analytic_ceiling_sigma_dx": 2.4534,
        "rows": rows,
    }
    with (args.output / "landau_binomial_passes.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output / "landau_binomial_passes.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
