#!/usr/bin/env python3
"""Analyze paired strong-Landau rates on common pre-bounce windows."""

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
from pic_kde_2d.validation.strong_landau_fit import (
    detect_first_bounce,
    fit_initial_decay,
    paired_bootstrap_ratio_fixed_window,
)


DEFAULT_SIGMAS = "0,1,1.10785,1.5,2,2.453"


def parse_floats(text: str) -> list[float]:
    return sorted({float(item.strip()) for item in text.split(",") if item.strip()})


def sigma_tag(value: float) -> str:
    return f"{value:.8g}".replace("-", "m").replace(".", "p")


def write_csv(path: pathlib.Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_histories(
    root: pathlib.Path,
    ppc: int,
    sigma: float,
    seeds: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    histories = []
    times = None
    for seed in seeds:
        archive = (
            root
            / f"ppc{ppc}"
            / f"sigma{sigma_tag(sigma)}"
            / f"seed{seed}"
            / "mode_history.npz"
        )
        with np.load(archive) as data:
            this_times = np.asarray(data["times"], dtype=np.float64)
            histories.append(np.asarray(data["mode"], dtype=np.complex128))
            if int(data["ppc"][0]) != ppc:
                raise ValueError(f"PPC mismatch in {archive}")
            if not np.isclose(float(data["sigma_dx"][0]), sigma):
                raise ValueError(f"sigma mismatch in {archive}")
        if times is None:
            times = this_times
        else:
            np.testing.assert_array_equal(times, this_times)
    assert times is not None
    return times, np.stack(histories)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--ppc", type=int, default=1024)
    parser.add_argument("--sigmas", default=DEFAULT_SIGMAS)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--cv-width", type=float, default=1.10785)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--fit-start", type=float, default=5.0)
    parser.add_argument("--fit-stop-max", type=float, default=30.0)
    parser.add_argument("--bounce-search-start", type=float, default=8.0)
    parser.add_argument("--bounce-search-stop", type=float, default=35.0)
    parser.add_argument("--bounce-slope-window", type=float, default=2.0)
    parser.add_argument("--bounce-persistence", type=int, default=5)
    parser.add_argument("--min-duration", type=float, default=3.0)
    parser.add_argument("--min-r-squared", type=float, default=0.90)
    parser.add_argument("--min-model-efoldings", type=float, default=0.0)
    parser.add_argument("--nboot", type=int, default=2000)
    parser.add_argument("--eps-gamma", type=float, default=0.10)
    args = parser.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    args.output.mkdir(parents=True, exist_ok=True)
    sigmas = parse_floats(args.sigmas)
    if 0.0 not in sigmas:
        raise ValueError("strong-Landau sigma grid must include the CIC baseline")
    if not any(np.isclose(sigma, args.cv_width) for sigma in sigmas):
        raise ValueError("strong-Landau sigma grid must include --cv-width")
    seeds = list(range(args.seeds))
    envelope_period = 2.0 * math.pi / langmuir_omega_r(0.4)
    histories_by_sigma: dict[float, np.ndarray] = {}
    bounce_by_sigma: dict[float, dict] = {}
    times = None
    for sigma in sigmas:
        this_times, histories = load_histories(args.input, args.ppc, sigma, seeds)
        if times is None:
            times = this_times
        else:
            np.testing.assert_array_equal(times, this_times)
        histories_by_sigma[sigma] = histories
        bounce_by_sigma[sigma] = detect_first_bounce(
            histories,
            this_times,
            envelope_period=envelope_period,
            search_start=args.bounce_search_start,
            search_stop=args.bounce_search_stop,
            slope_window=args.bounce_slope_window,
            persistence_points=args.bounce_persistence,
        )
    assert times is not None

    baseline = histories_by_sigma[0.0]
    baseline_bounce = bounce_by_sigma[0.0]
    rows: list[dict] = []
    for sigma in sigmas:
        current = histories_by_sigma[sigma]
        bounce = bounce_by_sigma[sigma]
        common_stop = min(
            args.fit_stop_max,
            float(baseline_bounce["safe_fit_stop"]),
            float(bounce["safe_fit_stop"]),
        )
        base_fit = fit_initial_decay(
            baseline,
            times,
            fit_start=args.fit_start,
            fit_stop=common_stop,
            envelope_period=envelope_period,
            min_duration=args.min_duration,
            min_r_squared=args.min_r_squared,
        )
        fit = fit_initial_decay(
            current,
            times,
            fit_start=args.fit_start,
            fit_stop=common_stop,
            envelope_period=envelope_period,
            min_duration=args.min_duration,
            min_r_squared=args.min_r_squared,
        )
        if sigma == 0.0:
            bootstrap = {
                "ratio_sem": 0.0,
                "ratio_ci_lo": 1.0,
                "ratio_ci_hi": 1.0,
                "gamma_ci_lo": fit["gamma"],
                "gamma_ci_hi": fit["gamma"],
                "bootstrap_valid_fraction": 1.0,
                "bootstrap_quality_fraction": float(fit["quality_pass"]),
            }
        else:
            bootstrap = paired_bootstrap_ratio_fixed_window(
                baseline,
                current,
                times,
                fit_start=args.fit_start,
                fit_stop=common_stop,
                envelope_period=envelope_period,
                n_boot=args.nboot,
                seed=93100 + int(round(1000.0 * sigma)),
                min_duration=args.min_duration,
                min_r_squared=args.min_r_squared,
            )
        ratio = float(fit["gamma"] / base_fit["gamma"])
        central_pass = abs(ratio - 1.0) <= args.eps_gamma
        ci_pass = bool(
            bootstrap["ratio_ci_lo"] >= 1.0 - args.eps_gamma
            and bootstrap["ratio_ci_hi"] <= 1.0 + args.eps_gamma
        )
        quality_pass = bool(
            fit["quality_pass"]
            and base_fit["quality_pass"]
            and fit["model_efoldings"] >= args.min_model_efoldings
            and base_fit["model_efoldings"] >= args.min_model_efoldings
        )
        rows.append(
            {
                "sigma_dx": sigma,
                "gamma": fit["gamma"],
                "gamma_cic_common_window": base_fit["gamma"],
                "gamma_ratio_to_cic": ratio,
                "ratio_sem": bootstrap["ratio_sem"],
                "ratio_ci_lo": bootstrap["ratio_ci_lo"],
                "ratio_ci_hi": bootstrap["ratio_ci_hi"],
                "gamma_ci_lo": bootstrap["gamma_ci_lo"],
                "gamma_ci_hi": bootstrap["gamma_ci_hi"],
                "bounce_detected": bounce["bounce_detected"],
                "bounce_time": bounce["bounce_time"],
                "cic_bounce_time": baseline_bounce["bounce_time"],
                "fit_t_start": fit["fit_t_start"],
                "fit_t_stop": fit["fit_t_stop"],
                "fit_duration": fit["fit_duration"],
                "fit_r_squared": fit["r_squared"],
                "cic_fit_r_squared": base_fit["r_squared"],
                "model_efoldings": fit["model_efoldings"],
                "quality_pass": quality_pass,
                "bootstrap_valid_fraction": bootstrap["bootstrap_valid_fraction"],
                "bootstrap_quality_fraction": bootstrap["bootstrap_quality_fraction"],
                "central_rate_gate": central_pass,
                "ci_supported_rate_gate": ci_pass,
                "gate_status": (
                    "UNRESOLVED"
                    if not quality_pass
                    else "PASS" if central_pass and ci_pass else "FAIL"
                ),
                "n_seeds": args.seeds,
            }
        )

    write_csv(args.output / "strong_landau_fidelity.csv", rows)
    cv_row = next(
        row for row in rows if np.isclose(row["sigma_dx"], args.cv_width)
    )
    result = {
        "protocol": {
            "alpha": args.alpha,
            "k_lambdaD": 0.4,
            "ppc": args.ppc,
            "seeds": seeds,
            "quiet_start_layout": "global",
            "initialization": "identical density-and-current moment-matched pipeline to weak Landau",
            "sigma_grid_dx": sigmas,
            "fit_rule": "common paired initial-decay interval censored before earlier first bounce",
            "eps_gamma": args.eps_gamma,
            "minimum_model_efoldings": args.min_model_efoldings,
        },
        "cv_width_dx": args.cv_width,
        "cv_result": cv_row,
        "cv_verdict": cv_row["gate_status"],
    }
    (args.output / "strong_landau_fidelity.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    fig, (axis_trace, axis_ratio) = plt.subplots(1, 2, figsize=(11.0, 4.5))
    colors = plt.cm.viridis(np.linspace(0.05, 0.90, len(sigmas)))
    for color, row in zip(colors, rows):
        sigma = float(row["sigma_dx"])
        bounce = bounce_by_sigma[sigma]
        axis_trace.semilogy(
            times,
            bounce["envelope"],
            color=color,
            label=rf"$\sigma={sigma:g}\Delta x$",
        )
        if bounce["bounce_detected"]:
            axis_trace.scatter(
                [bounce["bounce_time"]],
                [np.interp(bounce["bounce_time"], times, bounce["envelope"])],
                color=color,
                s=25,
                zorder=4,
            )
    axis_trace.axvspan(args.fit_start, min(row["fit_t_stop"] for row in rows), color="grey", alpha=0.12)
    axis_trace.set(xlabel=r"$t\omega_p$", ylabel=r"coherent $|\hat E_x(1)|$", title="Initial decay and first bounce")
    ratio = np.asarray([row["gamma_ratio_to_cic"] for row in rows])
    lo = np.asarray([row["ratio_ci_lo"] for row in rows])
    hi = np.asarray([row["ratio_ci_hi"] for row in rows])
    x = np.asarray(sigmas)
    axis_ratio.errorbar(x, ratio, yerr=np.vstack((ratio - lo, hi - ratio)), marker="o", capsize=3)
    axis_ratio.axhspan(0.9, 1.1, color="#65a765", alpha=0.14, label="10% rate gate")
    axis_ratio.axhline(1.0, color="black", lw=0.9)
    axis_ratio.axvline(args.cv_width, color="#7b2c8e", ls="--", label="LOO-CV")
    axis_ratio.set(xlabel=r"$\sigma/\Delta x$", ylabel=r"$\gamma_\sigma/\gamma_{\rm CIC}$", title="Bounce-censored fidelity")
    for axis in (axis_trace, axis_ratio):
        axis.grid(alpha=0.2)
        axis.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(args.output / "strong_landau_fidelity.png", dpi=240)
    plt.close(fig)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
