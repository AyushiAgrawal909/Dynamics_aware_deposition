#!/usr/bin/env python3
"""Audit the uniform-limit divergence and Wu-Qin AD update gate.

The unguarded LOO-CV curve is evaluated with the existing E2 estimator and
trajectory protocol at successively halved perturbation amplitudes.  The
Anderson-Darling gate is evaluated separately and passed through the exact
p-value-halving state machine implemented in ``WuQinWidthUpdater``.

This script does not replace the unguarded CV result.  It produces the separate
Wu-Qin controller result required for the selector comparison.
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import sys
import warnings

import numpy as np
from scipy import stats

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pic_kde_2d.core.grid2d import Grid2D
from pic_kde_2d.core.pic_loop import pic_step
from pic_kde_2d.methods.spectral_W_wrappers import make_spectral_W_gaussian_methods
from pic_kde_2d.methods.wu_qin_bandwidth import (
    WuQinWidthUpdater,
    identify_cv_minimum,
)
from scripts.run_E2_cv_selector import (
    make_particles,
    parse_float_list,
    run_one_seed,
)


def write_csv(path: pathlib.Path, rows: list[dict]) -> None:
    fields = sorted({key for row in rows for key in row}) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def uniformity_pvalue_at_time(
    seed: int,
    grid: Grid2D,
    args: argparse.Namespace,
) -> float:
    """Return a one-sample Anderson-Darling p-value against uniform x."""
    particles = make_particles(args.ppc, seed, args)
    deposit, gather = make_spectral_W_gaussian_methods(
        grid,
        args.live_sigma_dx * grid.dx,
        args.live_sigma_dx * grid.dy,
    )
    target_step = int(round(args.ad_eval_time / args.dt))
    for _ in range(target_step):
        pic_step(particles, grid, args.dt, deposit, gather)

    sample = np.mod(particles.x, grid.Lx) / grid.Lx
    rng = np.random.default_rng(args.ad_reference_seed + seed)
    if sample.size > args.ad_max_samples:
        sample = sample[rng.choice(sample.size, size=args.ad_max_samples, replace=False)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        result = stats.goodness_of_fit(
            stats.uniform,
            sample,
            known_params={"loc": 0.0, "scale": 1.0},
            statistic="ad",
            n_mc_samples=args.ad_mc_samples,
            rng=rng,
        )
    return float(result.pvalue)


def run_audit(args: argparse.Namespace) -> dict:
    args.output.mkdir(parents=True, exist_ok=True)
    grid = Grid2D(args.Nx, args.Ny, args.Lx, args.Ly)
    sigma_grid = parse_float_list(args.sigma_dx)
    alpha_grid = parse_float_list(args.alphas)
    if any(alpha_grid[index + 1] != 0.5 * alpha_grid[index] for index in range(len(alpha_grid) - 1)):
        raise ValueError("--alphas must be an exact descending halving sequence")

    seed_rows: list[dict] = []
    summary_rows: list[dict] = []
    for alpha in alpha_grid:
        run_args = argparse.Namespace(**vars(args))
        run_args.alpha = alpha
        p_values: list[float] = []
        cv_by_sigma: dict[float, list[float]] = {sigma: [] for sigma in sigma_grid}
        for seed in range(args.seed_start, args.seed_start + args.seeds):
            for row in run_one_seed(seed, sigma_grid, grid, run_args):
                sigma = float(row["sigma_dx"])
                score = float(row["cv_time_avg"])
                cv_by_sigma[sigma].append(score)
                seed_rows.append({"alpha": alpha, **row})
            p_value = uniformity_pvalue_at_time(seed, grid, run_args)
            p_values.append(p_value)
            seed_rows.append(
                {
                    "alpha": alpha,
                    "seed": seed,
                    "ad_pvalue": p_value,
                    "ad_eval_time": args.ad_eval_time,
                }
            )
            print(f"alpha={alpha:g} seed={seed} p_AD={p_value:.6g}", flush=True)

        mean_scores = np.asarray(
            [np.mean(cv_by_sigma[sigma]) for sigma in sigma_grid],
            dtype=np.float64,
        )
        selection = identify_cv_minimum(sigma_grid, mean_scores)
        median_p = float(np.median(p_values))
        updater = WuQinWidthUpdater(
            current_width=args.initial_sigma_dx,
            uniformity_threshold=args.uniformity_threshold,
            adjust_rate=args.adjust_rate,
        )
        event = updater.step(
            median_p,
            selection if median_p <= args.uniformity_threshold else None,
        )
        tail_slope = float(
            (mean_scores[-1] - mean_scores[-2])
            / (sigma_grid[-1] - sigma_grid[-2])
        )
        summary_rows.append(
            {
                "alpha": alpha,
                "ad_pvalue_median": median_p,
                "ad_pvalue_min": float(np.min(p_values)),
                "ad_pvalue_max": float(np.max(p_values)),
                "unguarded_cv_status": selection.status,
                "unguarded_sigma_relation": ">=" if not selection.identified else "=",
                "unguarded_sigma_dx": (
                    selection.lower_bound
                    if not selection.identified
                    else selection.bandwidth
                ),
                "cv_tail_slope": tail_slope,
                "ad_gate_reason": event["reason"],
                "ad_gate_triggered_cv": event["triggered_cv"],
                "ad_gate_accepted_target": event["accepted_target"],
                "ad_gated_sigma_dx": updater.current_width,
            }
        )

    p_medians = [float(row["ad_pvalue_median"]) for row in summary_rows]
    check = {
        "all_unguarded_cv_right_censored": all(
            row["unguarded_cv_status"] == "right_censored" for row in summary_rows
        ),
        "all_cv_tails_nonincreasing": all(
            float(row["cv_tail_slope"]) <= 0.0 for row in summary_rows
        ),
        "all_ad_gates_hold": all(
            not bool(row["ad_gate_accepted_target"]) for row in summary_rows
        ),
        "ad_pvalue_non_decreasing_as_alpha_halves": all(
            p_medians[index + 1] >= p_medians[index]
            for index in range(len(p_medians) - 1)
        ),
    }
    check["uniform_limit_divergence_confirmed"] = bool(
        check["all_unguarded_cv_right_censored"]
        and check["all_cv_tails_nonincreasing"]
        and check["all_ad_gates_hold"]
    )
    result = {
        "parameters": {
            "alphas": alpha_grid,
            "sigma_dx": sigma_grid,
            "seeds": args.seeds,
            "ppc": args.ppc,
            "ad_test": "scipy.stats.goodness_of_fit one-sample Anderson-Darling against U(0,1)",
            "ad_eval_time": args.ad_eval_time,
            "ad_max_samples": args.ad_max_samples,
            "ad_mc_samples": args.ad_mc_samples,
            "uniformity_threshold": args.uniformity_threshold,
            "pvalue_reoptimization_factor": 0.5,
            "adjust_rate": args.adjust_rate,
            "initial_sigma_dx": args.initial_sigma_dx,
        },
        "summary": summary_rows,
        "checkpoint": check,
    }
    write_csv(args.output / "wu_qin_alpha_halving_seed_values.csv", seed_rows)
    write_csv(args.output / "wu_qin_alpha_halving_summary.csv", summary_rows)
    (args.output / "wu_qin_alpha_halving_result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    if not check["uniform_limit_divergence_confirmed"]:
        raise SystemExit("WU-QIN ALPHA-HALVING CHECKPOINT FAILED")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=PROJECT_ROOT / "results" / "wu_qin_alpha_halving",
    )
    parser.add_argument("--alphas", default="0.001,0.0005,0.00025")
    parser.add_argument(
        "--sigma-dx",
        default="0.0,0.25,0.5,1.0,2.0,4.0,8.0,12.0,20.0,30.0,60.0",
    )
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--ppc", type=int, default=32)
    parser.add_argument("--Nx", type=int, default=128)
    parser.add_argument("--Ny", type=int, default=16)
    parser.add_argument("--Lx", type=float, default=64.0)
    parser.add_argument("--Ly", type=float, default=16.0)
    parser.add_argument("--kx-mode", type=int, default=12)
    parser.add_argument("--v-beam", type=float, default=0.8)
    parser.add_argument("--v-thermal", type=float, default=0.05)
    parser.add_argument("--live-sigma-dx", type=float, default=0.25)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--n-steps", type=int, default=80)
    parser.add_argument("--diag-stride", type=int, default=2)
    parser.add_argument("--eval-t-min", type=float, default=0.5)
    parser.add_argument("--eval-t-max", type=float, default=4.0)
    parser.add_argument("--ad-eval-time", type=float, default=2.0)
    parser.add_argument("--ad-reference-seed", type=int, default=918273)
    parser.add_argument("--ad-max-samples", type=int, default=4096)
    parser.add_argument("--ad-mc-samples", type=int, default=999)
    parser.add_argument("--uniformity-threshold", type=float, default=0.01)
    parser.add_argument("--adjust-rate", type=float, default=0.05)
    parser.add_argument("--initial-sigma-dx", type=float, default=0.0)
    parser.add_argument("--nboot", type=int, default=0)
    return parser


if __name__ == "__main__":
    run_audit(build_parser().parse_args())
