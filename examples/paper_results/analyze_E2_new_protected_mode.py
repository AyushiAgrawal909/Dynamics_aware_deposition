#!/usr/bin/env python3
"""Reduce an independent paired Gaussian test at a new protected E2 mode."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np


WORKSPACE = Path(__file__).resolve().parents[2]
for candidate in (
    WORKSPACE / "pic_kde_2d",
    WORKSPACE / "pic_kde_2d/scripts",
    Path(__file__).resolve().parent,
):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from analyze_linearized_delta_f_m12_sigma import estimate_growing_floor, fit_rate


def load_group(
    root: Path,
    label: str,
    seed_start: int,
    seeds: int,
    mode_x: int,
    mode_y: int,
) -> dict:
    amplitudes: list[np.ndarray] = []
    validity: list[np.ndarray] = []
    metadata: list[dict] = []
    times: np.ndarray | None = None
    for seed in range(seed_start, seed_start + seeds):
        path = root / label / f"seed{seed}"
        current = json.loads((path / "result.json").read_text(encoding="utf-8"))
        if int(current["seed"]) != seed:
            raise AssertionError(f"seed mismatch in {path}")
        if (int(current["kx_mode"]), int(current["ky_mode"])) != (mode_x, mode_y):
            raise AssertionError(f"mode mismatch in {path}")
        if current["benchmark"] != "two_stream" or int(current["ppc"]) != 512:
            raise AssertionError(f"benchmark or PPC mismatch in {path}")
        if not bool(current["target_mode_projection"]):
            raise AssertionError(f"target-mode projection disabled in {path}")
        with np.load(path / "mode_history.npz", allow_pickle=False) as archive:
            current_times = np.asarray(archive["times"], dtype=np.float64)
            amplitudes.append(np.abs(np.asarray(archive["mode"], dtype=np.complex128)))
            validity.append(np.asarray(archive["delta_f_ratio"], dtype=np.float64))
            if int(np.asarray(archive["seed"]).flat[0]) != seed:
                raise AssertionError(f"NPZ seed mismatch in {path}")
        if times is None:
            times = current_times
        else:
            np.testing.assert_array_equal(times, current_times)
        metadata.append(current)
    assert times is not None
    return {
        "label": label,
        "times": times,
        "amplitudes": np.stack(amplitudes),
        "validity": np.stack(validity),
        "metadata": metadata,
    }


def fit_r_squared(amplitudes: np.ndarray, times: np.ndarray, start: float, stop: float) -> float:
    mask = (times >= start) & (times <= stop)
    x = times[mask]
    y = np.log(np.maximum(np.mean(amplitudes, axis=0)[mask], 1.0e-300))
    slope, intercept = np.polyfit(x, y, 1)
    residual = float(np.sum((y - (slope * x + intercept)) ** 2))
    total = float(np.sum((y - np.mean(y)) ** 2))
    return 1.0 - residual / total if total > 0.0 else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--labels", default="base,c4")
    parser.add_argument("--mode-x", type=int, default=10)
    parser.add_argument("--mode-y", type=int, default=1)
    parser.add_argument("--seed-start", type=int, default=100)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--sigma-x-dx", type=float, default=0.0)
    parser.add_argument("--sigma-y-dy", type=float, default=0.24273459979982182)
    parser.add_argument("--fit-t-start", type=float, default=20.0)
    parser.add_argument("--fit-t-stop", type=float, default=44.0)
    parser.add_argument("--nboot", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=314159)
    parser.add_argument(
        "--skip-theory",
        action="store_true",
        help="reduce measurements without the local dispersion-audit source tree",
    )
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    labels = [item.strip() for item in args.labels.split(",") if item.strip()]
    if len(labels) != 2 or len(set(labels)) != 2:
        raise ValueError("exactly one baseline and one candidate label are required")
    groups = [
        load_group(args.input, label, args.seed_start, args.seeds, args.mode_x, args.mode_y)
        for label in labels
    ]
    np.testing.assert_array_equal(groups[0]["times"], groups[1]["times"])
    times = groups[0]["times"]
    if args.fit_t_start < times[0] or args.fit_t_stop > times[-1]:
        raise ValueError("fit window lies outside saved histories")
    for metadata in groups[0]["metadata"]:
        if float(metadata.get("sigma_x_dx", np.nan)) != 0.0 or float(
            metadata.get("sigma_y_dy", np.nan)
        ) != 0.0:
            raise AssertionError("baseline is not CIC")
    for metadata in groups[1]["metadata"]:
        if metadata.get("filter_kind", "gaussian") != "gaussian":
            raise AssertionError("candidate is not Gaussian")
        np.testing.assert_allclose(
            [float(metadata["sigma_x_dx"]), float(metadata["sigma_y_dy"])],
            [args.sigma_x_dx, args.sigma_y_dy],
            rtol=0.0,
            atol=1.0e-15,
        )

    central = [
        fit_rate(group["amplitudes"], times, args.fit_t_start, args.fit_t_stop)
        for group in groups
    ]
    rng = np.random.default_rng(args.bootstrap_seed)
    rates = [np.empty(args.nboot, dtype=np.float64) for _ in groups]
    ratios = [np.empty(args.nboot, dtype=np.float64) for _ in groups]
    for draw in range(args.nboot):
        indices = rng.integers(0, args.seeds, size=args.seeds)
        draw_rates = [
            fit_rate(group["amplitudes"][indices], times, args.fit_t_start, args.fit_t_stop)
            for group in groups
        ]
        if draw_rates[0] <= 0.0 or not np.all(np.isfinite(draw_rates)):
            raise AssertionError("non-finite paired bootstrap rate")
        for index, rate in enumerate(draw_rates):
            rates[index][draw] = rate
            ratios[index][draw] = rate / draw_rates[0]

    theory_record: dict | None = None
    if not args.skip_theory:
        from recompute_dispersion import ModeCase, gaussian_force_transfer, solve_level

        case = ModeCase(
            f"E2_m{args.mode_x}_n{args.mode_y}",
            "two_stream",
            128,
            16,
            64.0,
            16.0,
            args.mode_x,
            args.mode_y,
            0.05,
            0.8,
            0.02,
        )
        theory_baseline = solve_level(case, "grid_time_discrete", qmax=8, lmax=1024)
        force_transfer = gaussian_force_transfer(case, args.sigma_x_dx, args.sigma_y_dy)
        theory_candidate = solve_level(
            case,
            "grid_time_discrete",
            filter_force_transfer=force_transfer,
            initial=theory_baseline,
            qmax=8,
            lmax=1024,
        )
        theory_record = {
            "baseline_gamma": float(theory_baseline.real),
            "candidate_gamma": float(theory_candidate.real),
            "candidate_ratio": float(theory_candidate.real / theory_baseline.real),
            "force_transfer": force_transfer,
            "fitted_parameters": 0,
        }

    rows: list[dict] = []
    fit_mask = (times >= args.fit_t_start) & (times <= args.fit_t_stop)
    for index, group in enumerate(groups):
        floor = estimate_growing_floor(times, group["amplitudes"], args.fit_t_start)
        ratio_lo = float(np.percentile(ratios[index], 2.5))
        ratio_hi = float(np.percentile(ratios[index], 97.5))
        r2 = fit_r_squared(group["amplitudes"], times, args.fit_t_start, args.fit_t_stop)
        efoldings = float(central[index] * (args.fit_t_stop - args.fit_t_start))
        rows.append(
            {
                "label": group["label"],
                "mode_x": args.mode_x,
                "mode_y": args.mode_y,
                "seed_start": args.seed_start,
                "n_seeds": args.seeds,
                "sigma_x_dx": 0.0 if index == 0 else args.sigma_x_dx,
                "sigma_y_dy": 0.0 if index == 0 else args.sigma_y_dy,
                "gamma": central[index],
                "gamma_ci_lo": float(np.percentile(rates[index], 2.5)),
                "gamma_ci_hi": float(np.percentile(rates[index], 97.5)),
                "gamma_ratio_to_cic": central[index] / central[0],
                "ratio_ci_lo": ratio_lo,
                "ratio_ci_hi": ratio_hi,
                "fit_t_start": args.fit_t_start,
                "fit_t_stop": args.fit_t_stop,
                "fit_r_squared": r2,
                "model_efoldings": efoldings,
                "max_delta_f_ratio_fit_window": float(np.max(group["validity"][:, fit_mask])),
                **floor,
                "rate_gate": "PASS" if ratio_lo >= 0.90 and ratio_hi <= 1.10 else "FAIL",
                "duration_gate": "PASS" if efoldings >= 2.0 else "FAIL",
                "signal_gate": "PASS" if r2 >= 0.98 and floor["fit_start_snr"] >= 5.0 else "FAIL",
            }
        )
    candidate = rows[1]
    result = {
        "status": (
            "PASS"
            if all(candidate[key] == "PASS" for key in ("rate_gate", "duration_gate", "signal_gate"))
            else "FAIL"
        ),
        "benchmark": "independent new protected-mode E2 Gaussian test",
        "measured_mode": [args.mode_x, args.mode_y],
        "seed_partition": {
            "new_seed_ids": list(range(args.seed_start, args.seed_start + args.seeds)),
            "relationship_to_prior_campaigns": (
                "disjoint from prior seed ids 0--9"
                if set(range(args.seed_start, args.seed_start + args.seeds)).isdisjoint(range(10))
                else "overlaps prior seed ids 0--9"
            ),
        },
        "fit_window": [args.fit_t_start, args.fit_t_stop],
        "acceptance": {
            "rate": "two-sided paired 95% ratio interval inside [0.90,1.10]",
            "duration": "at least two candidate model e-foldings",
            "signal": "candidate R^2 >= 0.98 and fit-start SNR >= 5",
        },
        "grid_time_discrete_prediction": theory_record,
        "rows": rows,
    }
    args.output.mkdir(parents=True)
    with (args.output / "E2_new_protected_mode.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output / "E2_new_protected_mode.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
