#!/usr/bin/env python3
"""Reduce paired compensated-binomial E2 runs at the binding (12,1) mode."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np


WORKSPACE = Path(__file__).resolve().parents[2]
for candidate in (WORKSPACE / "pic_kde_2d", WORKSPACE / "pic_kde_2d" / "scripts"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from analyze_linearized_delta_f_m12_sigma import estimate_growing_floor, fit_rate


def load_group(root: Path, label: str, seeds: int) -> dict:
    amplitudes: list[np.ndarray] = []
    validity: list[np.ndarray] = []
    metadata: list[dict] = []
    times: np.ndarray | None = None
    for seed in range(seeds):
        path = root / label / f"seed{seed}"
        current = json.loads((path / "result.json").read_text(encoding="utf-8"))
        if int(current["seed"]) != seed:
            raise AssertionError(f"seed mismatch in {path}")
        if (int(current["kx_mode"]), int(current["ky_mode"])) != (12, 1):
            raise AssertionError(f"{path} did not measure binding mode (12,1)")
        if current["benchmark"] != "two_stream" or int(current["ppc"]) != 512:
            raise AssertionError(f"unexpected benchmark or PPC in {path}")
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


def fit_quality(amplitudes: np.ndarray, times: np.ndarray, start: float, stop: float) -> float:
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
    parser.add_argument(
        "--labels", default="base,p1n1,p2n2,p2n4,p3n4",
        help="comma-separated labels; first label is the paired CIC reference",
    )
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--nboot", type=int, default=5000)
    parser.add_argument("--fit-t-start", type=float, default=24.32)
    parser.add_argument("--fit-t-stop", type=float, default=59.68)
    parser.add_argument("--bootstrap-seed", type=int, default=84211)
    args = parser.parse_args()
    labels = [item.strip() for item in args.labels.split(",") if item.strip()]
    if len(labels) < 2 or len(labels) != len(set(labels)):
        raise ValueError("--labels must contain a unique baseline plus at least one candidate")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")

    groups = [load_group(args.input, label, args.seeds) for label in labels]
    for group in groups[1:]:
        np.testing.assert_array_equal(groups[0]["times"], group["times"])
    times = groups[0]["times"]
    if args.fit_t_start < times[0] or args.fit_t_stop > times[-1]:
        raise ValueError("fit window lies outside the saved histories")

    central = [
        fit_rate(group["amplitudes"], times, args.fit_t_start, args.fit_t_stop)
        for group in groups
    ]
    rng = np.random.default_rng(args.bootstrap_seed)
    rate_samples: list[list[float]] = [[] for _ in groups]
    ratio_samples: list[list[float]] = [[] for _ in groups]
    for _ in range(args.nboot):
        indices = rng.integers(0, args.seeds, size=args.seeds)
        rates = [
            fit_rate(group["amplitudes"][indices], times, args.fit_t_start, args.fit_t_stop)
            for group in groups
        ]
        if rates[0] <= 0.0 or not np.all(np.isfinite(rates)):
            continue
        for index, rate in enumerate(rates):
            rate_samples[index].append(float(rate))
            ratio_samples[index].append(float(rate / rates[0]))

    rows = []
    for index, (group, gamma) in enumerate(zip(groups, central)):
        rates = np.asarray(rate_samples[index], dtype=np.float64)
        ratios = np.asarray(ratio_samples[index], dtype=np.float64)
        if rates.size != args.nboot:
            raise AssertionError(f"only {rates.size}/{args.nboot} valid bootstraps for {group['label']}")
        floor = estimate_growing_floor(times, group["amplitudes"], args.fit_t_start)
        lo = float(np.percentile(ratios, 2.5))
        hi = float(np.percentile(ratios, 97.5))
        r2 = fit_quality(group["amplitudes"], times, args.fit_t_start, args.fit_t_stop)
        duration = args.fit_t_stop - args.fit_t_start
        metadata = group["metadata"][0]
        model_efoldings = float(max(0.0, gamma * duration))
        rows.append({
            "label": group["label"],
            "kx_mode": 12,
            "ky_mode": 1,
            "ppc": 512,
            "filter_kind": metadata.get("filter_kind", "gaussian"),
            "compensation_order": metadata.get("compensation_order"),
            "passes_x": metadata.get("binomial_passes_x"),
            "passes_y": metadata.get("binomial_passes_y"),
            "gamma": gamma,
            "gamma_ratio_to_cic": gamma / central[0],
            "ratio_sem": float(np.std(ratios)),
            "ratio_ci_lo": lo,
            "ratio_ci_hi": hi,
            "gamma_ci_lo": float(np.percentile(rates, 2.5)),
            "gamma_ci_hi": float(np.percentile(rates, 97.5)),
            "bootstrap_valid": int(rates.size),
            "fit_t_start": args.fit_t_start,
            "fit_t_stop": args.fit_t_stop,
            "fit_r_squared": r2,
            "model_efoldings": model_efoldings,
            **floor,
            "max_delta_f_ratio_fit_window": float(
                np.max(group["validity"][:, (times >= args.fit_t_start) & (times <= args.fit_t_stop)])
            ),
            "rate_gate": "PASS" if lo >= 0.90 and hi <= 1.10 else "FAIL",
            "duration_gate": "PASS" if model_efoldings >= 2.0 else "FAIL",
            "signal_gate": "PASS" if r2 >= 0.98 and floor["fit_start_snr"] >= 5.0 else "FAIL",
        })

    args.output.mkdir(parents=True)
    csv_path = args.output / "E2_compensated_corner.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result = {
        "benchmark": "projected linearized delta-f E2 binding corner",
        "measured_mode": [12, 1],
        "fit_window": [args.fit_t_start, args.fit_t_stop],
        "rate_definition": "slope of ln(mean_seed(abs(Ex_hat(12,1))))",
        "bootstrap_unit": "paired independent seed index; no temporal resampling",
        "acceptance": {
            "rate": "two-sided 95% ratio interval inside [0.90,1.10]",
            "duration": "candidate rate times shared window >= 2 e-foldings",
            "signal": "candidate R^2 >= 0.98 and fit-start amplitude/noise-floor >= 5",
        },
        "rows": rows,
    }
    (args.output / "E2_compensated_corner.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
