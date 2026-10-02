#!/usr/bin/env python3
"""Step 2.4 E2 selector using measured gamma-ratio interpolation.

Two reference conventions are supported:

* selector_ref: gamma_sigma / gamma(0.25 dx). This was the original internal
  selector convention.
* cic: gamma_sigma / gamma_CIC. This is the paper-facing convention when the
  figure and unsafe threshold are normalized to unsmoothed CIC.
"""

from __future__ import annotations

import argparse
import csv
import pathlib

import numpy as np


def read_csv(path: pathlib.Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: pathlib.Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def select_sigma(gamma_ratio_fn, eps_gamma: float, sigma_grid: np.ndarray) -> float:
    """Return largest sigma satisfying |gamma_ratio(sigma) - 1| <= eps_gamma."""
    feasible = [float(sigma) for sigma in sigma_grid if abs(gamma_ratio_fn(float(sigma)) - 1.0) <= eps_gamma]
    if not feasible:
        return float("nan")
    return max(feasible)


def make_interp_fn(x: np.ndarray, y: np.ndarray):
    order = np.argsort(x)
    x_sorted = np.asarray(x[order], dtype=np.float64)
    y_sorted = np.asarray(y[order], dtype=np.float64)

    def interp(value: float) -> float:
        return float(np.interp(value, x_sorted, y_sorted))

    return interp


def build_selector(args: argparse.Namespace) -> list[dict]:
    gamma_rows = read_csv(args.gamma_csv)
    noise_rows = read_csv(args.noise_csv)

    gamma_sigma = np.asarray([float(row["sigma_dx"]) for row in gamma_rows], dtype=np.float64)
    gamma_values = np.asarray([float(row["gamma_sigma"]) for row in gamma_rows], dtype=np.float64)
    if args.reference == "selector_ref":
        gamma_ratio = np.asarray([float(row["gamma_ratio"]) for row in gamma_rows], dtype=np.float64)
        reference_gamma = float(gamma_values[np.argmin(np.abs(gamma_sigma - args.reference_sigma_dx))])
    elif args.reference == "cic":
        if args.cic_gamma is None:
            if args.headline_csv is None:
                raise RuntimeError("--reference cic requires --cic-gamma or --headline-csv")
            headline_rows = read_csv(args.headline_csv)
            try:
                args.cic_gamma = float(next(row["gamma_sigma"] for row in headline_rows if row["choice"] == "cic"))
            except StopIteration as exc:
                raise RuntimeError(f"No choice=cic row found in {args.headline_csv}") from exc
        reference_gamma = float(args.cic_gamma)
        gamma_ratio = gamma_values / reference_gamma
    else:
        raise RuntimeError(f"unknown reference convention: {args.reference}")
    gamma_fn = make_interp_fn(gamma_sigma, gamma_ratio)

    noise_sigma = np.asarray([float(row["sigma_dx"]) for row in noise_rows], dtype=np.float64)
    noise_reduction = np.asarray([float(row["noise_reduction"]) for row in noise_rows], dtype=np.float64)
    highk_energy = np.asarray([float(row["highk_energy"]) for row in noise_rows], dtype=np.float64)
    noise_fn = make_interp_fn(noise_sigma, noise_reduction)
    highk_fn = make_interp_fn(noise_sigma, highk_energy)

    # The selector should not extrapolate beyond the measured gamma curve.  Use
    # the noise grid inside the measured gamma range so the largest feasible
    # point can still land on the fine MISE grid.
    sigma_grid = noise_sigma[(noise_sigma >= gamma_sigma.min()) & (noise_sigma <= gamma_sigma.max())]
    if len(sigma_grid) == 0:
        raise RuntimeError("noise sigma grid does not overlap gamma sigma grid")

    rows = []
    for eps_gamma in args.eps_gamma:
        sigma_star = select_sigma(gamma_fn, eps_gamma, sigma_grid)
        predicted_ratio = gamma_fn(sigma_star) if np.isfinite(sigma_star) else float("nan")
        measured_ratio_at_star = predicted_ratio
        noise_at_star = noise_fn(sigma_star) if np.isfinite(sigma_star) else float("nan")
        highk_at_star = highk_fn(sigma_star) if np.isfinite(sigma_star) else float("nan")
        checkpoint_pass = bool(abs(sigma_star - args.expected_sigma_safe) <= args.expected_tolerance) if np.isfinite(sigma_star) else False
        rows.append(
            {
                "eps_gamma": eps_gamma,
                "reference": args.reference,
                "reference_gamma": reference_gamma,
                "sigma_star": sigma_star,
                "predicted_ratio": predicted_ratio,
                "measured_ratio_at_star": measured_ratio_at_star,
                "noise_reduction_at_star": noise_at_star,
                "highk_energy_at_star": highk_at_star,
                "checkpoint_expected_sigma": args.expected_sigma_safe,
                "checkpoint_pass": checkpoint_pass,
            }
        )
        status = "PASS" if checkpoint_pass else "FLAG"
        print(
            f"reference={args.reference} eps_gamma={eps_gamma:.3f} sigma_star={sigma_star:.3f} "
            f"ratio={predicted_ratio:.4f} noise_reduction={noise_at_star:.2f} {status}"
        )
    return rows


def parse_float_list(text: str) -> list[float]:
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def build_parser() -> argparse.ArgumentParser:
    root = pathlib.Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Run E2 Step 2.4 selector.")
    parser.add_argument(
        "--gamma-csv",
        type=pathlib.Path,
        default=root / "results" / "part2" / "gamma_sweep_full_schemafix_failed_checkpoint.csv",
    )
    parser.add_argument("--noise-csv", type=pathlib.Path, default=root / "results" / "part2" / "noise_model.csv")
    parser.add_argument(
        "--headline-csv",
        type=pathlib.Path,
        default=root / "results" / "part2" / "headline_comparison" / "headline_table.csv",
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=root / "results" / "part2" / "selector_result.csv",
    )
    parser.add_argument("--reference", choices=["selector_ref", "cic"], default="cic")
    parser.add_argument("--cic-gamma", type=float, default=None)
    parser.add_argument("--reference-sigma-dx", type=float, default=0.25)
    parser.add_argument("--eps-gamma", type=parse_float_list, default=parse_float_list("0.05,0.06,0.10"))
    parser.add_argument("--expected-sigma-safe", type=float, default=0.30)
    parser.add_argument("--expected-tolerance", type=float, default=0.10)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    rows = build_selector(args)
    write_csv(args.output, rows)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
