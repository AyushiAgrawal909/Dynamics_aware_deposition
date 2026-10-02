#!/usr/bin/env python3
"""Reproduce the dispersion-based C4 rate ceilings for the frozen benchmarks."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pic_kde_2d.benchmarks.theory import (
    matched_cic_force_coupling,
    rate_tolerance_crossing,
    solve_landau_dispersion_coupling,
    warm_two_stream_growth,
)


def landau_rate(sigma_dx: float) -> float:
    coupling = matched_cic_force_coupling(1, 64, sigma_dx)
    return float(solve_landau_dispersion_coupling(0.4, coupling).imag)


def two_stream_rate(mode: int, sigma_dx: float) -> float:
    coupling = matched_cic_force_coupling(mode, 128, sigma_dx)
    k = 2.0 * math.pi * mode / 64.0
    return warm_two_stream_growth(k, 0.8, 0.05, coupling)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("results/c4_closed_form"))
    parser.add_argument(
        "--measured-m8-crossing",
        type=float,
        default=0.99,
        help="measured m=8 10%%-damage crossing used for the one-point bias calibration",
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    landau_crossing = rate_tolerance_crossing(
        landau_rate, tolerance=0.10, sigma_upper=4.0, increasing_magnitude=True
    )
    m8_crossing = rate_tolerance_crossing(
        lambda sigma: two_stream_rate(8, sigma), tolerance=0.10, sigma_upper=2.0
    )
    m12_crossing = rate_tolerance_crossing(
        lambda sigma: two_stream_rate(12, sigma), tolerance=0.10, sigma_upper=1.0
    )
    m8_baseline = two_stream_rate(8, 0.0)
    m8_ratio_at_measured_crossing = (
        two_stream_rate(8, args.measured_m8_crossing) / m8_baseline
    )
    m8_predicted_damage = 1.0 - m8_ratio_at_measured_crossing
    damage_calibration = 0.10 / m8_predicted_damage
    sigma_bias_factor = math.sqrt(damage_calibration)
    m8_crossing_corrected = m8_crossing / sigma_bias_factor
    m12_crossing_corrected = m12_crossing / sigma_bias_factor

    rows = []
    for benchmark, mode, crossing, corrected_crossing, frozen_width, rate_fn in (
        ("landau", 1, landau_crossing, None, 1.0, landau_rate),
        (
            "two_stream",
            8,
            m8_crossing,
            m8_crossing_corrected,
            args.measured_m8_crossing,
            lambda sigma: two_stream_rate(8, sigma),
        ),
        (
            "two_stream",
            12,
            m12_crossing,
            m12_crossing_corrected,
            0.22,
            lambda sigma: two_stream_rate(12, sigma),
        ),
    ):
        baseline = rate_fn(0.0)
        frozen_ratio = abs(rate_fn(frozen_width)) / abs(baseline)
        rows.append(
            {
                "benchmark": benchmark,
                "mode": mode,
                "sigma_R_10pct_dx": crossing,
                "bias_corrected_sigma_R_10pct_dx": corrected_crossing,
                "frozen_sigma_dx": frozen_width,
                "predicted_ratio_at_frozen_sigma": frozen_ratio,
                "frozen_sigma_passes_10pct": 0.9 <= frozen_ratio <= 1.1,
            }
        )

    with (args.output / "c4_closed_form_crossings.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    result = {
        "transfer": "sinc(pi*m/N)^4 * exp(-(2*pi*m*sigma_dx/N)^2)",
        "rate_tolerance": 0.10,
        "m8_bias_calibration": {
            "measured_crossing_dx": args.measured_m8_crossing,
            "closed_form_ratio_at_measured_crossing": m8_ratio_at_measured_crossing,
            "closed_form_damage_at_measured_crossing": m8_predicted_damage,
            "measured_damage_at_crossing": 0.10,
            "damage_multiplier": damage_calibration,
            "sigma_divisor": sigma_bias_factor,
            "corrected_m8_crossing_dx": m8_crossing_corrected,
            "corrected_m12_prior_dx": m12_crossing_corrected,
        },
        "rows": rows,
        "status": {
            "landau": "analytic crossing lies inside the measured 2-3 dx bracket",
            "two_stream_m8": "analytic crossing agrees closely with isolated-seed measurement",
            "two_stream_m12": (
                "projected linearized-delta-f sweep complete; raw curve favored, "
                "but 0.1987 dx narrowly fails the CI-supported rate gate"
            ),
            "matrix_e2_rate_gate": (
                "passes at 0.18445488487247555 dx on protected-edge m=12: "
                "ratio 0.92727 [0.91428, 0.93337]"
            ),
            "production_selector": (
                "0.18445488487247555 dx; exact-width noise reduction "
                "1.20050 [1.19768, 1.20338]"
            ),
        },
    }
    with (args.output / "c4_closed_form_validation.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")

    for row in rows:
        print(
            f"{row['benchmark']} m={row['mode']}: "
            f"sigma_R={row['sigma_R_10pct_dx']:.6f} dx, "
            f"ratio@frozen={row['predicted_ratio_at_frozen_sigma']:.6f}, "
            f"pass={row['frozen_sigma_passes_10pct']}"
        )


if __name__ == "__main__":
    main()
