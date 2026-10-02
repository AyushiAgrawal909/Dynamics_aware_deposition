#!/usr/bin/env python3
"""Audit exact binomial/Gaussian transfers and spectral/FD symbols.

This calculation is deterministic and does not modify simulations or the
manuscript.  It quantifies where ``sigma=sqrt(n/2) dx`` is only a low-k
Gaussian match and separates the audited Python spectral operator from the
auxiliary C++ centred-finite-difference operator.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np


CASES = (
    {
        "case": "weak_Landau_m1",
        "Nx": 64, "Ny": 16,
        "Lx": 2.0 * math.pi / 0.4, "Ly": 8.0,
        "mx": 1, "my": 0,
    },
    {
        "case": "E2_longitudinal_m12",
        "Nx": 128, "Ny": 16,
        "Lx": 64.0, "Ly": 16.0,
        "mx": 12, "my": 0,
    },
    {
        "case": "E2_oblique_m12_n1_isotropic_filter",
        "Nx": 128, "Ny": 16,
        "Lx": 64.0, "Ly": 16.0,
        "mx": 12, "my": 1,
    },
)
PASSES = (1, 2, 4, 8, 12, 16)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def one_axis_binomial(theta: float, passes: int) -> float:
    return float(math.cos(0.5 * theta) ** (2 * passes))


def one_axis_gaussian(theta: float, sigma_cells: float) -> float:
    return float(math.exp(-0.5 * (theta * sigma_cells) ** 2))


def transfer_rows() -> list[dict]:
    rows: list[dict] = []
    for case in CASES:
        theta_x = 2.0 * math.pi * case["mx"] / case["Nx"]
        theta_y = 2.0 * math.pi * case["my"] / case["Ny"]
        for passes in PASSES:
            sigma_low_k = math.sqrt(passes / 2.0)
            binomial = one_axis_binomial(theta_x, passes) * one_axis_binomial(
                theta_y, passes
            )
            gaussian = one_axis_gaussian(theta_x, sigma_low_k) * one_axis_gaussian(
                theta_y, sigma_low_k
            )
            theta_squared = theta_x**2 + theta_y**2
            sigma_exact_at_this_mode = math.sqrt(
                -2.0 * math.log(binomial) / theta_squared
            )
            rows.append(
                {
                    **case,
                    "theta_x_kdx": theta_x,
                    "theta_y_kdy": theta_y,
                    "passes_n": passes,
                    "sigma_low_k_sqrt_n_over_2_cells": sigma_low_k,
                    "sigma_exactly_matching_this_mode_cells": sigma_exact_at_this_mode,
                    "one_way_W_raw_binomial_exact": binomial,
                    "one_way_W_gaussian_low_k_match": gaussian,
                    "one_way_relative_difference_binomial_over_gaussian_minus_1": binomial / gaussian - 1.0,
                    "round_trip_W2_raw_binomial_exact": binomial**2,
                    "round_trip_W2_gaussian_low_k_match": gaussian**2,
                    "round_trip_relative_difference_binomial_over_gaussian_minus_1": (binomial / gaussian) ** 2 - 1.0,
                }
            )
    return rows


def operator_rows() -> list[dict]:
    rows: list[dict] = []
    for case in CASES:
        dx = case["Lx"] / case["Nx"]
        dy = case["Ly"] / case["Ny"]
        kx = 2.0 * math.pi * case["mx"] / case["Lx"]
        ky = 2.0 * math.pi * case["my"] / case["Ly"]
        k2 = kx**2 + ky**2
        spectral_green = 1.0 / k2
        spectral_gradient_magnitude = math.sqrt(k2)
        spectral_field_magnitude_per_density = spectral_gradient_magnitude * spectral_green
        lambda_fd = (
            4.0 / dx**2 * math.sin(0.5 * kx * dx) ** 2
            + 4.0 / dy**2 * math.sin(0.5 * ky * dy) ** 2
        )
        dx_fd = math.sin(kx * dx) / dx
        dy_fd = math.sin(ky * dy) / dy
        fd_gradient_magnitude = math.hypot(dx_fd, dy_fd)
        fd_field_magnitude_per_density = fd_gradient_magnitude / lambda_fd
        rows.append(
            {
                **case,
                "dx": dx,
                "dy": dy,
                "kx": kx,
                "ky": ky,
                "spectral_laplacian_symbol_k2": k2,
                "five_point_positive_laplacian_symbol_Lambda_h": lambda_fd,
                "spectral_gradient_magnitude": spectral_gradient_magnitude,
                "centered_fd_gradient_magnitude": fd_gradient_magnitude,
                "spectral_field_magnitude_per_density": spectral_field_magnitude_per_density,
                "fd_field_magnitude_per_density": fd_field_magnitude_per_density,
                "fd_over_spectral_field_amplitude": (
                    fd_field_magnitude_per_density / spectral_field_magnitude_per_density
                ),
                "fd_over_spectral_field_power": (
                    fd_field_magnitude_per_density / spectral_field_magnitude_per_density
                ) ** 2,
            }
        )
    return rows


def exact_pass_budget(N: int, mode: int, sigma_ceiling: float) -> int:
    angle = math.pi * mode / N
    log_binomial_round_trip_per_pass = 4.0 * math.log(math.cos(angle))
    log_gaussian_round_trip = -(2.0 * math.pi * mode * sigma_ceiling / N) ** 2
    return max(0, int(math.floor(
        log_gaussian_round_trip / log_binomial_round_trip_per_pass + 1.0e-12
    )))


def pass_budget_rows() -> list[dict]:
    specifications = (
        ("weak_Landau_m1", 64, 1, 2.4534, 12),
        ("E2_longitudinal_m12", 128, 12, 0.1844549, 0),
    )
    rows = []
    for case, cells, mode, ceiling, expected in specifications:
        exact = exact_pass_budget(cells, mode, ceiling)
        leading_order = int(math.floor(2.0 * ceiling**2 + 1.0e-12))
        if exact != expected:
            raise AssertionError(f"exact pass-budget regression changed for {case}")
        rows.append(
            {
                "case": case,
                "N": cells,
                "mode": mode,
                "gaussian_ceiling_sigma_cells": ceiling,
                "exact_protected_mode_pass_budget": exact,
                "leading_order_floor_2_sigma_squared": leading_order,
                "reported_integer_budget_changes": exact != expected,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--spectral-source", type=Path, required=True)
    parser.add_argument("--dispersion-source", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    transfers = transfer_rows()
    operators = operator_rows()
    budgets = pass_budget_rows()

    landau_n12 = next(
        row for row in transfers
        if row["case"] == "weak_Landau_m1" and row["passes_n"] == 12
    )
    e2_n4 = next(
        row for row in transfers
        if row["case"] == "E2_longitudinal_m12" and row["passes_n"] == 4
    )
    if abs(landau_n12["one_way_relative_difference_binomial_over_gaussian_minus_1"]) >= 3.0e-5:
        raise AssertionError("weak-Landau low-k match exceeded audited tolerance")
    if abs(e2_n4["one_way_relative_difference_binomial_over_gaussian_minus_1"]) <= 1.0e-3:
        raise AssertionError("E2 diagnostic no longer distinguishes exact and low-k filters")

    args.output.mkdir(parents=True)
    write_csv(args.output / "raw_binomial_vs_low_k_gaussian_transfer.csv", transfers)
    write_csv(args.output / "spectral_vs_finite_difference_symbols.csv", operators)
    write_csv(args.output / "exact_pass_budgets.csv", budgets)

    result = {
        "status": "PASS_WITH_CORRECTIONS_IDENTIFIED",
        "identity": {
            "raw_binomial_one_axis": "cos(k*dx/2)^(2n) = (1-s)^n",
            "gaussian_one_axis": "exp[-(k*dx*sigma_cells)^2/2]",
            "sigma_sqrt_n_over_2": "leading-order k*dx -> 0 match only",
        },
        "quantified_anchors": {
            "Landau_m1_n12_one_way_relative_difference": landau_n12[
                "one_way_relative_difference_binomial_over_gaussian_minus_1"
            ],
            "E2_m12_n4_one_way_relative_difference": e2_n4[
                "one_way_relative_difference_binomial_over_gaussian_minus_1"
            ],
        },
        "impact_classification": [
            {
                "item": "claim that raw binomial and Gaussian are equivalent at protected modes to 5--6 digits",
                "effect": "WORDING_AND_TABLE_METADATA",
                "action": "replace by low-k asymptotic match; retain exact W for every quantitative mode",
            },
            {
                "item": "raw-binomial simulations and measured rate/noise reductions",
                "effect": "NO_SIMULATION_OR_MEASUREMENT_CHANGE",
                "action": "production code already applies exact cos^(2n) transfer; sigma_eq is metadata only",
            },
            {
                "item": "integer Landau and E2 pass ceilings",
                "effect": "NO_NUMERICAL_CHANGE",
                "action": "existing pass-budget helper compares exact protected-mode responses; audited values remain 12 and 0",
            },
            {
                "item": "Gaussian-versus-binomial noise comparison",
                "effect": "INTERPRETATION_ONLY",
                "action": "do not describe n=2 and sigma=1 as identical filters outside the low-k limit",
            },
            {
                "item": "Python production force-noise operator",
                "effect": "EQUATION_WORDING",
                "action": "use exact spectral k and k^2 symbols; reserve D_h and Lambda_h for the separate C++ finite-difference chain",
            },
            {
                "item": "dispersion predictions used for Python campaigns",
                "effect": "NO_RECOMPUTATION_REQUIRED",
                "action": "audited root code already uses the continuum Fourier Green function and exact spectral derivative",
            },
            {
                "item": "C++ HYPRE conservation diagnostics",
                "effect": "NO_CHANGE",
                "action": "retain five-point Laplacian/implementation-specific gather description as a separate chain",
            },
        ],
        "input_sha256": {
            str(args.spectral_source): sha256(args.spectral_source),
            str(args.dispersion_source): sha256(args.dispersion_source),
        },
        "tables": {
            "transfers": "raw_binomial_vs_low_k_gaussian_transfer.csv",
            "operator_symbols": "spectral_vs_finite_difference_symbols.csv",
            "pass_budgets": "exact_pass_budgets.csv",
        },
    }
    (args.output / "operator_and_transfer_audit.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
