#!/usr/bin/env python3
"""Build Step 2.3 E2 noise model from existing MISE and spectrum scans."""

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


def highk_energy_from_cic_spectrum(
    spectrum_rows: list[dict],
    sigma_dx: float,
    nx: int,
    highk_min_mode: int,
) -> float:
    """Predict high-k energy after Gaussian W using the measured CIC spectrum.

    The spectrum file contains power after selected filters.  To define a
    high-k metric on the full MISE sigma grid, use the measured sigma=0
    spectrum and apply the Gaussian density-filter power transfer:

        P_sigma(m) = P_CIC(m) * exp(-(2*pi*m/Nx)^2 * sigma_cells^2)
    """
    total = 0.0
    for row in spectrum_rows:
        if float(row["sigma_dx"]) != 0.0:
            continue
        mode = int(row["mode"])
        if mode < highk_min_mode:
            continue
        k_cells = 2.0 * np.pi * mode / nx
        power_transfer = np.exp(-(k_cells * sigma_dx) ** 2)
        total += float(row["power"]) * float(power_transfer)
    return float(total)


def build_noise_model(args: argparse.Namespace) -> list[dict]:
    mise_rows = sorted(read_csv(args.mise_csv), key=lambda row: float(row["sigma_dx"]))
    spectrum_rows = read_csv(args.spectrum_csv)
    if not mise_rows:
        raise RuntimeError(f"empty MISE CSV: {args.mise_csv}")
    mise_ref = float(mise_rows[0]["mise_mean"])
    rows = []
    previous_mise = None
    monotone_mise = True
    for row in mise_rows:
        sigma_dx = float(row["sigma_dx"])
        mise = float(row["mise_mean"])
        if previous_mise is not None and mise > previous_mise * (1.0 + 1.0e-12):
            monotone_mise = False
        previous_mise = mise
        rows.append(
            {
                "sigma_dx": sigma_dx,
                "MISE": mise,
                "noise_reduction": mise_ref / mise,
                "highk_energy": highk_energy_from_cic_spectrum(
                    spectrum_rows,
                    sigma_dx=sigma_dx,
                    nx=args.Nx,
                    highk_min_mode=args.highk_min_mode,
                ),
            }
        )
    print(f"noise model rows: {len(rows)}")
    print(f"MISE monotone non-increasing: {monotone_mise}")
    print(f"sigma_min={rows[0]['sigma_dx']} MISE={rows[0]['MISE']:.6e}")
    print(f"sigma_max={rows[-1]['sigma_dx']} noise_reduction={rows[-1]['noise_reduction']:.2f}")
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build E2 Step 2.3 noise model.")
    root = pathlib.Path(__file__).resolve().parents[1]
    parser.add_argument("--mise-csv", type=pathlib.Path, default=root / "results" / "E2_mise_cv" / "E2_mise_summary.csv")
    parser.add_argument(
        "--spectrum-csv",
        type=pathlib.Path,
        default=root / "results" / "E2_spectrum" / "E2_spectrum_summary.csv",
    )
    parser.add_argument("--output", type=pathlib.Path, default=root / "results" / "part2" / "noise_model.csv")
    parser.add_argument("--Nx", type=int, default=128)
    parser.add_argument("--highk-min-mode", type=int, default=32)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    rows = build_noise_model(args)
    write_csv(args.output, rows)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
