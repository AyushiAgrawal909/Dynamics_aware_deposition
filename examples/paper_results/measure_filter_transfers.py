#!/usr/bin/env python3
"""Measure Paper 1 filter transfers with deterministic single-mode particles.

This is an operator test, not a time-dependent plasma simulation.  It imports
the recovered campaign copy in ``pic_kde_2d_landau_cv_stage`` because that copy
contains the binomial and compensated-binomial implementations used by the
August 2026 campaigns.  The newer top-level ``pic_kde_2d`` copy does not contain
those production functions.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import pathlib
import sys
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np


HERE = pathlib.Path(__file__).resolve().parent
DOWNLOADS = HERE.parents[1]
CAMPAIGN_ROOT = DOWNLOADS / "pic_kde_2d_landau_cv_stage"
if not CAMPAIGN_ROOT.is_dir():
    raise FileNotFoundError(f"recovered campaign copy not found: {CAMPAIGN_ROOT}")
sys.path.insert(0, str(CAMPAIGN_ROOT))

from pic_kde_2d.core.grid2d import Grid2D  # noqa: E402
from pic_kde_2d.methods.cic import deposit_cic_2d, gather_cic_2d  # noqa: E402
from pic_kde_2d.methods.spectral_W import (  # noqa: E402
    make_W_binomial,
    make_W_compensated_binomial,
    make_W_gaussian,
)
from pic_kde_2d.methods.spectral_W_wrappers import (  # noqa: E402
    make_spectral_W_methods,
)


@dataclass(frozen=True)
class ModeCase:
    name: str
    nx: int
    ny: int
    lx: float
    ly: float
    mx: int
    my: int


@dataclass(frozen=True)
class FilterCase:
    name: str
    kind: str
    passes: int = 0
    order: int = 0
    sigma_cells: float = 0.0


MODES = (
    ModeCase("weak_landau_m1", 64, 16, 2.0 * math.pi / 0.4, 8.0, 1, 0),
    ModeCase("E2_m12", 128, 16, 64.0, 16.0, 12, 0),
    # This is not substituted for the m=12 campaign.  It is a separate check
    # that both filtered axes are handled correctly for the later corner case.
    ModeCase("E2_oblique_m12_n1", 128, 16, 64.0, 16.0, 12, 1),
)

FILTERS = (
    FilterCase("gaussian_sigma1", "gaussian", sigma_cells=1.0),
    FilterCase("raw_binomial_n4", "binomial", passes=4),
    FilterCase("compensated_p1_n1", "compensated", passes=1, order=1),
    FilterCase("compensated_p2_n2", "compensated", passes=2, order=2),
    FilterCase("compensated_p1_n4", "compensated", passes=4, order=1),
    FilterCase("compensated_p2_n4", "compensated", passes=4, order=2),
    FilterCase("compensated_p3_n4", "compensated", passes=4, order=3),
    FilterCase("compensated_p1_n16", "compensated", passes=16, order=1),
)


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def analytic_axis_response(theta: float, spec: FilterCase) -> float:
    if spec.kind == "gaussian":
        return float(math.exp(-0.5 * (spec.sigma_cells * theta) ** 2))
    s = math.sin(0.5 * theta) ** 2
    base = (1.0 - s) ** spec.passes
    if spec.kind == "binomial":
        return float(base)
    if spec.kind != "compensated":
        raise ValueError(spec.kind)
    polynomial = 0.0
    for j in range(spec.order + 1):
        polynomial += math.comb(spec.passes + j - 1, j) * s**j
    return float(base * polynomial)


def analytic_response(case: ModeCase, spec: FilterCase) -> float:
    theta_x = 2.0 * math.pi * case.mx / case.nx
    theta_y = 2.0 * math.pi * case.my / case.ny
    return analytic_axis_response(theta_x, spec) * analytic_axis_response(theta_y, spec)


def build_filter(grid: Grid2D, spec: FilterCase) -> np.ndarray:
    if spec.kind == "gaussian":
        return make_W_gaussian(
            grid,
            spec.sigma_cells * grid.dx,
            spec.sigma_cells * grid.dy,
        )
    if spec.kind == "binomial":
        return make_W_binomial(grid, spec.passes, spec.passes)
    if spec.kind == "compensated":
        return make_W_compensated_binomial(
            grid,
            spec.passes,
            spec.passes,
            compensation_order=spec.order,
        )
    raise ValueError(spec.kind)


def deterministic_particles(
    grid: Grid2D,
    mx: int,
    my: int,
    subcells: int,
) -> tuple[SimpleNamespace, np.ndarray]:
    """Return a periodic subcell quadrature carrying one cosine charge mode."""
    if subcells < 2:
        raise ValueError("subcells must be at least 2")
    i = np.arange(grid.Nx)[:, None, None, None]
    j = np.arange(grid.Ny)[None, :, None, None]
    a = np.arange(subcells)[None, None, :, None]
    b = np.arange(subcells)[None, None, None, :]
    shape = (grid.Nx, grid.Ny, subcells, subcells)
    x = np.broadcast_to(
        (i + (a + 0.37) / subcells) * grid.dx,
        shape,
    ).ravel()
    y = np.broadcast_to(
        (j + (b + 0.61) / subcells) * grid.dy,
        shape,
    ).ravel()
    phase = 2.0 * np.pi * (mx * x / grid.Lx + my * y / grid.Ly)
    weights = np.cos(phase)
    return SimpleNamespace(x=x, y=y, w=weights), phase


def mode_coefficient(values: np.ndarray, case: ModeCase) -> complex:
    return complex(np.fft.fft2(values)[case.mx % case.nx, case.my % case.ny]) / (
        case.nx * case.ny
    )


def vector_mode_coefficient(
    ex: np.ndarray,
    ey: np.ndarray,
    case: ModeCase,
) -> complex:
    kx = 2.0 * np.pi * case.mx / case.lx
    ky = 2.0 * np.pi * case.my / case.ly
    knorm = math.hypot(kx, ky)
    return (kx * mode_coefficient(ex, case) + ky * mode_coefficient(ey, case)) / knorm


def cic_checks(
    particles: SimpleNamespace,
    phase: np.ndarray,
    grid: Grid2D,
    case: ModeCase,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float, float]:
    rho = deposit_cic_2d(
        particles.x,
        particles.y,
        particles.w,
        grid.Nx,
        grid.Ny,
        grid.dx,
        grid.dy,
    )
    particle_coefficient = np.sum(
        particles.w * np.exp(-1j * phase)
    ) / (grid.Lx * grid.Ly)
    cic_deposit_measured = abs(mode_coefficient(rho, case) / particle_coefficient)

    kx = 2.0 * np.pi * case.mx / case.lx
    ky = 2.0 * np.pi * case.my / case.ly
    knorm = math.hypot(kx, ky)
    node_phase = (
        kx * grid.x[:, None] + ky * grid.y[None, :]
    )
    ex_node = (kx / knorm) * np.sin(node_phase)
    ey_node = (ky / knorm) * np.sin(node_phase)
    ex_p, ey_p = gather_cic_2d(
        ex_node,
        ey_node,
        particles.x,
        particles.y,
        grid.Nx,
        grid.Ny,
        grid.dx,
        grid.dy,
    )
    force = (kx * ex_p + ky * ey_p) / knorm
    cic_gather_measured = 2.0 * abs(np.mean(force * np.exp(-1j * phase)))
    cic_predicted = float(
        np.sinc(case.mx / case.nx) ** 2
        * np.sinc(case.my / case.ny) ** 2
    )
    ex, ey = grid.poisson_solve(rho)
    return rho, ex, ey, cic_predicted, cic_deposit_measured, cic_gather_measured


def measure_one(
    case: ModeCase,
    spec: FilterCase,
    subcells: int,
    cache: dict[str, tuple],
) -> dict:
    grid = Grid2D(Nx=case.nx, Ny=case.ny, Lx=case.lx, Ly=case.ly)
    if case.name not in cache:
        particles, phase = deterministic_particles(
            grid,
            case.mx,
            case.my,
            subcells,
        )
        cache[case.name] = (
            particles,
            phase,
            *cic_checks(particles, phase, grid, case),
        )
    (
        particles,
        phase,
        rho_cic,
        ex_cic,
        ey_cic,
        cic_predicted,
        cic_deposit_measured,
        cic_gather_measured,
    ) = cache[case.name]
    del phase

    W = build_filter(grid, spec)
    expected = analytic_response(case, spec)
    implemented = float(W[case.mx % case.nx, case.my])
    deposit, gather = make_spectral_W_methods(W)
    rho_filtered = deposit(particles, grid)
    ex_filtered, ey_filtered = grid.poisson_solve(rho_filtered)

    rho_ratio = abs(mode_coefficient(rho_filtered, case) / mode_coefficient(rho_cic, case))
    field_ratio = abs(
        vector_mode_coefficient(ex_filtered, ey_filtered, case)
        / vector_mode_coefficient(ex_cic, ey_cic, case)
    )

    ex_p_cic, ey_p_cic = gather_cic_2d(
        ex_cic,
        ey_cic,
        particles.x,
        particles.y,
        grid.Nx,
        grid.Ny,
        grid.dx,
        grid.dy,
    )
    ex_p_filtered, ey_p_filtered = gather(
        ex_filtered,
        ey_filtered,
        particles,
        grid,
    )
    kx = 2.0 * np.pi * case.mx / case.lx
    ky = 2.0 * np.pi * case.my / case.ly
    knorm = math.hypot(kx, ky)
    force_cic = (kx * ex_p_cic + ky * ey_p_cic) / knorm
    force_filtered = (kx * ex_p_filtered + ky * ey_p_filtered) / knorm
    force_ratio = float(np.dot(force_filtered, force_cic) / np.dot(force_cic, force_cic))
    force_residual = float(
        np.linalg.norm(force_filtered - force_ratio * force_cic)
        / np.linalg.norm(force_cic)
    )

    if spec.kind in ("binomial", "compensated") and spec.passes > 0:
        if abs(float(W[grid.Nx // 2, 0])) > 1.0e-25:
            raise AssertionError(f"{spec.name}: x Nyquist is not zero")
        if abs(float(W[0, grid.Ny // 2])) > 1.0e-25:
            raise AssertionError(f"{spec.name}: y Nyquist is not zero")

    row = {
        "mode_case": case.name,
        "Nx": case.nx,
        "Ny": case.ny,
        "Lx": case.lx,
        "Ly": case.ly,
        "mode_x": case.mx,
        "mode_y": case.my,
        "kdx": 2.0 * math.pi * case.mx / case.nx,
        "kdy": 2.0 * math.pi * case.my / case.ny,
        "filter": spec.name,
        "kind": spec.kind,
        "passes_x": spec.passes if spec.kind != "gaussian" else None,
        "passes_y": spec.passes if spec.kind != "gaussian" else None,
        "compensation_order": spec.order if spec.kind == "compensated" else None,
        "sigma_x_cells": spec.sigma_cells if spec.kind == "gaussian" else None,
        "sigma_y_cells": spec.sigma_cells if spec.kind == "gaussian" else None,
        "W_analytic": expected,
        "W_implemented": implemented,
        "density_ratio_measured": float(rho_ratio),
        "density_ratio_predicted": expected,
        "grid_field_ratio_measured": float(field_ratio),
        "grid_field_ratio_predicted": expected,
        "particle_force_ratio_measured": force_ratio,
        "particle_force_ratio_predicted": expected**2,
        "particle_force_scaled_residual": force_residual,
        "cic_deposit_measured": cic_deposit_measured,
        "cic_gather_measured": cic_gather_measured,
        "cic_continuum_shape": cic_predicted,
    }

    strict = 2.0e-12
    for measured_name, measured, predicted in (
        ("implemented W", implemented, expected),
        ("density", rho_ratio, expected),
        ("grid field", field_ratio, expected),
        ("particle force", force_ratio, expected**2),
    ):
        if not math.isclose(float(measured), float(predicted), rel_tol=strict, abs_tol=strict):
            raise AssertionError(
                f"{case.name}/{spec.name} {measured_name}: "
                f"measured={measured:.17g}, predicted={predicted:.17g}"
            )
    if force_residual > 2.0e-12:
        raise AssertionError(
            f"{case.name}/{spec.name}: force is not a single scaled mode; "
            f"residual={force_residual:.3e}"
        )
    # The finite subcell quadrature approaches the continuum CIC sinc^2 shape.
    # This looser check is deliberately separate from the machine-precision W
    # checks above.
    for label, value in (
        ("deposit", cic_deposit_measured),
        ("gather", cic_gather_measured),
    ):
        relative_error = abs(value / cic_predicted - 1.0)
        if relative_error > 3.0e-4:
            raise AssertionError(
                f"{case.name}: CIC {label} factor error {relative_error:.3e}"
            )
    return row


def run(subcells: int) -> dict:
    rows: list[dict] = []
    cache: dict[str, tuple] = {}
    for case in MODES:
        for spec in FILTERS:
            rows.append(measure_one(case, spec, subcells, cache))

    anchor = next(
        row
        for row in rows
        if row["mode_case"] == "E2_m12"
        and row["filter"] == "compensated_p3_n4"
    )
    attenuation = 1.0 - anchor["particle_force_ratio_measured"]
    measured_protected_attenuation = 0.0028624
    if abs(attenuation / measured_protected_attenuation - 1.0) > 0.003:
        raise AssertionError(
            "E2 p=3,n=4 attenuation does not agree within 0.3%: "
            f"transfer={anchor['particle_force_ratio_measured']:.12f}, "
            f"attenuation={attenuation:.12f}"
        )

    source_files = {
        "spectral_W.py": CAMPAIGN_ROOT / "pic_kde_2d/methods/spectral_W.py",
        "spectral_W_wrappers.py": CAMPAIGN_ROOT
        / "pic_kde_2d/methods/spectral_W_wrappers.py",
        "cic.py": CAMPAIGN_ROOT / "pic_kde_2d/methods/cic.py",
        "grid2d.py": CAMPAIGN_ROOT / "pic_kde_2d/core/grid2d.py",
    }
    return {
        "audit": "deterministic single-Fourier-mode particle-field transfer",
        "campaign_source_root": str(CAMPAIGN_ROOT),
        "source_sha256": {
            str(path.relative_to(DOWNLOADS)): sha256(path)
            for path in source_files.values()
        },
        "subcells_per_axis_per_grid_cell": subcells,
        "particle_count_by_mode": {
            case.name: case.nx * case.ny * subcells**2 for case in MODES
        },
        "stage_definitions": {
            "density": "filtered CIC deposit / CIC deposit at target mode",
            "grid_field": "Poisson E from filtered deposit / Poisson E from CIC deposit",
            "particle_force": "filtered deposit, Poisson, field-side W, CIC gather / CIC chain",
            "cic_shape": "absolute deposit and gather factors versus sinc(mx/Nx)^2 sinc(my/Ny)^2",
        },
        "e2_p3_n4": {
            "measured_force_transfer": anchor["particle_force_ratio_measured"],
            "measured_attenuation": attenuation,
            "reported_protected_attenuation": measured_protected_attenuation,
            "relative_difference": attenuation / measured_protected_attenuation - 1.0,
        },
        "rows": rows,
    }


def write_outputs(result: dict, output: pathlib.Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    json_path = output / "filter_transfer_audit.json"
    csv_path = output / "filter_transfer_audit.csv"
    json_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    rows = result["rows"]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=HERE / "results/filter_transfer_audit",
    )
    parser.add_argument("--subcells", type=int, default=8)
    args = parser.parse_args()
    result = run(args.subcells)
    write_outputs(result, args.output)
    print(json.dumps(result["e2_p3_n4"], indent=2, sort_keys=True))
    print(f"wrote {args.output / 'filter_transfer_audit.json'}")
    print(f"wrote {args.output / 'filter_transfer_audit.csv'}")


if __name__ == "__main__":
    main()
