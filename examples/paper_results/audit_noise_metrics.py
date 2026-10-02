#!/usr/bin/env python3
"""Regenerate and audit the density/field error metrics used in Paper 1.

This script reads immutable per-seed CSV/JSON artifacts.  It does not rerun a
production PIC campaign.  The small synthetic checks exercise normalization,
Parseval scope, and the exact binned-CIC leave-one-out subtraction.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
import sys
from typing import Iterable

import numpy as np


REPRO = Path(__file__).resolve().parent
WORKSPACE = REPRO.parents[1]
PROJECT = WORKSPACE / "pic_kde_2d"
STAGE = WORKSPACE / "pic_kde_2d_landau_cv_stage"
RECOVERED = (
    PROJECT
    / "results_raw_rate_audit"
    / "recovered_meluxina_20260909"
    / "noise_metric_campaigns"
)
OUTPUT = REPRO / "results" / "noise_metric_audit"

# Import the preserved implementation that contains the exact LOO and all
# filter families.  It is intentionally ahead of the top-level campaign copy.
sys.path.insert(0, str(STAGE))
from pic_kde_2d.conventions import marker_grid_shape  # noqa: E402
from pic_kde_2d.core.grid2d import Grid2D  # noqa: E402
from pic_kde_2d.core.particles2d import (  # noqa: E402
    init_landau_2d,
    init_two_stream_2d,
)
from pic_kde_2d.methods.cic import deposit_cic_2d, gather_cic_2d  # noqa: E402
from pic_kde_2d.methods.lscv import (  # noqa: E402
    cic_filter_self_matrix,
    cic_weight_second_moment,
    lscv_score,
    mean_cic_filter_self,
)
from pic_kde_2d.methods.spectral_W import (  # noqa: E402
    apply_W_to_grid,
    make_W_binomial,
    make_W_gaussian,
)


GAUSSIAN_CAMPAIGNS = (
    {
        "name": "table4_per_cell_4867515",
        "job": "4867515",
        "root": RECOVERED / "p1_landau_noise_4867515",
        "layout": "randomized per-cell velocity quadrature",
        "layout_evidence": (
            "contemporaneous audit record; result schema predates the "
            "quiet_start_layout field"
        ),
    },
    {
        "name": "aligned_global_4868286",
        "job": "4868286",
        "root": RECOVERED / "p1_landau_noise_4868286",
        "layout": "globally shuffled velocity quadrature",
        "layout_evidence": "serialized as quiet_start_layout=global",
    },
)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"empty CSV: {path}")
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise RuntimeError(f"refusing to write an empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"expected object in {path}")
    return value


def ratio_bootstrap(
    baseline: np.ndarray,
    candidate: np.ndarray,
    *,
    nboot: int,
    seed: int,
) -> tuple[float, float, float]:
    """Ratio of ensemble means with paired independent-seed resampling."""
    if baseline.shape != candidate.shape or baseline.ndim != 1:
        raise ValueError("paired vectors must be one-dimensional and equal length")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, baseline.size, size=(nboot, baseline.size))
    draws = np.mean(baseline[indices], axis=1) / np.mean(candidate[indices], axis=1)
    return (
        float(np.mean(baseline) / np.mean(candidate)),
        float(np.percentile(draws, 2.5)),
        float(np.percentile(draws, 97.5)),
    )


def mean_bootstrap(
    values: np.ndarray, *, nboot: int, seed: int
) -> tuple[float, float, float]:
    return ratio_bootstrap(
        values,
        np.ones_like(values),
        nboot=nboot,
        seed=seed,
    )


def assert_summary_matches(
    generated: list[dict],
    archived_path: Path,
    key_fields: tuple[str, ...],
    value_fields: tuple[str, ...],
) -> None:
    archived = read_csv(archived_path)
    for row in generated:
        matches = [
            old
            for old in archived
            if all(str(old[key]) == str(row[key]) for key in key_fields)
        ]
        if len(matches) != 1:
            raise AssertionError(
                f"expected one archived match for {key_fields}: {row}"
            )
        old = matches[0]
        for field in value_fields:
            np.testing.assert_allclose(
                float(row[field]),
                float(old[field]),
                rtol=2.0e-14,
                atol=1.0e-16,
                err_msg=f"archive mismatch in {archived_path}: {field}",
            )


def regenerate_gaussian() -> list[dict]:
    output: list[dict] = []
    for campaign in GAUSSIAN_CAMPAIGNS:
        raw: list[dict[str, str]] = []
        for seed in range(10):
            raw.extend(
                read_csv(
                    campaign["root"]
                    / f"seed{seed}"
                    / "mise"
                    / "landau_mise_seed_values.csv"
                )
            )
        sigmas = sorted({float(row["sigma_dx"]) for row in raw})
        matrix = {
            sigma: np.asarray(
                [
                    float(
                        next(
                            row["mise_time_avg"]
                            for row in raw
                            if int(row["seed"]) == seed
                            and float(row["sigma_dx"]) == sigma
                        )
                    )
                    for seed in range(10)
                ],
                dtype=np.float64,
            )
            for sigma in sigmas
        }
        baseline = matrix[0.0]
        generated = []
        for sigma in sigmas:
            values = matrix[sigma]
            reduction = ratio_bootstrap(
                baseline,
                values,
                nboot=10_000,
                seed=91_000 + int(1000 * sigma),
            )
            mse = mean_bootstrap(
                values,
                nboot=10_000,
                seed=92_000 + int(1000 * sigma),
            )
            generated.append(
                {
                    "sigma_dx": sigma,
                    "mise_mean": mse[0],
                    "mise_ci_lo": mse[1],
                    "mise_ci_hi": mse[2],
                    "noise_reduction": reduction[0],
                    "noise_reduction_ci_lo": reduction[1],
                    "noise_reduction_ci_hi": reduction[2],
                    "n_seeds": 10,
                }
            )
            output.append(
                {
                    "campaign": campaign["name"],
                    "job": campaign["job"],
                    "quiet_start_layout": campaign["layout"],
                    "sigma_dx": sigma,
                    "grid_density_mse_mean": mse[0],
                    "grid_density_mse_ci_lo": mse[1],
                    "grid_density_mse_ci_hi": mse[2],
                    "noise_power_reduction": reduction[0],
                    "noise_power_reduction_ci_lo": reduction[1],
                    "noise_power_reduction_ci_hi": reduction[2],
                    "ppc": 1024,
                    "reference_ppc": 16384,
                    "time_window": "[5,30]",
                    "n_times_per_seed": 51,
                    "n_seeds": 10,
                }
            )
        assert_summary_matches(
            generated,
            campaign["root"] / "reduced" / "landau_noise_summary.csv",
            ("sigma_dx",),
            (
                "mise_mean",
                "mise_ci_lo",
                "mise_ci_hi",
                "noise_reduction",
                "noise_reduction_ci_lo",
                "noise_reduction_ci_hi",
            ),
        )
    return output


def load_filter_seed_rows(root: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in sorted(root.glob("seed*/binomial_noise_seed.csv")):
        rows.extend(read_csv(path))
    if sorted({int(row["seed"]) for row in rows}) != list(range(10)):
        raise AssertionError(f"incomplete seed set: {root}")
    return rows


def regenerate_filter_group(
    *,
    campaign: str,
    benchmark: str,
    family: str,
    order: int | None,
    root: Path,
    archived: Path,
    ppc: int,
    reference_ppc: int,
    time_window: str,
    n_times: int,
) -> list[dict]:
    rows = load_filter_seed_rows(root)
    if "compensation_order" in rows[0] and order is not None:
        rows = [row for row in rows if int(row["compensation_order"]) == order]
    passes = sorted({int(row["passes"]) for row in rows})
    matrix = np.asarray(
        [
            [
                float(
                    next(
                        row["mise"]
                        for row in rows
                        if int(row["seed"]) == seed
                        and int(row["passes"]) == count
                    )
                )
                for count in passes
            ]
            for seed in range(10)
        ],
        dtype=np.float64,
    )
    rng = np.random.default_rng(85_231)
    indices = rng.integers(0, 10, size=(4000, 10))
    boot_means = np.mean(matrix[indices], axis=1)
    boot_ratios = boot_means[:, [0]] / boot_means
    means = np.mean(matrix, axis=0)
    reductions = means[0] / means
    generated = []
    output = []
    for index, count in enumerate(passes):
        row = {
            "passes": count,
            "mise_mean": float(means[index]),
            "mise_ci_lo": float(np.percentile(boot_means[:, index], 2.5)),
            "mise_ci_hi": float(np.percentile(boot_means[:, index], 97.5)),
            "noise_reduction": float(reductions[index]),
            "noise_reduction_ci_lo": float(np.percentile(boot_ratios[:, index], 2.5)),
            "noise_reduction_ci_hi": float(np.percentile(boot_ratios[:, index], 97.5)),
            "n_seeds": 10,
        }
        generated.append(row)
        output.append(
            {
                "campaign": campaign,
                "benchmark": benchmark,
                "filter_family": family,
                "compensation_order": "" if order is None else order,
                "passes": count,
                "grid_density_mse_mean": row["mise_mean"],
                "grid_density_mse_ci_lo": row["mise_ci_lo"],
                "grid_density_mse_ci_hi": row["mise_ci_hi"],
                "noise_power_reduction": row["noise_reduction"],
                "noise_power_reduction_ci_lo": row["noise_reduction_ci_lo"],
                "noise_power_reduction_ci_hi": row["noise_reduction_ci_hi"],
                "ppc": ppc,
                "reference_ppc": reference_ppc,
                "time_window": time_window,
                "n_times_per_seed": n_times,
                "n_seeds": 10,
            }
        )
    assert_summary_matches(
        generated,
        archived,
        ("passes",),
        (
            "mise_mean",
            "mise_ci_lo",
            "mise_ci_hi",
            "noise_reduction",
            "noise_reduction_ci_lo",
            "noise_reduction_ci_hi",
        ),
    )
    return output


def regenerate_filters() -> list[dict]:
    specs = [
        (
            "raw_binomial_landau_5048431",
            "weak Landau",
            "raw_binomial",
            None,
            RECOVERED / "binomial_noise_landau_weak_5048431" / "shards",
            RECOVERED
            / "binomial_noise_landau_weak_5048431"
            / "analysis"
            / "binomial_noise_summary.csv",
            1024,
            16384,
            "[5,30]",
            251,
        ),
        (
            "raw_binomial_e2_5048435",
            "E2",
            "raw_binomial",
            None,
            RECOVERED / "binomial_noise_e2_5048435" / "shards",
            RECOVERED
            / "binomial_noise_e2_5048435"
            / "analysis"
            / "binomial_noise_summary.csv",
            32,
            1024,
            "[0.5,4]",
            36,
        ),
    ]
    for benchmark, short, ppc, ref, window, times in (
        ("landau", "weak Landau", 1024, 16384, "[5,30]", 251),
        ("e2", "E2", 32, 1024, "[0.5,4]", 36),
    ):
        specs.append(
            (
                f"compensated_p1_{benchmark}_20260811",
                short,
                "compensated_binomial",
                1,
                PROJECT
                / "results_raw_rate_audit"
                / "compensated_binomial_20260811"
                / "noise"
                / benchmark,
                PROJECT
                / "results"
                / "compensated_binomial_20260811"
                / "analysis"
                / f"{benchmark}_noise"
                / "binomial_noise_summary.csv",
                ppc,
                ref,
                window,
                times,
            )
        )
        for order in (2, 3):
            specs.append(
                (
                    f"compensated_p{order}_{benchmark}_20260811",
                    short,
                    "compensated_binomial",
                    order,
                    PROJECT
                    / "results_raw_rate_audit"
                    / "higher_order_compensation_20260811"
                    / "noise"
                    / benchmark,
                    PROJECT
                    / "results"
                    / "higher_order_compensation_20260811"
                    / "analysis"
                    / f"p{order}"
                    / f"{benchmark}_noise"
                    / "binomial_noise_summary.csv",
                    ppc,
                    ref,
                    window,
                    times,
                )
            )
    output: list[dict] = []
    for spec in specs:
        output.extend(
            regenerate_filter_group(
                campaign=spec[0],
                benchmark=spec[1],
                family=spec[2],
                order=spec[3],
                root=spec[4],
                archived=spec[5],
                ppc=spec[6],
                reference_ppc=spec[7],
                time_window=spec[8],
                n_times=spec[9],
            )
        )
    return output


def paired_noise_bootstrap(
    baseline: np.ndarray, candidate: np.ndarray, nboot: int, seed: int
) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(nboot):
        indices = rng.integers(0, baseline.size, size=baseline.size)
        values.append(float(np.mean(baseline[indices]) / np.mean(candidate[indices])))
    array = np.asarray(values)
    return (
        float(np.mean(baseline) / np.mean(candidate)),
        float(np.percentile(array, 2.5)),
        float(np.percentile(array, 97.5)),
    )


def regenerate_e2_density_field() -> list[dict]:
    raw_root = (
        PROJECT
        / "results_raw_rate_audit"
        / "recovered_meluxina_20260909"
        / "e2_joint_corner_4913408"
        / "noise"
    )
    archived = load_json(
        PROJECT / "results" / "e2_joint_corner_4913408" / "joint_corner_sweep.json"
    )["rows"]
    labels = [row["label"] for row in archived]
    by_label = {
        label: {"density": [], "field": [], "charge": [], "neutrality": []}
        for label in labels
    }
    for seed in range(10):
        for row in read_csv(raw_root / f"seed{seed}" / "joint_noise_seed.csv"):
            entry = by_label[row["label"]]
            entry["density"].append(float(row["density_mse"]))
            entry["field"].append(float(row["field_mse"]))
            entry["charge"].append(float(row["max_relative_charge_error"]))
            entry["neutrality"].append(float(row["max_neutrality_defect"]))
    baseline = by_label["base"]
    output = []
    for index, old in enumerate(archived):
        label = old["label"]
        current = by_label[label]
        density = paired_noise_bootstrap(
            np.asarray(baseline["density"]),
            np.asarray(current["density"]),
            5000,
            59_200 + index,
        )
        field = paired_noise_bootstrap(
            np.asarray(baseline["field"]),
            np.asarray(current["field"]),
            5000,
            59_300 + index,
        )
        for actual, key in zip(
            (*density, *field),
            (
                "density_noise_reduction",
                "density_noise_ci_lo",
                "density_noise_ci_hi",
                "field_noise_reduction",
                "field_noise_ci_lo",
                "field_noise_ci_hi",
            ),
        ):
            np.testing.assert_allclose(actual, float(old[key]), rtol=2e-14, atol=1e-16)
        output.append(
            {
                "label": label,
                "sigma_x_dx": old["sigma_x_dx"],
                "sigma_y_dy": old["sigma_y_dy"],
                "density_error_power_reduction": density[0],
                "density_ci_lo": density[1],
                "density_ci_hi": density[2],
                "grid_field_error_power_reduction": field[0],
                "field_ci_lo": field[1],
                "field_ci_hi": field[2],
                "max_relative_charge_error": max(current["charge"]),
                "max_neutrality_defect": max(current["neutrality"]),
                "ppc": 32,
                "reference_ppc": 1024,
                "time_window": "[0.5,4]",
                "n_times_per_seed": 36,
                "n_seeds": 10,
            }
        )
    return output


def normalization_and_parseval_checks() -> dict:
    grid = Grid2D(Nx=8, Ny=4, Lx=4.0, Ly=2.0)
    mx, my = marker_grid_shape(grid.Nx, grid.Ny, 4)
    particles = {
        "landau": init_landau_2d(
            Nx_marker=mx,
            Ny_marker=my,
            Lx=grid.Lx,
            Ly=grid.Ly,
            v_thermal=1.0,
            alpha=0.01,
            kx_mode=1,
            omega_real=1.2,
            gamma=-0.06,
            rng=np.random.default_rng(7),
        ),
        "e2": init_two_stream_2d(
            Nx_marker=mx,
            Ny_marker=my,
            Lx=grid.Lx,
            Ly=grid.Ly,
            v_beam=0.8,
            v_thermal=0.05,
            alpha=0.001,
            kx_mode=1,
            rng=np.random.default_rng(9),
        ),
    }
    charge_rows = []
    for species, part in particles.items():
        rho = deposit_cic_2d(
            part.x, part.y, part.w, grid.Nx, grid.Ny, grid.dx, grid.dy
        )
        for filter_name, W in (
            ("CIC", None),
            ("Gaussian", make_W_gaussian(grid, grid.dx, grid.dy)),
            ("raw binomial n=4", make_W_binomial(grid, 4, 4)),
        ):
            candidate = rho if W is None else apply_W_to_grid(rho, W, grid)
            deposited = float(np.sum(candidate) * grid.dx * grid.dy)
            expected = float(np.sum(part.w))
            np.testing.assert_allclose(expected, -grid.Lx * grid.Ly, atol=2e-15)
            np.testing.assert_allclose(deposited, expected, atol=4e-15)
            charge_rows.append(
                {
                    "species": species,
                    "filter": filter_name,
                    "particle_charge": expected,
                    "deposited_charge": deposited,
                    "mean_electron_charge_density": float(np.mean(candidate)),
                    "mean_normalized_number_density": float(-np.mean(candidate)),
                    "absolute_charge_defect": abs(deposited - expected),
                }
            )

    rng = np.random.default_rng(112)
    error = rng.normal(size=(grid.Nx, grid.Ny))
    error += 0.31
    error += 0.17 * np.cos(2.0 * np.pi * np.arange(grid.Nx)[:, None] / grid.Nx)
    real_mse = float(np.mean(error**2))
    fft = np.fft.fft2(error)
    spectral_all = float(np.sum(np.abs(fft) ** 2) / error.size**2)
    np.testing.assert_allclose(real_mse, spectral_all, rtol=2e-15, atol=2e-15)
    k0 = float(np.abs(fft[0, 0]) ** 2 / error.size**2)
    protected_pair = float(
        (np.abs(fft[1, 0]) ** 2 + np.abs(fft[-1, 0]) ** 2) / error.size**2
    )
    return {
        "charge_rows": charge_rows,
        "parseval": {
            "real_space_grid_mean_square": real_mse,
            "all_resolved_modes_power": spectral_all,
            "k0_power_in_control": k0,
            "protected_mode_pair_power_in_control": protected_pair,
            "absolute_parseval_difference": abs(real_mse - spectral_all),
            "scope": (
                "all resolved modes; no Fourier mask. k=0 is formally included "
                "in density MSE but vanishes for production pairs by charge "
                "normalization; the protected mode is included"
            ),
        },
    }


def explicit_loo_score(
    x: np.ndarray,
    y: np.ndarray,
    grid: Grid2D,
    W: np.ndarray | None,
) -> tuple[float, float, float]:
    n = x.size
    rho = deposit_cic_2d(
        x,
        y,
        np.full(n, 1.0 / n),
        grid.Nx,
        grid.Ny,
        grid.dx,
        grid.dy,
    )
    moment = cic_weight_second_moment(x, y, grid)
    self_matrix = cic_filter_self_matrix(grid, W)
    mean_self = mean_cic_filter_self(moment, self_matrix)
    fast = lscv_score(rho, W, mean_self, n, grid)
    smoothed = rho if W is None else apply_W_to_grid(rho, W, grid)
    integrated_square = float(np.sum(smoothed**2) * grid.dx * grid.dy)
    evaluations = []
    for omitted in range(n):
        keep = np.arange(n) != omitted
        loo = deposit_cic_2d(
            x[keep],
            y[keep],
            np.full(n - 1, 1.0 / (n - 1)),
            grid.Nx,
            grid.Ny,
            grid.dx,
            grid.dy,
        )
        if W is not None:
            loo = apply_W_to_grid(loo, W, grid)
        value, _ = gather_cic_2d(
            loo,
            np.zeros_like(loo),
            x[omitted : omitted + 1],
            y[omitted : omitted + 1],
            grid.Nx,
            grid.Ny,
            grid.dx,
            grid.dy,
        )
        evaluations.append(float(value[0]))
    explicit = integrated_square - 2.0 * float(np.mean(evaluations))
    return fast, explicit, mean_self


def loo_checks() -> dict:
    grid = Grid2D(Nx=7, Ny=6, Lx=3.5, Ly=2.4)
    rng = np.random.default_rng(918)
    x = rng.random(11) * grid.Lx
    y = rng.random(11) * grid.Ly
    rows = []
    for label, W in (
        ("CIC", None),
        ("Gaussian sigma=1 cell", make_W_gaussian(grid, grid.dx, grid.dy)),
        ("raw binomial n=4", make_W_binomial(grid, 4, 4)),
    ):
        fast, explicit, mean_self = explicit_loo_score(x, y, grid, W)
        np.testing.assert_allclose(fast, explicit, rtol=2e-14, atol=2e-14)
        rows.append(
            {
                "filter": label,
                "quadratic_form_score": fast,
                "explicit_particle_loo_score": explicit,
                "absolute_difference": abs(fast - explicit),
                "particle_phase_averaged_unit_self": mean_self,
            }
        )

    archived_rows = []
    for job, alpha in (("landau_lscv_static_5008392", 0.01), ("landau_lscv_static_5008393", 0.5)):
        for seed in range(10):
            path = RECOVERED / job / f"seed{seed}" / "landau_lscv_seed_values.csv"
            for row in read_csv(path):
                archived_rows.append(
                    {
                        "job": job.rsplit("_", 1)[-1],
                        "alpha": alpha,
                        "seed": int(row["seed"]),
                        "sigma_dx": float(row["sigma_dx"]),
                        "lscv_score": float(row["lscv_time_avg"]),
                        "mean_particle_phase_self": float(row["mean_self_time_avg"]),
                        "n_times": int(row["n_times"]),
                    }
                )
    write_csv(OUTPUT / "lscv_self_terms.csv", archived_rows)
    return {
        "controlled_explicit_comparison": rows,
        "archived_self_term_rows": len(archived_rows),
        "archived_output": "lscv_self_terms.csv",
        "formula": (
            "CV=int rho_H^2 - 2*N/(N-1)*(int rho_H*rho_CIC "
            "- mean_self/N); mean_self is evaluated from each snapshot's "
            "actual bilinear CIC weights"
        ),
    }


def row_at(rows: Iterable[dict], **match) -> dict:
    selected = [
        row
        for row in rows
        if all(str(row[key]) == str(value) for key, value in match.items())
    ]
    if len(selected) != 1:
        raise AssertionError(f"expected one row for {match}, found {len(selected)}")
    return selected[0]


def main() -> dict:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    gaussian = regenerate_gaussian()
    filters = regenerate_filters()
    e2 = regenerate_e2_density_field()
    normalization = normalization_and_parseval_checks()
    loo = loo_checks()

    table4 = row_at(gaussian, campaign="table4_per_cell_4867515", sigma_dx=1.0)
    aligned = row_at(gaussian, campaign="aligned_global_4868286", sigma_dx=1.0)
    raw_n2 = row_at(
        filters,
        campaign="raw_binomial_landau_5048431",
        passes=2,
    )
    np.testing.assert_allclose(table4["noise_power_reduction"], 2.559393052975873)
    np.testing.assert_allclose(aligned["noise_power_reduction"], 5.643223264073615)
    np.testing.assert_allclose(raw_n2["noise_power_reduction"], 5.8715086641784495)

    anchors = [
        {
            "source": "Table 4 selector-matrix reducer",
            "job": "4867515",
            "filter": "Gaussian sigma=1 dx",
            "quiet_start_layout": "randomized per-cell velocity quadrature",
            "noise_power_reduction": table4["noise_power_reduction"],
            "ci_lo": table4["noise_power_reduction_ci_lo"],
            "ci_hi": table4["noise_power_reduction_ci_hi"],
            "grid_density_mse_cic": row_at(
                gaussian, campaign="table4_per_cell_4867515", sigma_dx=0.0
            )["grid_density_mse_mean"],
            "grid_density_mse_filtered": table4["grid_density_mse_mean"],
        },
        {
            "source": "Fig. 7a filter-family figure Gaussian overlay",
            "job": "4868286",
            "filter": "Gaussian sigma=1 dx",
            "quiet_start_layout": "globally shuffled velocity quadrature",
            "noise_power_reduction": aligned["noise_power_reduction"],
            "ci_lo": aligned["noise_power_reduction_ci_lo"],
            "ci_hi": aligned["noise_power_reduction_ci_hi"],
            "grid_density_mse_cic": row_at(
                gaussian, campaign="aligned_global_4868286", sigma_dx=0.0
            )["grid_density_mse_mean"],
            "grid_density_mse_filtered": aligned["grid_density_mse_mean"],
        },
        {
            "source": "Fig. 7a raw-binomial reducer (matched low-k width)",
            "job": "5048431",
            "filter": "raw binomial n=2, sigma_eq=1 dx",
            "quiet_start_layout": "globally shuffled velocity quadrature",
            "noise_power_reduction": raw_n2["noise_power_reduction"],
            "ci_lo": raw_n2["noise_power_reduction_ci_lo"],
            "ci_hi": raw_n2["noise_power_reduction_ci_hi"],
            "grid_density_mse_cic": row_at(
                filters, campaign="raw_binomial_landau_5048431", passes=0
            )["grid_density_mse_mean"],
            "grid_density_mse_filtered": raw_n2["grid_density_mse_mean"],
        },
    ]
    write_csv(OUTPUT / "weak_landau_gaussian_noise.csv", gaussian)
    write_csv(OUTPUT / "filter_family_noise.csv", filters)
    write_csv(OUTPUT / "e2_grid_density_field_noise.csv", e2)
    write_csv(OUTPUT / "weak_landau_sigma1_comparison.csv", anchors)

    audit = {
        "status": "PASS",
        "archive_reproduction": {
            "gaussian_rows": len(gaussian),
            "filter_family_rows": len(filters),
            "e2_density_field_rows": len(e2),
            "all_archived_central_values_and_bootstrap_intervals": "matched",
        },
        "weak_landau_sigma1_comparison": anchors,
        "metric_definition": {
            "stored_per_seed_quantity": (
                "time average of the realized grid mean squared charge-density "
                "error mean_g[(rho_candidate-rho_reference)^2]"
            ),
            "ensemble_quantity": (
                "mean across ten independent seeds of the per-seed time-averaged ISE; "
                "called density MISE only after this ensemble average"
            ),
            "noise_reduction": (
                "ratio of CIC error power to filtered error power (ratio of ensemble "
                "means); it is not an amplitude ratio"
            ),
            "fourier_scope": normalization["parseval"]["scope"],
            "cic_baseline": (
                "low-PPC unfiltered CIC snapshot density compared with the paired "
                "high-PPC unfiltered CIC snapshot density; both trajectories use "
                "the live matched Gaussian sigma=0.25-cell coupling"
            ),
            "reference": (
                "paired high-PPC CIC density (16384 PPC for Landau, 1024 PPC for E2), "
                "not the filtered candidate and not an analytic density"
            ),
            "selection_independence": (
                "the MISE width and its reported noise reduction use the same ten "
                "seeds, so this is in-sample estimator evaluation rather than a "
                "held-out validation; the reference itself does not contain the candidate"
            ),
            "field_metric": (
                "E2 field MSE is grid-node mean[(Ex-Ex_ref)^2+(Ey-Ey_ref)^2] "
                "after one density filter and the Poisson solve"
            ),
            "particle_force_warning": (
                "the field MSE does not apply the return-path filter and does not gather "
                "to particles; production particle force uses the full C-W-A-W-C^T chain"
            ),
        },
        "normalization_and_parseval_checks": normalization,
        "loo_self_contribution_check": loo,
        "difference_explaining_2p559_vs_5p643": (
            "the particle velocity-quadrature layout changed from randomized per-cell "
            "to globally shuffled; metric, Fourier scope, PPC, high-PPC reference, "
            "time window, and 51-snapshot cadence are otherwise the same"
        ),
        "figure_source_correction": (
            "Fig. 7a obtains Gaussian 5.643 from job 4868286 via LANDAU_NOISE; "
            "the raw-binomial reducer's n=2 matched-width value is 5.872, not 5.643"
        ),
    }
    (OUTPUT / "noise_metric_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, indent=2, sort_keys=True))
    return audit


if __name__ == "__main__":
    main()
