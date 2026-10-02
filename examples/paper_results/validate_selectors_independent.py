#!/usr/bin/env python3
"""Held-out selector comparison using the recovered job-5049182 snapshots.

The fixed split is declared here rather than chosen from the results: seeds
0--4 select a width and seeds 5--9 evaluate it.  MISE and LOO-CV receive the
same 22-member isotropic Gaussian candidate grid and the same five training
realisations.  Fixed/model rules do not consume a snapshot-search budget.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
from pathlib import Path

import numpy as np


WORKSPACE = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = (
    WORKSPACE
    / "pic_kde_2d/results_raw_rate_audit/recovered_meluxina_20260909"
    / "landau_ppc_dynamic_selectors_5049182"
)
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "results/selector_independent_audit"
TRAIN_SEEDS = np.arange(0, 5, dtype=np.int64)
TEST_SEEDS = np.arange(5, 10, dtype=np.int64)
SILVERMAN_X_DX = 1.8329728493314705
SILVERMAN_Y_DY = 0.45824321233286763
C4_RATE_CI_WIDTH = 1.0
C4_PRINCIPAL_CEILING = 2.4534381425710894


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"refusing to write an empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_snapshots(
    root: Path, ppc: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict]]:
    densities: list[np.ndarray] = []
    scores: list[np.ndarray] = []
    metadata: list[dict] = []
    times: np.ndarray | None = None
    sigmas: np.ndarray | None = None
    for seed in range(10):
        seed_root = root / f"ppc{ppc}" / f"seed{seed}"
        with np.load(seed_root / "selector_snapshots.npz", allow_pickle=False) as archive:
            current_times = np.asarray(archive["times"], dtype=np.float64)
            current_sigmas = np.asarray(archive["sigmas"], dtype=np.float64)
            densities.append(np.asarray(archive["probability_density"], dtype=np.float64))
            scores.append(np.asarray(archive["lscv_scores"], dtype=np.float64))
        current_metadata = json.loads((seed_root / "metadata.json").read_text(encoding="utf-8"))
        if int(current_metadata["seed"]) != seed or int(current_metadata["ppc"]) != ppc:
            raise AssertionError(f"metadata mismatch in {seed_root}")
        metadata.append(current_metadata)
        if times is None:
            times, sigmas = current_times, current_sigmas
        else:
            np.testing.assert_array_equal(times, current_times)
            np.testing.assert_array_equal(sigmas, current_sigmas)
    assert times is not None and sigmas is not None
    return times, sigmas, np.stack(densities), np.stack(scores), metadata


def gaussian_transfer(
    nx: int,
    ny: int,
    dx: float,
    dy: float,
    sigma_x_dx: float,
    sigma_y_dy: float,
) -> np.ndarray:
    kx = 2.0 * np.pi * np.fft.fftfreq(nx, d=dx)
    ky = 2.0 * np.pi * np.fft.rfftfreq(ny, d=dy)
    return np.exp(
        -0.5
        * (
            (sigma_x_dx * dx * kx[:, None]) ** 2
            + (sigma_y_dy * dy * ky[None, :]) ** 2
        )
    )


def filter_density(density: np.ndarray, transfer: np.ndarray) -> np.ndarray:
    spectrum = np.fft.rfft2(density, axes=(-2, -1))
    return np.fft.irfft2(spectrum * transfer, s=density.shape[-2:], axes=(-2, -1)).real


def per_seed_error(
    low: np.ndarray,
    reference: np.ndarray,
    transfer: np.ndarray,
    cell_area: float,
) -> np.ndarray:
    candidate = filter_density(low, transfer)
    return np.mean(
        np.sum((candidate - reference) ** 2, axis=(-2, -1)) * cell_area,
        axis=1,
    )


def ratio_interval(
    baseline: np.ndarray,
    candidate: np.ndarray,
    nboot: int,
    seed: int,
) -> tuple[float, float, float]:
    if baseline.shape != candidate.shape:
        raise ValueError("paired test vectors do not have the same shape")
    central = float(np.mean(baseline) / np.mean(candidate))
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, baseline.size, size=(nboot, baseline.size))
    samples = np.mean(baseline[indices], axis=1) / np.mean(candidate[indices], axis=1)
    return central, float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))


def bootstrap_selected(
    train_objective: np.ndarray,
    test_errors: np.ndarray,
    test_baseline: np.ndarray,
    sigmas: np.ndarray,
    nboot: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Bootstrap the independent training and test seed sets separately."""
    rng = np.random.default_rng(seed)
    widths = np.empty(nboot, dtype=np.float64)
    ratios = np.empty(nboot, dtype=np.float64)
    for draw in range(nboot):
        train_indices = rng.integers(0, train_objective.shape[0], size=train_objective.shape[0])
        test_indices = rng.integers(0, test_errors.shape[0], size=test_errors.shape[0])
        width_index = int(np.argmin(np.mean(train_objective[train_indices], axis=0)))
        widths[draw] = sigmas[width_index]
        ratios[draw] = float(
            np.mean(test_baseline[test_indices])
            / np.mean(test_errors[test_indices, width_index])
        )
    return widths, ratios


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--nboot", type=int, default=5000)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    args.output.mkdir(parents=True)

    times, sigmas, low, low_lscv, low_metadata = load_snapshots(args.input, 1024)
    ref_times, ref_sigmas, reference, _, ref_metadata = load_snapshots(args.input, 4096)
    np.testing.assert_array_equal(times, ref_times)
    np.testing.assert_array_equal(sigmas, ref_sigmas)
    expected_sigmas = np.asarray(
        [0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 2, 2.5, 3, 4, 5, 6, 7, 8, 10, 12, 16, 20, 30, 45, 60],
        dtype=np.float64,
    )
    np.testing.assert_array_equal(sigmas, expected_sigmas)
    if set(TRAIN_SEEDS).intersection(set(TEST_SEEDS)) or set(np.r_[TRAIN_SEEDS, TEST_SEEDS]) != set(range(10)):
        raise AssertionError("the fixed train/test split is not a disjoint partition")
    for metadata in low_metadata + ref_metadata:
        if metadata["initializer"] != "regular positions, globally shuffled velocity quadrature":
            raise AssertionError("unexpected particle-loading protocol")
        if metadata["eval_window"] != [5.0, 30.0] or int(metadata["n_snapshots"]) != 51:
            raise AssertionError("unexpected selector evaluation window")

    nx, ny = low.shape[-2:]
    lx = 2.0 * math.pi / 0.4
    ly = 8.0
    dx, dy = lx / nx, ly / ny
    cell_area = dx * dy
    grid_errors = np.empty((10, len(sigmas)), dtype=np.float64)
    for index, sigma in enumerate(sigmas):
        transfer = gaussian_transfer(nx, ny, dx, dy, float(sigma), float(sigma))
        grid_errors[:, index] = per_seed_error(low, reference, transfer, cell_area)
    lscv = np.mean(low_lscv, axis=1)
    if lscv.shape != grid_errors.shape:
        raise AssertionError("MISE and LOO-CV candidate matrices differ in shape")

    train_mise = grid_errors[TRAIN_SEEDS]
    train_lscv = lscv[TRAIN_SEEDS]
    test_grid_errors = grid_errors[TEST_SEEDS]
    test_baseline = test_grid_errors[:, 0]
    mise_index = int(np.argmin(np.mean(train_mise, axis=0)))
    lscv_index = int(np.argmin(np.mean(train_lscv, axis=0)))
    hindsight_index = int(np.argmin(np.mean(test_grid_errors, axis=0)))
    largest_c4_grid_index = int(np.flatnonzero(sigmas <= C4_PRINCIPAL_CEILING)[-1])

    selected_bootstrap: dict[str, tuple[np.ndarray, np.ndarray]] = {
        "MISE held-out": bootstrap_selected(
            train_mise,
            test_grid_errors,
            test_baseline,
            sigmas,
            args.nboot,
            19001,
        ),
        "LOO-CV held-out": bootstrap_selected(
            train_lscv,
            test_grid_errors,
            test_baseline,
            sigmas,
            args.nboot,
            19002,
        ),
    }

    method_specs = [
        {
            "method": "CIC",
            "sigma_x_dx": 0.0,
            "sigma_y_dy": 0.0,
            "selection": "fixed baseline",
            "candidate_budget": 0,
            "grid_index": 0,
        },
        {
            "method": "C4 rate-CI operating point",
            "sigma_x_dx": C4_RATE_CI_WIDTH,
            "sigma_y_dy": C4_RATE_CI_WIDTH,
            "selection": "pre-existing paired-rate certification",
            "candidate_budget": 0,
            "grid_index": int(np.where(sigmas == C4_RATE_CI_WIDTH)[0][0]),
        },
        {
            "method": "C4 model, common grid",
            "sigma_x_dx": float(sigmas[largest_c4_grid_index]),
            "sigma_y_dy": float(sigmas[largest_c4_grid_index]),
            "selection": "largest common-grid width below principal-CIC 10% ceiling",
            "candidate_budget": len(sigmas),
            "grid_index": largest_c4_grid_index,
        },
        {
            "method": "Silverman isotropized x-width",
            "sigma_x_dx": SILVERMAN_X_DX,
            "sigma_y_dy": SILVERMAN_X_DX,
            "selection": "actual scalar width used by the weak-Landau manuscript campaigns",
            "candidate_budget": 0,
            "grid_index": None,
        },
        {
            "method": "Silverman per direction",
            "sigma_x_dx": SILVERMAN_X_DX,
            "sigma_y_dy": SILVERMAN_Y_DY,
            "selection": "formula h_j=(L_j/sqrt(12))*N^(-1/6)",
            "candidate_budget": 0,
            "grid_index": None,
        },
        {
            "method": "MISE held-out",
            "sigma_x_dx": float(sigmas[mise_index]),
            "sigma_y_dy": float(sigmas[mise_index]),
            "selection": "minimum training MISE on common isotropic grid",
            "candidate_budget": len(sigmas),
            "grid_index": mise_index,
        },
        {
            "method": "LOO-CV held-out",
            "sigma_x_dx": float(sigmas[lscv_index]),
            "sigma_y_dy": float(sigmas[lscv_index]),
            "selection": "minimum training LOO-CV on common isotropic grid",
            "candidate_budget": len(sigmas),
            "grid_index": lscv_index,
        },
        {
            "method": "test-set hindsight optimum",
            "sigma_x_dx": float(sigmas[hindsight_index]),
            "sigma_y_dy": float(sigmas[hindsight_index]),
            "selection": "diagnostic only; not an eligible selector",
            "candidate_budget": len(sigmas),
            "grid_index": hindsight_index,
        },
    ]

    hindsight_error = float(np.mean(test_grid_errors[:, hindsight_index]))
    rows: list[dict] = []
    for method_index, spec in enumerate(method_specs):
        if spec["grid_index"] is None:
            transfer = gaussian_transfer(
                nx,
                ny,
                dx,
                dy,
                float(spec["sigma_x_dx"]),
                float(spec["sigma_y_dy"]),
            )
            errors = per_seed_error(low[TEST_SEEDS], reference[TEST_SEEDS], transfer, cell_area)
        else:
            errors = test_grid_errors[:, int(spec["grid_index"])]
        ratio, ci_lo, ci_hi = ratio_interval(
            test_baseline,
            errors,
            args.nboot,
            19100 + method_index,
        )
        width_ci_lo = width_ci_hi = float(spec["sigma_x_dx"])
        nested_ci_lo, nested_ci_hi = ci_lo, ci_hi
        selection_stability = 1.0
        if spec["method"] in selected_bootstrap:
            boot_widths, boot_ratios = selected_bootstrap[spec["method"]]
            width_ci_lo = float(np.percentile(boot_widths, 2.5))
            width_ci_hi = float(np.percentile(boot_widths, 97.5))
            nested_ci_lo = float(np.percentile(boot_ratios, 2.5))
            nested_ci_hi = float(np.percentile(boot_ratios, 97.5))
            selection_stability = float(np.mean(boot_widths == spec["sigma_x_dx"]))
        rows.append(
            {
                **{key: value for key, value in spec.items() if key != "grid_index"},
                "train_seeds": "0,1,2,3,4" if "held-out" in spec["method"] else "not used",
                "test_seeds": "5,6,7,8,9",
                "train_width_ci_lo": width_ci_lo,
                "train_width_ci_hi": width_ci_hi,
                "train_selection_stability": selection_stability,
                "test_density_error_mean": float(np.mean(errors)),
                "test_noise_power_reduction": ratio,
                "test_ratio_ci_lo": ci_lo,
                "test_ratio_ci_hi": ci_hi,
                "nested_train_test_ratio_ci_lo": nested_ci_lo,
                "nested_train_test_ratio_ci_hi": nested_ci_hi,
                "test_error_regret_vs_hindsight": float(np.mean(errors) / hindsight_error - 1.0),
                "n_train_seeds": 5,
                "n_test_seeds": 5,
                "n_snapshots_per_seed": len(times),
            }
        )

    split_rows: list[dict] = []
    for selector, objective in (("MISE", grid_errors), ("LOO-CV", lscv)):
        for train_tuple in itertools.combinations(range(10), 5):
            train = np.asarray(train_tuple, dtype=np.int64)
            test = np.asarray(sorted(set(range(10)) - set(train_tuple)), dtype=np.int64)
            selected_index = int(np.argmin(np.mean(objective[train], axis=0)))
            test_best = int(np.argmin(np.mean(grid_errors[test], axis=0)))
            split_rows.append(
                {
                    "selector": selector,
                    "train_seeds": ",".join(str(value) for value in train),
                    "test_seeds": ",".join(str(value) for value in test),
                    "selected_sigma_dx": float(sigmas[selected_index]),
                    "test_hindsight_sigma_dx": float(sigmas[test_best]),
                    "test_noise_power_reduction": float(
                        np.mean(grid_errors[test, 0]) / np.mean(grid_errors[test, selected_index])
                    ),
                    "test_error_regret": float(
                        np.mean(grid_errors[test, selected_index])
                        / np.mean(grid_errors[test, test_best])
                        - 1.0
                    ),
                }
            )

    write_csv(args.output / "heldout_selector_comparison.csv", rows)
    write_csv(args.output / "split_sensitivity.csv", split_rows)
    result = {
        "status": "PASS",
        "source_job": 5049182,
        "source_snapshot_schema": ["times", "sigmas", "probability_density", "lscv_scores"],
        "fixed_split": {"train": TRAIN_SEEDS.tolist(), "test": TEST_SEEDS.tolist()},
        "metric": "paired time-averaged integrated squared probability-density error against PPC-4096; reduction is CIC error power divided by candidate error power",
        "candidate_family": "MISE and LOO-CV use the identical 22-width isotropic Gaussian grid",
        "training_budget_each_data_selector": {
            "independent_seeds": 5,
            "snapshots_per_seed": len(times),
            "candidate_widths": len(sigmas),
            "width_snapshot_evaluations": int(5 * len(times) * len(sigmas)),
        },
        "silverman_protocol_warning": (
            "the weak-Landau manuscript campaigns apply h_x/dx=1.8329728493 isotropically in cell units; "
            "the stated per-direction formula instead gives h_y/dy=0.4582432123"
        ),
        "rows": rows,
        "split_sensitivity": {
            selector: {
                "n_balanced_splits": sum(row["selector"] == selector for row in split_rows),
                "selected_widths": sorted(
                    {row["selected_sigma_dx"] for row in split_rows if row["selector"] == selector}
                ),
                "median_test_regret": float(
                    np.median([row["test_error_regret"] for row in split_rows if row["selector"] == selector])
                ),
                "max_test_regret": float(
                    np.max([row["test_error_regret"] for row in split_rows if row["selector"] == selector])
                ),
            }
            for selector in ("MISE", "LOO-CV")
        },
        "claims": {
            "density_optimality": "MISE is only optimal on the tested 22-width isotropic Gaussian grid for this held-out metric",
            "dynamics_optimality": "not established by this density-only held-out audit",
            "global_or_anisotropic_optimality": "not established; requires the separate dense anisotropic campaign",
        },
    }
    (args.output / "selector_independent_audit.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
