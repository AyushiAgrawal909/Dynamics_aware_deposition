#!/usr/bin/env python3
"""Reduce the paired ``(12,1)`` joint-width rate and noise campaign."""

from __future__ import annotations

import argparse
import csv
import json
import math
import pathlib
import sys

import numpy as np


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pic_kde_2d.benchmarks.theory import warm_two_stream_oblique_growth
from pic_kde_2d.core.grid2d import Grid2D
from pic_kde_2d.methods.spectral_W import make_W_gaussian
from pic_kde_2d.validation.landau_fit import fit_gamma_ensemble_fixed_window


def load_rate_group(path: pathlib.Path, seeds: int) -> dict:
    """Load one complete paired rate group."""
    histories = []
    validity = []
    seed_ids = []
    metadata = None
    times = None
    for seed_path in sorted(path.glob("seed*")):
        result_path = seed_path / "result.json"
        history_path = seed_path / "mode_history.npz"
        if not result_path.exists() or not history_path.exists():
            continue
        this_metadata = json.loads(result_path.read_text(encoding="utf-8"))
        if metadata is None:
            metadata = this_metadata
        else:
            for key in ("sigma_x_dx", "sigma_y_dy", "kx_mode", "ky_mode"):
                if this_metadata[key] != metadata[key]:
                    raise RuntimeError(f"{path}: mixed metadata for {key}")
        with np.load(history_path) as data:
            this_times = np.asarray(data["times"], dtype=np.float64)
            histories.append(np.abs(np.asarray(data["mode"], dtype=np.complex128)))
            validity.append(np.asarray(data["delta_f_ratio"], dtype=np.float64))
            seed_ids.append(int(np.asarray(data["seed"]).reshape(-1)[0]))
        if times is None:
            times = this_times
        else:
            np.testing.assert_allclose(times, this_times, rtol=0.0, atol=0.0)
    if len(histories) != seeds or sorted(seed_ids) != list(range(seeds)):
        raise RuntimeError(f"{path}: incomplete paired rate ensemble")
    assert metadata is not None and times is not None
    order = np.argsort(seed_ids)
    return {
        "label": path.name,
        "sigma_x_dx": float(metadata["sigma_x_dx"]),
        "sigma_y_dy": float(metadata["sigma_y_dy"]),
        "times": times,
        "amplitudes": np.stack(histories)[order],
        "validity": np.stack(validity)[order],
    }


def fit_rate(
    amplitudes: np.ndarray,
    times: np.ndarray,
    t_start: float,
    t_stop: float,
) -> float:
    fit = fit_gamma_ensemble_fixed_window(
        [row for row in amplitudes],
        times,
        t_fit_start=t_start,
        t_fit_stop=t_stop,
        amp_floor=1.0e-30,
    )
    return float(fit["gamma"])


def fit_quality(
    amplitudes: np.ndarray,
    times: np.ndarray,
    t_start: float,
    t_stop: float,
) -> tuple[float, float]:
    mask = (times >= t_start) & (times <= t_stop)
    x = times[mask]
    y = np.log(np.maximum(np.mean(amplitudes, axis=0)[mask], 1.0e-300))
    slope, intercept = np.polyfit(x, y, 1)
    residual = float(np.sum((y - (slope * x + intercept)) ** 2))
    total = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - residual / total if total > 0.0 else 0.0
    return r2, float(slope * (x[-1] - x[0]))


def paired_rate_bootstrap(
    groups: list[dict],
    t_start: float,
    t_stop: float,
    nboot: int,
) -> list[dict]:
    """Bootstrap all configurations with common seed indices."""
    rng = np.random.default_rng(59120)
    n_seeds = groups[0]["amplitudes"].shape[0]
    rates = [[] for _ in groups]
    ratios = [[] for _ in groups]
    for _ in range(nboot):
        indices = rng.integers(0, n_seeds, size=n_seeds)
        boot_rates = [
            fit_rate(
                group["amplitudes"][indices],
                group["times"],
                t_start,
                t_stop,
            )
            for group in groups
        ]
        baseline = boot_rates[0]
        if baseline <= 0.0 or not all(np.isfinite(boot_rates)):
            continue
        for index, rate in enumerate(boot_rates):
            rates[index].append(rate)
            ratios[index].append(rate / baseline)
    output = []
    for rate_values, ratio_values in zip(rates, ratios):
        rate_array = np.asarray(rate_values)
        ratio_array = np.asarray(ratio_values)
        output.append(
            {
                "gamma_sem": float(np.std(rate_array)),
                "gamma_ci_lo": float(np.percentile(rate_array, 2.5)),
                "gamma_ci_hi": float(np.percentile(rate_array, 97.5)),
                "ratio_sem": float(np.std(ratio_array)),
                "ratio_ci_lo": float(np.percentile(ratio_array, 2.5)),
                "ratio_ci_hi": float(np.percentile(ratio_array, 97.5)),
                "bootstrap_valid": int(ratio_array.size),
            }
        )
    return output


def load_noise(noise_root: pathlib.Path, seeds: int) -> dict[str, dict]:
    """Load the complete per-seed density and field metrics."""
    by_label: dict[str, dict[str, list]] = {}
    seen_seeds = set()
    for path in sorted(noise_root.glob("seed*/joint_noise_seed.csv")):
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        seed = int(rows[0]["seed"])
        seen_seeds.add(seed)
        for row in rows:
            label = row["label"]
            entry = by_label.setdefault(
                label,
                {
                    "sigma_x_dx": float(row["sigma_x_dx"]),
                    "sigma_y_dy": float(row["sigma_y_dy"]),
                    "density_mse": [],
                    "field_mse": [],
                    "neutrality": [],
                    "charge_error": [],
                },
            )
            entry["density_mse"].append(float(row["density_mse"]))
            entry["field_mse"].append(float(row["field_mse"]))
            entry["neutrality"].append(float(row["max_neutrality_defect"]))
            entry["charge_error"].append(float(row["max_relative_charge_error"]))
    if seen_seeds != set(range(seeds)):
        raise RuntimeError(f"incomplete noise seeds: {sorted(seen_seeds)}")
    for entry in by_label.values():
        for key in ("density_mse", "field_mse", "neutrality", "charge_error"):
            entry[key] = np.asarray(entry[key], dtype=np.float64)
    return by_label


def paired_noise_bootstrap(
    baseline: np.ndarray,
    candidate: np.ndarray,
    nboot: int,
    seed: int,
) -> tuple[float, float, float]:
    """Return ratio-of-means and its paired seed-bootstrap interval."""
    central = float(np.mean(baseline) / np.mean(candidate))
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(nboot):
        indices = rng.integers(0, baseline.size, size=baseline.size)
        values.append(
            float(np.mean(baseline[indices]) / np.mean(candidate[indices]))
        )
    array = np.asarray(values)
    return (
        central,
        float(np.percentile(array, 2.5)),
        float(np.percentile(array, 97.5)),
    )


def white_noise_proxy(
    grid: Grid2D,
    sigma_x_dx: float,
    sigma_y_dy: float,
) -> float:
    """Resolved-grid white-density-noise reduction from one Gaussian filter."""
    W = make_W_gaussian(
        grid,
        sigma_x_dx * grid.dx,
        sigma_y_dy * grid.dy,
    )
    weights = np.full(W.shape[1], 2.0)
    weights[0] = 1.0
    if grid.Ny % 2 == 0:
        weights[-1] = 1.0
    weighted_power = float(np.sum(W**2 * weights[np.newaxis, :]))
    n_modes = float(grid.Nx * grid.Ny)
    return n_modes / weighted_power


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rate-root", type=pathlib.Path, required=True)
    parser.add_argument("--noise-root", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--nboot", type=int, default=5000)
    parser.add_argument("--fit-t-start", type=float, default=24.32)
    parser.add_argument("--fit-t-stop", type=float, default=59.68)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    groups = [
        load_rate_group(path, args.seeds)
        for path in sorted(args.rate_root.iterdir())
        if path.is_dir()
    ]
    groups.sort(
        key=lambda group: (
            group["sigma_x_dx"] != 0.0 or group["sigma_y_dy"] != 0.0,
            group["sigma_x_dx"],
            group["sigma_y_dy"],
        )
    )
    if not groups or (groups[0]["sigma_x_dx"], groups[0]["sigma_y_dy"]) != (0.0, 0.0):
        raise RuntimeError("the first sorted rate group must be the CIC baseline")
    for group in groups[1:]:
        np.testing.assert_allclose(
            group["times"], groups[0]["times"], rtol=0.0, atol=0.0
        )
    rate_uncertainty = paired_rate_bootstrap(
        groups, args.fit_t_start, args.fit_t_stop, args.nboot
    )
    central_rates = [
        fit_rate(
            group["amplitudes"],
            group["times"],
            args.fit_t_start,
            args.fit_t_stop,
        )
        for group in groups
    ]
    baseline_rate = central_rates[0]
    noise = load_noise(args.noise_root, args.seeds)
    baseline_noise = noise[groups[0]["label"]]

    grid = Grid2D(Nx=128, Ny=16, Lx=64.0, Ly=16.0)
    kx = 2.0 * math.pi * 12.0 / 64.0
    ky = 2.0 * math.pi / 16.0
    cic = float(np.sinc(12.0 / 128.0) ** 4 * np.sinc(1.0 / 16.0) ** 4)
    gamma_warm_baseline = warm_two_stream_oblique_growth(
        kx, ky, 0.8, 0.05, cic
    )

    rows = []
    for index, (group, gamma, uncertainty) in enumerate(
        zip(groups, central_rates, rate_uncertainty)
    ):
        label = group["label"]
        if label not in noise:
            raise RuntimeError(f"missing noise configuration {label}")
        r2, efoldings = fit_quality(
            group["amplitudes"],
            group["times"],
            args.fit_t_start,
            args.fit_t_stop,
        )
        fit_mask = (
            (group["times"] >= args.fit_t_start)
            & (group["times"] <= args.fit_t_stop)
        )
        max_validity_fit = float(
            np.max(group["validity"][:, fit_mask])
        )
        density = paired_noise_bootstrap(
            baseline_noise["density_mse"],
            noise[label]["density_mse"],
            args.nboot,
            seed=59200 + index,
        )
        field = paired_noise_bootstrap(
            baseline_noise["field_mse"],
            noise[label]["field_mse"],
            args.nboot,
            seed=59300 + index,
        )
        sigma_x = group["sigma_x_dx"]
        sigma_y = group["sigma_y_dy"]
        qx = (2.0 * math.pi * 12.0 * sigma_x / 128.0) ** 2
        qy = (2.0 * math.pi * sigma_y / 16.0) ** 2
        gamma_warm = warm_two_stream_oblique_growth(
            kx, ky, 0.8, 0.05, cic * math.exp(-(qx + qy))
        )
        ratio = gamma / baseline_rate
        rate_pass = bool(
            uncertainty["ratio_ci_lo"] >= 0.90
            and uncertainty["ratio_ci_hi"] <= 1.10
            and r2 >= 0.98
            and efoldings >= 2.0
            and max_validity_fit < 0.5
        )
        rows.append(
            {
                "label": label,
                "sigma_x_dx": sigma_x,
                "sigma_y_dy": sigma_y,
                "qx": qx,
                "qy": qy,
                "q_total": qx + qy,
                "gamma": gamma,
                "gamma_ratio": ratio,
                **uncertainty,
                "fit_r2": r2,
                "fit_efoldings": efoldings,
                "max_delta_f_ratio_fit_window": max_validity_fit,
                "warm_ratio": gamma_warm / gamma_warm_baseline,
                "rate_gate_pass": rate_pass,
                "white_noise_proxy": white_noise_proxy(grid, sigma_x, sigma_y),
                "density_noise_reduction": density[0],
                "density_noise_ci_lo": density[1],
                "density_noise_ci_hi": density[2],
                "field_noise_reduction": field[0],
                "field_noise_ci_lo": field[1],
                "field_noise_ci_hi": field[2],
                "density_noise_gate_pass": density[1] > 1.0,
                "field_noise_gate_pass": field[1] > 1.0,
                "max_neutrality_defect": float(np.max(noise[label]["neutrality"])),
                "max_relative_charge_error": float(
                    np.max(noise[label]["charge_error"])
                ),
                "n_seeds": args.seeds,
            }
        )

    with (args.output / "joint_corner_sweep.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    feasible = [row for row in rows if row["rate_gate_pass"]]
    selected_density = (
        max(feasible, key=lambda row: row["density_noise_reduction"])
        if feasible
        else None
    )
    selected_field = (
        max(feasible, key=lambda row: row["field_noise_reduction"])
        if feasible
        else None
    )
    pure_y_rows = sorted(
        [row for row in rows if row["sigma_x_dx"] == 0.0],
        key=lambda row: row["sigma_y_dy"],
    )
    pure_y_monotone = bool(
        all(
            left["gamma_ratio"] >= right["gamma_ratio"]
            for left, right in zip(pure_y_rows[:-1], pure_y_rows[1:])
        )
    )
    result = {
        "benchmark": "E2 coherently seeded binding corner (12,1)",
        "paired_common_random_numbers": True,
        "fit_window": [args.fit_t_start, args.fit_t_stop],
        "n_seeds": args.seeds,
        "rate_tolerance": 0.10,
        "pure_y_rate_monotone": pure_y_monotone,
        "selected_density_metric": selected_density,
        "selected_field_metric": selected_field,
        "density_and_field_select_same_point": bool(
            selected_density is not None
            and selected_field is not None
            and selected_density["label"] == selected_field["label"]
        ),
        "rows": rows,
        "status": "pass" if selected_density is not None else "no_feasible_width",
    }
    (args.output / "joint_corner_sweep.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
