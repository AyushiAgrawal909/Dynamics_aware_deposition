#!/usr/bin/env python3
"""Recompute Paper 1 conservation diagnostics from archived raw outputs.

This reducer deliberately keeps unlike quantities separate:

* charge error is the change in represented particle charge, normalized by the
  nonzero initial electron charge;
* energy error is reported both relative to initial total energy and to each
  realization's peak field energy;
* temporal samples are never bootstrapped as independent data.  One slope is
  fitted per run and the independent seed-level slopes are bootstrapped;
* the grad-h comparison reports x components and two-dimensional norms side by
  side rather than substituting one for the other.

The input directories are preserved verbatim.  All generated files are placed
under reproducibility/results/conservation_audit.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import pathlib
import re
from collections import defaultdict

import numpy as np


HERE = pathlib.Path(__file__).resolve().parent
DOWNLOADS = HERE.parents[1]
RESULTS = DOWNLOADS / "pic_kde_2d" / "results"
ENERGY_ROOT = RESULTS / "energy_conservation_5048437"
MOMENTUM_ROOT = RESULTS / "hexapic_momentum_audit"
GRAD_H_ROOT = MOMENTUM_ROOT / "grad_h_full_4855533"
PHASE_ROOT = MOMENTUM_ROOT / "grid_phase_full_4858823"
ONE_STEP_PHASE = RESULTS / "grid_phase_one_step_4858774" / "grid_phase_summary.csv"
TOLERANCE_ROOT = MOMENTUM_ROOT / "hypre_tolerance_one_step_5195488"


def bootstrap_mean_ci(values: list[float], rng: np.random.Generator, nboot: int) -> tuple[float, float, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size < 2 or not np.all(np.isfinite(array)):
        raise AssertionError(f"invalid bootstrap sample: {array}")
    samples = array[rng.integers(0, array.size, size=(nboot, array.size))]
    means = np.mean(samples, axis=1)
    return float(np.mean(array)), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def write_csv(path: pathlib.Path, rows: list[dict]) -> None:
    if not rows:
        raise AssertionError(f"refusing to write an empty table: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: pathlib.Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def analyze_energy(nboot: int) -> tuple[list[dict], list[dict], dict, dict]:
    archives = sorted(ENERGY_ROOT.glob("*/*/seed*/energy_history.npz"))
    if len(archives) != 90:
        raise AssertionError(f"expected 90 energy histories, found {len(archives)}")

    per_seed: list[dict] = []
    traces: dict[tuple[str, str], list[tuple[np.ndarray, np.ndarray, np.ndarray]]] = defaultdict(list)
    for archive in archives:
        benchmark, case, seed_tag = archive.parts[-4:-1]
        seed = int(seed_tag.removeprefix("seed"))
        params_path = archive.with_name("run_parameters.json")
        params = json.loads(params_path.read_text(encoding="utf-8"))
        with np.load(archive) as data:
            times = np.asarray(data["times"], dtype=np.float64)
            field = np.asarray(data["field_energy"], dtype=np.float64)
            kinetic = np.asarray(data["kinetic_energy"], dtype=np.float64)
            total = np.asarray(data["total_energy"], dtype=np.float64)
            charge = np.asarray(data["total_charge"], dtype=np.float64)
        for array in (times, field, kinetic, total, charge):
            if array.ndim != 1 or array.size != times.size or not np.all(np.isfinite(array)):
                raise AssertionError(f"malformed history in {archive}")
        np.testing.assert_allclose(total, field + kinetic, rtol=0.0, atol=2.0e-13)
        if int(params["seed"]) != seed or params["benchmark"] != benchmark or params["case"] != case:
            raise AssertionError(f"path/parameter mismatch in {archive}")

        delta = total - total[0]
        rel_total = delta / abs(float(total[0]))
        peak_field = float(np.max(field))
        rel_field = delta / peak_field
        charge_change = charge - charge[0]
        charge_scale = abs(float(charge[0]))
        if charge_scale == 0.0:
            raise AssertionError(f"zero represented particle charge in {archive}")
        slope = float(np.polyfit(times, rel_total, 1)[0])
        row = {
            "benchmark": benchmark,
            "case": case,
            "seed": seed,
            "ppc": int(params["ppc"]),
            "alpha": float(params["alpha"]),
            "sigma_x_dx": float(params["sigma_x_dx"]),
            "sigma_y_dy": float(params["sigma_y_dy"]),
            "dt": float(params["dt"]),
            "t_end": float(times[-1]),
            "diagnostic_dt": float(times[1] - times[0]),
            "initial_total_energy": float(total[0]),
            "peak_field_energy": peak_field,
            "max_abs_charge_change_over_initial_particle_charge": float(np.max(np.abs(charge_change)) / charge_scale),
            "max_abs_energy_change_over_initial_total": float(np.max(np.abs(rel_total))),
            "final_energy_change_over_initial_total": float(rel_total[-1]),
            "energy_drift_slope_per_time": slope,
            "max_abs_energy_change_over_peak_field": float(np.max(np.abs(rel_field))),
            "final_energy_change_over_peak_field": float(rel_field[-1]),
        }
        # Recompute and verify all quantities used by the archived reducer.
        np.testing.assert_allclose(
            row["max_abs_energy_change_over_initial_total"],
            float(params["max_abs_relative_energy_drift"]),
            rtol=2.0e-13,
            atol=2.0e-16,
        )
        np.testing.assert_allclose(
            row["final_energy_change_over_initial_total"],
            float(params["final_relative_energy_drift"]),
            rtol=2.0e-13,
            atol=2.0e-16,
        )
        np.testing.assert_allclose(
            slope,
            float(params["relative_energy_drift_slope_per_time"]),
            rtol=2.0e-12,
            atol=2.0e-17,
        )
        np.testing.assert_allclose(
            row["max_abs_energy_change_over_peak_field"],
            float(params["max_abs_drift_over_peak_field_energy"]),
            rtol=2.0e-13,
            atol=2.0e-16,
        )
        per_seed.append(row)
        traces[(benchmark, case)].append((times, rel_total, rel_field))

    expected_groups = {(b, c) for b in ("landau_weak", "landau_strong", "e2") for c in ("cic", "c4", "large")}
    if set(traces) != expected_groups:
        raise AssertionError(f"incomplete energy matrix: {sorted(traces)}")

    rng = np.random.default_rng(20260909)
    aggregate: list[dict] = []
    for benchmark, case in sorted(traces):
        rows = sorted(
            (row for row in per_seed if row["benchmark"] == benchmark and row["case"] == case),
            key=lambda row: row["seed"],
        )
        if [row["seed"] for row in rows] != list(range(10)):
            raise AssertionError(f"missing seeds for {benchmark}/{case}")
        maximum = bootstrap_mean_ci([row["max_abs_energy_change_over_initial_total"] for row in rows], rng, nboot)
        final = bootstrap_mean_ci([row["final_energy_change_over_initial_total"] for row in rows], rng, nboot)
        slope = bootstrap_mean_ci([row["energy_drift_slope_per_time"] for row in rows], rng, nboot)
        field_scaled = bootstrap_mean_ci([row["max_abs_energy_change_over_peak_field"] for row in rows], rng, nboot)
        charge = max(row["max_abs_charge_change_over_initial_particle_charge"] for row in rows)
        aggregate.append(
            {
                "benchmark": benchmark,
                "case": case,
                "ppc": rows[0]["ppc"],
                "alpha": rows[0]["alpha"],
                "sigma_x_dx": rows[0]["sigma_x_dx"],
                "sigma_y_dy": rows[0]["sigma_y_dy"],
                "dt": rows[0]["dt"],
                "t_end": rows[0]["t_end"],
                "n_seeds": len(rows),
                "max_charge_change_over_initial_particle_charge": charge,
                "max_abs_dE_over_E0_mean": maximum[0],
                "max_abs_dE_over_E0_ci_lo": maximum[1],
                "max_abs_dE_over_E0_ci_hi": maximum[2],
                "final_dE_over_E0_mean": final[0],
                "final_dE_over_E0_ci_lo": final[1],
                "final_dE_over_E0_ci_hi": final[2],
                "slope_dE_over_E0_per_time_mean": slope[0],
                "slope_dE_over_E0_per_time_ci_lo": slope[1],
                "slope_dE_over_E0_per_time_ci_hi": slope[2],
                "slope_ci_excludes_zero": bool(slope[1] > 0.0 or slope[2] < 0.0),
                "max_abs_dE_over_peak_field_mean": field_scaled[0],
                "max_abs_dE_over_peak_field_ci_lo": field_scaled[1],
                "max_abs_dE_over_peak_field_ci_hi": field_scaled[2],
            }
        )

    trace_summary: dict = {}
    for key, values in traces.items():
        benchmark, case = key
        reference_times = values[0][0]
        for times, _, _ in values[1:]:
            np.testing.assert_array_equal(times, reference_times)
        trace_summary[f"{benchmark}/{case}"] = {
            "times": reference_times,
            "mean_rel_total": np.mean([value[1] for value in values], axis=0),
            "mean_rel_field": np.mean([value[2] for value in values], axis=0),
        }

    metadata = {
        "charge_definition": "max_t |Q(t)-Q(0)| / |Q(0)| for represented electron macro-particle charge",
        "energy_total_definition": "K=0.5*sum_p |w_p|*(vx^2+vy^2); U_E=0.5*sum_grid(Ex^2+Ey^2)*dx*dy; E=K+U_E",
        "slope_definition": "ordinary least-squares slope of (E(t)-E(0))/|E(0)| over every diagnostic time in one run",
        "uncertainty_definition": f"percentile bootstrap of {nboot} resamples of 10 independent seed-level statistics; temporal samples are not resampled",
        "important_limit": "all archived runs use one dt, so these histories do not establish time-step convergence or bounded error as dt->0",
    }
    return per_seed, aggregate, trace_summary, metadata


KEY_VALUE = re.compile(r"([A-Za-z_]+)=([^ ]+)")


def parse_last_prefixed_line(path: pathlib.Path, prefix: str) -> dict[str, float]:
    matches = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.startswith(prefix)]
    if not matches:
        raise AssertionError(f"no {prefix!r} line in {path}")
    values: dict[str, float] = {}
    for key, value in KEY_VALUE.findall(matches[-1]):
        values[key] = float(value)
    return values


def parse_momentum_run(path: pathlib.Path) -> dict[str, float]:
    final: dict[str, float] | None = None
    max_manual_residual = 0.0
    max_iterations = 0
    sum_force_x = 0.0
    sum_force_y = 0.0
    sum_abs_force_x = 0.0
    sum_abs_force_y = 0.0
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("HYPRE_AUDIT step="):
            values = {key: float(value) for key, value in KEY_VALUE.findall(line)}
            max_manual_residual = max(max_manual_residual, abs(values["manual_relative_residual"]))
            max_iterations = max(max_iterations, int(values["iterations"]))
        elif line.startswith("MOMENTUM_AUDIT step="):
            final = {key: float(value) for key, value in KEY_VALUE.findall(line)}
            sum_force_x += final["force_x"]
            sum_force_y += final["force_y"]
            sum_abs_force_x += final["abs_force_x"]
            sum_abs_force_y += final["abs_force_y"]
    if final is None:
        raise AssertionError(f"no momentum lines in {path}")
    return {
        **final,
        "max_manual_relative_residual": max_manual_residual,
        "max_solver_iterations": max_iterations,
        "cumulative_relative_force_x": abs(sum_force_x) / max(sum_abs_force_x, 1.0e-300),
        "cumulative_relative_force_y": abs(sum_force_y) / max(sum_abs_force_y, 1.0e-300),
    }


def parse_charge_run(path: pathlib.Path) -> dict[str, float]:
    max_particle_norm = 0.0
    max_signed_grid_relative = 0.0
    max_abs_weight_relative = 0.0
    count = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("KDE_CHARGE_AUDIT "):
            continue
        values = {key: float(value) for key, value in KEY_VALUE.findall(line)}
        scale = max(abs(values["expected_abs"]), 1.0e-300)
        max_particle_norm = max(max_particle_norm, abs(values["max_particle_norm_error"]))
        max_signed_grid_relative = max(max_signed_grid_relative, abs(values["signed_grid_defect"]) / scale)
        max_abs_weight_relative = max(max_abs_weight_relative, abs(values["abs_weight_defect"]) / scale)
        count += 1
    if count == 0:
        raise AssertionError(f"no charge-audit lines in {path}")
    return {
        "charge_audit_samples": count,
        "max_particle_normalization_error_reparsed": max_particle_norm,
        "max_assembled_signed_defect_over_expected_abs": max_signed_grid_relative,
        "max_abs_weight_defect_over_expected_abs": max_abs_weight_relative,
    }


def analyze_momentum() -> tuple[list[dict], list[dict], dict]:
    source = json.loads((GRAD_H_ROOT / "grad_h_summary.json").read_text(encoding="utf-8"))
    cases = {row["case"]: row for row in source["cases"]}
    required = {"discontinuous_negative_control", "smooth_with_grad_h", "smooth_without_grad_h", "uniform_smooth"}
    if set(cases) != required:
        raise AssertionError(f"unexpected grad-h cases: {set(cases)}")
    on = cases["smooth_with_grad_h"]
    off = cases["smooth_without_grad_h"]
    discontinuous = cases["discontinuous_negative_control"]
    uniform = cases["uniform_smooth"]
    ratios = {
        "grad_h_x_drift_on_over_off": abs(on["momentum_drift_x"] / off["momentum_drift_x"]),
        "grad_h_norm_drift_on_over_off": on["final_momentum_drift_norm"] / off["final_momentum_drift_norm"],
        "discontinuous_over_smooth_off_x_drift": abs(discontinuous["momentum_drift_x"] / off["momentum_drift_x"]),
        "discontinuous_over_smooth_off_norm_drift": discontinuous["final_momentum_drift_norm"] / off["final_momentum_drift_norm"],
        "uniform_over_smooth_off_norm_drift": uniform["final_momentum_drift_norm"] / off["final_momentum_drift_norm"],
    }
    np.testing.assert_allclose(ratios["grad_h_x_drift_on_over_off"], 0.5717812769266952, rtol=1.0e-12)
    np.testing.assert_allclose(ratios["grad_h_norm_drift_on_over_off"], 1.0000136794945573, rtol=1.0e-12)

    grad_rows = []
    for case in sorted(cases):
        row = cases[case]
        charge = parse_charge_run(GRAD_H_ROOT / case / "run.log")
        grad_rows.append(
            {
                "case": case,
                "steps": int(row["step"]) + 1,
                "momentum_drift_x": row["momentum_drift_x"],
                "momentum_drift_y": row["momentum_drift_y"],
                "momentum_drift_norm": row["final_momentum_drift_norm"],
                "integrated_force_x": row["integrated_force_dt_x"],
                "integrated_force_y": row["integrated_force_dt_y"],
                "integrated_force_norm": row["integrated_force_norm"],
                "cumulative_relative_force_x": row["cumulative_relative_force_x"],
                "cumulative_relative_force_y": row["cumulative_relative_force_y"],
                "max_manual_relative_residual": row["max_manual_relative_residual"],
                "max_solver_iterations": int(row["max_solver_iterations"]),
                "max_particle_norm_error": row["max_particle_norm_error"],
                **charge,
            }
        )

    one_step = {float(row["phase_cells"]): row for row in read_csv(ONE_STEP_PHASE)}
    full: dict[float, dict] = {}
    # Phase zero was deliberately reused from the smooth-with-grad-h run.
    full[0.0] = cases["smooth_with_grad_h"]
    for case_dir in sorted(PHASE_ROOT.glob("phase_*")):
        tag = case_dir.name.removeprefix("phase_").replace("p", ".")
        phase = float(tag)
        full[phase] = parse_momentum_run(case_dir / "run.log")
    expected_phases = {0.0, 0.125, 0.25, 0.375, 0.5}
    if set(one_step) != expected_phases or set(full) != expected_phases:
        raise AssertionError(f"phase matrix incomplete: one={set(one_step)}, full={set(full)}")

    phase_rows = []
    for phase in sorted(expected_phases):
        one = one_step[phase]
        long = full[phase]
        drift_norm = math.hypot(float(long["momentum_drift_x"]), float(long["momentum_drift_y"]))
        phase_rows.append(
            {
                "phase_cells": phase,
                "one_step_force_x": float(one["first_force_x"]),
                "one_step_momentum_drift_x": float(one["final_momentum_drift_x"]),
                "one_step_manual_relative_residual": float(one["max_manual_relative_residual"]),
                "full_1500_momentum_drift_x": float(long["momentum_drift_x"]),
                "full_1500_momentum_drift_y": float(long["momentum_drift_y"]),
                "full_1500_momentum_drift_norm": drift_norm,
                "full_1500_cumulative_relative_force_x": float(long["cumulative_relative_force_x"]),
                "full_1500_cumulative_relative_force_y": float(long["cumulative_relative_force_y"]),
                "full_1500_manual_relative_residual_last": float(long["max_manual_relative_residual"]),
            }
        )

    metadata = {
        "grad_h_ratios": ratios,
        "interpretation": {
            "42.8_percent": "x-momentum component only; the 2D norm changes by +0.0013679%",
            "1817_factor": "discontinuous/smooth-without-grad-h x component only; the 2D norm ratio is about 16.03",
            "grid_phase": "one-step x-force reverses with mesh phase, establishing a grid-anchored residual but not by itself separating iterative-solver from spatial-discretization error",
        },
    }
    return grad_rows, phase_rows, metadata


def analyze_tolerance_scan() -> tuple[list[dict], dict]:
    expected_hashes = {
        "HEXAPIC": "5f29c47c00ef0ddb6908a7994eadd68cdf28cee61502c73c7bb1e8ef44e760d4",
        "field_solver_hypre.cpp": "e6334c421041b46709deff9068fc139a29a78bdb8786e3785eafebd7961c5d8c",
        "kde_deposition.cpp": "41c98df4d24d9377cce9a588f521d0e9418778673613c0f6105fdd5afd73434d",
        "kde_deposition.hpp": "4aeb177ff22fb01b4a5a08285a0b2aa7e2b028b3147251e590776fa76c44a9df",
        "particles.cpp": "d71f09a0b4be82f23189801970dab0d6a6a069df6f60728f3f704c8ffeccb869",
    }
    revision = (TOLERANCE_ROOT / "code_revision.txt").read_text(encoding="utf-8").strip()
    if revision != "ed6dd96d16d0b70a53bf9486fd78a270ab3486d6":
        raise AssertionError(f"unexpected tolerance-scan revision: {revision}")
    observed_hashes = {}
    for line in (TOLERANCE_ROOT / "code_sha256.txt").read_text(encoding="utf-8").splitlines():
        digest, source = line.split(maxsplit=1)
        observed_hashes[pathlib.Path(source).name] = digest
    if observed_hashes != expected_hashes:
        raise AssertionError(f"tolerance-scan source hashes changed: {observed_hashes}")

    rows = []
    for case_dir in sorted(TOLERANCE_ROOT.glob("tol_*")):
        configured = parse_last_prefixed_line(case_dir / "run.log", "HYPRE_AUDIT configured_tolerance=")
        hypre = parse_last_prefixed_line(case_dir / "run.log", "HYPRE_AUDIT step=0")
        momentum = parse_last_prefixed_line(case_dir / "run.log", "MOMENTUM_AUDIT step=0")
        charge = parse_last_prefixed_line(case_dir / "run.log", "KDE_CHARGE_AUDIT step=0")
        derivative = parse_last_prefixed_line(case_dir / "run.log", "KDE_DERIVATIVE_AUDIT step=0")
        rows.append(
            {
                "configured_tolerance": configured["configured_tolerance"],
                "iterations": int(hypre["iterations"]),
                "reported_relative_residual": hypre["final_relative_residual"],
                "manual_relative_residual": hypre["manual_relative_residual"],
                "force_x": momentum["force_x"],
                "abs_force_x": momentum["abs_force_x"],
                "relative_force_x": momentum["relative_x"],
                "momentum_drift_x": momentum["momentum_drift_x"],
                "max_particle_normalization_error": charge["max_particle_norm_error"],
                "shape_derivative_relative_error": derivative["relative_error"],
                "assembled_signed_defect_over_expected_abs": abs(charge["signed_grid_defect"]) / charge["expected_abs"],
                "abs_weight_defect_over_expected_abs": abs(charge["abs_weight_defect"]) / charge["expected_abs"],
            }
        )
    rows.sort(key=lambda row: row["configured_tolerance"])
    if [row["configured_tolerance"] for row in rows] != [1.0e-11, 1.0e-9, 1.0e-7, 1.0e-5]:
        raise AssertionError(f"unexpected tolerance matrix: {rows}")
    force = np.asarray([row["force_x"] for row in rows])
    relative_force = np.asarray([row["relative_force_x"] for row in rows])
    summary = {
        "job_id": 5195488,
        "revision": revision,
        "source_hashes": observed_hashes,
        "manual_residual_range": [min(row["manual_relative_residual"] for row in rows), max(row["manual_relative_residual"] for row in rows)],
        "force_x_range": [float(np.min(force)), float(np.max(force))],
        "force_x_fractional_span": float(np.ptp(force) / abs(np.mean(force))),
        "relative_force_x_range": [float(np.min(relative_force)), float(np.max(relative_force))],
        "conclusion": "six orders of magnitude in achieved PFMG residual do not reduce the one-step force floor; solver tolerance is not its controlling cause",
    }
    return rows, summary


def make_energy_plot(trace_summary: dict, output: pathlib.Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required to generate conservation plots") from exc
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 8,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    names = [("landau_weak", "weak Landau"), ("landau_strong", "strong Landau"), ("e2", "E2")]
    colors = {"cic": "#172a46", "c4": "#9b2226", "large": "#6c757d"}
    labels = {"cic": "CIC", "c4": "C4", "large": "large-width control"}
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.45), constrained_layout=True)
    for axis, (benchmark, title) in zip(axes, names):
        for case in ("cic", "c4", "large"):
            trace = trace_summary[f"{benchmark}/{case}"]
            axis.plot(trace["times"], trace["mean_rel_field"], color=colors[case], label=labels[case], lw=1.2)
        axis.axhline(0.0, color="0.5", lw=0.7, ls="--")
        axis.grid(alpha=0.25)
        axis.set_title(title)
        axis.set_xlabel(r"$t\,\omega_{pe}$")
    axes[0].set_ylabel(r"ensemble mean $\Delta E/\max_t U_E$")
    axes[0].legend(frameon=False, fontsize=7, loc="best")
    fig.savefig(output / "energy_drift_normalized.pdf", bbox_inches="tight")
    fig.savefig(output / "energy_drift_normalized.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_momentum_plot(phase_rows: list[dict], output: pathlib.Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required to generate conservation plots") from exc
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 8,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    phase = np.asarray([row["phase_cells"] for row in phase_rows])
    force = np.asarray([row["one_step_force_x"] for row in phase_rows])
    drift_x = np.asarray([row["full_1500_momentum_drift_x"] for row in phase_rows])
    drift_norm = np.asarray([row["full_1500_momentum_drift_norm"] for row in phase_rows])
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.45), constrained_layout=True)
    axes[0].plot(phase, force, "o-", color="#172a46", lw=1.2, ms=4)
    axes[0].axhline(0.0, color="0.5", lw=0.7, ls="--")
    axes[0].set_xlabel(r"mesh phase $\delta/\Delta x$")
    axes[0].set_ylabel(r"one-step $F_x$")
    axes[0].set_title("grid-phase fingerprint")
    axes[0].grid(alpha=0.25)
    axes[1].plot(phase, np.abs(drift_x), "o-", color="#9b2226", label=r"$|\Delta P_x|$", lw=1.2, ms=4)
    axes[1].plot(phase, drift_norm, "s--", color="#6c757d", label=r"$\|\Delta\mathbf{P}\|_2$", lw=1.0, ms=3.5)
    axes[1].set_yscale("log")
    axes[1].set_xlabel(r"mesh phase $\delta/\Delta x$")
    axes[1].set_ylabel("1500-step momentum drift")
    axes[1].set_title("component versus vector norm")
    axes[1].legend(frameon=False, fontsize=7)
    axes[1].grid(alpha=0.25)
    fig.savefig(output / "momentum_phase_audit.pdf", bbox_inches="tight")
    fig.savefig(output / "momentum_phase_audit.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_tolerance_plot(rows: list[dict], output: pathlib.Path) -> None:
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 8,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    residual = np.asarray([row["manual_relative_residual"] for row in rows])
    force = np.abs(np.asarray([row["force_x"] for row in rows]))
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.35), constrained_layout=True)
    axes[0].loglog(
        [row["configured_tolerance"] for row in rows],
        residual,
        "o-",
        color="#172a46",
    )
    axes[0].loglog([1.0e-11, 1.0e-5], [1.0e-11, 1.0e-5], "--", color="0.5", lw=0.8, label="residual = tolerance")
    axes[0].set_xlabel("configured PFMG tolerance")
    axes[0].set_ylabel("manual relative residual")
    axes[0].set_title("solver convergence")
    axes[0].legend(frameon=False, fontsize=7)
    axes[0].grid(alpha=0.25)
    force_change_ppm = 1.0e6 * (force / force[0] - 1.0)
    axes[1].semilogx(residual, force_change_ppm, "o-", color="#9b2226")
    axes[1].axhline(0.0, color="0.5", lw=0.8, ls="--")
    axes[1].set_xlabel("manual relative residual")
    axes[1].set_ylabel(r"$10^6[F_x/F_x(10^{-11})-1]$")
    axes[1].set_title("force floor is tolerance-independent")
    axes[1].grid(alpha=0.25)
    fig.savefig(output / "hypre_tolerance_scan.pdf", bbox_inches="tight")
    fig.savefig(output / "hypre_tolerance_scan.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=pathlib.Path, default=HERE / "results" / "conservation_audit")
    parser.add_argument("--nboot", type=int, default=10000)
    args = parser.parse_args()
    for required in (ENERGY_ROOT, GRAD_H_ROOT, PHASE_ROOT, TOLERANCE_ROOT):
        if not required.is_dir():
            raise FileNotFoundError(required)
    args.output.mkdir(parents=True, exist_ok=True)

    per_seed, energy_rows, traces, energy_metadata = analyze_energy(args.nboot)
    grad_rows, phase_rows, momentum_metadata = analyze_momentum()
    tolerance_rows, tolerance_metadata = analyze_tolerance_scan()
    write_csv(args.output / "energy_per_seed.csv", per_seed)
    write_csv(args.output / "energy_ensemble.csv", energy_rows)
    write_csv(args.output / "grad_h_momentum.csv", grad_rows)
    write_csv(args.output / "grid_phase_momentum.csv", phase_rows)
    write_csv(args.output / "hypre_tolerance_scan.csv", tolerance_rows)
    np.savez_compressed(
        args.output / "energy_ensemble_traces.npz",
        **{
            f"{key.replace('/', '__')}__{field}": np.asarray(value[field])
            for key, value in traces.items()
            for field in ("times", "mean_rel_total", "mean_rel_field")
        },
    )
    make_energy_plot(traces, args.output)
    make_momentum_plot(phase_rows, args.output)
    make_tolerance_plot(tolerance_rows, args.output)
    result = {
        "inputs": {
            "energy": str(ENERGY_ROOT.relative_to(DOWNLOADS)),
            "grad_h": str(GRAD_H_ROOT.relative_to(DOWNLOADS)),
            "grid_phase_full": str(PHASE_ROOT.relative_to(DOWNLOADS)),
            "grid_phase_one_step": str(ONE_STEP_PHASE.relative_to(DOWNLOADS)),
            "hypre_tolerance_scan": str(TOLERANCE_ROOT.relative_to(DOWNLOADS)),
        },
        "energy_metadata": energy_metadata,
        "momentum_metadata": momentum_metadata,
        "energy_rows": energy_rows,
        "grad_h_rows": grad_rows,
        "grid_phase_rows": phase_rows,
        "hypre_tolerance_rows": tolerance_rows,
        "hypre_tolerance_metadata": tolerance_metadata,
    }
    (args.output / "conservation_audit.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
