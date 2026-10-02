#!/usr/bin/env python3
"""Build the remaining Paper-1 existing-data evidence package.

This script does not run a particle simulation and does not edit either
manuscript.  It reads the frozen/recovered reductions, checks their schemas,
adds matched finite-marker and estimator summaries, audits the finite-PPC
Landau reference, and writes source tables plus vector figures.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np


HERE = Path(__file__).resolve().parent
DOWNLOADS = HERE.parents[1]
RESULTS = HERE / "results"
PROJECT = DOWNLOADS / "pic_kde_2d"
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from pic_kde_2d.validation.landau_censored_fit import fit_censored_mode  # noqa: E402


ESTIMATOR_DIR = RESULTS / "e2_estimator_comparison_complete_20260912_with_extension"
MARKER_DIR = RESULTS / "e2_seed_rate_variation_20260912_with_extension"
DRIFT_DIR = RESULTS / "e2_beam_drift_scan_analysis_20260912_complete_extension_r9"
INTERIOR_DIR = RESULTS / "e2_interior_noise_replay_20260911"
LANDAU_ROOT = RESULTS / "weak_landau_confirmatory_v2_production_20260911"
LANDAU_RESULT = RESULTS / "weak_landau_confirmatory_v2_stage2_analysis_20260911" / "confirmatory_result.json"
LANDAU_CONFIG = HERE / "weak_landau_confirmatory_v2_global_velocity.json"
JOINT_CORNER = (
    RESULTS / "rate_uncertainty_audit_v8" / "e2_gaussian_joint_corner_recovered"
    / "joint_corner_sweep.csv"
)
COMP_ACCURACY = RESULTS / "evidence_closure_20260911" / "e2_compensated_prediction_accuracy.csv"
FILTER_NOISE = RESULTS / "noise_metric_audit" / "filter_family_noise.csv"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise AssertionError(f"missing CSV schema: {path}")
        return list(reader)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise AssertionError(f"refusing to write empty table: {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def f(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    return float(value) if value not in ("", None) else float("nan")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def percentile_interval(values: np.ndarray) -> tuple[float, float]:
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def marker_group_for_campaign(campaign: str) -> str:
    if campaign == "registered_drift_scan_5201231":
        return "current_drift_seeds_200_209"
    return "historical_seeds_0_9_reconstructed"


def baseline_for_campaign(campaign: str) -> str:
    return "cic" if campaign == "registered_drift_scan_5201231" else "base"


def registered_interval(
    campaign: str, v_beam: float, candidate: str, drift: list[dict], joint: list[dict], comp: list[dict]
) -> tuple[float, float]:
    if campaign == "registered_drift_scan_5201231":
        row = next(
            r for r in drift
            if abs(f(r, "v_beam") - v_beam) < 1e-12 and r["arm"] == candidate
        )
        return f(row, "ratio_ci_lo"), f(row, "ratio_ci_hi")
    if campaign == "historical_table5_corner_4913408":
        row = next(r for r in joint if r["label"] == candidate)
        return f(row, "ratio_ci_lo"), f(row, "ratio_ci_hi")
    row = next(r for r in comp if r["label"] == candidate)
    return f(row, "measured_ci_lo"), f(row, "measured_ci_hi")


def build_estimand_table(output: Path, rng: np.random.Generator) -> list[dict]:
    sensitivity = read_csv(ESTIMATOR_DIR / "paired_individual_rate_bootstrap_sensitivity.csv")
    drift = read_csv(DRIFT_DIR / "E2_beam_drift_scan.csv")
    joint = read_csv(JOINT_CORNER)
    comp = read_csv(COMP_ACCURACY)
    marker = read_csv(MARKER_DIR / "finite_marker_linear_response.csv")
    groups: dict[tuple[str, float, str], list[dict]] = {}
    for row in marker:
        groups.setdefault((row["marker_group"], f(row, "v_beam"), row["arm"]), []).append(row)

    rows: list[dict] = []
    for source in sensitivity:
        campaign = source["campaign"]
        vb = f(source, "v_beam")
        candidate = source["candidate"]
        baseline = source["baseline"]
        reg_lo, reg_hi = registered_interval(campaign, vb, candidate, drift, joint, comp)
        group = marker_group_for_campaign(campaign)
        marker_base = baseline_for_campaign(campaign)
        base_rows = sorted(groups[(group, vb, marker_base)], key=lambda r: int(r["seed"]))
        cand_rows = sorted(groups[(group, vb, candidate)], key=lambda r: int(r["seed"]))
        if [r["seed"] for r in base_rows] != [r["seed"] for r in cand_rows]:
            raise AssertionError(f"finite-marker pairing mismatch: {campaign} {candidate}")
        pred0 = np.array([f(r, "finite_marker_diagonal_gamma") for r in base_rows])
        pred1 = np.array([f(r, "finite_marker_diagonal_gamma") for r in cand_rows])
        obs0 = np.array([f(r, "observed_individual_gamma") for r in base_rows])
        obs1 = np.array([f(r, "observed_individual_gamma") for r in cand_rows])
        good = np.isfinite(pred0) & np.isfinite(pred1) & np.isfinite(obs0) & np.isfinite(obs1)
        pred0, pred1, obs0, obs1 = pred0[good], pred1[good], obs0[good], obs1[good]
        if pred0.size:
            draws = rng.integers(0, pred0.size, size=(5000, pred0.size))
            pred_ratios = pred1[draws].mean(axis=1) / pred0[draws].mean(axis=1)
            pred_lo, pred_hi = percentile_interval(pred_ratios)
            pred_ratio = float(np.mean(pred1) / np.mean(pred0))
            obs_residual = obs1 / obs0 - pred1 / pred0
            res_draws = obs_residual[draws].mean(axis=1)
            res_lo, res_hi = percentile_interval(res_draws)
        else:
            pred_ratio = pred_lo = pred_hi = res_lo = res_hi = float("nan")
        rows.append(
            {
                "campaign": campaign,
                "v_beam": vb,
                "baseline": baseline,
                "candidate": candidate,
                "window": source["window"],
                "n_paired_seeds": int(source["n_paired_seeds"]),
                "registered_mean_amplitude_ratio": f(source, "registered_ratio_of_mean_amplitude_rates_primary"),
                "registered_paired_ci_lo": reg_lo,
                "registered_paired_ci_hi": reg_hi,
                "ratio_of_individual_rate_means": f(source, "ratio_of_individual_rate_means_sensitivity"),
                "individual_rate_paired_ci_lo": f(source, "paired_bootstrap_ci_lo_sensitivity"),
                "individual_rate_paired_ci_hi": f(source, "paired_bootstrap_ci_hi_sensitivity"),
                "initialization_finite_marker_ratio": pred_ratio,
                "finite_marker_paired_ci_lo": pred_lo,
                "finite_marker_paired_ci_hi": pred_hi,
                "observed_seedwise_ratio_minus_finite_marker_ratio_mean": float(np.mean(obs_residual)) if pred0.size else float("nan"),
                "ratio_residual_paired_ci_lo": res_lo,
                "ratio_residual_paired_ci_hi": res_hi,
                "finite_marker_source_status": cand_rows[0]["source_status"] if cand_rows else "NO_CONVERGED_ROOT",
                "registered_signal_status": (
                    "PASS" if source.get("baseline_registered_signal_gate") == "PASS"
                    and source.get("candidate_registered_signal_gate") == "PASS" else
                    "UNRESOLVED_OR_HISTORICAL_GATE_NOT_RECORDED"
                ),
                "interpretation": (
                    "primary registered estimator plus labelled individual-rate sensitivity; "
                    "finite-marker prediction uses initialization, not fitted rates"
                ),
            }
        )
    write_csv(output / "e2_estimand_comparison.csv", rows)
    return rows


def build_marker_residual_table(output: Path, rng: np.random.Generator) -> list[dict]:
    rows = read_csv(MARKER_DIR / "finite_marker_linear_response.csv")
    grouped: dict[tuple[str, float, str], list[dict]] = {}
    for row in rows:
        grouped.setdefault((row["marker_group"], f(row, "v_beam"), row["arm"]), []).append(row)
    out: list[dict] = []
    for (group, vb, arm), values in sorted(grouped.items()):
        pred = np.array([f(r, "finite_marker_diagonal_gamma") for r in values])
        obs = np.array([f(r, "observed_individual_gamma") for r in values])
        good = np.isfinite(pred) & np.isfinite(obs)
        pred, obs = pred[good], obs[good]
        residual = obs - pred
        if residual.size:
            draws = rng.integers(0, residual.size, size=(5000, residual.size))
            mean_draws = residual[draws].mean(axis=1)
            lo, hi = percentile_interval(mean_draws)
            corr = float(np.corrcoef(pred, obs)[0, 1]) if residual.size > 1 else float("nan")
        else:
            lo = hi = corr = float("nan")
        out.append(
            {
                "marker_group": group,
                "source_status": values[0]["source_status"],
                "v_beam": vb,
                "arm": arm,
                "n_roots": int(pred.size),
                "finite_marker_gamma_mean": float(np.mean(pred)) if pred.size else float("nan"),
                "observed_individual_gamma_mean": float(np.mean(obs)) if obs.size else float("nan"),
                "observed_minus_prediction_mean": float(np.mean(residual)) if residual.size else float("nan"),
                "residual_paired_ci_lo": lo,
                "residual_paired_ci_hi": hi,
                "observed_prediction_rmse": float(np.sqrt(np.mean(residual**2))) if residual.size else float("nan"),
                "observed_prediction_correlation": corr,
                "statistical_unit": "seed",
                "predictor_scope": values[0]["approximation_limit"],
            }
        )
    write_csv(output / "finite_marker_prediction_residuals.csv", out)
    return out


def load_landau_arrays(arm: str, reference: bool = False) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    seeds = range(202609300, 202609340)
    collected: dict[str, list[np.ndarray]] = {}
    times = None
    for seed in seeds:
        if reference:
            path = LANDAU_ROOT / "reference" / f"seed{seed}" / "cic_reference" / "history_and_noise_snapshots.npz"
        else:
            path = LANDAU_ROOT / "low" / f"seed{seed}" / arm / "history_and_noise_snapshots.npz"
        with np.load(path, allow_pickle=False) as archive:
            current_times = np.asarray(archive["times"])
            if times is None:
                times = current_times
            else:
                np.testing.assert_array_equal(times, current_times)
            for key in ("mode", "snapshot_times", "rho_snapshots", "Ex_snapshots", "Ey_snapshots"):
                collected.setdefault(key, []).append(np.asarray(archive[key]))
    assert times is not None
    return times, {key: np.stack(value) for key, value in collected.items()}


def fit_ppc4096_reference(output: Path) -> dict:
    config = json.loads(LANDAU_CONFIG.read_text(encoding="utf-8"))
    times, arrays = load_landau_arrays("cic", reference=True)
    rule = config["fit_rule"]
    result = fit_censored_mode(
        arrays["mode"], times,
        fit_t_start=rule["fit_t_start"], fit_t_stop_max=rule["fit_t_stop_max"],
        noise_floor_start=rule["noise_floor_t_start"], noise_floor_stop=rule["noise_floor_t_stop"],
        noise_factor=rule["noise_threshold_factor"], envelope_period=rule["envelope_period"],
        persistence_points=rule["persistence_points"], min_points=rule["min_points"],
        min_duration=rule["min_duration"], min_model_efoldings=rule["min_fitted_amplitude_efoldings"],
        min_r_squared=rule["min_fit_r_squared"],
    )
    result.update(
        {
            "ppc": 4096,
            "n_seeds": 40,
            "role": "finite-PPC CIC reference originally generated for error-power metrics",
            "decision": "FAIL_DURATION; it does not supply two fitted amplitude e-foldings",
            "new_campaign_implication": (
                "The existing higher-PPC reference does not repair the duration gate; "
                "a larger repeat is not justified by the ideal 1/sqrt(N) scaling argument alone."
            ),
        }
    )
    (output / "weak_landau_ppc4096_reference_fit.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def build_reference_decomposition(output: Path) -> list[dict]:
    _, reference = load_landau_arrays("cic", reference=True)
    out: list[dict] = []
    for arm in ("cic", "c4_sigma1", "lscv_job5008392"):
        _, current = load_landau_arrays(arm)
        for metric, keys in (
            ("density_error_power", ("rho_snapshots",)),
            ("pre_return_grid_field_error_power", ("Ex_snapshots", "Ey_snapshots")),
        ):
            parts = {name: 0.0 for name in ("pair_mse", "squared_mean_difference", "arm_seed_variance", "reference_seed_variance", "seed_covariance")}
            for key in keys:
                x = current[key].astype(float)
                r = reference[key].astype(float)
                mx, mr = x.mean(axis=0), r.mean(axis=0)
                parts["pair_mse"] += float(np.mean((x - r) ** 2))
                parts["squared_mean_difference"] += float(np.mean((mx - mr) ** 2))
                parts["arm_seed_variance"] += float(np.mean((x - mx) ** 2))
                parts["reference_seed_variance"] += float(np.mean((r - mr) ** 2))
                parts["seed_covariance"] += float(np.mean((x - mx) * (r - mr)))
            reconstructed = (
                parts["squared_mean_difference"] + parts["arm_seed_variance"]
                + parts["reference_seed_variance"] - 2.0 * parts["seed_covariance"]
            )
            if not math.isclose(reconstructed, parts["pair_mse"], rel_tol=2e-13, abs_tol=1e-18):
                raise AssertionError("finite-reference decomposition failed")
            out.append(
                {
                    "arm": arm,
                    "metric": metric,
                    **parts,
                    "decomposition_reconstruction": reconstructed,
                    "reference_variance_fraction_of_pair_mse": parts["reference_seed_variance"] / parts["pair_mse"],
                    "minus_two_covariance_fraction_of_pair_mse": -2.0 * parts["seed_covariance"] / parts["pair_mse"],
                    "reference_ppc": 4096,
                    "low_ppc": 1024,
                    "window": "51 snapshots, t=[5,30], cadence=0.5",
                    "reference_limit": "finite independent PPC-4096 realization; not continuum truth",
                }
            )
    write_csv(output / "weak_landau_finite_reference_decomposition.csv", out)
    return out


def build_smoothing_source(output: Path) -> list[dict]:
    joint = read_csv(JOINT_CORNER)
    drift = read_csv(DRIFT_DIR / "E2_beam_drift_scan.csv")
    rows: list[dict] = []
    for r in joint:
        if r["label"] in {"base", "y012", "y018", "y02428", "y030"}:
            rows.append(
                {
                    "campaign": "historical_table5_corner_4913408",
                    "source_status": "historical source revision unknown",
                    "sigma_y_dy": f(r, "sigma_y_dy"),
                    "predicted_rate_ratio": f(r, "warm_ratio"),
                    "measured_registered_ratio": f(r, "gamma_ratio"),
                    "measured_ci_lo": f(r, "ratio_ci_lo"),
                    "measured_ci_hi": f(r, "ratio_ci_hi"),
                    "point_role": "historical directional sweep",
                }
            )
    for arm, role in (("constrained_gaussian", "R=0.95 interior"), ("fixed_gaussian", "model boundary control")):
        r = next(x for x in drift if abs(f(x, "v_beam") - 0.8) < 1e-12 and x["arm"] == arm)
        rows.append(
            {
                "campaign": "registered_drift_scan_5201231",
                "source_status": "exact current audited implementation",
                "sigma_y_dy": f(r, "sigma_y_dy"),
                "predicted_rate_ratio": f(r, "predicted_rate_ratio"),
                "measured_registered_ratio": f(r, "gamma_ratio_to_cic"),
                "measured_ci_lo": f(r, "ratio_ci_lo"),
                "measured_ci_hi": f(r, "ratio_ci_hi"),
                "point_role": role,
            }
        )
    write_csv(output / "e2_smoothing_budget_source.csv", rows)
    return rows


def build_practical_table(output: Path) -> list[dict]:
    landau = json.loads(LANDAU_RESULT.read_text(encoding="utf-8"))
    noise = landau["same_campaign_noise_metrics"]["arms"]
    interior = read_csv(INTERIOR_DIR / "interior_noise_summary.csv")
    drift = read_csv(DRIFT_DIR / "E2_beam_drift_scan.csv")
    comp_rate = read_csv(COMP_ACCURACY)
    comp_noise = read_csv(FILTER_NOISE)
    rows: list[dict] = []

    # The exact JSON nesting is intentionally resolved by named keys, not by position.
    for arm_id, label in (("c4_sigma1", "weak Landau C4 sigma=1"), ("lscv_job5008392", "weak Landau fixed CV-derived 7.2443")):
        ratio_raw = landau["rates"]["bootstrap"]["comparisons"][arm_id]
        ratio = {
            "central": ratio_raw["central_ratio_to_cic"],
            "ci_lo": ratio_raw["ratio_ci_lo"],
            "ci_hi": ratio_raw["ratio_ci_hi"],
        }
        n = noise[arm_id]["density_error_power_reduction_vs_cic"]
        rows.append(
            {
                "label": label,
                "dynamics_campaign": "weak_landau_confirmatory_v2",
                "noise_campaign": "weak_landau_confirmatory_v2",
                "comparison_type": "same campaign/configuration and paired seeds",
                "rate_estimator": "registered coherent complex-mean amplitude; arm-specific frozen windows",
                "rate_ratio": ratio["central"], "rate_ci_lo": ratio["ci_lo"], "rate_ci_hi": ratio["ci_hi"],
                "density_error_power_reduction": n["central"], "noise_ci_lo": n["ci_lo"], "noise_ci_hi": n["ci_hi"],
                "scientific_status": "FAIL_CIC_DURATION" if arm_id == "c4_sigma1" else "DYNAMICS_NOT_PRESERVED",
            }
        )

    interior_row = next(
        r for r in interior
        if r.get("metric") == "density_mse" and r.get("label") == "R095_interior_vb080"
    )
    drift_row = next(r for r in drift if abs(f(r, "v_beam") - 0.8) < 1e-12 and r["arm"] == "constrained_gaussian")
    rows.append(
        {
            "label": "E2 Gaussian R=0.95 interior",
            "dynamics_campaign": "registered_drift_scan_5201231",
            "noise_campaign": "e2_interior_noise_replay_20260911",
            "comparison_type": "compatible settings but separate projected-rate/full-f-error campaigns",
            "rate_estimator": "registered arithmetic-mean amplitude slope",
            "rate_ratio": f(drift_row, "gamma_ratio_to_cic"), "rate_ci_lo": f(drift_row, "ratio_ci_lo"), "rate_ci_hi": f(drift_row, "ratio_ci_hi"),
            "density_error_power_reduction": f(interior_row, "noise_power_reduction_vs_cic"),
            "noise_ci_lo": f(interior_row, "paired_bootstrap_ci_lo"),
            "noise_ci_hi": f(interior_row, "paired_bootstrap_ci_hi"),
            "scientific_status": "RATE_PRESERVED; MODEST_GRID_ERROR_BENEFIT; NOT_END_TO_END",
        }
    )
    for name in ("p2n2", "p3n4"):
        rr = next(r for r in comp_rate if r["label"] == name)
        nn = next(
            r for r in comp_noise
            if r["filter_family"] == "compensated_binomial"
            and int(r["compensation_order"]) == int(name[1])
            and int(r["passes"]) == int(name[3])
            and r["benchmark"] == "E2"
        )
        rows.append(
            {
                "label": f"E2 compensated {name}",
                "dynamics_campaign": "compensated_corner_5195640",
                "noise_campaign": nn["campaign"],
                "comparison_type": "separate historical projected-rate and full-f density-error campaigns",
                "rate_estimator": "registered arithmetic-mean amplitude slope",
                "rate_ratio": f(rr, "measured_rate_ratio"), "rate_ci_lo": f(rr, "measured_ci_lo"), "rate_ci_hi": f(rr, "measured_ci_hi"),
                "density_error_power_reduction": f(nn, "noise_power_reduction"),
                "noise_ci_lo": f(nn, "noise_power_reduction_ci_lo"),
                "noise_ci_hi": f(nn, "noise_power_reduction_ci_hi"),
                "scientific_status": "RATE_PRESERVED_IN_PROJECTED_TEST; NOT_SAME_CAMPAIGN_PRACTICAL_PROOF",
            }
        )
    write_csv(output / "practical_benefit_comparison.csv", rows)
    return rows


def configure_plotting():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update(
        {
            "font.family": "serif", "font.size": 8.5, "axes.labelsize": 9,
            "axes.titlesize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
            "legend.fontsize": 7.2, "axes.spines.top": False, "axes.spines.right": False,
            "axes.grid": True, "grid.alpha": 0.22, "figure.dpi": 150,
        }
    )
    return plt


def save_vector(fig, output: Path, stem: str) -> None:
    fig.savefig(output / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(output / f"{stem}.svg", bbox_inches="tight")
    fig.savefig(output / f"{stem}.png", dpi=300, bbox_inches="tight")


def make_figures(output: Path, estimands: list[dict], smoothing: list[dict], practical: list[dict]) -> None:
    plt = configure_plotting()

    # 1. Audited model boundary/interior and separately identified measurements.
    fig, ax = plt.subplots(figsize=(7.0, 3.0))
    historical = sorted((r for r in smoothing if r["campaign"].startswith("historical")), key=lambda r: r["sigma_y_dy"])
    x = np.array([r["sigma_y_dy"] for r in historical]); y = np.array([r["predicted_rate_ratio"] for r in historical])
    ax.plot(x, y, color="#333333", lw=1.5, label="audited grid-time prediction (saved roots)")
    ax.axhspan(0.90, 1.10, color="#4daf4a", alpha=0.10, label="10% preservation band")
    ax.axvline(0.2427346, color="#777777", ls=":", lw=1.0)
    ax.errorbar(x, [r["measured_registered_ratio"] for r in historical],
                yerr=[[r["measured_registered_ratio"]-r["measured_ci_lo"] for r in historical],
                      [r["measured_ci_hi"]-r["measured_registered_ratio"] for r in historical]],
                fmt="o", mfc="white", mec="#7b3294", ecolor="#7b3294", ms=4,
                label="historical Table-5 measurements (revision unknown)")
    for r, color in ((next(q for q in smoothing if q["point_role"].startswith("R=0.95")), "#0571b0"),
                     (next(q for q in smoothing if q["point_role"].startswith("model boundary")), "#ca0020")):
        ax.errorbar(r["sigma_y_dy"], r["measured_registered_ratio"],
                    yerr=[[r["measured_registered_ratio"]-r["measured_ci_lo"]],
                          [r["measured_ci_hi"]-r["measured_registered_ratio"]]],
                    fmt="s", color=color, ms=5, label=r["point_role"] + " (current scan)")
    ax.annotate("R=0.95 interior", xy=(0.1738592384, 0.95), xytext=(0.115, 0.915),
                arrowprops={"arrowstyle": "-", "lw": .7}, color="#0571b0")
    ax.text(0.245, 1.018, "model 10% boundary", rotation=90, va="top", color="#555555")
    ax.set(xlabel=r"directional Gaussian width $\sigma_y/\Delta y$", ylabel=r"growth-rate ratio $\gamma/\gamma_{\rm CIC}$",
           xlim=(-0.01, 0.315), ylim=(0.82, 1.035), title="E2 smoothing budget: prediction, interior choice, and measured points")
    ax.legend(ncol=2, frameon=False, loc="lower left")
    fig.tight_layout(); save_vector(fig, output, "fig_e2_smoothing_budget"); plt.close(fig)

    # 2. Current exact and historical nominal finite-marker comparisons.
    raw = read_csv(MARKER_DIR / "finite_marker_linear_response.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0), sharex=False, sharey=False)
    colors = {"cic": "#1b365d", "fixed_gaussian": "#b2182b", "constrained_gaussian": "#2166ac"}
    current = [r for r in raw if r["source_status"] == "EXACT_CURRENT_AUDITED_INITIALIZER" and f(r,"v_beam") <= .8]
    for arm in ("cic", "fixed_gaussian", "constrained_gaussian"):
        q = [r for r in current if r["arm"] == arm]
        axes[0].scatter([f(r,"finite_marker_diagonal_gamma") for r in q], [f(r,"observed_individual_gamma") for r in q],
                        s=12, alpha=.75, color=colors[arm], label=arm.replace("_", " "))
    lim=(0.08,.34); axes[0].plot(lim,lim,"--",color="#777777",lw=1); axes[0].set(xlim=lim,ylim=lim,title="(a) Current audited initializations",xlabel="initialization predictor",ylabel="individual fitted rate")
    axes[0].legend(frameon=False)
    nominal=[r for r in raw if r["source_status"] != "EXACT_CURRENT_AUDITED_INITIALIZER"]
    hist=[r for r in nominal if r["arm"].startswith("y") or r["arm"].startswith("joint") or r["arm"]=="base"]
    comp=[r for r in nominal if r["arm"].startswith("p")]
    axes[1].scatter([f(r,"finite_marker_diagonal_gamma") for r in hist],[f(r,"observed_individual_gamma") for r in hist],s=12,alpha=.65,color="#7b3294",label="Table 5 (nominal init.)")
    axes[1].scatter([f(r,"finite_marker_diagonal_gamma") for r in comp],[f(r,"observed_individual_gamma") for r in comp],s=12,alpha=.65,color="#e08214",label="job 5195640 (nominal init.)")
    lim=(.04,.15); axes[1].plot(lim,lim,"--",color="#777777",lw=1); axes[1].set(xlim=lim,ylim=lim,title="(b) Historical marker arrays unavailable",xlabel="nominal initialization predictor",ylabel="individual fitted rate")
    axes[1].legend(frameon=False)
    fig.tight_layout(); save_vector(fig, output, "fig_e2_finite_marker_predictions"); plt.close(fig)

    # 3. Registered mixture estimator versus individual-rate mean.
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0), sharey=True)
    for ax, candidate, title, color in ((axes[0],"fixed_gaussian","fixed Gaussian", "#b2182b"),(axes[1],"constrained_gaussian","R=0.95 interior", "#2166ac")):
        q=sorted([r for r in estimands if r["campaign"]=="registered_drift_scan_5201231" and r["candidate"]==candidate],key=lambda r:r["v_beam"])
        valid=[r for r in q if r["v_beam"] <= .8]
        xv=np.array([r["v_beam"] for r in valid])
        reg=np.array([r["registered_mean_amplitude_ratio"] for r in valid]); reglo=np.array([r["registered_paired_ci_lo"] for r in valid]); reghi=np.array([r["registered_paired_ci_hi"] for r in valid])
        ind=np.array([r["ratio_of_individual_rate_means"] for r in valid]); indlo=np.array([r["individual_rate_paired_ci_lo"] for r in valid]); indhi=np.array([r["individual_rate_paired_ci_hi"] for r in valid])
        ax.axhspan(.9,1.1,color="#4daf4a",alpha=.10)
        ax.errorbar(xv-.0015,reg,yerr=[reg-reglo,reghi-reg],fmt="o-",ms=4,color=color,label="registered mean-amplitude slope")
        ax.errorbar(xv+.0015,ind,yerr=[ind-indlo,indhi-ind],fmt="s--",ms=4,color="#333333",label="mean individual rates (sensitivity)")
        unresolved=next(r for r in q if abs(r["v_beam"]-.818)<1e-12)
        ax.plot(.818,unresolved["registered_mean_amplitude_ratio"],marker="x",ms=7,color="#999999")
        ax.annotate("unresolved",(.818,unresolved["registered_mean_amplitude_ratio"]),xytext=(-25,9),textcoords="offset points",fontsize=7,color="#777777")
        ax.set(title=title,xlabel=r"beam drift $v_b$")
        ax.legend(frameon=False,loc="lower left")
    axes[0].set_ylabel(r"filtered/CIC growth-rate ratio")
    axes[0].set_ylim(.86,1.025)
    fig.tight_layout(); save_vector(fig, output, "fig_e2_estimands"); plt.close(fig)

    # 4. Practical rate/noise trade evidence with same/split campaign encoded.
    fig, ax = plt.subplots(figsize=(7.0, 3.1))
    label_offsets = {
        "weak Landau C4 sigma=1": (7, 13),
        "weak Landau fixed CV-derived 7.2443": (7, -5),
        "E2 Gaussian R=0.95 interior": (7, 13),
        "E2 compensated p2n2": (7, -16),
        "E2 compensated p3n4": (7, 3),
    }
    for r in practical:
        filled = r["comparison_type"].startswith("same campaign")
        marker = "o" if filled else "s"
        ax.errorbar(r["density_error_power_reduction"],r["rate_ratio"],
                    xerr=[[r["density_error_power_reduction"]-r["noise_ci_lo"]],[r["noise_ci_hi"]-r["density_error_power_reduction"]]],
                    yerr=[[r["rate_ratio"]-r["rate_ci_lo"]],[r["rate_ci_hi"]-r["rate_ratio"]]],
                    fmt=marker,mfc="#2166ac" if filled else "white",mec="#2166ac",ecolor="#777777",ms=5)
        ax.annotate(
            r["label"], (r["density_error_power_reduction"], r["rate_ratio"]),
            xytext=label_offsets[r["label"]], textcoords="offset points", fontsize=7,
        )
    ax.axhspan(.9,1.1,color="#4daf4a",alpha=.10)
    ax.axhline(1,color="#777777",ls="--",lw=1)
    ax.set(xlabel="density-error-power reduction (CIC/candidate)",ylabel="filtered/CIC rate ratio",xlim=(.8,4.9),ylim=(.82,1.95),title="Measured benefit versus dynamic preservation (campaign provenance encoded)")
    ax.text(.99,.845,"filled circle: same campaign\nopen square: compatible but split campaigns",fontsize=7,va="bottom")
    fig.tight_layout(); save_vector(fig, output, "fig_practical_benefit_vs_rate"); plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.mkdir(parents=True)
    rng = np.random.default_rng(20260914)
    estimands = build_estimand_table(output, rng)
    marker = build_marker_residual_table(output, rng)
    ppc4096 = fit_ppc4096_reference(output)
    reference = build_reference_decomposition(output)
    smoothing = build_smoothing_source(output)
    practical = build_practical_table(output)
    make_figures(output, estimands, smoothing, practical)
    inputs = [
        ESTIMATOR_DIR / "paired_individual_rate_bootstrap_sensitivity.csv",
        MARKER_DIR / "finite_marker_linear_response.csv", DRIFT_DIR / "E2_beam_drift_scan.csv",
        INTERIOR_DIR / "interior_noise_summary.csv", LANDAU_RESULT, LANDAU_CONFIG,
        JOINT_CORNER, COMP_ACCURACY, FILTER_NOISE,
    ]
    manifest = {
        "analysis_role": "existing-data evidence completion; no simulations; no manuscript edits",
        "bootstrap_draws": 5000, "bootstrap_seed": 20260914,
        "inputs_sha256": {str(p.relative_to(DOWNLOADS)): sha256(p) for p in inputs},
        "output_files_sha256": {},
        "key_results": {
            "e2_estimand_rows": len(estimands), "finite_marker_summary_rows": len(marker),
            "weak_landau_ppc4096_fit": ppc4096,
            "finite_reference_decomposition_rows": len(reference),
            "practical_comparison_rows": len(practical),
        },
    }
    for path in sorted(output.iterdir()):
        if path.name != "analysis_manifest.json":
            manifest["output_files_sha256"][path.name] = sha256(path)
    (output / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
