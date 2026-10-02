#!/usr/bin/env python3
"""Apply the Paper 1 validation gates without conflating their components."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


WORKSPACE = Path(__file__).resolve().parents[2]
PAPER_RESULTS = WORKSPACE / "pic_kde_2d" / "results"
AUDIT = Path(__file__).resolve().parent / "results" / "rate_uncertainty_audit_v8"
RECOVERED = WORKSPACE / "pic_kde_2d" / "results_raw_rate_audit" / "recovered_meluxina_20260909"
RECOVERED_20260911 = (
    WORKSPACE / "pic_kde_2d" / "results_raw_rate_audit" / "recovered_meluxina_20260911"
)
RECOVERED_M12_REDUCTION = (
    Path(__file__).resolve().parent
    / "results/recovered_delta_f_m12_sigma_4872745_20260911/m12_sigma_sweep.csv"
)
RECOVERED_REFERENCE_REDUCTION = (
    Path(__file__).resolve().parent
    / "results/recovered_delta_f_reference_4872599_20260911/linearized_delta_f_reference.csv"
)
CONFIRMATORY_V2 = (
    Path(__file__).resolve().parent
    / "results/weak_landau_confirmatory_v2_stage2_analysis_20260911/confirmatory_result.json"
)
EVIDENCE_CLOSURE = (
    Path(__file__).resolve().parent
    / "results/evidence_closure_20260911/e2_c4_signal_resolvability.json"
)
LOW, HIGH = 0.90, 1.10
MIN_EFOLDINGS = 2.0


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def find_row(path: Path, **criteria: str) -> dict[str, str]:
    matches = [
        row for row in read_csv(path)
        if all(row.get(key) == value for key, value in criteria.items())
    ]
    if len(matches) != 1:
        raise AssertionError(f"expected one row in {path} for {criteria}, found {len(matches)}")
    return matches[0]


def f(value: Any) -> float | None:
    if value in (None, "", "nan", "None"):
        return None
    return float(value)


def rate_gate(lo: float | None, hi: float | None) -> str:
    if lo is None or hi is None:
        return "NOT_MEASURED"
    return "PASS" if lo >= LOW and hi <= HIGH else "FAIL"


def duration_gate(efoldings: float | None) -> str:
    if efoldings is None:
        return "NOT_MEASURED"
    return "PASS" if efoldings >= MIN_EFOLDINGS else "FAIL"


def overall(row: dict[str, Any]) -> str:
    # A resolved model, rate, duration, or signal failure is a failure.
    # A missing component prevents a pass but must not be converted into a
    # physical failure.
    if (
        row["model_admissibility"] == "FAIL"
        or row["rate_gate"] == "FAIL"
        or row["duration_gate"] == "FAIL"
        or row["signal_resolvability"] == "FAIL"
        or row["baseline_duration_gate"] == "FAIL"
    ):
        return "FAIL"
    required = (
        row["model_admissibility"], row["rate_gate"], row["duration_gate"],
        row["signal_resolvability"], row["protected_mode_coverage"],
        row["raw_traceability"],
    )
    baseline_duration = row["baseline_duration_gate"]
    return "PASS" if (
        all(item == "PASS" for item in required)
        and baseline_duration in {"PASS", "NOT_APPLICABLE"}
    ) else "UNRESOLVED"


def make_row(
    *, condition: str, benchmark: str, method: str, setting: str,
    measured_mode: str, protected_scope: str, model: str,
    ratio: float | None, lo: float | None, hi: float | None,
    start: float | None, stop: float | None, efoldings: float | None,
    r2: float | None, snr: float | None, signal: str, coverage: str,
    raw: str, evidence: str, note: str = "", prediction: float | None = None,
    baseline_duration: str = "NOT_APPLICABLE",
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "condition": condition,
        "benchmark": benchmark,
        "method": method,
        "setting": setting,
        "measured_mode": measured_mode,
        "protected_scope": protected_scope,
        "model_admissibility": model,
        "rate_ratio": ratio,
        "rate_ratio_ci_lo": lo,
        "rate_ratio_ci_hi": hi,
        "rate_gate": rate_gate(lo, hi),
        "fit_t_start": start,
        "fit_t_stop": stop,
        "model_efoldings": efoldings,
        "duration_gate": duration_gate(efoldings),
        "baseline_duration_gate": baseline_duration,
        "fit_r_squared": r2,
        "fit_start_snr": snr,
        "signal_resolvability": signal,
        "protected_mode_coverage": coverage,
        "raw_traceability": raw,
        "prediction_only_ratio": prediction,
        "evidence": evidence,
        "note": note,
    }
    row["overall_status"] = overall(row)
    return row


def weak_confirmatory_v2_rows() -> list[dict[str, Any]]:
    """Add the fresh global-loading confirmation without pooling development data."""
    result = json.loads(CONFIRMATORY_V2.read_text(encoding="utf-8"))
    if result["promotion_status"] != "FAIL" or result["development_seeds_pooled"]:
        raise AssertionError("unexpected v2 confirmation status or seed pooling")
    if result["n_fresh_seeds"] != 40:
        raise AssertionError("v2 confirmation is not the fixed 40-seed sample")
    central = result["rates"]["central"]
    bootstrap = result["rates"]["bootstrap"]
    evidence = str(CONFIRMATORY_V2.relative_to(WORKSPACE))

    def independent_signal_gate(fit: dict[str, Any]) -> str:
        # Keep signal/fit quality separate from the two-e-fold duration gate.
        passed = (
            fit["gamma"] < 0.0
            and fit["r_squared"] >= 0.90
            and fit["fit_start_snr"] >= 5.0
            and fit["n_points"] >= 30
            and fit["duration"] >= 5.0
        )
        return "PASS" if passed else "FAIL"

    cic = central["cic"]
    cic_reference = bootstrap["cic_absolute_ratio_to_discrete_reference"]
    rows = [make_row(
        condition="weak_confirm_v2_cic_absolute",
        benchmark="weak Landau",
        method="CIC absolute confirmation",
        setting="global velocities; PPC 1024; 40 fresh seeds",
        measured_mode="(1,0)", protected_scope="(1,0)", model="PASS",
        ratio=f(cic_reference["central"]), lo=f(cic_reference["ci_lo"]),
        hi=f(cic_reference["ci_hi"]), start=5.0, stop=30.1,
        efoldings=f(cic["efoldings"]), r2=f(cic["r_squared"]),
        snr=f(cic["fit_start_snr"]), signal=independent_signal_gate(cic),
        coverage="PASS", raw="PASS", evidence=evidence,
        note=("absolute amplitude-rate ratio to the preregistered fully discrete "
              "CIC reference; rate interval passes but duration fails"),
    )]

    for condition, arm, method, setting in [
        ("weak_confirm_v2_c4", "c4_sigma1", "Gaussian C4", "sigma=1 cell; global velocities; 40 fresh seeds"),
        ("weak_confirm_v2_lscv", "lscv_job5008392", "LOO-CV comparator", "sigma=7.244293147028428 cells; 40 fresh seeds"),
    ]:
        fit = central[arm]
        comparison = bootstrap["comparisons"][arm]
        rows.append(make_row(
            condition=condition, benchmark="weak Landau", method=method,
            setting=setting, measured_mode="(1,0)", protected_scope="(1,0)",
            model="PASS", ratio=f(comparison["central_ratio_to_cic"]),
            lo=f(comparison["ratio_ci_lo"]), hi=f(comparison["ratio_ci_hi"]),
            start=5.0, stop=f(fit["duration"]) + 5.0,
            efoldings=f(fit["efoldings"]), r2=f(fit["r_squared"]),
            snr=f(fit["fit_start_snr"]), signal=independent_signal_gate(fit),
            coverage="PASS", raw="PASS", evidence=evidence,
            baseline_duration="FAIL",
            note=("paired interval uses the frozen arm-specific windows; the "
                  "confirmation cannot pass because its CIC baseline spans only "
                  "1.721386 fitted amplitude e-foldings"),
        ))
    return rows


def weak_gaussian_rows() -> list[dict[str, Any]]:
    grid = AUDIT / "weak_gaussian_grid/landau_sigma_refinement.csv"
    mise = AUDIT / "weak_gaussian_mise6/landau_sigma_refinement.csv"
    cv = AUDIT / "weak_gaussian_cv6p61/landau_sigma_refinement.csv"
    silver = AUDIT / "weak_gaussian_silverman_recovered/landau_sigma_refinement.csv"
    specs = [
        ("weak_cic", "CIC reference", "0", grid, {"ppc": "1024", "sigma_dx": "0.0"}, "PASS"),
        ("weak_c4", "Gaussian C4", "sigma=1", grid, {"ppc": "1024", "sigma_dx": "1.0"}, "PASS"),
        ("weak_mise", "Gaussian MISE", "sigma=6", mise, {"ppc": "1024", "sigma_dx": "6.0"}, "PASS"),
        ("weak_loo_proxy", "LOO-CV proxy", "sigma=6.61", cv, {"ppc": "1024", "sigma_dx": "6.61"}, "PASS"),
        ("weak_silverman", "Silverman", "sigma=1.83297", silver, {"ppc": "1024", "sigma_dx": "1.8329728493314705"}, "PASS"),
    ]
    rows = []
    for condition, method, setting, path, criteria, raw in specs:
        source = find_row(path, **criteria)
        ratio = f(source["gamma_ratio_to_cic"])
        lo, hi = f(source["ratio_ci_lo"]), f(source["ratio_ci_hi"])
        signal = "PASS" if (
            source["quality_pass"] == "True"
            and f(source["fit_r_squared"]) >= 0.90
            and f(source["bootstrap_valid_fraction"]) >= 0.80
            and f(source["bootstrap_quality_fraction"]) >= 0.80
        ) else "FAIL"
        rows.append(make_row(
            condition=condition, benchmark="weak Landau", method=method,
            setting=setting, measured_mode="(1,0)", protected_scope="(1,0)",
            model="PASS", ratio=ratio, lo=lo, hi=hi,
            start=f(source["fit_t_start"]), stop=f(source["fit_t_stop"]),
            efoldings=f(source["model_efoldings"]), r2=f(source["fit_r_squared"]),
            snr=f(source["fit_start_snr"]), signal=signal, coverage="PASS",
            raw="PASS" if raw == "PASS" else "UNRESOLVED",
            evidence=str(path.relative_to(WORKSPACE)),
            note=("baseline ratio is definitional" if condition == "weak_cic" else ""),
        ))
    # A dedicated paired rerun now measures the actual LOO-CV optimum.  Keep
    # the historical sigma=6.61 proxy above as a distinct row.
    selected_path = (
        RECOVERED / "weak_landau_lscv_exact_5195682" / "analysis"
        / "landau_sigma_refinement.csv"
    )
    source = find_row(selected_path, ppc="1024", sigma_dx="7.244293147028428")
    rows.append(make_row(
        condition="weak_loo_selected", benchmark="weak Landau", method="LOO-CV",
        setting="selected sigma=7.244293147028428", measured_mode="(1,0)",
        protected_scope="(1,0)", model="PASS",
        ratio=f(source["gamma_ratio_to_cic"]), lo=f(source["ratio_ci_lo"]),
        hi=f(source["ratio_ci_hi"]), start=f(source["fit_t_start"]),
        stop=f(source["fit_t_stop"]), efoldings=f(source["model_efoldings"]),
        r2=f(source["fit_r_squared"]), snr=f(source["fit_start_snr"]),
        signal="PASS" if source["quality_pass"] == "True" else "FAIL",
        coverage="PASS", raw="PASS", evidence=str(selected_path.relative_to(WORKSPACE)),
        note=("direct paired measurement at the selected width; floor-censored "
              "with zero quality-passing bootstrap fraction"),
    ))
    # The inactive controller is exactly the baseline, but inherits its failed
    # duration gate and provides no smoothing/noise validation.
    baseline = rows[0]
    rows.append(make_row(
        condition="weak_wu_qin_inert", benchmark="weak Landau",
        method="Wu-Qin AD-gated", setting="held sigma=0", measured_mode="(1,0)",
        protected_scope="(1,0)", model="PASS", ratio=1.0, lo=1.0, hi=1.0,
        start=baseline["fit_t_start"], stop=baseline["fit_t_stop"],
        efoldings=baseline["model_efoldings"], r2=baseline["fit_r_squared"],
        snr=baseline["fit_start_snr"], signal="PASS", coverage="PASS", raw="PASS",
        evidence=baseline["evidence"], note="identical to CIC because the controller never opens",
    ))
    return rows


def weak_filter_rows() -> list[dict[str, Any]]:
    specs = [
        ("weak_raw_n4", "raw binomial", "n=4", "weak_raw_binomial", "4"),
        ("weak_raw_n8", "raw binomial", "n=8", "weak_raw_binomial", "8"),
        ("weak_comp_p1_n16", "compensated binomial", "p=1,n=16", "weak_compensated_p1", "16"),
        ("weak_comp_p2_n4", "compensated binomial", "p=2,n=4", "weak_compensated_p2", "4"),
        ("weak_comp_p3_n4", "compensated binomial", "p=3,n=4", "weak_compensated_p3", "4"),
    ]
    rows = []
    for condition, method, setting, campaign, passes in specs:
        path = AUDIT / campaign / "landau_binomial_passes.csv"
        source = find_row(path, passes=passes)
        rows.append(make_row(
            condition=condition, benchmark="weak Landau", method=method,
            setting=setting, measured_mode="(1,0)", protected_scope="(1,0)",
            model="PASS", ratio=f(source["gamma_ratio_to_cic"]),
            lo=f(source["ratio_ci_lo"]), hi=f(source["ratio_ci_hi"]),
            start=f(source["fit_t_start"]), stop=f(source["fit_t_stop"]),
            efoldings=f(source["model_efoldings"]), r2=f(source["fit_r_squared"]),
            snr=None, signal="PASS" if source["quality_pass"] == "True" else "FAIL",
            coverage="PASS", raw="PASS", evidence=str(path.relative_to(WORKSPACE)),
        ))
    return rows


def strong_rows() -> list[dict[str, Any]]:
    specs = [
        ("strong_a05_fit3", "alpha=0.5 CV", "strong_alpha0p5_fit3", "1.10785"),
        ("strong_a05_fit5", "alpha=0.5 CV sensitivity", "strong_alpha0p5_fit5", "1.10785"),
        ("strong_a025_cv", "alpha=0.25 CV", "strong_alpha0p25_cv_exact", "2.8208226"),
    ]
    rows = []
    for condition, method, campaign, sigma in specs:
        path = AUDIT / campaign / "strong_landau_fidelity.csv"
        source = find_row(path, sigma_dx=sigma)
        rows.append(make_row(
            condition=condition, benchmark="strong Landau", method=method,
            setting=f"sigma={sigma}", measured_mode="(1,0)", protected_scope="(1,0)",
            model="UNRESOLVED", ratio=f(source["gamma_ratio_to_cic"]),
            lo=f(source["ratio_ci_lo"]), hi=f(source["ratio_ci_hi"]),
            start=f(source["fit_t_start"]), stop=f(source["fit_t_stop"]),
            efoldings=f(source["model_efoldings"]), r2=f(source["fit_r_squared"]),
            snr=None, signal="PASS" if f(source["fit_r_squared"]) >= 0.90 else "FAIL",
            coverage="PASS", raw="PASS", evidence=str(path.relative_to(WORKSPACE)),
            note="finite-amplitude linear model is curtailed by the first bounce",
        ))
    for condition, label, campaign in [
        ("strong_raw_binomial_n16_fit3", "fit start 3", "strong_raw_binomial_fit3_recovered"),
        ("strong_raw_binomial_n16_fit5", "fit start 5 sensitivity", "strong_raw_binomial_fit5_recovered"),
    ]:
        path = AUDIT / campaign / "landau_binomial_passes.csv"
        source = find_row(path, passes="16")
        rows.append(make_row(
            condition=condition, benchmark="strong Landau", method="raw binomial",
            setting=f"n=16; {label}", measured_mode="(1,0)", protected_scope="(1,0)",
            model="UNRESOLVED", ratio=f(source["gamma_ratio_to_cic"]),
            lo=f(source["ratio_ci_lo"]), hi=f(source["ratio_ci_hi"]),
            start=f(source["fit_t_start"]), stop=f(source["fit_t_stop"]),
            efoldings=f(source["model_efoldings"]), r2=f(source["fit_r_squared"]),
            snr=None, signal="PASS" if f(source["fit_r_squared"]) >= 0.90 else "FAIL",
            coverage="PASS", raw="PASS", evidence=str(path.relative_to(WORKSPACE)),
            note="raw campaign recovered from MeluXina; finite-amplitude result is window-sensitive",
        ))
    return rows


def e2_summary_rows() -> list[dict[str, Any]]:
    joint_path = AUDIT / "e2_gaussian_joint_corner_recovered/joint_corner_sweep.csv"
    signal_audit = json.loads(EVIDENCE_CLOSURE.read_text(encoding="utf-8"))
    if signal_audit["signal_resolvability"] != "PASS":
        raise AssertionError("E2 C4 signal-resolvability audit does not pass")
    baseline = find_row(joint_path, label="base")
    baseline_duration = duration_gate(f(baseline["fit_efoldings"]))
    rows = []
    for condition, label, setting in [
        ("e2_c4_corner", "y02428", "sigma_x=0,sigma_y=0.2428"),
        ("e2_c4_failure", "y030", "sigma_x=0,sigma_y=0.30"),
    ]:
        source = find_row(joint_path, label=label)
        audited_snr = (
            f(signal_audit["floor"]["fit_start_snr"])
            if condition == "e2_c4_corner"
            else None
        )
        signal = "PASS" if (
            f(source["fit_r2"]) >= 0.98
            and f(source["fit_efoldings"]) >= MIN_EFOLDINGS
            and f(source["max_delta_f_ratio_fit_window"]) < 0.5
            and int(source["bootstrap_valid"]) == 5000
            and (audited_snr is None or audited_snr >= 5.0)
        ) else "FAIL"
        rows.append(make_row(
            condition=condition, benchmark="E2 near-marginal", method="Gaussian C4",
            setting=setting, measured_mode="(12,1)", protected_scope="binding corner (12,1)",
            model="PASS", ratio=f(source["gamma_ratio"]), lo=f(source["ratio_ci_lo"]),
            hi=f(source["ratio_ci_hi"]), start=24.32, stop=59.68,
            efoldings=f(source["fit_efoldings"]), r2=f(source["fit_r2"]), snr=audited_snr,
            signal=signal, coverage="PASS", raw="PASS",
            evidence=str(joint_path.relative_to(WORKSPACE)),
            baseline_duration=baseline_duration,
            note=(
                "regenerated from ten recovered raw histories; the original joint reducer "
                "stored no floor. The post hoc evidence audit applies the existing "
                "pre-emergence floor estimator and gives fit-start SNR 11.213 for C4; "
                "the shared-window CIC baseline spans 4.484 e-foldings"
                if condition == "e2_c4_corner"
                else "regenerated from ten recovered raw histories; original joint reducer has no explicit noise-floor SNR"
            ),
        ))
    rows.extend([
        make_row(
            condition="e2_silverman_erased", benchmark="E2 near-marginal",
            method="configured directional Silverman formula",
            setting="sigma_x=5.819,sigma_y=0.727", measured_mode="not measured",
            protected_scope="binding corner (12,1)", model="FAIL", ratio=None, lo=None, hi=None,
            start=None, stop=None, efoldings=None, r2=None, snr=None, signal="NOT_MEASURED",
            coverage="PASS", raw="UNRESOLVED",
            evidence="paper1_source_package/reproducibility/SELECTOR_INDEPENDENT_AUDIT.md",
            note=("analytic q exceeds the protected-mode budget; the supplied scalar noise and "
                  "rate launchers do not implement the configured pair, so no measured rate is substituted"),
        ),
        make_row(
            condition="e2_mise_cv_erased", benchmark="E2 near-marginal", method="MISE / LOO-CV",
            setting="sigma>=60", measured_mode="not measured", protected_scope="binding corner (12,1)",
            model="FAIL", ratio=None, lo=None, hi=None, start=None, stop=None,
            efoldings=0.0, r2=None, snr=None, signal="FAIL", coverage="PASS",
            raw="UNRESOLVED", evidence="pic_kde_2d/paper1_selector_comparison/results/comparison_cells.csv",
            note="mode transfer is operationally zero; no rate measurement substituted",
        ),
        make_row(
            condition="e2_wu_qin_inert", benchmark="E2 near-marginal", method="Wu-Qin AD-gated",
            setting="held sigma=0", measured_mode="binding baseline claimed", protected_scope="(12,1)",
            model="UNRESOLVED", ratio=1.0, lo=1.0, hi=1.0, start=None, stop=None,
            efoldings=4.483681269108191, r2=0.9981418050002847, snr=None,
            signal="PASS", coverage="PASS", raw="PASS",
            evidence=str(joint_path.relative_to(WORKSPACE)),
            note="controller is inert; CIC fidelity is reproduced, but no smoothing method is exercised",
        ),
    ])
    new_mode_path = (
        Path(__file__).resolve().parent
        / "results/selector_independent_audit/e2_new_mode_m10n1/E2_new_protected_mode.json"
    )
    new_mode = json.loads(new_mode_path.read_text(encoding="utf-8"))
    candidate = next(row for row in new_mode["rows"] if row["label"] == "c4")
    rows.append(make_row(
        condition="e2_c4_new_mode_m10n1", benchmark="E2 independent transfer",
        method="Gaussian C4", setting="sigma_x=0,sigma_y=0.2427346",
        measured_mode="(10,1)", protected_scope="independent test mode (10,1)",
        model="PASS", ratio=f(candidate["gamma_ratio_to_cic"]),
        lo=f(candidate["ratio_ci_lo"]), hi=f(candidate["ratio_ci_hi"]),
        start=f(candidate["fit_t_start"]), stop=f(candidate["fit_t_stop"]),
        efoldings=f(candidate["model_efoldings"]), r2=f(candidate["fit_r_squared"]),
        snr=f(candidate["fit_start_snr"]), signal=candidate["signal_gate"],
        coverage="PASS", raw="PASS", evidence=str(new_mode_path.relative_to(WORKSPACE)),
        note="new seeds 100--109; fixed predeclared mode and fit window",
        prediction=f(new_mode["grid_time_discrete_prediction"]["candidate_ratio"]),
    ))
    return rows


def e2_binomial_rows() -> list[dict[str, Any]]:
    specs = [
        ("e2_raw_n1", "raw binomial", "n=1", AUDIT / "e2_raw_binomial_recovered/E2_binomial_passes.csv", "1", "PASS"),
        ("e2_comp_p1_n1", "compensated binomial", "p=1,n=1", AUDIT / "e2_longitudinal_compensated_p1/E2_binomial_passes.csv", "1", "PASS"),
        ("e2_comp_p2_n2", "compensated binomial", "p=2,n=2", AUDIT / "e2_longitudinal_compensated_p2/E2_binomial_passes.csv", "2", "PASS"),
        ("e2_comp_p2_n4", "compensated binomial", "p=2,n=4", AUDIT / "e2_longitudinal_compensated_p2/E2_binomial_passes.csv", "4", "PASS"),
        ("e2_comp_p3_n4", "compensated binomial", "p=3,n=4", AUDIT / "e2_longitudinal_compensated_p3/E2_binomial_passes.csv", "4", "PASS"),
    ]
    rows = []
    for condition, method, setting, path, passes, raw in specs:
        source = find_row(path, passes=passes)
        duration = f(source["fit_t_stop"]) - f(source["fit_t_start"])
        efoldings = max(0.0, f(source["gamma"]) * duration)
        snr = f(source["fit_start_snr"])
        signal = "PASS" if snr is not None and snr >= 5.0 and efoldings >= MIN_EFOLDINGS else "FAIL"
        row = make_row(
            condition=condition, benchmark="E2 near-marginal", method=method,
            setting=setting, measured_mode="(12,0)", protected_scope="longitudinal (12,0)",
            model="PASS", ratio=f(source["gamma_ratio_to_cic"]),
            lo=f(source["ratio_ci_lo"]), hi=f(source["ratio_ci_hi"]),
            start=f(source["fit_t_start"]), stop=f(source["fit_t_stop"]),
            efoldings=efoldings, r2=f(source["fit_r_squared"]), snr=snr,
            signal=signal, coverage="PASS", raw=raw,
            evidence=str(path.relative_to(WORKSPACE)),
            note="actual current-rate e-foldings recomputed; archived fit_efoldings repeats CIC-window value",
        )
        rows.append(row)

    predictions = read_csv(Path(__file__).resolve().parent / "results/dispersion_audit/filter_rate_predictions.csv")
    corner_path = (
        RECOVERED / "e2_compensated_corner_5195640" / "analysis"
        / "E2_compensated_corner.csv"
    )
    corner_baseline = find_row(corner_path, label="base")
    corner_baseline_duration = duration_gate(f(corner_baseline["model_efoldings"]))
    for condition, label, order, passes in [
        ("e2_comp_p1_n1_corner", "p1n1", "1", "1"),
        ("e2_comp_p2_n2_corner", "p2n2", "2", "2"),
        ("e2_comp_p2_n4_corner", "p2n4", "2", "4"),
        ("e2_comp_p3_n4_corner", "p3n4", "3", "4"),
    ]:
        match = next(
            row for row in predictions
            if row["evaluated_mode"] == "(12,1)"
            and row["compensation_order"] == order and row["passes"] == passes
            and row["theory_level"] == "grid_time_discrete"
        )
        source = find_row(corner_path, label=label)
        measured = f(source["gamma_ratio_to_cic"])
        predicted = f(match["predicted_rate_ratio"])
        prediction_inside_ci = f(source["ratio_ci_lo"]) <= predicted <= f(source["ratio_ci_hi"])
        rows.append(make_row(
            condition=condition, benchmark="E2 near-marginal",
            method="compensated binomial", setting=f"p={order},n={passes}",
            measured_mode="(12,1)", protected_scope="binding corner (12,1)",
            model="PASS" if f(source["max_delta_f_ratio_fit_window"]) < 0.5 else "FAIL",
            ratio=measured, lo=f(source["ratio_ci_lo"]),
            hi=f(source["ratio_ci_hi"]), start=f(source["fit_t_start"]),
            stop=f(source["fit_t_stop"]), efoldings=f(source["model_efoldings"]),
            r2=f(source["fit_r_squared"]), snr=f(source["fit_start_snr"]),
            signal="PASS" if source["signal_gate"] == "PASS" else "FAIL",
            coverage="PASS", raw="PASS", evidence=str(corner_path.relative_to(WORKSPACE)),
            baseline_duration=corner_baseline_duration,
            note=("direct paired projected-mode measurement; independently verified-transfer "
                  f"dispersion prediction is {predicted:.6f}, "
                  f"{100.0 * abs(predicted / measured - 1.0):.3f}% from the measured central "
                  f"ratio and {'inside' if prediction_inside_ci else 'outside'} its 95% interval; "
                  "prediction calibration and measured 10% rate preservation are separate; "
                  "shared-window CIC baseline spans 4.484 e-foldings"),
            prediction=predicted,
        ))
    return rows


def e2_gaussian_rows() -> list[dict[str, Any]]:
    path = RECOVERED_M12_REDUCTION
    raw_root = RECOVERED_20260911 / "df_m12_sigma_4872745"
    if len(list(raw_root.glob("sigma_*/seed*/mode_history.npz"))) != 50:
        raise AssertionError("recovered m12 Gaussian sweep is incomplete")
    rows = []
    for condition, sigma in [("e2_gaussian_sigma01845", "0.18445488487247555"),
                             ("e2_gaussian_sigma01987", "0.19872036040909125")]:
        source = find_row(path, sigma_dx=sigma)
        duration = f(source["fit_t_stop"]) - f(source["fit_t_start"])
        efoldings = f(source["gamma"]) * duration
        rows.append(make_row(
            condition=condition, benchmark="E2 near-marginal", method="isotropic Gaussian",
            setting=f"sigma={sigma}", measured_mode="(12,0)", protected_scope="longitudinal (12,0)",
            model="PASS" if f(source["max_delta_f_ratio"]) < 0.5 else "FAIL",
            ratio=f(source["gamma_ratio"]), lo=f(source["ratio_ci_lo"]),
            hi=f(source["ratio_ci_hi"]), start=f(source["fit_t_start"]),
            stop=f(source["fit_t_stop"]), efoldings=efoldings, r2=None,
            snr=f(source["fit_start_snr"]), signal="PASS" if f(source["fit_start_snr"]) >= 5 else "FAIL",
            coverage="PASS", raw="PASS", evidence=str(path.relative_to(WORKSPACE)),
            note=("all 50 per-seed histories recovered from MeluXina job 4872745; "
                  "sigma=0 is the dereferenced job-4872599 baseline"),
        ))
    return rows


def reference_rows() -> list[dict[str, Any]]:
    path = RECOVERED_REFERENCE_REDUCTION
    raw_root = RECOVERED_20260911 / "df_linear_ref_4872599"
    if len(list(raw_root.glob("**/mode_history.npz"))) != 30:
        raise AssertionError("recovered delta-f reference ensemble is incomplete")
    predictions = read_csv(Path(__file__).resolve().parent / "results/dispersion_audit/dispersion_predictions.csv")
    specs = [
        ("df_landau_reference", "landau", "1", "weak_landau_m1", "(1,0)"),
        ("df_e2_m8_reference", "two_stream", "8", "E2_m8", "(8,0)"),
        ("df_e2_m12_reference", "two_stream", "12", "E2_m12", "(12,0)"),
    ]
    rows = []
    for condition, benchmark, mode, case, label in specs:
        source = find_row(path, benchmark=benchmark, mode=mode)
        prediction = next(
            row for row in predictions
            if row["case"] == case and row["level"] == "grid_time_discrete"
        )
        theory = abs(f(prediction["lambda_real_rate"]))
        measured = abs(f(source["gamma"]))
        a, b = abs(f(source["gamma_ci_lo"])), abs(f(source["gamma_ci_hi"]))
        lo, hi = min(a, b) / theory, max(a, b) / theory
        rows.append(make_row(
            condition=condition,
            benchmark="weak Landau" if benchmark == "landau" else "E2 reference",
            method="linearized delta-f reference", setting=f"mode={mode}",
            measured_mode=label, protected_scope=label, model="PASS",
            ratio=measured / theory, lo=lo, hi=hi,
            start=f(source["fit_t_start"]), stop=f(source["fit_t_stop"]),
            efoldings=f(source["fit_efoldings"]), r2=f(source["fit_r2"]), snr=None,
            signal="PASS" if source["fit_quality_pass"] == "True" else "FAIL",
            coverage="PASS", raw="PASS", evidence=str(path.relative_to(WORKSPACE)),
            note=("rate ratio is measured absolute rate divided by verified grid/time prediction; "
                  "all 30 raw reference histories recovered from MeluXina job 4872599"),
        ))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    args.output.mkdir(parents=True)
    rows = (
        weak_gaussian_rows() + weak_confirmatory_v2_rows() + weak_filter_rows() + strong_rows()
        + e2_summary_rows() + e2_binomial_rows() + e2_gaussian_rows()
        + reference_rows()
    )
    with (args.output / "validation_status.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "criteria": {
            "rate": "two-sided 95% rate-ratio interval wholly inside [0.90,1.10]",
            "duration": "at least 2.0 model e-foldings for the measured arm",
            "signal": "fit-specific quality; E2 amplitude SNR >=5 and >=2 current-rate e-foldings where available",
            "overall": "PASS requires model, rate, duration, signal, protected-mode coverage and raw traceability all PASS",
            "prediction_policy": "predictions are reported separately and never satisfy a measurement gate",
            "baseline_duration": (
                "The fresh v2 confirmation preregistered a separate two-e-folding check for both "
                "arms because their censoring windows differ. Retrospective NOT_APPLICABLE values "
                "mean that the older candidate-only schema had no separate baseline column; they "
                "must not be read as automatic baseline passes. For the E2 Gaussian corner the "
                "shared-window CIC baseline is measured and passes with 4.484 e-foldings."
            ),
        },
        "counts": {status: sum(row["overall_status"] == status for row in rows)
                   for status in ("PASS", "FAIL", "UNRESOLVED")},
        "weak_landau_overall_passes": [
            row["condition"] for row in rows
            if row["benchmark"] == "weak Landau" and row["overall_status"] == "PASS"
        ],
        "e2_binding_corner_overall_passes": [
            row["condition"] for row in rows
            if "binding corner" in row["protected_scope"] and row["overall_status"] == "PASS"
        ],
        "rows": len(rows),
    }
    (args.output / "validation_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
