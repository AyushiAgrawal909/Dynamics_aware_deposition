#!/usr/bin/env python3
"""Rebuild Paper 1 rate fits and seed-bootstrap intervals from raw histories.

This driver never edits the historical campaign directories.  It invokes the
recovered campaign analyzers into a new output tree, compares their tabular
results with the archived reductions, and records the exact commands and raw
input inventory.  It intentionally reports unavailable raw campaigns instead
of treating manuscript tables or transfer predictions as measurements.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np


WORKSPACE = Path(__file__).resolve().parents[2]
RAW_ROOT = WORKSPACE / "pic_kde_2d" / "results_raw_rate_audit"
RECOVERED = RAW_ROOT / "recovered_meluxina_20260909"
STAGE = WORKSPACE / "pic_kde_2d_landau_cv_stage"
GAUSSIAN_ANALYZER = WORKSPACE / "pic_kde_2d" / "scripts" / "run_landau_sigma_refinement.py"
BINOMIAL_ANALYZER = STAGE / "scripts" / "analyze_landau_binomial_passes.py"
E2_ANALYZER = STAGE / "scripts" / "analyze_E2_binomial_passes.py"
STRONG_ANALYZER = STAGE / "scripts" / "analyze_strong_landau_fidelity.py"
JOINT_ANALYZER = WORKSPACE / "pic_kde_2d" / "scripts" / "analyze_E2_joint_corner_sweep.py"


def _campaigns(output_root: Path) -> list[dict[str, Any]]:
    """Return exact recovered reduction commands and archived comparison files."""

    py = sys.executable

    def row(name: str, analyzer: Path, args: list[str], reference: Path,
            product: str, observable: str, mode: str) -> dict[str, Any]:
        return {
            "name": name,
            "command": [py, str(analyzer), *args, "--output", str(output_root / name)],
            "reference": reference,
            "product": product,
            "observable": observable,
            "mode": mode,
        }

    return [
        row(
            "weak_gaussian_grid",
            GAUSSIAN_ANALYZER,
            ["analyze", "--input", str(RAW_ROOT / "landau_sigma_refinement_4864819/shards"),
             "--ppcs", "32,1024", "--sigma-grid", "0,0.3,0.5,1,2,3,5,8,12",
             "--seeds", "10", "--nboot", "1000"],
            RAW_ROOT / "landau_sigma_refinement_4864819/analysis/landau_sigma_refinement.csv",
            "landau_sigma_refinement.csv", "abs(mean_seed(complex Ex_hat(1,0)))", "(1,0)",
        ),
        row(
            "weak_gaussian_mise6",
            GAUSSIAN_ANALYZER,
            ["analyze", "--input", str(RAW_ROOT / "landau_mise6_rate_4872322/shards"),
             "--ppcs", "1024", "--sigma-grid", "0,6", "--seeds", "10", "--nboot", "2000"],
            RAW_ROOT / "landau_mise6_rate_4872322/analysis/landau_sigma_refinement.csv",
            "landau_sigma_refinement.csv", "abs(mean_seed(complex Ex_hat(1,0)))", "(1,0)",
        ),
        row(
            "weak_gaussian_cv6p61",
            GAUSSIAN_ANALYZER,
            ["analyze", "--input", str(RAW_ROOT / "landau_cv_rate_5007754/shards"),
             "--ppcs", "1024", "--sigma-grid", "0,6.61", "--seeds", "10", "--nboot", "2000"],
            RAW_ROOT / "landau_cv_rate_5007754/analysis/landau_sigma_refinement.csv",
            "landau_sigma_refinement.csv", "abs(mean_seed(complex Ex_hat(1,0)))", "(1,0)",
        ),
        row(
            "weak_gaussian_silverman_recovered",
            GAUSSIAN_ANALYZER,
            ["analyze", "--input", str(RECOVERED / "p1_landau_rates_4866573/combined"),
             "--ppcs", "1024", "--sigma-grid", "0,1.8329728493314705,60",
             "--seeds", "10", "--nboot", "1000"],
            RECOVERED / "p1_landau_rates_4866573/analysis/landau_sigma_refinement.csv",
            "landau_sigma_refinement.csv", "abs(mean_seed(complex Ex_hat(1,0)))", "(1,0)",
        ),
        row(
            "weak_raw_binomial",
            BINOMIAL_ANALYZER,
            ["--input", str(RAW_ROOT / "landau_binomial_weak_5048412/shards"),
             "--alpha", "0.01", "--seeds", "10", "--nboot", "2000",
             "--filter-family", "binomial", "--passes", "0", "1", "2", "4", "8", "12", "13", "16"],
            RAW_ROOT / "landau_binomial_weak_5048412/analysis/landau_binomial_passes.csv",
            "landau_binomial_passes.csv", "abs(mean_seed(complex Ex_hat(1,0)))", "(1,0)",
        ),
        row(
            "weak_compensated_p1",
            BINOMIAL_ANALYZER,
            ["--input", str(RAW_ROOT / "compensated_binomial_20260811/rates/landau/shards"),
             "--alpha", "0.01", "--seeds", "10", "--nboot", "2000",
             "--filter-family", "compensated_binomial", "--compensation-order", "1",
             "--passes", "0", "1", "2", "4", "8", "16"],
            RAW_ROOT / "compensated_binomial_20260811/analysis/landau_rate/landau_binomial_passes.csv",
            "landau_binomial_passes.csv", "abs(mean_seed(complex Ex_hat(1,0)))", "(1,0)",
        ),
        row(
            "weak_compensated_p2",
            BINOMIAL_ANALYZER,
            ["--input", str(RAW_ROOT / "higher_order_compensation_20260811/rates/p2/landau/shards"),
             "--alpha", "0.01", "--seeds", "10", "--nboot", "2000",
             "--filter-family", "compensated_binomial", "--compensation-order", "2",
             "--passes", "0", "1", "2", "4"],
            RAW_ROOT / "higher_order_compensation_20260811/analysis/p2/landau_rate/landau_binomial_passes.csv",
            "landau_binomial_passes.csv", "abs(mean_seed(complex Ex_hat(1,0)))", "(1,0)",
        ),
        row(
            "weak_compensated_p3",
            BINOMIAL_ANALYZER,
            ["--input", str(RAW_ROOT / "higher_order_compensation_20260811/rates/p3/landau/shards"),
             "--alpha", "0.01", "--seeds", "10", "--nboot", "2000",
             "--filter-family", "compensated_binomial", "--compensation-order", "3",
             "--passes", "0", "1", "2", "4"],
            RAW_ROOT / "higher_order_compensation_20260811/analysis/p3/landau_rate/landau_binomial_passes.csv",
            "landau_binomial_passes.csv", "abs(mean_seed(complex Ex_hat(1,0)))", "(1,0)",
        ),
        row(
            "strong_alpha0p5_fit5",
            STRONG_ANALYZER,
            ["--input", str(RAW_ROOT / "strong_landau_fidelity_5048278/shards"),
             "--ppc", "1024", "--seeds", "10", "--nboot", "2000", "--fit-start", "5"],
            RAW_ROOT / "strong_landau_fidelity_5048278/analysis/strong_landau_fidelity.csv",
            "strong_landau_fidelity.csv", "abs(mean_seed(complex Ex_hat(1,0)))", "(1,0)",
        ),
        row(
            "strong_alpha0p5_fit3",
            STRONG_ANALYZER,
            ["--input", str(RAW_ROOT / "strong_landau_fidelity_5048278/shards"),
             "--ppc", "1024", "--seeds", "10", "--nboot", "2000", "--fit-start", "3"],
            RAW_ROOT / "strong_landau_fidelity_5048278/analysis_fit3/strong_landau_fidelity.csv",
            "strong_landau_fidelity.csv", "abs(mean_seed(complex Ex_hat(1,0)))", "(1,0)",
        ),
        row(
            "strong_alpha0p25_grid",
            STRONG_ANALYZER,
            ["--input", str(RAW_ROOT / "strong_landau_alpha025_20260811/rate_shards"),
             "--alpha", "0.25", "--cv-width", "1.3958", "--sigmas", "0,1,1.25,1.3958,1.5,2,2.453",
             "--fit-start", "3", "--fit-stop-max", "30", "--bounce-search-start", "8",
             "--bounce-search-stop", "35", "--min-duration", "5", "--min-r-squared", "0.98",
             "--min-model-efoldings", "3", "--seeds", "10", "--nboot", "2000"],
            RAW_ROOT / "strong_landau_alpha025_20260811/analysis/fidelity/strong_landau_fidelity.csv",
            "strong_landau_fidelity.csv", "abs(mean_seed(complex Ex_hat(1,0)))", "(1,0)",
        ),
        row(
            "strong_alpha0p25_cv_exact",
            STRONG_ANALYZER,
            ["--input", str(RAW_ROOT / "strong_landau_alpha025_20260811/rate_shards"),
             "--alpha", "0.25", "--cv-width", "2.8208226", "--sigmas", "0,2.8208226",
             "--fit-start", "3", "--fit-stop-max", "30", "--bounce-search-start", "8",
             "--bounce-search-stop", "35", "--min-duration", "5", "--min-r-squared", "0.98",
             "--min-model-efoldings", "3", "--seeds", "10", "--nboot", "2000"],
            RAW_ROOT / "strong_landau_alpha025_20260811/analysis/cv_width_exact/strong_landau_fidelity.csv",
            "strong_landau_fidelity.csv", "abs(mean_seed(complex Ex_hat(1,0)))", "(1,0)",
        ),
        row(
            "strong_raw_binomial_fit3_recovered",
            BINOMIAL_ANALYZER,
            ["--input", str(RECOVERED / "landau_binomial_strong_5048414/shards"),
             "--alpha", "0.5", "--seeds", "10", "--nboot", "2000",
             "--filter-family", "binomial", "--strong-fit-start", "3",
             "--passes", "0", "1", "2", "4", "8", "12", "13", "16"],
            RECOVERED / "landau_binomial_strong_5048414/analysis_fit3/landau_binomial_passes.csv",
            "landau_binomial_passes.csv", "abs(mean_seed(complex Ex_hat(1,0)))", "(1,0)",
        ),
        row(
            "strong_raw_binomial_fit5_recovered",
            BINOMIAL_ANALYZER,
            ["--input", str(RECOVERED / "landau_binomial_strong_5048414/shards"),
             "--alpha", "0.5", "--seeds", "10", "--nboot", "2000",
             "--filter-family", "binomial", "--strong-fit-start", "5",
             "--passes", "0", "1", "2", "4", "8", "12", "13", "16"],
            RECOVERED / "landau_binomial_strong_5048414/analysis/landau_binomial_passes.csv",
            "landau_binomial_passes.csv", "abs(mean_seed(complex Ex_hat(1,0)))", "(1,0)",
        ),
        row(
            "e2_raw_binomial_recovered",
            E2_ANALYZER,
            ["--input", str(RECOVERED / "E2_binomial_5048416/shards"),
             "--seeds", "10", "--nboot", "2000", "--filter-family", "binomial",
             "--passes", "0", "1", "2", "4", "8", "12", "13", "16"],
            RECOVERED / "E2_binomial_5048416/analysis/E2_binomial_passes.csv",
            "E2_binomial_passes.csv", "mean_seed(abs(Ex_hat(12,0)))", "(12,0)",
        ),
        row(
            "e2_longitudinal_compensated_p1",
            E2_ANALYZER,
            ["--input", str(RAW_ROOT / "compensated_binomial_20260811/rates/e2/shards"),
             "--seeds", "10", "--nboot", "2000", "--filter-family", "compensated_binomial",
             "--compensation-order", "1", "--passes", "0", "1", "2", "4", "8", "16"],
            RAW_ROOT / "compensated_binomial_20260811/analysis/e2_rate/E2_binomial_passes.csv",
            "E2_binomial_passes.csv", "mean_seed(abs(Ex_hat(12,0)))", "(12,0), not binding corner",
        ),
        row(
            "e2_longitudinal_compensated_p2",
            E2_ANALYZER,
            ["--input", str(RAW_ROOT / "higher_order_compensation_20260811/rates/p2/e2/shards"),
             "--seeds", "10", "--nboot", "2000", "--filter-family", "compensated_binomial",
             "--compensation-order", "2", "--passes", "0", "1", "2", "4"],
            RAW_ROOT / "higher_order_compensation_20260811/analysis/p2/e2_rate/E2_binomial_passes.csv",
            "E2_binomial_passes.csv", "mean_seed(abs(Ex_hat(12,0)))", "(12,0), not binding corner",
        ),
        row(
            "e2_longitudinal_compensated_p3",
            E2_ANALYZER,
            ["--input", str(RAW_ROOT / "higher_order_compensation_20260811/rates/p3/e2/shards"),
             "--seeds", "10", "--nboot", "2000", "--filter-family", "compensated_binomial",
             "--compensation-order", "3", "--passes", "0", "1", "2", "4"],
            RAW_ROOT / "higher_order_compensation_20260811/analysis/p3/e2_rate/E2_binomial_passes.csv",
            "E2_binomial_passes.csv", "mean_seed(abs(Ex_hat(12,0)))", "(12,0), not binding corner",
        ),
        row(
            "e2_gaussian_joint_corner_recovered",
            JOINT_ANALYZER,
            ["--rate-root", str(RECOVERED / "e2_joint_corner_4913408/rate"),
             "--noise-root", str(RECOVERED / "e2_joint_corner_4913408/noise"),
             "--seeds", "10", "--nboot", "5000", "--fit-t-start", "24.32",
             "--fit-t-stop", "59.68"],
            RECOVERED / "e2_joint_corner_4913408/analysis/joint_corner_sweep.csv",
            "joint_corner_sweep.csv", "mean_seed(abs(Ex_hat(12,1)))", "(12,1)",
        ),
    ]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _number(text: str) -> float | None:
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def compare_csv(rebuilt: Path, archived: Path) -> dict[str, Any]:
    got = _read_csv(rebuilt)
    expected = _read_csv(archived)
    result: dict[str, Any] = {
        "archived": str(archived.relative_to(WORKSPACE)),
        "archived_sha256": _sha256(archived),
        "rebuilt": str(rebuilt.relative_to(WORKSPACE)),
        "rebuilt_sha256": _sha256(rebuilt),
        "rows_archived": len(expected),
        "rows_rebuilt": len(got),
        "max_abs_numeric_difference": 0.0,
        "mismatch_count": 0,
        "examples": [],
        "added_columns_in_rebuilt": [],
        "missing_archived_columns": [],
    }
    if len(got) != len(expected):
        result["mismatch_count"] += abs(len(got) - len(expected)) + 1
    for index, (left, right) in enumerate(zip(got, expected)):
        missing = sorted(set(right) - set(left))
        added = sorted(set(left) - set(right))
        result["missing_archived_columns"] = sorted(
            set(result["missing_archived_columns"]) | set(missing)
        )
        result["added_columns_in_rebuilt"] = sorted(
            set(result["added_columns_in_rebuilt"]) | set(added)
        )
        if missing:
            result["mismatch_count"] += 1
            result["examples"].append(f"row {index}: missing archived fields {missing}")
            continue
        for key in right:
            a, b = _number(left[key]), _number(right[key])
            if a is not None and b is not None:
                if math.isnan(a) and math.isnan(b):
                    continue
                delta = abs(a - b)
                result["max_abs_numeric_difference"] = max(
                    result["max_abs_numeric_difference"], delta
                )
                # Cross-version NumPy/SciPy root and least-squares roundoff can
                # move derived response coefficients in the 11th decimal.  This
                # is a reproduction-comparison tolerance, not a physics gate;
                # the manuscript's 10% and duration criteria are untouched.
                tolerance = 1.0e-10 * max(1.0, abs(a), abs(b))
                equal = delta <= tolerance
            else:
                equal = left[key] == right[key]
            if not equal:
                result["mismatch_count"] += 1
                if len(result["examples"]) < 8:
                    result["examples"].append(
                        f"row {index}, {key}: rebuilt={left[key]!r}, archived={right[key]!r}"
                    )
    if result["mismatch_count"]:
        result["status"] = "FAIL"
    elif result["added_columns_in_rebuilt"]:
        result["status"] = "PASS_WITH_SCHEMA_EXTENSION"
    else:
        result["status"] = "PASS"
    return result


def inventory_raw() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    e2_modes: list[dict[str, Any]] = []
    for campaign in sorted(path for path in RAW_ROOT.iterdir() if path.is_dir()):
        npz_paths = sorted(campaign.rglob("*.npz"))
        rate_paths = [
            path for path in npz_paths
            if path.name in {"history.npz", "mode_history.npz"}
        ]
        seeds: list[int] = []
        lengths: set[int] = set()
        schemas: set[str] = set()
        for path in rate_paths:
            with np.load(path, allow_pickle=False) as archive:
                schemas.add("|".join(sorted(archive.files)))
                if "times" in archive:
                    lengths.add(int(len(archive["times"])))
                if "seed" in archive and archive["seed"].size:
                    seeds.append(int(archive["seed"].flat[0]))
            result_path = path.parent / "result.json"
            if result_path.exists():
                metadata = json.loads(result_path.read_text(encoding="utf-8"))
                if metadata.get("benchmark") == "two_stream":
                    e2_modes.append({
                        "file": str(path.relative_to(WORKSPACE)),
                        "kx_mode": metadata.get("kx_mode", "unknown"),
                        "ky_mode": metadata.get("ky_mode", "unknown"),
                        "ppc": metadata.get("ppc", "unknown"),
                        "alpha": metadata.get("alpha", "unknown"),
                        "dt": metadata.get("dt", "unknown"),
                    })
        rows.append({
            "campaign": campaign.name,
            "rate_npz_count": len(rate_paths),
            "seed_values": ";".join(map(str, sorted(set(seeds)))) if seeds else "",
            "trace_lengths": ";".join(map(str, sorted(lengths))),
            "schema_count": len(schemas),
            "status": "raw histories present" if rate_paths else "no raw rate histories",
        })
    return rows, e2_modes


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty table {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        help="new directory; existing paths are refused")
    parser.add_argument("--campaign", action="append", default=[],
                        help="run only named campaign(s); default is all")
    parser.add_argument("--list", action="store_true", help="list campaign names and exit")
    args = parser.parse_args()
    if args.list:
        campaigns = _campaigns(Path("OUTPUT_ROOT"))
        print("\n".join(item["name"] for item in campaigns))
        return
    if args.output is None:
        parser.error("--output is required unless --list is used")
    args.output = args.output.resolve()
    campaigns = _campaigns(args.output)
    selected = [item for item in campaigns if not args.campaign or item["name"] in args.campaign]
    unknown = sorted(set(args.campaign) - {item["name"] for item in campaigns})
    if unknown:
        raise ValueError(f"unknown campaigns: {unknown}")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {args.output}")
    args.output.mkdir(parents=True)

    inventory, e2_modes = inventory_raw()
    _write_csv(args.output / "raw_history_inventory.csv", inventory)
    _write_csv(args.output / "e2_measured_mode_inventory.csv", e2_modes)
    for row in e2_modes:
        file_name = str(row["file"])
        measured = (row["kx_mode"], row["ky_mode"])
        expected = (12, 1) if "e2_joint_corner_4913408" in file_name else (12, 0)
        if measured != expected:
            raise AssertionError(
                f"unexpected measured mode for {file_name}: {measured}, expected {expected}"
            )

    env = os.environ.copy()
    # The preserved stage copy's E2 reducer imports two bare helper modules
    # that are absent from stage/scripts but present in the supplied top-level
    # research tree.  Keep this mixed-source dependency explicit in commands
    # and in the audit report; it prevents claiming an immutable source bundle.
    helper_scripts = WORKSPACE / "pic_kde_2d" / "scripts"
    env["PYTHONPATH"] = (
        str(STAGE) + os.pathsep + str(helper_scripts) + os.pathsep
        + env.get("PYTHONPATH", "")
    )
    comparison_rows: list[dict[str, Any]] = []
    command_rows: list[dict[str, Any]] = []
    for index, campaign in enumerate(selected, start=1):
        out = args.output / campaign["name"]
        command_text = subprocess.list2cmdline(campaign["command"])
        command_rows.append({
            "campaign": campaign["name"], "command": command_text,
            "observable": campaign["observable"], "measured_mode": campaign["mode"],
        })
        print(f"[{index}/{len(selected)}] {campaign['name']}", flush=True)
        process = subprocess.run(
            campaign["command"], cwd=WORKSPACE, env=env,
            text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )
        (args.output / f"{campaign['name']}.log").write_text(
            process.stdout, encoding="utf-8"
        )
        if process.returncode:
            raise RuntimeError(
                f"{campaign['name']} failed with {process.returncode}; see its log"
            )
        compared = compare_csv(out / campaign["product"], campaign["reference"])
        comparison_rows.append({
            "campaign": campaign["name"],
            "status": compared["status"],
            "rows_rebuilt": compared["rows_rebuilt"],
            "rows_archived": compared["rows_archived"],
            "max_abs_numeric_difference": compared["max_abs_numeric_difference"],
            "mismatch_count": compared["mismatch_count"],
            "archived": compared["archived"],
            "rebuilt": compared["rebuilt"],
        })
        (args.output / f"{campaign['name']}_comparison.json").write_text(
            json.dumps(compared, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        if not compared["status"].startswith("PASS"):
            print(
                f"  recorded archived-reduction mismatch for {campaign['name']}",
                flush=True,
            )

    _write_csv(args.output / "commands.csv", command_rows)
    _write_csv(args.output / "reproduction_comparison.csv", comparison_rows)
    unresolved = [
        {
            "claim": "E2 binding-corner (12,1) compensated-filter rates",
            "status": "raw histories not recovered",
            "consequence": "longitudinal (12,0) measurements cannot certify the corner",
        },
        {
            "claim": "E2 Gaussian m=12 and oblique reference rates",
            "status": "summary only in supplied and remote result trees",
            "consequence": "archived intervals cannot be regenerated from raw output",
        },
        {
            "claim": "linearized delta-f reference and band intervals",
            "status": "summary only in supplied and remote result trees",
            "consequence": "archived intervals cannot be regenerated from raw output",
        },
    ]
    summary = {
        "rate_definition": {
            "reported_gamma": "slope of ln(field-mode amplitude); units omega_pe",
            "field_energy_rate": "2*gamma; not the quantity tabulated as gamma",
            "weak_landau_ensemble": "coherent complex seed mean, then absolute value and one-period geometric envelope",
            "strong_landau_ensemble": "coherent complex seed mean, then absolute value and one-period geometric envelope",
            "e2_ensemble": "absolute value per seed, then arithmetic seed mean",
            "uncertainty_unit": "independent seed; paired CIC/filter seed indices",
            "temporal_resampling": "none",
            "weak_window": "candidate [5,30], independently censored at persistent 5x late-floor crossing",
            "strong_window": "common paired pre-bounce stop; start is campaign parameter",
            "e2_window": "one CIC-selected common growth window for all compared filters",
        },
        "source_limitation": (
            "the preserved stage E2 analyzer requires analyze_E2_mode_growth.py "
            "and analyze_linearized_delta_f_m12_sigma.py from the supplied "
            "top-level research tree because those helpers are absent from "
            "pic_kde_2d_landau_cv_stage/scripts"
        ),
        "recovery_note": (
            "MeluXina recovery on 2026-09-09 added raw strong-binomial, raw E2-binomial, "
            "weak-Silverman, and Gaussian E2 (12,1) joint-corner campaigns; cluster originals "
            "were not changed"
        ),
        "campaigns_run": [item["name"] for item in selected],
        "all_reproduced": all(row["status"].startswith("PASS") for row in comparison_rows),
        "unresolved": unresolved,
    }
    (args.output / "audit_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
