#!/usr/bin/env python3
"""Rebuild the affected Paper-1 completion tables/figures from cached raw data."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


HERE = Path(__file__).resolve().parent
DOWNLOADS = HERE.parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(command: list[str], env: dict[str, str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=DOWNLOADS, env=env, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_root.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.mkdir(parents=True)
    mpl_cache = output / "_matplotlib_cache"
    mpl_cache.mkdir()
    env = os.environ.copy()
    env.update({"MPLBACKEND": "Agg", "MPLCONFIGDIR": str(mpl_cache), "XDG_CACHE_HOME": str(mpl_cache)})
    py = sys.executable

    config = HERE / "results/e2_beam_drift_scan_design_20260911/frozen_campaign.json"
    scan_root = DOWNLOADS / "pic_kde_2d/results_raw_rate_audit/recovered_meluxina_20260911/e2_beam_drift_scan_5201231/runs"
    extension = DOWNLOADS / "pic_kde_2d/results_raw_rate_audit/recovered_meluxina_20260912/e2_beam_drift_scan_extension_5202227"
    historical = DOWNLOADS / "pic_kde_2d/results_raw_rate_audit/recovered_meluxina_20260909/e2_joint_corner_4913408"
    compensated = DOWNLOADS / "pic_kde_2d/results_raw_rate_audit/recovered_meluxina_20260909/e2_compensated_corner_5195640"
    predictions = HERE / "results/evidence_closure_20260911/e2_compensated_prediction_accuracy.csv"
    dense_noise = DOWNLOADS / "pic_kde_2d/results_raw_rate_audit/recovered_meluxina_20260909/e2_anisotropic_dense_5197174"

    run([
        py, str(HERE / "audit_exact_filter_and_spectral_operators.py"),
        "--output", str(output / "operator_transfer"),
        "--spectral-source", str(DOWNLOADS / "pic_kde_2d_landau_cv_stage/pic_kde_2d/methods/spectral_W.py"),
        "--dispersion-source", str(HERE / "recompute_dispersion.py"),
    ], env)
    run([
        py, str(HERE / "analyze_E2_beam_drift_scan.py"),
        "--config", str(config), "--input", str(scan_root),
        "--extension-input", str(extension), "--output", str(output / "drift_scan"),
    ], env)
    registered = output / "drift_scan/E2_beam_drift_scan.json"
    run([
        py, str(HERE / "complete_E2_estimator_comparison.py"),
        "--config", str(config), "--scan-root", str(scan_root),
        "--extension-root", str(extension), "--registered", str(registered),
        "--historical-root", str(historical), "--compensated-root", str(compensated),
        "--compensated-predictions", str(predictions), "--output", str(output / "estimator"),
    ], env)
    run([
        py, str(HERE / "diagnose_E2_seed_rate_variation.py"),
        "--config", str(config), "--scan-root", str(scan_root),
        "--extension-root", str(extension), "--registered", str(registered),
        "--historical-root", str(historical), "--compensated-root", str(compensated),
        "--estimator-results", str(output / "estimator"), "--output", str(output / "seed_diagnostic"),
    ], env)
    run([
        py, str(HERE / "measure_E2_interior_noise_replay.py"),
        "--saved-root", str(dense_noise), "--output", str(output / "interior_noise"),
        "--nboot", "5000",
    ], env)
    run([
        py, str(HERE / "audit_E2_vb0818_extension.py"),
        "--extension", str(extension), "--combined-analysis", str(output / "drift_scan"),
        "--output", str(output / "extension_audit"),
    ], env)
    run([
        py, str(HERE / "make_E2_estimator_seed_diagnostic.py"),
        "--estimator-summary", str(output / "estimator/estimator_summary.csv"),
        "--finite-marker", str(output / "seed_diagnostic/finite_marker_linear_response.csv"),
        "--registered", str(registered), "--output", str(output / "estimator_figure"),
    ], env)

    products = sorted(
        path for path in output.rglob("*")
        if path.is_file() and "_matplotlib_cache" not in path.parts
    )
    manifest = {
        "status": "PASS_AFFECTED_RESULTS_REBUILT_FROM_CACHED_RAW_DATA",
        "simulation_submitted": False,
        "product_count": len(products),
        "products_sha256": {
            str(path.relative_to(output)): sha256(path) for path in products
        },
    }
    (output / "rebuild_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": manifest["status"], "product_count": len(products)}, indent=2))


if __name__ == "__main__":
    main()
