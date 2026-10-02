#!/usr/bin/env python3
"""Run one arm/seed from the frozen E2 beam-drift campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
PROJECT_CANDIDATES = (
    HERE.parent,
    HERE.parents[1] / "pic_kde_2d_landau_cv_stage",
)
PROJECT_ROOT = next(
    (
        candidate
        for candidate in PROJECT_CANDIDATES
        if (candidate / "scripts/run_linearized_delta_f_mode.py").is_file()
    ),
    None,
)
if PROJECT_ROOT is None:
    raise FileNotFoundError("could not locate the audited PIC campaign source")
SCRIPTS = PROJECT_ROOT / "scripts"
for candidate in (PROJECT_ROOT, SCRIPTS):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import run_linearized_delta_f_mode as production  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--config-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--drift-index", type=int, required=True)
    parser.add_argument(
        "--arm",
        choices=("cic", "fixed_gaussian", "constrained_gaussian"),
        required=True,
    )
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    if sha256(args.config) != args.config_sha256:
        raise AssertionError("frozen campaign hash mismatch")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config["status"] != "FROZEN_BEFORE_SIMULATION_RESULTS":
        raise AssertionError("campaign is not frozen")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    if args.seed not in config["seeds"]:
        raise ValueError("seed is not in the frozen set")
    scan = config["scan_rows"][args.drift_index]
    if args.arm == "cic":
        sigma_x, sigma_y = 0.0, 0.0
        predicted_gamma = scan["grid_time_discrete_cic_gamma"]
        predicted_ratio = 1.0
    elif args.arm == "fixed_gaussian":
        sigma_x = scan["fixed_sigma_x_dx"]
        sigma_y = scan["fixed_sigma_y_dy"]
        predicted_gamma = scan["fixed_predicted_gamma"]
        predicted_ratio = scan["fixed_predicted_rate_ratio"]
    else:
        sigma_x = scan["constrained_sigma_x_dx"]
        sigma_y = scan["constrained_sigma_y_dy"]
        predicted_gamma = scan["constrained_predicted_gamma"]
        predicted_ratio = scan["constrained_predicted_rate_ratio"]

    production_args = production.build_parser().parse_args(
        [
            "--output", str(args.output),
            "--benchmark", "two_stream",
            "--mode", "12",
            "--ky-mode", "1",
            "--seed", str(args.seed),
            "--ppc", str(config["ppc"]),
            "--alpha", str(config["alpha"]),
            "--filter-kind", "gaussian",
            "--sigma-x-dx", str(sigma_x),
            "--sigma-y-dy", str(sigma_y),
            "--Nx", str(config["grid"]["Nx"]),
            "--Ny", str(config["grid"]["Ny"]),
            "--Lx", str(config["grid"]["Lx"]),
            "--Ly", str(config["grid"]["Ly"]),
            "--v-beam", str(scan["v_beam"]),
            "--v-thermal", str(config["thermal_velocity"]["x"]),
            "--dt", str(config["dt"]),
            "--t-end", str(scan["t_end"]),
            "--diag-stride", str(config["diagnostic_stride"]),
            "--project-target-mode",
        ]
    )
    result = production.run(production_args)
    manifest = {
        "campaign_id": config["campaign_id"],
        "frozen_campaign_sha256": args.config_sha256,
        "pairing_key": f"vbeam={scan['v_beam']:.3f}/seed={args.seed}",
        "drift_index": args.drift_index,
        "arm": args.arm,
        "seed": args.seed,
        "v_beam_plus": scan["v_beam_plus"],
        "v_beam_minus": scan["v_beam_minus"],
        "sigma_x_dx": sigma_x,
        "sigma_y_dy": sigma_y,
        "predicted_gamma": predicted_gamma,
        "predicted_rate_ratio": predicted_ratio,
        "fit_t_start": scan["fit_t_start"],
        "fit_t_stop": scan["fit_t_stop"],
        "t_end": scan["t_end"],
        "grid": config["grid"],
        "ppc": config["ppc"],
        "alpha": config["alpha"],
        "dt": config["dt"],
        "diagnostic_stride": config["diagnostic_stride"],
        "target_mode_projection": config["target_mode_projection"],
        "scope": config["scope"],
        "production_result": result,
    }
    (args.output / "campaign_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
