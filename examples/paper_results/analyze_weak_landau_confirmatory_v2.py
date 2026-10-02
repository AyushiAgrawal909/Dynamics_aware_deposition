#!/usr/bin/env python3
"""Locked two-stage reducer entry point for the global-velocity v2 campaign."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import analyze_weak_landau_confirmatory as base  # noqa: E402


DEFAULT_CONFIG = HERE / "weak_landau_confirmatory_v2_global_velocity.json"
EXPECTED_CAMPAIGN = "weak_landau_confirmatory_v2_global_velocity"


def load_config(path: Path) -> tuple[dict, Path, str]:
    resolved = path.resolve()
    config = json.loads(resolved.read_text(encoding="utf-8"))
    if config["campaign_id"] != EXPECTED_CAMPAIGN:
        raise AssertionError("unexpected v2 campaign")
    confirmation = config["confirmation"]
    if confirmation["n_seeds"] != 40 or not confirmation["fixed_sample"]:
        raise AssertionError("v2 sample is not fixed at 40")
    if not confirmation["sequential_seed_addition_forbidden"]:
        raise AssertionError("v2 sequential seed addition must be forbidden")
    if not config["benchmark"]["velocity_assignment_scope"].startswith("global"):
        raise AssertionError("v2 reducer requires global velocity assignment")
    return config, resolved, base.file_sha256(resolved)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    sub = parser.add_subparsers(dest="command", required=True)
    windows = sub.add_parser("select-windows")
    windows.add_argument("--input-low", type=Path, required=True)
    windows.add_argument("--output", type=Path, required=True)
    analyze = sub.add_parser("analyze")
    analyze.add_argument("--input-low", type=Path, required=True)
    analyze.add_argument("--input-reference", type=Path, required=True)
    analyze.add_argument("--window-decisions", type=Path, required=True)
    analyze.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config, config_path, config_hash = load_config(args.config)
    if args.command == "select-windows":
        base.select_windows(config, config_path, config_hash, args.input_low, args.output)
    else:
        base.comparative_analysis(
            config,
            config_path,
            config_hash,
            args.input_low,
            args.input_reference,
            args.window_decisions,
            args.output,
        )


if __name__ == "__main__":
    main()
