#!/usr/bin/env python3
"""Fail-closed conservation precheck for the G0 locality profile.

The G0 covariance calculation uses source-dependent spectral Gaussians. The
validated grad-h audit uses the production variational wrapped-Gaussian KDE.
Before comparing force or momentum numbers, this script verifies that the
variational chain can represent the requested width without clamping or an
invalid periodic-image truncation.

If representability fails, grad-h force and integrated drift are deliberately
reported as null. Measuring the silently clamped profile would answer a
different question. Static refresh work is exactly zero because no controller
refresh occurs.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import re
import sys
from pathlib import Path
from typing import Optional

import numpy as np


def load_g0_module(script: Path):
    spec = importlib.util.spec_from_file_location("g0_locality_for_conservation", script)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {script}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def parse_constant(source: str, name: str) -> float:
    pattern = rf"{re.escape(name)}\s*=\s*([0-9]+(?:\.[0-9]*)?)"
    match = re.search(pattern, source)
    if match is None:
        raise ValueError(f"cannot find {name} in production KDE source")
    return float(match.group(1))


def parse_periodic_images(source: str) -> int:
    pattern = r"for\s*\(\s*int\s+k\s*=\s*-([0-9]+)\s*;\s*k\s*<=\s*([0-9]+)"
    match = re.search(pattern, source)
    if match is None or match.group(1) != match.group(2):
        raise ValueError("cannot find symmetric wrapped-Gaussian image loop")
    return int(match.group(1))


def periodic_image_fraction(span_cells: int, sigma_cells: float, images: int) -> float:
    """Fraction of the d=0 wrapped-Gaussian image sum retained by +/-images."""
    required = max(images + 1, int(math.ceil(8.0 * sigma_cells / span_cells)) + 1)
    indices = np.arange(-required, required + 1, dtype=np.float64)
    terms = np.exp(-0.5 * (indices * span_cells / sigma_cells) ** 2)
    kept = terms[(indices >= -images) & (indices <= images)]
    return float(np.sum(kept) / np.sum(terms))


def profile_gradients(g0, config: dict, requested_sigma: float) -> dict:
    entry = next(item for item in config["geometries"] if item["intended"])
    geometry = g0.build_geometry(entry, config["packet"])
    x = np.arange(geometry.Nx, dtype=np.float64) * geometry.dx
    _, sigma = g0.c0_log_width_profile(
        x,
        geometry.Lx,
        float(config["packet"]["protected_core_fraction_x"]),
        float(config["packet"]["blend_fraction_x"]),
        float(config["sigma_safe_cells"]),
        requested_sigma,
    )
    d_sigma_d_cell = 0.5 * (np.roll(sigma, -1) - np.roll(sigma, 1))
    log_sigma = np.log(sigma)
    d_log_sigma_d_cell = 0.5 * (np.roll(log_sigma, -1) - np.roll(log_sigma, 1))
    return {
        "Nx": geometry.Nx,
        "Ny": geometry.Ny,
        "max_abs_dsigma_dcell": float(np.max(np.abs(d_sigma_d_cell))),
        "max_abs_dlog_sigma_dcell": float(np.max(np.abs(d_log_sigma_d_cell))),
    }


def capped_gain(g0, config: dict, cap: float) -> float:
    capped = dict(config)
    capped["sigma_free_cells"] = [cap]
    entry = next(item for item in capped["geometries"] if item["intended"])
    geometry = g0.build_geometry(entry, capped["packet"])
    rows, _ = g0.analyze_geometry(capped, geometry)
    return float(rows[0]["field_noise_gain"])


def main() -> None:
    parser = argparse.ArgumentParser()
    root = Path(__file__).resolve().parents[1]
    repository = root.parents[1]
    parser.add_argument(
        "--config", type=Path, default=root / "configs" / "h2_g0_locality.json"
    )
    parser.add_argument(
        "--ceiling-result",
        type=Path,
        default=root
        / "results"
        / "G0_locality_headroom_ceiling_baseline"
        / "g0_summary.json",
    )
    parser.add_argument(
        "--kde-source",
        type=Path,
        default=repository / "source" / "kde_deposition.cpp",
    )
    parser.add_argument("--kde-max-cells", type=float)
    parser.add_argument("--periodic-images", type=int)
    parser.add_argument("--requested-sigma", type=float)
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "results" / "G0_conservation_precheck",
    )
    args = parser.parse_args()

    g0 = load_g0_module(root / "scripts" / "run_G0_locality_headroom.py")
    config = g0.load_config(args.config)
    config["sigma_safe_cells"] = 2.453
    source_text: Optional[str] = None
    if args.kde_source.exists():
        source_text = args.kde_source.read_text()
    kde_max = (
        float(args.kde_max_cells)
        if args.kde_max_cells is not None
        else parse_constant(source_text or "", "KDE_H_MAX_CELLS")
    )
    images = (
        int(args.periodic_images)
        if args.periodic_images is not None
        else parse_periodic_images(source_text or "")
    )

    requested_sigma = args.requested_sigma
    if requested_sigma is None:
        ceiling = json.loads(args.ceiling_result.read_text())
        intended = next(item for item in ceiling["summaries"] if item["intended"])
        requested_sigma = float(intended["best_admissible_sigma_free_cells"])
    if requested_sigma <= 0.0:
        raise ValueError("requested sigma must be positive")

    gradients = profile_gradients(g0, config, requested_sigma)
    span_x = gradients["Nx"] - 1
    span_y = gradients["Ny"] - 1
    x_image_fraction = periodic_image_fraction(span_x, requested_sigma, images)
    y_image_fraction = periodic_image_fraction(span_y, requested_sigma, images)
    width_representable = requested_sigma <= kde_max
    image_sum_representable = min(x_image_fraction, y_image_fraction) >= 1.0 - 1.0e-12
    representable = width_representable and image_sum_representable
    supported_gain = capped_gain(g0, config, min(requested_sigma, kde_max))

    status = "NEEDS_FORCE_AUDIT" if representable else "FAIL_UNREPRESENTABLE"
    result = {
        "status": status,
        "h2_authorized": False,
        "requested_profile": {
            "sigma_safe_cells": config["sigma_safe_cells"],
            "sigma_free_cells": requested_sigma,
            **gradients,
        },
        "production_variational_chain": {
            "kde_max_cells": kde_max,
            "periodic_images_each_side": images,
            "width_representable_without_clamp": width_representable,
            "wrapped_image_fraction_x_at_d0": x_image_fraction,
            "wrapped_image_fraction_y_at_d0": y_image_fraction,
            "periodic_image_sum_accurate": image_sum_representable,
            "field_gain_at_largest_supported_width": supported_gain,
        },
        "conservation_metrics": {
            "grad_h_force_magnitude": None,
            "integrated_momentum_drift": None,
            "static_refresh_work_signed": 0.0,
            "static_refresh_work_absolute": 0.0,
            "reason_not_measured": (
                None
                if representable
                else "the validated variational chain clamps or misrepresents the requested profile"
            ),
        },
        "decision": (
            "RUN_FORCE_AUDIT"
            if representable
            else "STOP_H2_VARIATIONAL_PROFILE_NOT_IMPLEMENTED"
        ),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "g0_conservation_precheck.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
