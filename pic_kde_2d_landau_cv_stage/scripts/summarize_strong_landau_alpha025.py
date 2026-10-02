#!/usr/bin/env python3
"""Combine the alpha=0.25 LOO-CV selector and bounce-censored fidelity result."""

from __future__ import annotations

import argparse
import json
import pathlib


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cv", type=pathlib.Path, required=True)
    parser.add_argument("--fidelity", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    cv = json.loads(args.cv.read_text(encoding="utf-8"))
    fidelity = json.loads(args.fidelity.read_text(encoding="utf-8"))
    selected = float(cv["sigma_cv_quadratic_dx"])
    tested = float(fidelity["cv_width_dx"])
    row = fidelity["cv_result"]
    result = {
        "alpha": 0.25,
        "measured_dynamic_lscv_width_dx": selected,
        "lscv_width_ci_lo": cv["sigma_cv_quadratic_ci_lo"],
        "lscv_width_ci_hi": cv["sigma_cv_quadratic_ci_hi"],
        "tested_scaled_prior_width_dx": tested,
        "width_difference_dx": selected - tested,
        "tested_width_rate_result": row,
        "duration_gate": {
            "required_model_efoldings": fidelity["protocol"][
                "minimum_model_efoldings"
            ],
            "measured_model_efoldings": row["model_efoldings"],
            "passed": row["quality_pass"],
        },
        "density_selector_verdict_at_scaled_prior": fidelity["cv_verdict"],
        "interpretation_guard": (
            "The measured CV minimum is certified directly only if it lies within "
            "the tested-width resolution around the scaled prior."
        ),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "strong_landau_alpha025_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
