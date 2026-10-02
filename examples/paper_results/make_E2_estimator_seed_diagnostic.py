#!/usr/bin/env python3
"""Make the post-hoc E2 estimator/finite-marker diagnostic figure."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np


HERE = Path(__file__).resolve().parent
FIGURE_HELPERS = HERE.parents[1] / "pic_kde_2d/paper1_selector_comparison/figures"
if str(FIGURE_HELPERS) not in sys.path:
    sys.path.insert(0, str(FIGURE_HELPERS))


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--estimator-summary", type=Path, required=True)
    parser.add_argument("--finite-marker", type=Path, required=True)
    parser.add_argument("--registered", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")

    estimator = read_csv(args.estimator_summary)
    marker = read_csv(args.finite_marker)
    registered = json.loads(args.registered.read_text(encoding="utf-8"))
    registered_rows = {
        (float(row["v_beam"]), row["arm"]): row for row in registered["rows"]
    }

    summary_rows = []
    for row in estimator:
        if row["campaign"] != "registered_drift_scan_5201231" or row["arm"] != "cic":
            continue
        v_beam = float(row["v_beam"])
        marker_row = next(
            item
            for item in read_csv(args.finite_marker.with_name("finite_marker_response_summary.csv"))
            if item["marker_group"] == "current_drift_seeds_200_209"
            and item["arm"] == "cic"
            and float(item["v_beam"]) == v_beam
        )
        reg = registered_rows[(v_beam, "cic")]
        summary_rows.append(
            {
                "v_beam": v_beam,
                "registered_mean_amplitude_gamma": float(row["registered_mean_amplitude_gamma"]),
                "mean_individual_gamma": float(row["individual_gamma_mean"]),
                "finite_marker_gamma_mean": float(marker_row["finite_marker_gamma_mean"]),
                "signal_gate": reg["signal_gate"],
                "analysis_role": "POST_HOC_ESTIMATOR_AND_FINITE_MARKER_DIAGNOSTIC",
            }
        )
    summary_rows.sort(key=lambda row: row["v_beam"])
    np.testing.assert_allclose(
        [row["registered_mean_amplitude_gamma"] for row in summary_rows if row["v_beam"] == 0.8],
        [0.14474589901399387],
        rtol=0.0,
        atol=1.0e-12,
    )

    seed_rows = []
    for row in marker:
        if (
            row["marker_group"] == "current_drift_seeds_200_209"
            and row["arm"] == "cic"
            and float(row["v_beam"]) in (0.8, 0.818)
            and row["finite_marker_root_status"] == "CONVERGED"
        ):
            seed_rows.append(
                {
                    "v_beam": float(row["v_beam"]),
                    "seed": int(row["seed"]),
                    "observed_individual_gamma": float(row["observed_individual_gamma"]),
                    "finite_marker_gamma": float(row["finite_marker_diagonal_gamma"]),
                    "signal_scope": (
                        "registered comparison signal-valid"
                        if float(row["v_beam"]) == 0.8
                        else "registered comparison signal-invalid; descriptive only"
                    ),
                }
            )
    if len(seed_rows) != 20:
        raise AssertionError("expected ten paired finite-marker roots at each of two drifts")

    args.output.mkdir(parents=True)
    write_csv(args.output / "estimator_by_drift.csv", summary_rows)
    write_csv(args.output / "finite_marker_seed_comparison.csv", seed_rows)

    import matplotlib.pyplot as plt

    try:
        from paperstyle import COLORS, DOUBLE_COL, apply_style

        apply_style()
        width = DOUBLE_COL
        navy = COLORS["CIC"]
        green = COLORS["C4"]
    except ImportError:
        plt.rcParams.update(
            {
                "font.family": "serif",
                "font.size": 9,
                "axes.labelsize": 9,
                "xtick.labelsize": 8,
                "ytick.labelsize": 8,
                "axes.spines.top": False,
                "axes.spines.right": False,
                "axes.grid": True,
                "grid.alpha": 0.25,
            }
        )
        width, navy, green = 7.0, "#14213D", "#2A7F62"
    red = "#8B1E1E"
    grey = "#737373"

    fig, axes = plt.subplots(1, 2, figsize=(width, 2.8))
    valid = [row for row in summary_rows if row["signal_gate"] == "PASS"]
    invalid = [row for row in summary_rows if row["signal_gate"] != "PASS"]
    x = np.asarray([row["v_beam"] for row in valid])
    axes[0].plot(
        x,
        [row["registered_mean_amplitude_gamma"] for row in valid],
        "o-",
        color=navy,
        label=r"$\ln\langle |\widehat E_s|\rangle_s$ slope",
    )
    axes[0].plot(
        x,
        [row["mean_individual_gamma"] for row in valid],
        "s-",
        color=red,
        label="mean individual slope",
    )
    axes[0].plot(
        x,
        [row["finite_marker_gamma_mean"] for row in valid],
        "^--",
        color=green,
        label="finite-marker response",
    )
    for row in invalid:
        axes[0].axvline(row["v_beam"], color=grey, linestyle=":", linewidth=0.8)
        axes[0].text(
            row["v_beam"] - 0.002,
            0.98,
            "unresolved",
            transform=axes[0].get_xaxis_transform(),
            rotation=90,
            va="top",
            ha="right",
            fontsize=7,
            color=grey,
        )
    axes[0].set_xlabel(r"beam drift $v_b$")
    axes[0].set_ylabel(r"growth-rate estimator $\gamma/\omega_p$")
    axes[0].set_title("(a) Estimator dependence")
    axes[0].legend(frameon=False, fontsize=7.1)

    symbols = {0.8: ("o", navy, "0.800 (signal-valid)"), 0.818: ("x", red, "0.818 (descriptive)")}
    all_values = []
    for value, (symbol, color, label) in symbols.items():
        selected = [row for row in seed_rows if row["v_beam"] == value]
        predicted = np.asarray([row["finite_marker_gamma"] for row in selected])
        observed = np.asarray([row["observed_individual_gamma"] for row in selected])
        all_values.extend(predicted)
        all_values.extend(observed)
        corr = float(np.corrcoef(predicted, observed)[0, 1])
        axes[1].scatter(
            predicted,
            observed,
            marker=symbol,
            color=color,
            s=24,
            linewidths=1.0,
            label=rf"$v_b={label}$, $r={corr:.3f}$",
        )
    lower, upper = min(all_values) - 0.006, max(all_values) + 0.006
    axes[1].plot([lower, upper], [lower, upper], "--", color=grey, linewidth=1.0)
    axes[1].set_xlim(lower, upper)
    axes[1].set_ylim(lower, upper)
    axes[1].set_xlabel(r"finite-marker predicted $\gamma/\omega_p$")
    axes[1].set_ylabel(r"individual fitted $\gamma/\omega_p$")
    axes[1].set_title("(b) Loading explains seed spread")
    axes[1].legend(frameon=False, fontsize=7.0, loc="upper left")

    fig.tight_layout(w_pad=1.2)
    for suffix, kwargs in (("pdf", {}), ("svg", {}), ("png", {"dpi": 300})):
        fig.savefig(args.output / f"fig_E2_estimator_seed_diagnostic.{suffix}", **kwargs)
    plt.close(fig)

    result = {
        "status": "POST_HOC_DIAGNOSTIC_COMPLETE_REGISTERED_RESULTS_UNCHANGED",
        "scope": (
            "Panel (a) omits the signal-invalid v_b=0.818 rates; panel (b) shows them "
            "only as descriptive finite-marker diagnostics."
        ),
        "source_tables": ["estimator_by_drift.csv", "finite_marker_seed_comparison.csv"],
    }
    (args.output / "figure_manifest.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
