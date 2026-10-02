#!/usr/bin/env python3
"""Reduce the direct-width weak-Landau PPC convergence campaign."""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from paper1_selector_comparison.figures.paperstyle import COLORS, DOUBLE_COL, apply_style
from pic_kde_2d.benchmarks.theory import landau_gamma_theory
from pic_kde_2d.validation.landau_censored_fit import (
    coherent_amplitude,
    fit_censored_mode,
    geometric_envelope,
)


PPCS = (8, 32, 128, 512, 1024, 4096)
CHOICES = ("cic", "c4", "mise", "lscv")
COLOR = {"cic": COLORS["CIC"], "c4": COLORS["C4"], "mise": COLORS["MISE"], "lscv": COLORS["LOO_CV"]}
LABEL = {"cic": "CIC", "c4": "C4", "mise": "MISE-optimal", "lscv": "LOO-CV"}


def load_group(root: pathlib.Path, ppc: int, choice: str):
    histories, energies, metadata = [], [], []
    times = None
    for seed in range(10):
        shard = root / f"ppc{ppc}" / choice / f"seed{seed}"
        metadata.append(json.loads((shard / "metadata.json").read_text()))
        with np.load(shard / "history.npz") as data:
            current = np.asarray(data["times"], dtype=np.float64)
            histories.append(np.asarray(data["mode"], dtype=np.complex128))
            energies.append(np.asarray(data["field_energy"], dtype=np.float64))
        if times is None:
            times = current
        else:
            np.testing.assert_array_equal(times, current)
    if len({item["seed"] for item in metadata}) != 10:
        raise RuntimeError(f"{ppc}/{choice}: seed duplication")
    return times, np.stack(histories), np.stack(energies), metadata


def gamma_bootstrap(histories, times, nboot, seed):
    rng = np.random.default_rng(seed)
    gammas, strict = [], 0
    for _ in range(nboot):
        indices = rng.integers(0, histories.shape[0], size=histories.shape[0])
        fit = fit_censored_mode(histories[indices], times)
        strict_fit = fit_censored_mode(
            histories[indices], times, min_r_squared=0.98, min_model_efoldings=2.0
        )
        if np.isfinite(fit["gamma"]):
            gammas.append(fit["gamma"])
            strict += int(strict_fit["quality_pass"])
    values = np.asarray(gammas)
    return {
        "gamma_ci_lo": float(np.percentile(values, 2.5)) if values.size else float("nan"),
        "gamma_ci_hi": float(np.percentile(values, 97.5)) if values.size else float("nan"),
        "bootstrap_valid_fraction": values.size / float(nboot),
        "bootstrap_strict_fraction": strict / float(nboot),
    }


def field_l2_error(histories, energies, times, fit, theory_gamma, area):
    field_norm = np.sqrt(2.0 * np.mean(energies, axis=0) / area)
    start = float(fit["fit_t_start"])
    stop = float(fit["fit_t_stop"])
    mask = (times >= start) & (times <= stop)
    if not np.any(mask):
        return float("nan")
    anchor = int(np.flatnonzero(mask)[0])
    analytic = field_norm[anchor] * np.exp(theory_gamma * (times - times[anchor]))
    return float(
        np.sqrt(np.mean((field_norm[mask] - analytic[mask]) ** 2))
        / np.sqrt(np.mean(analytic[mask] ** 2))
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=pathlib.Path, required=True)
    parser.add_argument("--selector-root", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--nboot", type=int, default=2000)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    theory = float(landau_gamma_theory(0.4, 1.0))
    area = (2.0 * np.pi / 0.4) * 8.0
    rows = []
    for ppc in PPCS:
        for choice_index, choice in enumerate(CHOICES):
            times, histories, energies, metadata = load_group(args.input, ppc, choice)
            fit = fit_censored_mode(histories, times)
            strict = fit_censored_mode(histories, times, min_r_squared=0.98, min_model_efoldings=2.0)
            bootstrap = gamma_bootstrap(histories, times, args.nboot, 82000 + ppc + choice_index)
            amplitude = coherent_amplitude(histories)
            envelope, _ = geometric_envelope(amplitude, times, 4.889)
            floor = float(fit["noise_floor"])
            resolved = float(max(0.0, np.log(envelope[0] / max(5.0 * floor, 1.0e-300))))
            gamma = float(fit["gamma"])
            gamma_interval = np.asarray(
                [bootstrap["gamma_ci_lo"], bootstrap["gamma_ci_hi"]], dtype=np.float64
            )
            rate_interval = np.abs(gamma_interval - theory) / abs(theory) * 100.0
            rate_ci_low = (
                0.0
                if np.all(np.isfinite(gamma_interval))
                and gamma_interval[0] <= theory <= gamma_interval[1]
                else float(np.nanmin(rate_interval))
            )
            rows.append(
                {
                    "ppc": ppc,
                    "choice": choice,
                    "sigma_dx": float(metadata[0]["sigma_dx"]),
                    "width_source": metadata[0]["width_source"],
                    "gamma": gamma,
                    "gamma_ci_lo": bootstrap["gamma_ci_lo"],
                    "gamma_ci_hi": bootstrap["gamma_ci_hi"],
                    "rate_error_percent": 100.0 * abs(gamma - theory) / abs(theory) if np.isfinite(gamma) else float("nan"),
                    "rate_error_ci_lo": rate_ci_low,
                    "rate_error_ci_hi": float(np.nanmax(rate_interval)),
                    "rel_field_l2_error": field_l2_error(histories, energies, times, fit, theory, area),
                    "resolved_efoldings": resolved,
                    "noise_floor": floor,
                    "fit_status": fit["status"],
                    "fit_quality_pass": fit["quality_pass"],
                    "strict_quality_pass": strict["quality_pass"],
                    "fit_r_squared": fit["r_squared"],
                    "fit_t_start": fit["fit_t_start"],
                    "fit_t_stop": fit["fit_t_stop"],
                    "bootstrap_valid_fraction": bootstrap["bootstrap_valid_fraction"],
                    "bootstrap_strict_fraction": bootstrap["bootstrap_strict_fraction"],
                }
            )
    with (args.output / "landau_ppc_convergence.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result = {
        "rows": rows,
        "selector_root": str(args.selector_root),
        "widths_measured_directly": True,
        "width_scaling_law_used": False,
        "bump_on_tail": "deferred: no equilibrium initializer or kinetic dielectric branch in current model",
    }
    (args.output / "landau_ppc_convergence.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    apply_style()
    figure, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL, 2.8))
    for choice in CHOICES:
        selected = [row for row in rows if row["choice"] == choice]
        x = np.asarray([row["ppc"] for row in selected])
        rate = np.asarray([row["rate_error_percent"] for row in selected])
        efold = np.asarray([row["resolved_efoldings"] for row in selected])
        rate_low = np.asarray([row["rate_error_ci_lo"] for row in selected])
        rate_high = np.asarray([row["rate_error_ci_hi"] for row in selected])
        axes[0].errorbar(
            x,
            rate,
            yerr=np.vstack((np.maximum(0.0, rate - rate_low), np.maximum(0.0, rate_high - rate))),
            marker="o",
            color=COLOR[choice],
            label=LABEL[choice],
            capsize=2,
        )
        axes[1].plot(x, efold, marker="o", color=COLOR[choice], label=LABEL[choice])
        for axis, values in ((axes[0], rate), (axes[1], efold)):
            for ppc_value, value, row in zip(x, values, selected):
                if not row["fit_quality_pass"]:
                    axis.scatter([ppc_value], [value], facecolors="none", edgecolors=COLOR[choice], zorder=4)
    axes[0].axhline(15.0, color=COLORS["MISE"], linestyle="--", label="15% theory gate")
    axes[0].set_ylabel(r"$|\gamma-\gamma_{th}|/|\gamma_{th}|$ (%)")
    axes[0].set_title("Rate fidelity")
    axes[1].set_ylabel("resolved e-foldings above 5x floor")
    axes[1].set_title("Measurability")
    for axis in axes:
        axis.set_xscale("log", base=2)
        axis.set_xlabel("particles per cell")
    axes[0].legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(args.output / "fig_ppc_regime_map.pdf")
    figure.savefig(args.output / "fig_ppc_regime_map.png")
    plt.close(figure)
    print(json.dumps({"n_rows": len(rows), "output": str(args.output)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
