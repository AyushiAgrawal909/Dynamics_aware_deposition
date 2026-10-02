#!/usr/bin/env python3
"""Materialize the frozen weak-Landau complex histories and envelope fits.

This is a diagnostic-only postprocessor.  It reads the already completed
Stage-1 decision and Stage-2 coherent-history artifacts, reproduces the exact
geometric envelope and fixed-window slope, and refuses to alter either input.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[1]
PROJECT = WORKSPACE / "pic_kde_2d"
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from pic_kde_2d.validation.landau_censored_fit import geometric_envelope  # noqa: E402


DEFAULT_HISTORIES = (
    HERE
    / "results/weak_landau_confirmatory_v2_stage2_analysis_20260911"
    / "coherent_ensemble_histories.npz"
)
DEFAULT_DECISIONS = (
    HERE
    / "results/weak_landau_confirmatory_v2_stage1_windows_20260911"
    / "window_decisions.json"
)
DEFAULT_CONFIG = HERE / "weak_landau_confirmatory_v2_global_velocity.json"
DEFAULT_OUTPUT = HERE / "results/weak_landau_confirmatory_v2_window_diagnostic_20260911"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--histories", type=Path, default=DEFAULT_HISTORIES)
    parser.add_argument("--decisions", type=Path, default=DEFAULT_DECISIONS)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite diagnostic directory: {output}")

    config = json.loads(args.config.read_text(encoding="utf-8"))
    stage1 = json.loads(args.decisions.read_text(encoding="utf-8"))
    if stage1["campaign_id"] != config["campaign_id"]:
        raise AssertionError("campaign mismatch between config and Stage 1")
    if stage1["rate_ratios_computed"]:
        raise AssertionError("Stage-1 file unexpectedly contains rate ratios")
    if sha256(args.config.resolve()) != stage1["config_sha256"]:
        raise AssertionError("configuration hash differs from the frozen Stage-1 value")

    with np.load(args.histories, allow_pickle=False) as archive:
        expected = {
            "times",
            "cic_coherent_complex_mode",
            "c4_sigma1_coherent_complex_mode",
            "lscv_job5008392_coherent_complex_mode",
        }
        if set(archive.files) != expected:
            raise AssertionError(f"unexpected coherent-history schema: {archive.files}")
        times = np.asarray(archive["times"], dtype=np.float64)
        modes = {
            "cic": np.asarray(archive["cic_coherent_complex_mode"], dtype=np.complex128),
            "c4_sigma1": np.asarray(
                archive["c4_sigma1_coherent_complex_mode"], dtype=np.complex128
            ),
        }

    period = float(config["fit_rule"]["envelope_period"])
    required_points = int(config["fit_rule"]["envelope_points_at_frozen_cadence"])
    products: dict[str, dict[str, np.ndarray | float | int]] = {}
    report_arms: dict[str, dict[str, object]] = {}
    for arm, mode in modes.items():
        if mode.shape != times.shape:
            raise AssertionError(f"{arm} complex mode shape does not match time")
        amplitude = np.abs(mode)
        envelope, points = geometric_envelope(amplitude, times, period)
        if points != required_points:
            raise AssertionError(f"{arm} envelope uses {points} points, expected {required_points}")
        decision = stage1["decisions"][arm]
        start = float(decision["accepted_t_start"])
        stop = float(decision["accepted_t_stop"])
        accepted = (times >= start) & (times <= stop) & np.isfinite(envelope) & (envelope > 0)
        slope, intercept = np.polyfit(times[accepted], np.log(envelope[accepted]), 1)
        if not np.isclose(slope, float(decision["gamma"]), rtol=0.0, atol=5.0e-14):
            raise AssertionError(f"{arm} slope changed: {slope} versus {decision['gamma']}")
        fit_envelope = np.full(times.shape, np.nan, dtype=np.float64)
        fit_envelope[accepted] = np.exp(intercept + slope * times[accepted])
        efoldings = abs(float(slope)) * (stop - start)
        if not np.isclose(
            efoldings,
            float(decision["fitted_amplitude_efoldings"]),
            rtol=0.0,
            atol=5.0e-13,
        ):
            raise AssertionError(f"{arm} fitted e-foldings changed")
        products[arm] = {
            "mode": mode,
            "amplitude": amplitude,
            "envelope": envelope,
            "fit_envelope": fit_envelope,
            "accepted": accepted,
        }
        report_arms[arm] = {
            "accepted_t_start": start,
            "accepted_t_stop": stop,
            "ending_reason": decision["ending_reason"],
            "ending_detail": decision["ending_detail"],
            "censor_t": decision["censor_t"],
            "noise_floor": decision["noise_floor"],
            "five_floor_threshold": 5.0 * float(decision["noise_floor"]),
            "gamma": float(slope),
            "fitted_amplitude_efoldings": efoldings,
            "fit_r_squared": decision["fit_r_squared"],
            "fit_start_signal_to_floor": decision["fit_start_signal_to_floor"],
            "signal_to_threshold_capacity_from_fit_start": decision[
                "signal_to_threshold_capacity_from_fit_start"
            ],
            "envelope_points": points,
        }

    output.mkdir(parents=True)
    csv_path = output / "complex_histories_and_envelopes.csv"
    fieldnames = ["time"]
    for arm in modes:
        fieldnames.extend(
            [
                f"{arm}_real",
                f"{arm}_imag",
                f"{arm}_amplitude",
                f"{arm}_envelope",
                f"{arm}_fitted_envelope",
                f"{arm}_accepted",
            ]
        )
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, time in enumerate(times):
            row: dict[str, object] = {"time": float(time)}
            for arm in modes:
                product = products[arm]
                mode = product["mode"]
                assert isinstance(mode, np.ndarray)
                for key, value in (
                    (f"{arm}_real", mode[index].real),
                    (f"{arm}_imag", mode[index].imag),
                    (f"{arm}_amplitude", product["amplitude"][index]),
                    (f"{arm}_envelope", product["envelope"][index]),
                    (f"{arm}_fitted_envelope", product["fit_envelope"][index]),
                    (f"{arm}_accepted", int(product["accepted"][index])),
                ):
                    row[key] = value
            writer.writerow(row)

    colors = {"cic": "#14213d", "c4_sigma1": "#9b2226"}
    labels = {"cic": "CIC", "c4_sigma1": r"C4, $\sigma=1\,\Delta x$"}
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8), sharex=True, sharey=True)
    for axis, arm in zip(axes, ("cic", "c4_sigma1")):
        product = products[arm]
        decision = report_arms[arm]
        axis.semilogy(times, product["amplitude"], color=colors[arm], alpha=0.32, lw=0.75, label=r"$|\langle \hat E_1\rangle|$")
        axis.semilogy(times, product["envelope"], color=colors[arm], lw=1.5, label="oscillation envelope")
        axis.semilogy(times, product["fit_envelope"], color="black", ls="--", lw=1.1, label="frozen fit")
        axis.axhline(decision["noise_floor"], color="0.35", ls=":", lw=1.0, label="measured floor")
        axis.axhline(decision["five_floor_threshold"], color="0.55", ls="-.", lw=0.9, label=r"$5\times$ floor")
        axis.axvspan(decision["accepted_t_start"], decision["accepted_t_stop"], color="0.5", alpha=0.10)
        axis.axvline(decision["censor_t"], color=colors[arm], ls=":", lw=0.9)
        axis.set_title(
            f"{labels[arm]}: $N_e={decision['fitted_amplitude_efoldings']:.3f}$\n"
            f"window [{decision['accepted_t_start']:.1f}, {decision['accepted_t_stop']:.1f}]"
        )
        axis.set_xlim(0.0, float(times[-1]))
        axis.grid(alpha=0.25)
        axis.spines[["top", "right"]].set_visible(False)
        axis.set_xlabel(r"$t\,\omega_p$")
    axes[0].set_ylabel(r"mode amplitude / envelope")
    axes[1].legend(frameon=False, fontsize=7, loc="upper right")
    fig.tight_layout(pad=0.7, w_pad=0.8)
    fig.savefig(output / "weak_landau_window_diagnostic.pdf")
    fig.savefig(output / "weak_landau_window_diagnostic.png", dpi=300)
    plt.close(fig)

    report = {
        "campaign_id": config["campaign_id"],
        "diagnostic_only": True,
        "rate_ratios_recomputed": False,
        "primary_reducer_operation": (
            "average the complex Fourier coefficient across all 40 seeds, take its magnitude, "
            "then apply a one-period geometric envelope"
        ),
        "inputs": {
            "config": str(args.config.resolve()),
            "config_sha256": sha256(args.config.resolve()),
            "stage1_decisions": str(args.decisions.resolve()),
            "stage1_decisions_sha256": sha256(args.decisions.resolve()),
            "stage2_histories": str(args.histories.resolve()),
            "stage2_histories_sha256": sha256(args.histories.resolve()),
        },
        "arms": report_arms,
        "artifacts": {
            "complex_history_csv": csv_path.name,
            "figure_pdf": "weak_landau_window_diagnostic.pdf",
            "figure_png": "weak_landau_window_diagnostic.png",
        },
    }
    report_path = output / "window_diagnostic.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
