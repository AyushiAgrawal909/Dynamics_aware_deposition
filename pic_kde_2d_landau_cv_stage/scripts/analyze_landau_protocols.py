#!/usr/bin/env python3
"""Compare two-peak and certified Landau fits and analytic field decay."""

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

from paper1_selector_comparison.figures.paperstyle import (
    COLORS,
    DOUBLE_COL,
    apply_style,
)
from pic_kde_2d.benchmarks.theory import landau_gamma_theory
from pic_kde_2d.validation.landau_censored_fit import (
    coherent_amplitude,
    fit_censored_mode,
    geometric_envelope,
)


METHODS = {
    "CIC": ("landau_sigma_refinement_4864819/shards/ppc1024/sigma0", "CIC"),
    "C4": ("landau_sigma_refinement_4864819/shards/ppc1024/sigma1", "C4"),
    "MISE": ("landau_mise6_rate_4872322/shards/ppc1024/sigma6", "MISE"),
}


def load_histories(root: pathlib.Path) -> tuple[np.ndarray, np.ndarray, list[int]]:
    records = []
    for path in root.glob("seed*/mode_history.npz"):
        with np.load(path) as data:
            records.append(
                (
                    int(np.asarray(data["seed"]).ravel()[0]),
                    np.asarray(data["times"], dtype=np.float64),
                    np.asarray(data["mode"], dtype=np.complex128),
                )
            )
    records.sort(key=lambda item: item[0])
    if len(records) != 10:
        raise RuntimeError(f"{root}: expected 10 seeds, found {len(records)}")
    times = records[0][1]
    for _, current, _ in records[1:]:
        np.testing.assert_array_equal(times, current)
    return times, np.stack([item[2] for item in records]), [item[0] for item in records]


def first_two_peak_slope(mode_histories: np.ndarray, times: np.ndarray) -> dict:
    amplitude = coherent_amplitude(mode_histories)
    peaks = np.flatnonzero(
        (amplitude[1:-1] > amplitude[:-2])
        & (amplitude[1:-1] >= amplitude[2:])
    ) + 1
    peaks = peaks[times[peaks] > 0.0]
    if peaks.size < 2:
        return {"gamma": float("nan"), "t1": float("nan"), "t2": float("nan")}
    first, second = int(peaks[0]), int(peaks[1])
    return {
        "gamma": float(np.log(amplitude[second] / amplitude[first]) / (times[second] - times[first])),
        "t1": float(times[first]),
        "t2": float(times[second]),
        "a1": float(amplitude[first]),
        "a2": float(amplitude[second]),
    }


def bootstrap_two_peak(histories: np.ndarray, times: np.ndarray, nboot: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(nboot):
        indices = rng.integers(0, histories.shape[0], size=histories.shape[0])
        value = first_two_peak_slope(histories[indices], times)["gamma"]
        if np.isfinite(value):
            values.append(value)
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def normalized_l2_error(
    histories: np.ndarray,
    times: np.ndarray,
    fit: dict,
    theory_gamma: float,
) -> tuple[float, np.ndarray, np.ndarray]:
    amplitude = coherent_amplitude(histories)
    envelope, _ = geometric_envelope(amplitude, times, 4.889)
    start = float(fit["fit_t_start"])
    stop = float(fit["fit_t_stop"])
    mask = (times >= start) & (times <= stop)
    if not np.any(mask):
        return float("nan"), np.full_like(times, np.nan), envelope
    anchor = int(np.flatnonzero(mask)[0])
    analytic = envelope[anchor] * np.exp(theory_gamma * (times - times[anchor]))
    numerator = np.sqrt(np.mean((envelope[mask] - analytic[mask]) ** 2))
    denominator = np.sqrt(np.mean(analytic[mask] ** 2))
    return float(numerator / denominator), analytic, envelope


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--nboot", type=int, default=4000)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    theory = float(landau_gamma_theory(0.4, 1.0))
    rows = []
    traces = {}
    seed_reference = None
    for index, (method, (relative, color_key)) in enumerate(METHODS.items()):
        times, histories, seeds = load_histories(args.results_root / relative)
        if seed_reference is None:
            seed_reference = seeds
        elif seeds != seed_reference:
            raise RuntimeError(f"{method}: paired seed set differs from CIC")
        two_peak = first_two_peak_slope(histories, times)
        peak_ci = bootstrap_two_peak(histories, times, args.nboot, 71300 + index)
        certified = fit_censored_mode(histories, times)
        l2_error, analytic, envelope = normalized_l2_error(histories, times, certified, theory)
        rows.append(
            {
                "method": method,
                "two_peak_gamma": two_peak["gamma"],
                "two_peak_ci_lo": peak_ci[0],
                "two_peak_ci_hi": peak_ci[1],
                "two_peak_t1": two_peak["t1"],
                "two_peak_t2": two_peak["t2"],
                "certified_gamma": certified["gamma"],
                "certified_r_squared": certified["r_squared"],
                "certified_status": certified["status"],
                "certified_floor_censored": certified["floor_censored"],
                "certified_fit_t_start": certified["fit_t_start"],
                "certified_fit_t_stop": certified["fit_t_stop"],
                "noise_floor": certified["noise_floor"],
                "relative_l2_field_error": l2_error,
                "theory_gamma": theory,
            }
        )
        traces[method] = (times, envelope, analytic, color_key, certified)

    with (args.output / "landau_protocol_comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output / "landau_protocol_comparison.json").write_text(
        json.dumps({"rows": rows}, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    apply_style()
    figure, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL, 2.8))
    x = np.arange(len(rows), dtype=np.float64)
    peak_values = np.asarray([row["two_peak_gamma"] for row in rows])
    peak_low = np.asarray([row["two_peak_ci_lo"] for row in rows])
    peak_high = np.asarray([row["two_peak_ci_hi"] for row in rows])
    certified_values = np.asarray([row["certified_gamma"] for row in rows])
    axes[0].errorbar(
        x - 0.08,
        peak_values,
        yerr=np.vstack((peak_values - peak_low, peak_high - peak_values)),
        fmt="o",
        color=COLORS["WuQin"],
        label="first-two-peak protocol",
        capsize=2,
    )
    axes[0].scatter(x + 0.08, certified_values, color=[COLORS[row["method"]] for row in rows], label="certified fit")
    axes[0].axhline(theory, color=COLORS["theory"], linestyle="--", label="kinetic theory")
    axes[0].set_xticks(x, [row["method"] for row in rows])
    axes[0].set_ylabel(r"fitted $\gamma$")
    axes[0].set_title("Protocol changes the rate verdict")
    axes[0].legend(fontsize=7)

    for method, (times, envelope, analytic, color_key, certified) in traces.items():
        mask = times <= 30.0
        norm = envelope[np.argmin(np.abs(times - 5.0))]
        axes[1].plot(times[mask], envelope[mask] / norm, color=COLORS[color_key], label=method)
        if method == "CIC":
            axes[1].plot(times[mask], analytic[mask] / norm, color=COLORS["theory"], linestyle="--", label=r"$e^{\gamma_{th}t}$")
        axes[1].axvspan(
            certified["fit_t_start"], certified["fit_t_stop"], color=COLORS[color_key], alpha=0.035
        )
    axes[1].set_yscale("log")
    axes[1].set_xlabel("time")
    axes[1].set_ylabel(r"normalized $\|E(t)\|_2$")
    axes[1].set_title("Analytic-decay benchmark")
    axes[1].legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(args.output / "fig_wuqin_protocol_and_field_norm.pdf")
    figure.savefig(args.output / "fig_wuqin_protocol_and_field_norm.png")
    plt.close(figure)
    print(json.dumps({"rows": rows}, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
