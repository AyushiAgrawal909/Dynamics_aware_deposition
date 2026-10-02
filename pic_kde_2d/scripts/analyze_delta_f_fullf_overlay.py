#!/usr/bin/env python3
"""Compare nonlinear full-f and linearized delta-f Landau traces at alpha=0.1."""

from __future__ import annotations

import argparse
import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_histories(root: pathlib.Path, seeds: int) -> tuple[np.ndarray, np.ndarray]:
    histories = []
    times = None
    for seed in range(seeds):
        path = root / f"seed{seed}" / "mode_history.npz"
        with np.load(path) as data:
            this_times = np.asarray(data["times"], dtype=np.float64)
            histories.append(np.asarray(data["mode"], dtype=np.complex128))
        if times is None:
            times = this_times
        else:
            np.testing.assert_allclose(times, this_times, rtol=0.0, atol=0.0)
    assert times is not None
    return times, np.stack(histories)


def slope(times: np.ndarray, amplitude: np.ndarray, start: float, stop: float) -> float:
    mask = (times >= start) & (times <= stop)
    return float(np.polyfit(times[mask], np.log(np.maximum(amplitude[mask], 1.0e-300)), 1)[0])


def persistent_crossing(values: np.ndarray, threshold: float, persistence: int) -> int | None:
    above = values > threshold
    for index in range(0, len(values) - persistence + 1):
        if np.all(above[index : index + persistence]):
            return index
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fullf", type=pathlib.Path, required=True)
    parser.add_argument("--deltaf", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--seeds", type=int, default=5)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    times_full, full_histories = load_histories(args.fullf, args.seeds)
    times_delta, delta_histories = load_histories(args.deltaf, args.seeds)
    np.testing.assert_allclose(times_full, times_delta, rtol=0.0, atol=0.0)
    times = times_full
    full_amp = np.mean(np.abs(full_histories), axis=0)
    delta_amp = np.mean(np.abs(delta_histories), axis=0)

    relative_difference = np.abs(full_amp - delta_amp) / np.maximum(delta_amp, 1.0e-300)
    eligible = np.flatnonzero(times >= 5.0)
    crossing_local = persistent_crossing(relative_difference[eligible], 0.25, 5)
    divergence_time = (
        float(times[eligible[crossing_local]]) if crossing_local is not None else float("nan")
    )
    late = times >= 40.0
    full_late = float(np.mean(full_amp[late]))
    delta_late = float(np.mean(delta_amp[late]))
    result = {
        "alpha": 0.1,
        "n_seeds": args.seeds,
        "fullf_early_gamma_5_15": slope(times, full_amp, 5.0, 15.0),
        "deltaf_early_gamma_5_15": slope(times, delta_amp, 5.0, 15.0),
        "first_persistent_25pct_divergence_time": divergence_time,
        "fullf_late_amplitude_mean_t40_60": full_late,
        "deltaf_late_amplitude_mean_t40_60": delta_late,
        "late_amplitude_ratio_fullf_over_deltaf": full_late / delta_late,
        "interpretation": (
            "linearized delta-f omits nonlinear trapping; a sustained late-time "
            "full-f excess or bounce structure is the physics-level failure"
        ),
    }
    (args.output / "delta_f_fullf_overlay.json").write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )

    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    ax.semilogy(times, full_amp, label="full-f, alpha=0.1", color="#c14924", lw=2.0)
    ax.semilogy(times, delta_amp, label="linearized delta-f, alpha=0.1", color="#247ba0", lw=2.0)
    if np.isfinite(divergence_time):
        ax.axvline(divergence_time, color="#333333", ls=":", label="25% persistent divergence")
    ax.set(xlabel=r"$t\,\omega_p$", ylabel=r"$|\hat E_x(m=1)|$", title="Delta-f validity failure: nonlinear Landau")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.output / "delta_f_fullf_overlay.png", dpi=220)
    plt.close(fig)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
