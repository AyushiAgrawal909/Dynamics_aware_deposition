#!/usr/bin/env python3
"""Analyze coherently seeded linearized-delta-f E2 modes ``(12, my)``."""

from __future__ import annotations

import argparse
import csv
import json
import math
import pathlib
import sys

import numpy as np


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from analyze_E2_mode_growth import bootstrap_ensemble_rate  # noqa: E402
from pic_kde_2d.benchmarks.theory import (  # noqa: E402
    warm_two_stream_oblique_growth,
)
from pic_kde_2d.validation.landau_fit import (  # noqa: E402
    fit_gamma_ensemble_fixed_window,
)


def parse_int_list(text: str) -> list[int]:
    """Parse a unique sorted comma-separated list of nonnegative integers."""
    values = sorted({int(value.strip()) for value in text.split(",") if value.strip()})
    if not values or values[0] < 0:
        raise argparse.ArgumentTypeError("mode list must contain nonnegative integers")
    return values


def load_mode(
    root: pathlib.Path,
    mode_y: int,
    seeds: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load a complete paired ensemble for one transverse mode."""
    histories = []
    validity = []
    seed_ids = []
    times = None
    for path in sorted((root / f"my{mode_y:02d}").glob("seed*/mode_history.npz")):
        with np.load(path) as data:
            this_times = np.asarray(data["times"], dtype=np.float64)
            histories.append(np.abs(np.asarray(data["mode"], dtype=np.complex128)))
            validity.append(np.asarray(data["delta_f_ratio"], dtype=np.float64))
            seed_ids.append(int(np.asarray(data["seed"]).reshape(-1)[0]))
        if times is None:
            times = this_times
        else:
            np.testing.assert_allclose(times, this_times, rtol=0.0, atol=0.0)
    if len(histories) != seeds or sorted(seed_ids) != list(range(seeds)):
        raise RuntimeError(
            f"my={mode_y}: expected seeds 0..{seeds - 1}, found {sorted(seed_ids)}"
        )
    assert times is not None
    order = np.argsort(seed_ids)
    return (
        times,
        np.stack(histories)[order],
        np.stack(validity)[order],
    )


def cic_coupling(mode_x: int, mode_y: int, nx: int, ny: int) -> float:
    """Matched 2D CIC deposit-gather coupling at one grid mode."""
    return float(np.sinc(mode_x / nx) ** 4 * np.sinc(mode_y / ny) ** 4)


def fit_r_squared(
    times: np.ndarray,
    amplitude: np.ndarray,
    t_start: float,
    t_stop: float,
) -> tuple[float, float]:
    """Return R2 and fitted e-foldings on the frozen common window."""
    mask = (times >= t_start) & (times <= t_stop)
    x = times[mask]
    y = np.log(np.maximum(amplitude[mask], 1.0e-300))
    slope, intercept = np.polyfit(x, y, 1)
    residual = float(np.sum((y - (slope * x + intercept)) ** 2))
    total = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - residual / total if total > 0.0 else 0.0
    return r2, float(slope * (x[-1] - x[0]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--mode-x", type=int, default=12)
    parser.add_argument("--mode-y", type=parse_int_list, default=parse_int_list("0,1,2,3"))
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--nboot", type=int, default=5000)
    parser.add_argument("--fit-t-start", type=float, default=24.32)
    parser.add_argument("--fit-t-stop", type=float, default=59.68)
    parser.add_argument("--Nx", type=int, default=128)
    parser.add_argument("--Ny", type=int, default=16)
    parser.add_argument("--Lx", type=float, default=64.0)
    parser.add_argument("--Ly", type=float, default=16.0)
    parser.add_argument("--v-beam", type=float, default=0.8)
    parser.add_argument("--v-thermal", type=float, default=0.05)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    rows = []
    for mode_y in args.mode_y:
        times, amplitudes, validity = load_mode(args.input, mode_y, args.seeds)
        validity_envelope = np.max(validity, axis=0)
        validity_crossings = np.flatnonzero(validity_envelope >= 0.5)
        if validity_crossings.size:
            crossing_index = int(validity_crossings[0])
            valid_stop_index = max(crossing_index - 1, 0)
            first_validity_crossing = float(times[crossing_index])
        else:
            valid_stop_index = len(times) - 1
            first_validity_crossing = float("nan")
        effective_t_stop = min(
            args.fit_t_stop,
            float(times[valid_stop_index]),
        )
        if effective_t_stop <= args.fit_t_start:
            raise RuntimeError(
                f"my={mode_y}: delta-f validity fails before the fit starts"
            )
        fit = fit_gamma_ensemble_fixed_window(
            [row for row in amplitudes],
            times,
            t_fit_start=args.fit_t_start,
            t_fit_stop=effective_t_stop,
            amp_floor=1.0e-30,
        )
        boot = bootstrap_ensemble_rate(
            amplitudes,
            times,
            args.fit_t_start,
            effective_t_stop,
            args.nboot,
            seed=58000 + mode_y,
        )
        ensemble = np.mean(amplitudes, axis=0)
        r2, efoldings = fit_r_squared(
            times, ensemble, args.fit_t_start, effective_t_stop
        )
        fit_mask = (
            (times >= args.fit_t_start)
            & (times <= effective_t_stop)
        )
        max_validity_fit = float(np.max(validity[:, fit_mask]))
        kx = 2.0 * math.pi * args.mode_x / args.Lx
        ky = 2.0 * math.pi * mode_y / args.Ly
        coupling = cic_coupling(
            args.mode_x, mode_y, args.Nx, args.Ny
        )
        gamma_warm = warm_two_stream_oblique_growth(
            kx,
            ky,
            args.v_beam,
            args.v_thermal,
            coupling,
        )
        gamma = float(fit["gamma"])
        clean_unstable = bool(
            gamma > 0.0
            and boot["ci_lo"] > 0.0
            and r2 >= 0.98
            and efoldings >= 2.0
            and max_validity_fit < 0.5
        )
        rows.append(
            {
                "mode_x": args.mode_x,
                "mode_y": mode_y,
                "kx": kx,
                "ky": ky,
                "k_magnitude": float(np.hypot(kx, ky)),
                "matched_cic_coupling": coupling,
                "gamma": gamma,
                "gamma_sem": boot["sem"],
                "gamma_ci_lo": boot["ci_lo"],
                "gamma_ci_hi": boot["ci_hi"],
                "gamma_warm_cic": gamma_warm,
                "relative_error_vs_warm": (
                    abs(gamma - gamma_warm) / gamma_warm
                    if gamma_warm > 0.0
                    else float("nan")
                ),
                "fit_t_start": args.fit_t_start,
                "fit_t_stop": effective_t_stop,
                "fit_r2": r2,
                "fit_efoldings": efoldings,
                "max_delta_f_ratio_full_run": float(np.max(validity)),
                "max_delta_f_ratio_fit_window": max_validity_fit,
                "first_validity_crossing": first_validity_crossing,
                "clean_unstable": clean_unstable,
                "n_seeds": args.seeds,
            }
        )

    with (args.output / "oblique_growth.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    unstable_oblique = [
        row["mode_y"]
        for row in rows
        if row["mode_y"] > 0 and row["clean_unstable"]
    ]
    result = {
        "benchmark": "E2 two-stream coherently seeded oblique modes",
        "mode_x": args.mode_x,
        "mode_y_tested": args.mode_y,
        "fit_window": [args.fit_t_start, args.fit_t_stop],
        "n_seeds": args.seeds,
        "rows": rows,
        "measured_unstable_oblique_modes": unstable_oblique,
        "sigma_y_4_adoption_gate_pass": len(unstable_oblique) == 0,
        "status": (
            "oblique_instability_detected"
            if unstable_oblique
            else "no_oblique_instability_detected"
        ),
    }
    (args.output / "oblique_growth.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
