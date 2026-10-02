#!/usr/bin/env python3
"""Diagnose the historical ten-seed weak-Landau CIC/C4 fit windows.

This script deliberately reuses the archived reducer implementation.  It
does not refit with a longer or more permissive interval.  In addition to the
human-readable tables and figure, it preserves the complex per-seed mode
histories used by the calculation.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
from pathlib import Path
import sys

import numpy as np


WORKSPACE = Path(__file__).resolve().parents[2]
PROJECT = WORKSPACE / "pic_kde_2d"
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from pic_kde_2d.benchmarks.theory import langmuir_omega_r  # noqa: E402
from pic_kde_2d.validation.landau_censored_fit import (  # noqa: E402
    _first_persistent_crossing,
    coherent_amplitude,
    fit_censored_mode,
    geometric_envelope,
    quiet_tail_floor,
)


RAW_DEFAULT = (
    PROJECT
    / "results_raw_rate_audit/landau_sigma_refinement_4864819/shards/ppc1024"
)
OUTPUT_DEFAULT = (
    Path(__file__).resolve().parent
    / "results/weak_landau_confirmatory_diagnosis"
)
ARMS = {
    "CIC": {"directory": "sigma0", "sigma_dx": 0.0, "color": "#14213D"},
    "C4": {"directory": "sigma1", "sigma_dx": 1.0, "color": "#2A7F62"},
}
FIT = {
    "fit_t_start": 5.0,
    "fit_t_stop_max": 30.0,
    "noise_floor_start": 70.0,
    "noise_floor_stop": 90.0,
    "noise_factor": 5.0,
    "envelope_period": 2.0 * math.pi / langmuir_omega_r(0.4),
    "persistence_points": 5,
    "min_points": 30,
    "min_duration": 5.0,
    "min_model_efoldings": 0.5,
    "min_r_squared": 0.75,
}
REFERENCE_GAMMA = -0.06612795869074942


def seed_number(path: Path) -> int:
    return int(path.parent.name.removeprefix("seed"))


def load_arm(root: Path, directory: str, sigma_dx: float) -> tuple[np.ndarray, np.ndarray, list[int]]:
    paths = sorted((root / directory).glob("seed*/mode_history.npz"), key=seed_number)
    if not paths:
        raise FileNotFoundError(f"no mode histories below {root / directory}")
    histories: list[np.ndarray] = []
    seeds: list[int] = []
    times: np.ndarray | None = None
    for path in paths:
        with np.load(path, allow_pickle=False) as archive:
            required = {"times", "mode", "ppc", "sigma_dx", "seed"}
            missing = required.difference(archive.files)
            if missing:
                raise KeyError(f"{path} lacks {sorted(missing)}")
            current_times = np.asarray(archive["times"], dtype=np.float64)
            mode = np.asarray(archive["mode"], dtype=np.complex128)
            seed = int(archive["seed"][0])
            if int(archive["ppc"][0]) != 1024:
                raise AssertionError(f"unexpected PPC in {path}")
            if not np.isclose(float(archive["sigma_dx"][0]), sigma_dx):
                raise AssertionError(f"unexpected sigma in {path}")
        if seed != seed_number(path):
            raise AssertionError(f"seed/path mismatch in {path}")
        if mode.shape != current_times.shape:
            raise AssertionError(f"mode/time mismatch in {path}")
        if times is None:
            times = current_times
        else:
            np.testing.assert_array_equal(times, current_times)
        histories.append(mode)
        seeds.append(seed)
    assert times is not None
    if seeds != list(range(10)):
        raise AssertionError(f"expected historical seeds 0..9, found {seeds}")
    return times, np.stack(histories), seeds


def detailed_fit(histories: np.ndarray, times: np.ndarray) -> tuple[dict, np.ndarray, np.ndarray]:
    raw_amplitude = coherent_amplitude(histories)
    envelope, envelope_points = geometric_envelope(
        raw_amplitude, times, FIT["envelope_period"]
    )
    floor = quiet_tail_floor(
        envelope,
        times,
        tail_start=FIT["noise_floor_start"],
        tail_stop=FIT["noise_floor_stop"],
    )
    candidates = np.flatnonzero(
        (times >= FIT["fit_t_start"])
        & (times <= FIT["fit_t_stop_max"])
        & np.isfinite(envelope)
        & (envelope > 0.0)
    )
    threshold = FIT["noise_factor"] * floor["noise_floor"]
    crossing = _first_persistent_crossing(
        envelope, candidates, threshold, FIT["persistence_points"]
    )
    fit = fit_censored_mode(histories, times, **FIT)
    if crossing is not None:
        protected_index = crossing - envelope_points // 2
        ending_reason = "noise_threshold"
        ending_detail = (
            f"first {FIT['persistence_points']} consecutive centered-envelope samples "
            f"at or below {FIT['noise_factor']}*floor begin at t={times[crossing]:.1f}; "
            f"the {envelope_points // 2}-sample centered-envelope look-ahead guard "
            f"excludes t>={times[protected_index]:.1f}"
        )
        censor_t = float(times[crossing])
        protected_stop = float(times[protected_index])
    elif FIT["fit_t_stop_max"] < times[-1]:
        ending_reason = "analysis_cap"
        ending_detail = f"no persistent floor crossing before the frozen t={FIT['fit_t_stop_max']} cap"
        censor_t = float("nan")
        protected_stop = float("nan")
    else:
        ending_reason = "simulation_end"
        ending_detail = "no persistent floor crossing before the final recorded sample"
        censor_t = float("nan")
        protected_stop = float("nan")
    result = dict(fit)
    result.update(
        {
            "ending_reason": ending_reason,
            "ending_detail": ending_detail,
            "censor_t_reconstructed": censor_t,
            "protected_stop_exclusive": protected_stop,
            "signal_to_threshold_capacity_from_t0": float(
                math.log(result["initial_snr"] / FIT["noise_factor"])
            ),
            "signal_to_threshold_capacity_from_fit_start": float(
                math.log(result["fit_start_snr"] / FIT["noise_factor"])
            ),
            "simulation_t_stop": float(times[-1]),
            "hard_cap_reference_efoldings": float(
                abs(REFERENCE_GAMMA) * (FIT["fit_t_stop_max"] - FIT["fit_t_start"])
            ),
            "two_reference_efolding_t_stop": float(
                FIT["fit_t_start"] + 2.0 / abs(REFERENCE_GAMMA)
            ),
        }
    )
    return result, raw_amplitude, envelope


def subset_floor_scaling(histories: np.ndarray, times: np.ndarray) -> tuple[list[dict], dict]:
    rows: list[dict] = []
    medians: list[float] = []
    for count in range(1, histories.shape[0] + 1):
        floors = []
        for indices in itertools.combinations(range(histories.shape[0]), count):
            amplitude = coherent_amplitude(histories[list(indices)])
            envelope, _ = geometric_envelope(amplitude, times, FIT["envelope_period"])
            floors.append(
                quiet_tail_floor(
                    envelope,
                    times,
                    tail_start=FIT["noise_floor_start"],
                    tail_stop=FIT["noise_floor_stop"],
                )["noise_floor"]
            )
        values = np.asarray(floors, dtype=np.float64)
        median = float(np.median(values))
        medians.append(median)
        rows.append(
            {
                "n_seeds": count,
                "n_subsets": len(values),
                "median_noise_floor": median,
                "q025_noise_floor": float(np.quantile(values, 0.025)),
                "q975_noise_floor": float(np.quantile(values, 0.975)),
            }
        )
    exponent, intercept = np.polyfit(
        np.log(np.arange(1, histories.shape[0] + 1, dtype=np.float64)),
        np.log(np.asarray(medians)),
        1,
    )
    tail_mask = (times >= 80.0) & (times <= 85.0)
    coherent_rms = float(np.sqrt(np.mean(np.abs(np.mean(histories[:, tail_mask], axis=0)) ** 2)))
    seed_scatter_rms = float(
        np.sqrt(np.mean(np.var(histories[:, tail_mask], axis=0, ddof=1)))
    )
    random_sem_10 = seed_scatter_rms / math.sqrt(histories.shape[0])
    residual_quadrature = math.sqrt(max(0.0, coherent_rms**2 - random_sem_10**2))
    projected_rms_40 = math.hypot(residual_quadrature, random_sem_10 / 2.0)
    summary = {
        "log_log_subset_median_exponent": float(exponent),
        "log_log_subset_median_intercept": float(intercept),
        "observed_scaling_projection_floor_40_over_10": float(4.0**exponent),
        "observed_scaling_projection_capacity_gain": float(-exponent * math.log(4.0)),
        "ideal_random_amplitude_floor_40_over_10": 0.5,
        "ideal_random_capacity_gain": float(math.log(2.0)),
        "tail_80_85_coherent_rms_10": coherent_rms,
        "tail_80_85_seed_scatter_rms": seed_scatter_rms,
        "tail_80_85_random_sem_10": random_sem_10,
        "tail_80_85_residual_quadrature_estimate": residual_quadrature,
        "tail_80_85_projected_rms_40": projected_rms_40,
        "tail_80_85_projected_rms_40_over_10": projected_rms_40 / coherent_rms,
        "warning": (
            "The exponent and quadrature projection are diagnostics from only ten development "
            "realisations, not confirmatory estimates or stopping-rule inputs."
        ),
    }
    return rows, summary


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_figure(output: Path, records: dict[str, dict], times: np.ndarray) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    style_root = PROJECT / "paper1_selector_comparison/figures"
    if str(style_root) not in sys.path:
        sys.path.insert(0, str(style_root))
    try:
        from paperstyle import apply_style

        apply_style()
    except ImportError:
        plt.rcParams.update({"font.family": "serif", "font.size": 9})

    figure, axes = plt.subplots(2, 2, figsize=(7.0, 5.0), sharex="col")
    for row, (name, spec) in enumerate(ARMS.items()):
        record = records[name]
        histories = record["histories"]
        coherent = np.mean(histories, axis=0)
        color = spec["color"]
        axes[row, 0].plot(times, coherent.real, color=color, linewidth=1.0, label=r"Re $\langle\hat E_x\rangle$")
        axes[row, 0].plot(times, coherent.imag, color=color, linewidth=0.9, linestyle="--", label=r"Im $\langle\hat E_x\rangle$")
        axes[row, 0].axvspan(record["fit"]["fit_t_start"], record["fit"]["fit_t_stop"], color="0.75", alpha=0.2)
        axes[row, 0].set_ylabel(f"{name}\ncomplex mode")
        axes[row, 0].legend(frameon=False, ncol=2, loc="upper right")

        envelope = record["envelope"]
        raw_amplitude = record["raw_amplitude"]
        fit = record["fit"]
        axes[row, 1].semilogy(times, raw_amplitude, color=color, alpha=0.35, linewidth=0.6, label="coherent magnitude")
        axes[row, 1].semilogy(times, envelope, color=color, linewidth=1.1, label="one-period envelope")
        accepted = (times >= fit["fit_t_start"]) & (times <= fit["fit_t_stop"])
        fitted = np.exp(np.polyval(np.polyfit(times[accepted], np.log(envelope[accepted]), 1), times[accepted]))
        axes[row, 1].semilogy(times[accepted], fitted, color="black", linestyle="--", linewidth=0.9, label="accepted fit")
        axes[row, 1].axhline(fit["noise_floor"], color="0.35", linestyle=":", linewidth=0.9, label="floor")
        axes[row, 1].axhline(fit["threshold"], color="0.55", linestyle="-.", linewidth=0.8, label=r"$5\times$ floor")
        axes[row, 1].axvspan(fit["fit_t_start"], fit["fit_t_stop"], color="0.75", alpha=0.2)
        axes[row, 1].axvline(fit["censor_t"], color=color, linestyle=":", linewidth=0.9)
        axes[row, 1].set_ylabel(f"{name}\nmode amplitude")
        axes[row, 1].legend(frameon=False, ncol=2, loc="upper right")
        axes[row, 1].text(
            0.03,
            0.08,
            rf"$t_{{fit}}=[{fit['fit_t_start']:.1f},{fit['fit_t_stop']:.1f}]$, "
            rf"$N_e={fit['model_efoldings']:.3f}$",
            transform=axes[row, 1].transAxes,
        )
    axes[1, 0].set_xlabel(r"$t\,\omega_p$")
    axes[1, 1].set_xlabel(r"$t\,\omega_p$")
    axes[0, 0].set_title("coherent complex Fourier history")
    axes[0, 1].set_title("magnitude, envelope, and accepted fit")
    for axis in axes.ravel():
        axis.set_xlim(0.0, 90.0)
    figure.tight_layout()
    figure.savefig(output / "weak_landau_window_diagnosis.pdf")
    figure.savefig(output / "weak_landau_window_diagnosis.png", dpi=300)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=RAW_DEFAULT)
    parser.add_argument("--output", type=Path, default=OUTPUT_DEFAULT)
    args = parser.parse_args()
    source = args.input.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing diagnostic directory: {output}")
    output.mkdir(parents=True)

    records: dict[str, dict] = {}
    common_times: np.ndarray | None = None
    scaling_rows: list[dict] = []
    archive_payload: dict[str, np.ndarray] = {}
    for name, spec in ARMS.items():
        times, histories, seeds = load_arm(source, spec["directory"], spec["sigma_dx"])
        if common_times is None:
            common_times = times
        else:
            np.testing.assert_array_equal(common_times, times)
        fit, raw_amplitude, envelope = detailed_fit(histories, times)
        rows, scaling = subset_floor_scaling(histories, times)
        for row in rows:
            scaling_rows.append({"arm": name, **row})
        records[name] = {
            "source_directory": str((source / spec["directory"]).resolve()),
            "sigma_dx": spec["sigma_dx"],
            "seeds": seeds,
            "n_seeds": len(seeds),
            "reducer_operation": "abs(mean(complex Fourier coefficient over seeds))",
            "fit": fit,
            "noise_scaling": scaling,
            "histories": histories,
            "raw_amplitude": raw_amplitude,
            "envelope": envelope,
        }
        key = name.lower()
        archive_payload[f"{key}_complex_mode_histories"] = histories
        archive_payload[f"{key}_coherent_complex_mode"] = np.mean(histories, axis=0)
        archive_payload[f"{key}_coherent_amplitude"] = raw_amplitude
        archive_payload[f"{key}_geometric_envelope"] = envelope

    assert common_times is not None
    archive_payload["times"] = common_times
    np.savez_compressed(output / "weak_landau_complex_histories_and_envelopes.npz", **archive_payload)
    write_csv(output / "ten_seed_subset_floor_scaling.csv", scaling_rows)
    summary = {
        "source": str(source),
        "development_data_only": True,
        "historical_fit_configuration": FIT,
        "reference_gamma": REFERENCE_GAMMA,
        "two_amplitude_efolding_duration": 2.0 / abs(REFERENCE_GAMMA),
        "two_amplitude_efolding_stop_for_t_start_5": 5.0 + 2.0 / abs(REFERENCE_GAMMA),
        "reducer_order": {
            "central_trace": "average complex Fourier coefficients across phase-aligned seeds, then take magnitude",
            "envelope": "centered geometric mean of magnitude over one fitted oscillation period",
            "rate": "one log-linear fit to that coherent-ensemble envelope",
            "not_used": ["mean of positive magnitudes", "mean of per-seed fitted rates"],
        },
        "arms": {
            name: {
                key: value
                for key, value in record.items()
                if key not in {"histories", "raw_amplitude", "envelope"}
            }
            for name, record in records.items()
        },
    }
    (output / "window_diagnosis.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    make_figure(output, records, common_times)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
