#!/usr/bin/env python3
"""Diagnose the failed v1 weak-Landau confirmatory window without rate ratios.

This audit compares the 40-seed per-cell velocity-quadrature campaign with the
preserved ten-seed global-velocity development histories.  It records only
loading support, coherent-mode/envelope behavior, phase coherence, and the
already-frozen Stage-1 decisions.  It deliberately computes no comparative
damping-rate ratio and cannot promote the v1 campaign.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[1]
PROJECT = WORKSPACE / "pic_kde_2d"
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from pic_kde_2d.core.particles2d import _inverse_normal_cdf  # noqa: E402
from pic_kde_2d.validation.landau_censored_fit import (  # noqa: E402
    coherent_amplitude,
    geometric_envelope,
)


DEFAULT_PRODUCTION = HERE / "results/weak_landau_confirmatory_production_20260911/low"
DEFAULT_STAGE1 = HERE / "results/weak_landau_confirmatory_stage1_windows_20260911/window_decisions.json"
DEFAULT_DEVELOPMENT = (
    PROJECT / "results_raw_rate_audit/landau_sigma_refinement_4864819/shards/ppc1024"
)
DEFAULT_CONFIG = HERE / "weak_landau_confirmatory_v1.json"
DEFAULT_OUTPUT = HERE / "results/weak_landau_confirmatory_stage1_failure_diagnosis_20260911"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tree_manifest_hash(root: Path, paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        relative = path.relative_to(root).as_posix()
        digest.update(f"{sha256(path)}  {relative}\n".encode("utf-8"))
    return digest.hexdigest()


def inspect_npz_schema(path: Path, required: set[str]) -> dict:
    with np.load(path) as archive:
        keys = list(archive.files)
        missing = required.difference(keys)
        if missing:
            raise KeyError(f"{path} lacks {sorted(missing)}; found {keys}")
        return {
            key: {"shape": list(archive[key].shape), "dtype": str(archive[key].dtype)}
            for key in keys
        }


def load_confirmatory(root: Path, arm: str) -> tuple[np.ndarray, np.ndarray, list[Path]]:
    paths = sorted(root.glob(f"seed*/{arm}/history_and_noise_snapshots.npz"))
    if len(paths) != 40:
        raise AssertionError(f"expected 40 {arm} histories, found {len(paths)}")
    histories = []
    times = None
    for path in paths:
        with np.load(path) as archive:
            current_times = np.asarray(archive["times"], dtype=np.float64)
            mode = np.asarray(archive["mode"], dtype=np.complex128)
        if times is None:
            times = current_times
        else:
            np.testing.assert_array_equal(times, current_times)
        histories.append(mode)
    assert times is not None
    return times, np.stack(histories), paths


def load_development(root: Path, sigma_tag: str) -> tuple[np.ndarray, np.ndarray, list[Path]]:
    paths = sorted((root / sigma_tag).glob("seed*/mode_history.npz"))
    if len(paths) != 10:
        raise AssertionError(f"expected 10 {sigma_tag} histories, found {len(paths)}")
    histories = []
    times = None
    for path in paths:
        with np.load(path) as archive:
            current_times = np.asarray(archive["times"], dtype=np.float64)
            mode = np.asarray(archive["mode"], dtype=np.complex128)
        if times is None:
            times = current_times
        else:
            np.testing.assert_array_equal(times, current_times)
        histories.append(mode)
    assert times is not None
    return times, np.stack(histories), paths


def first_index(times: np.ndarray, value: float) -> int:
    return int(np.argmin(np.abs(times - value)))


def summarize_history(
    times: np.ndarray,
    histories: np.ndarray,
    period: float,
    checkpoints: list[float],
) -> tuple[dict, list[dict], np.ndarray, np.ndarray]:
    mean_mode = np.mean(histories, axis=0)
    amplitude = coherent_amplitude(histories)
    envelope, points = geometric_envelope(amplitude, times, period)
    window = np.flatnonzero((times >= 5.0) & (times <= 45.0))
    minimum_index = int(window[np.argmin(envelope[window])])
    rows = []
    for checkpoint in checkpoints:
        index = first_index(times, checkpoint)
        mean_magnitude = float(np.mean(np.abs(histories[:, index])))
        unit_phase = histories[:, index] / np.maximum(np.abs(histories[:, index]), 1.0e-300)
        rows.append(
            {
                "time": float(times[index]),
                "coherent_amplitude": float(amplitude[index]),
                "geometric_envelope": float(envelope[index]),
                "mean_seed_magnitude": mean_magnitude,
                "complex_mean_coherence": float(amplitude[index] / mean_magnitude),
                "circular_phase_resultant": float(abs(np.mean(unit_phase))),
                "complex_relative_seed_scatter": float(
                    np.std(histories[:, index]) / max(abs(mean_mode[index]), 1.0e-300)
                ),
            }
        )
    return (
        {
            "n_seeds": int(histories.shape[0]),
            "n_times": int(histories.shape[1]),
            "time_start": float(times[0]),
            "time_stop": float(times[-1]),
            "diagnostic_cadence": float(np.median(np.diff(times))),
            "centered_envelope_points": int(points),
            "minimum_envelope_t_5_45": float(times[minimum_index]),
            "minimum_envelope_5_45": float(envelope[minimum_index]),
            "envelope_at_45_over_minimum_5_45": float(
                envelope[first_index(times, 45.0)] / envelope[minimum_index]
            ),
        },
        rows,
        mean_mode,
        envelope,
    )


def normal_pdf(values: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * values**2) / math.sqrt(2.0 * math.pi)


def json_safe(value):
    """Replace non-finite numbers in inherited Stage-1 data with JSON null."""
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def make_figure(
    output: Path,
    times: np.ndarray,
    envelopes: dict[str, np.ndarray],
    velocity_nodes: np.ndarray,
    phase_velocity: float,
) -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 8,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    navy = "#102a43"
    red = "#8b1e2d"
    figure, axes = plt.subplots(1, 2, figsize=(7.0, 2.8))

    axis = axes[0]
    axis.semilogy(times, envelopes["confirmatory_cic"], color=navy, label="v1 per-cell CIC (40)")
    axis.semilogy(times, envelopes["confirmatory_c4"], color=red, label="v1 per-cell C4 (40)")
    axis.semilogy(times, envelopes["development_cic"], color=navy, linestyle="--", label="global-velocity CIC (10)")
    axis.semilogy(times, envelopes["development_c4"], color=red, linestyle="--", label="global-velocity C4 (10)")
    axis.axvline(5.0, color="0.55", linestyle=":", linewidth=0.9)
    axis.axvline(35.2444, color="0.55", linestyle=":", linewidth=0.9)
    axis.text(5.6, 0.032, "frozen start", color="0.35")
    axis.text(35.9, 0.032, "2-e-fold target", color="0.35")
    axis.set_xlim(0.0, 50.0)
    axis.set_ylim(2.5e-4, 4.5e-2)
    axis.set_xlabel(r"$t\,\omega_p$")
    axis.set_ylabel(r"envelope of $|\langle \widehat E_x(1,0)\rangle|$")
    axis.set_title("(a) v1 has seed-coherent regrowth")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False, fontsize=6.8, loc="lower left")

    axis = axes[1]
    velocity = np.linspace(0.0, 4.0, 500)
    axis.plot(velocity, normal_pdf(velocity), color="0.35", linewidth=1.2, label="Maxwellian")
    positive_nodes = velocity_nodes[velocity_nodes >= 0.0]
    for node in positive_nodes:
        axis.vlines(node, 0.0, 0.018, color=navy, linewidth=0.7, alpha=0.85)
    vmax = float(np.max(velocity_nodes))
    axis.axvline(vmax, color=navy, linestyle="--", linewidth=1.2)
    axis.axvline(phase_velocity, color=red, linestyle="-", linewidth=1.4)
    axis.annotate(
        rf"largest node $={vmax:.3f}$",
        xy=(vmax, normal_pdf(np.asarray([vmax]))[0]),
        xytext=(0.82, 0.29),
        arrowprops={"arrowstyle": "-", "color": navy, "lw": 0.8},
        color=navy,
    )
    axis.annotate(
        rf"resonance $v_\phi={phase_velocity:.3f}$" + "\n(no longitudinal node)",
        xy=(phase_velocity, normal_pdf(np.asarray([phase_velocity]))[0]),
        xytext=(2.85, 0.32),
        arrowprops={"arrowstyle": "-", "color": red, "lw": 0.8},
        color=red,
        ha="center",
    )
    axis.text(0.06, 0.025, "rug: 16 nonnegative nodes of the 32-point $v_x$ rule", color=navy)
    axis.set_xlim(0.0, 4.0)
    axis.set_ylim(0.0, 0.43)
    axis.set_xlabel(r"$v_x/v_{\rm th}$")
    axis.set_ylabel("probability density")
    axis.set_title("(b) per-cell velocity rule misses resonance")
    axis.grid(alpha=0.25)

    figure.tight_layout(w_pad=1.5)
    figure.savefig(output / "fig_weak_landau_loading_failure.pdf")
    figure.savefig(output / "fig_weak_landau_loading_failure.png", dpi=300)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--production", type=Path, default=DEFAULT_PRODUCTION)
    parser.add_argument("--stage1", type=Path, default=DEFAULT_STAGE1)
    parser.add_argument("--development", type=Path, default=DEFAULT_DEVELOPMENT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    args.output.mkdir(parents=True)

    config = json.loads(args.config.read_text(encoding="utf-8"))
    stage1 = json.loads(args.stage1.read_text(encoding="utf-8"))
    if stage1["rate_ratios_computed"] is not False:
        raise AssertionError("Stage-1 rate-ratio lock is absent")
    if stage1["n_seeds"] != 40 or not stage1["all_pairing_hashes_verified"]:
        raise AssertionError("Stage-1 40-seed pairing checks did not pass")
    if stage1["config_sha256"] != sha256(args.config):
        raise AssertionError("Stage-1/config hash mismatch")

    production_example = next(args.production.glob("seed*/cic/history_and_noise_snapshots.npz"))
    development_example = next((args.development / "sigma0").glob("seed*/mode_history.npz"))
    schemas = {
        "confirmatory": inspect_npz_schema(
            production_example,
            {"times", "mode", "field_energy", "kinetic_energy", "total_charge"},
        ),
        "development": inspect_npz_schema(development_example, {"times", "mode", "ppc", "sigma_dx", "seed"}),
    }

    period = float(config["fit_rule"]["envelope_period"])
    checkpoints = [0.0, 5.0, 10.0, 20.0, 30.0, 35.0, 40.0, 45.0]
    cases = {
        "confirmatory_cic": load_confirmatory(args.production, "cic"),
        "confirmatory_c4": load_confirmatory(args.production, "c4_sigma1"),
        "development_cic": load_development(args.development, "sigma0"),
        "development_c4": load_development(args.development, "sigma1"),
    }
    summaries = {}
    rows = []
    envelopes = {}
    shared_times = None
    input_paths: list[Path] = []
    for name, (times, histories, paths) in cases.items():
        if shared_times is None:
            shared_times = times
        else:
            np.testing.assert_array_equal(shared_times, times)
        summary, diagnostics, _, envelope = summarize_history(times, histories, period, checkpoints)
        summaries[name] = summary
        envelopes[name] = envelope
        input_paths.extend(paths)
        for row in diagnostics:
            rows.append({"case": name, **row})
    assert shared_times is not None

    benchmark = config["benchmark"]
    grid = config["grid"]
    ppc = int(benchmark["ppc"])
    longitudinal_nodes = int(round(math.sqrt(ppc)))
    if longitudinal_nodes**2 != ppc:
        raise AssertionError("v1 per-cell tensor is expected to be square")
    quantiles = (np.arange(longitudinal_nodes, dtype=np.float64) + 0.5) / longitudinal_nodes
    velocity_nodes = _inverse_normal_cdf(quantiles) * float(benchmark["velocity_thermal"])
    k = 2.0 * math.pi * int(benchmark["kx_mode"]) / float(grid["Lx"])
    omega_r = 2.0 * math.pi / period
    phase_velocity = omega_r / k
    max_velocity = float(np.max(velocity_nodes))
    missing_one_sided_mass = 0.5 * math.erfc(max_velocity / math.sqrt(2.0))
    resonance_tail_mass = 0.5 * math.erfc(phase_velocity / math.sqrt(2.0))

    if not max_velocity < phase_velocity:
        raise AssertionError("expected the disputed per-cell rule to miss the resonance")
    if summaries["confirmatory_cic"]["envelope_at_45_over_minimum_5_45"] <= 2.0:
        raise AssertionError("expected strong deterministic CIC regrowth in v1")
    if summaries["development_cic"]["envelope_at_45_over_minimum_5_45"] > 1.05:
        raise AssertionError("global-velocity development trace unexpectedly regrows before t=45")

    fieldnames = [
        "case",
        "time",
        "coherent_amplitude",
        "geometric_envelope",
        "mean_seed_magnitude",
        "complex_mean_coherence",
        "circular_phase_resultant",
        "complex_relative_seed_scatter",
    ]
    with (args.output / "ensemble_diagnostics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    production_files = sorted(args.production.glob("seed*/*/*"))
    result = {
        "status": "FAIL_LOADING_ADMISSIBILITY",
        "scope": "Stage-1 loading/envelope diagnosis only; no rate ratios computed",
        "rate_ratios_computed": False,
        "input_schemas": schemas,
        "input_hashes": {
            "config_sha256": sha256(args.config),
            "stage1_sha256": sha256(args.stage1),
            "production_tree_manifest_sha256": tree_manifest_hash(args.production, production_files),
            "history_files_manifest_sha256": tree_manifest_hash(WORKSPACE, input_paths),
        },
        "stage1_decisions": json_safe(stage1["decisions"]),
        "loading_implementation": {
            "spatial_marker_layout": "regular 2048 x 512 lattice; 1024 markers in every field-grid cell",
            "velocity_layout": "32 x 32 tensor inverse-normal quadrature independently permuted within each cell",
            "longitudinal_distinct_unperturbed_velocities_per_cell": longitudinal_nodes,
            "longitudinal_velocity_min": float(np.min(velocity_nodes)),
            "longitudinal_velocity_max": max_velocity,
            "omega_r": omega_r,
            "k": k,
            "resonant_phase_velocity": phase_velocity,
            "largest_velocity_below_resonance": bool(max_velocity < phase_velocity),
            "maxwellian_one_sided_mass_above_largest_node": missing_one_sided_mass,
            "maxwellian_one_sided_mass_above_resonance": resonance_tail_mass,
            "code_links": [
                "pic_kde_2d/pic_kde_2d/core/particles2d.py:139",
                "pic_kde_2d/pic_kde_2d/core/particles2d.py:144",
                "pic_kde_2d/pic_kde_2d/core/particles2d.py:157",
                "pic_kde_2d/scripts/run_weak_landau_confirmatory.py:178",
                "pic_kde_2d/scripts/run_landau_sigma_refinement.py:72",
            ],
        },
        "ensemble_behavior": summaries,
        "diagnosis": (
            "The v1 interpretation of per-cell loading discretizes vx to 32 inverse-normal nodes per cell. "
            "Its largest node lies below the k*lambda_D=0.4 phase velocity, eliminating the resonant population. "
            "All seeds share this velocity support, so ensemble averaging cannot remove the deterministic regrowth. "
            "The [70,90] value is therefore recurrence/regrowth, not an independent-particle noise floor."
        ),
        "comparison_scope": (
            "The preserved job-4864819 development traces use the same regular spatial marker lattice but a global "
            "inverse-CDF velocity template; they continue damping through the two-e-fold target."
        ),
        "required_action": (
            "Do not run Stage 2 or reuse v1 for validation. First confirm a corrected velocity-loading convention "
            "with a small fresh-seed pilot; any replacement 40-seed campaign requires a new frozen specification and seeds."
        ),
    }
    (args.output / "stage1_loading_failure_diagnosis.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    make_figure(args.output, shared_times, envelopes, velocity_nodes, phase_velocity)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
