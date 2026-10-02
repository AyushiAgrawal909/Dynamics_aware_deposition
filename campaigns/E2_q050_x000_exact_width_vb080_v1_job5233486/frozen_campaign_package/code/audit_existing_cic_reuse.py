#!/usr/bin/env python3
"""Audit archived CIC histories for deterministic reuse in the q050 campaign."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import sys

import numpy as np


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_runner(path: Path):
    spec = importlib.util.spec_from_file_location("q050_runner", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_hash_manifest(path: Path) -> dict[str, str]:
    values = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        digest, filename = line.split(maxsplit=1)
        values[Path(filename).name] = digest
    return values


def linear_fit(times: np.ndarray, amplitudes: np.ndarray, start: float, stop: float):
    mask = (
        (times >= start)
        & (times <= stop)
        & np.isfinite(amplitudes)
        & (amplitudes > 1.0e-300)
    )
    if np.count_nonzero(mask) < 5:
        raise RuntimeError("fewer than five finite positive fit samples")
    x = times[mask]
    y = np.log(amplitudes[mask])
    slope, intercept = np.polyfit(x, y, 1)
    fitted = slope * x + intercept
    residual = float(np.sum((y - fitted) ** 2))
    total = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = 1.0 - residual / total if total > 0.0 else 0.0
    return float(slope), float(r_squared), int(np.count_nonzero(mask))


def estimate_floor(
    times: np.ndarray, amplitudes: np.ndarray, fit_start: float, duration: float
) -> dict:
    dt = float(np.median(np.diff(times)))
    width = max(5, int(round(duration / dt)))
    stop_limit = int(np.searchsorted(times, fit_start, side="left"))
    if stop_limit < width:
        raise RuntimeError("insufficient pre-fit samples for floor estimate")
    candidates = []
    for begin in range(stop_limit - width + 1):
        end = begin + width
        candidates.append((float(np.median(amplitudes[begin:end])), begin, end))
    floor, begin, end = min(candidates, key=lambda item: item[0])
    fit_index = int(np.searchsorted(times, fit_start, side="left"))
    start_amplitude = float(amplitudes[fit_index])
    return {
        "noise_floor": floor,
        "floor_t_start": float(times[begin]),
        "floor_t_stop": float(times[end - 1]),
        "fit_start_amplitude": start_amplitude,
        "fit_start_snr": start_amplitude / max(floor, 1.0e-300),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--config-sha256", required=True)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--legacy-campaign-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    runner = load_runner(Path(__file__).with_name("run_E2_q050_exact_arm.py"))
    if sha256_file(args.config) != args.config_sha256:
        raise AssertionError("frozen config hash mismatch")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    reuse = config["existing_cic_reuse"]
    runner.verify_source(args.project, config)
    expected_runtime = reuse["production_runtime"]
    if platform.python_version() != expected_runtime["observed_python"]:
        raise AssertionError("production Python version mismatch")
    if np.__version__ != expected_runtime["observed_numpy"]:
        raise AssertionError("production NumPy version mismatch")

    legacy_git = (args.legacy_campaign_root / "git_revision.txt").read_text().strip()
    if legacy_git != reuse["git_revision"]:
        raise AssertionError("legacy git revision mismatch")
    frozen_path = args.legacy_campaign_root / "frozen_campaign.json"
    if sha256_file(frozen_path) != reuse["frozen_campaign_sha256"]:
        raise AssertionError("legacy frozen campaign hash mismatch")
    legacy_hashes = read_hash_manifest(args.legacy_campaign_root / "input_sha256.txt")
    if legacy_hashes.get("run_E2_beam_drift_seed.py") != reuse["runner_sha256"]:
        raise AssertionError("legacy wrapper hash mismatch")
    legacy_unhashed = []
    for relative, expected in config["source_identity"]["sha256"].items():
        observed = legacy_hashes.get(Path(relative).name)
        if observed is None:
            legacy_unhashed.append(relative)
        elif observed != expected:
            raise AssertionError(f"legacy source hash mismatch for {relative}")

    for candidate in (args.project, args.project / "scripts"):
        if str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
    import run_linearized_delta_f_mode as production

    rows = []
    amplitudes_by_seed = []
    reference_times = None
    file_hashes = {}
    for seed in config["seeds"]:
        directory = args.input / f"seed{seed}"
        paths = {
            "result": directory / "result.json",
            "manifest": directory / "campaign_manifest.json",
            "history": directory / "mode_history.npz",
        }
        for name, path in paths.items():
            if not path.is_file():
                raise FileNotFoundError(path)
            file_hashes[str(path.relative_to(args.input))] = sha256_file(path)
        result = json.loads(paths["result"].read_text(encoding="utf-8"))
        manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
        if manifest["frozen_campaign_sha256"] != reuse["frozen_campaign_sha256"]:
            raise AssertionError(f"legacy campaign hash mismatch for seed {seed}")
        expected_fields = {
            "seed": seed,
            "ppc": config["loading"]["ppc"],
            "alpha": config["initial_perturbation_alpha"],
            "kx_mode": config["mode"][0],
            "ky_mode": config["mode"][1],
            "dt": config["dt"],
            "t_end": config["t_end"],
            "sigma_x_dx": 0.0,
            "sigma_y_dy": 0.0,
            "target_mode_projection": True,
        }
        for key, expected in expected_fields.items():
            if result.get(key) != expected:
                raise AssertionError(
                    f"seed {seed}: {key} expected {expected!r}, got {result.get(key)!r}"
                )
        if manifest["grid"] != config["grid"] | {"dx": 0.5, "dy": 1.0}:
            # The legacy manifest omits derived spacings.
            legacy_grid = {k: config["grid"][k] for k in ("Nx", "Ny", "Lx", "Ly")}
            if manifest["grid"] != legacy_grid:
                raise AssertionError(f"legacy grid mismatch for seed {seed}")
        if manifest["v_beam_plus"] != config["equilibrium"]["v_beam_plus"]:
            raise AssertionError(f"beam drift mismatch for seed {seed}")

        production_args = runner.make_args(
            production, config, "cic", int(seed), Path("unused-audit-output")
        )
        grid, particles = production.build_markers(production_args)
        state_hash, array_hashes = runner.hash_initial_state(particles)
        expected_hash = reuse["production_runtime_reconstructed_initial_state_sha256"][str(seed)]
        if state_hash != expected_hash:
            raise AssertionError(
                f"production-runtime reconstructed state hash mismatch for seed {seed}"
            )
        deposit_unprojected, _ = production.make_delta_f_methods(W=None)
        rho = production.project_density_to_mode(
            deposit_unprojected(particles, grid), grid, config["mode"][0], config["mode"][1]
        )
        Ex, Ey = grid.poisson_solve(rho)
        reconstructed_mode0 = grid.mode_amplitude(Ex, config["mode"][0], config["mode"][1])
        with np.load(paths["history"], allow_pickle=False) as archive:
            times = np.asarray(archive["times"], dtype=np.float64)
            mode = np.asarray(archive["mode"], dtype=np.complex128)
            validity = np.asarray(archive["delta_f_ratio"], dtype=np.float64)
            archived_seed = int(np.asarray(archive["seed"]).flat[0])
        if reference_times is None:
            reference_times = times
        else:
            np.testing.assert_array_equal(reference_times, times)
        expected_samples = int(round(config["t_end"] / config["diagnostic_cadence"])) + 1
        if archived_seed != seed or times.size != expected_samples:
            raise AssertionError(f"history shape/seed mismatch for seed {seed}")
        if times[0] != 0.0 or abs(times[-1] - config["t_end"]) > 1.0e-12:
            raise AssertionError(f"history endpoints mismatch for seed {seed}")
        if not np.allclose(
            np.diff(times), config["diagnostic_cadence"], rtol=0.0, atol=2.0e-14
        ):
            raise AssertionError(f"diagnostic cadence mismatch for seed {seed}")
        mode0_delta = float(abs(mode[0] - reconstructed_mode0))
        if mode0_delta > reuse["time_zero_mode_absolute_tolerance"]:
            raise AssertionError(f"time-zero mode mismatch for seed {seed}")
        amplitudes = np.abs(mode)
        amplitudes_by_seed.append(amplitudes)
        fit_start = float(config["analysis"]["fit_t_start"])
        fit_stop = float(config["analysis"]["fit_t_stop"])
        fit_duration = fit_stop - fit_start
        gamma, r_squared, fit_samples = linear_fit(
            times, amplitudes, fit_start, fit_stop
        )
        individual_floor = estimate_floor(
            times,
            amplitudes,
            fit_start,
            float(config["analysis"]["floor_window_duration"]),
        )
        individual_efoldings = gamma * fit_duration
        individual_checks = {
            "history_complete_pass": True,
            "initialization_reconstruction_pass": True,
            "time_zero_mode_pass": True,
            "individual_fit_r_squared_pass": (
                r_squared >= config["acceptance"]["individual_fit_r_squared_min"]
            ),
            "individual_delta_f_pass": (
                float(np.max(validity)) < config["acceptance"]["delta_f_ratio_max"]
            ),
            "individual_positive_rate_observed": gamma > 0.0,
            "individual_efoldings_at_least_two_diagnostic": (
                individual_efoldings
                >= config["acceptance"]["fitted_efoldings_min"]
            ),
            "individual_fit_start_snr_at_least_five_diagnostic": (
                individual_floor["fit_start_snr"]
                >= config["acceptance"]["fit_start_snr_min"]
            ),
        }
        rows.append(
            {
                "seed": seed,
                "historical_saved_particle_array_hash": "NOT_SAVED",
                "production_runtime_reconstructed_initial_state_sha256": state_hash,
                "archived_mode0_real": float(mode[0].real),
                "archived_mode0_imag": float(mode[0].imag),
                "reconstructed_mode0_real": float(reconstructed_mode0.real),
                "reconstructed_mode0_imag": float(reconstructed_mode0.imag),
                "mode0_absolute_difference": mode0_delta,
                "history_samples": int(times.size),
                "history_t_start": float(times[0]),
                "history_t_stop": float(times[-1]),
                "max_delta_f_ratio": float(np.max(validity)),
                "individual_gamma": gamma,
                "individual_fit_r_squared": r_squared,
                "individual_fit_samples": fit_samples,
                "individual_fitted_efoldings": individual_efoldings,
                "individual_fit_start_snr_diagnostic": individual_floor[
                    "fit_start_snr"
                ],
                **individual_checks,
                "x_sha256": array_hashes["x"],
                "y_sha256": array_hashes["y"],
                "vx_sha256": array_hashes["vx"],
                "vy_sha256": array_hashes["vy"],
                "w_full_sha256": array_hashes["w_full"],
                "w_delta_sha256": array_hashes["w_delta"],
            }
        )

    if reference_times is None:
        raise RuntimeError("no CIC histories loaded")
    amplitude_matrix = np.stack(amplitudes_by_seed)
    ensemble_amplitude = np.mean(amplitude_matrix, axis=0)
    fit_start = float(config["analysis"]["fit_t_start"])
    fit_stop = float(config["analysis"]["fit_t_stop"])
    ensemble_gamma, ensemble_r_squared, ensemble_fit_samples = linear_fit(
        reference_times, ensemble_amplitude, fit_start, fit_stop
    )
    ensemble_floor = estimate_floor(
        reference_times,
        ensemble_amplitude,
        fit_start,
        float(config["analysis"]["floor_window_duration"]),
    )
    individual_rates = np.asarray(
        [row["individual_gamma"] for row in rows], dtype=np.float64
    )
    rng = np.random.default_rng(int(config["analysis"]["bootstrap_seed"]))
    indices = rng.integers(
        0,
        len(rows),
        size=(int(config["analysis"]["bootstrap_draws"]), len(rows)),
    )
    individual_mean_draws = np.mean(individual_rates[indices], axis=1)
    individual_mean_ci = np.percentile(individual_mean_draws, [2.5, 97.5])
    registered_quality = {
        "all_histories_complete": all(row["history_complete_pass"] for row in rows),
        "all_initialization_reconstructions_pass": all(
            row["initialization_reconstruction_pass"] for row in rows
        ),
        "all_individual_fits_linear": all(
            row["individual_fit_r_squared_pass"] for row in rows
        ),
        "all_individual_delta_f_valid": all(
            row["individual_delta_f_pass"] for row in rows
        ),
        "ensemble_fit_start_snr": (
            ensemble_floor["fit_start_snr"]
            >= config["acceptance"]["fit_start_snr_min"]
        ),
        "ensemble_linearity": (
            ensemble_r_squared >= config["acceptance"]["fit_r_squared_min"]
        ),
        "ensemble_fitted_duration": (
            ensemble_gamma * (fit_stop - fit_start)
            >= config["acceptance"]["fitted_efoldings_min"]
        ),
        "positive_individual_mean_growth": float(individual_mean_ci[0]) > 0.0,
        "no_seed_rejection": len(rows) == len(config["seeds"]),
    }
    if not all(registered_quality.values()):
        raise AssertionError("one or more recovered CIC quality gates failed")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    audit = {
        "status": "PASS_PRODUCTION_RUNTIME_RECONSTRUCTION_AND_CIC_QUALITY",
        "config_sha256": args.config_sha256,
        "legacy_campaign": reuse["campaign_id"],
        "legacy_campaign_root": str(args.legacy_campaign_root),
        "legacy_git_revision": legacy_git,
        "legacy_input_manifest_unhashed_files": legacy_unhashed,
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "expected": expected_runtime,
        },
        "pairing_evidence": (
            "Job 5201231 did not save particle arrays or array hashes. Pairing is "
            "established by deterministic reconstruction in the production runtime "
            "from the archived wrapper, frozen configuration, explicitly hashed "
            "initializer sources and seed. The new q050 runner must match these "
            "reconstructed aggregate hashes before integration, and the reducer "
            "must match all q050 per-array hashes to this audit. Matching seed "
            "numbers alone is not used as evidence."
        ),
        "historical_saved_particle_array_hash_available": False,
        "historical_saved_particle_array_hash": None,
        "pairing_decision": "ADEQUATE_FOR_CONDITIONAL_REUSE_WITH_EXPLICIT_PROVENANCE_LIMITATION",
        "registered_cic_quality_gates": registered_quality,
        "all_registered_cic_quality_gates_pass": True,
        "ensemble_amplitude_result": {
            "gamma": ensemble_gamma,
            "fit_r_squared": ensemble_r_squared,
            "fit_samples": ensemble_fit_samples,
            "fitted_efoldings": ensemble_gamma * (fit_stop - fit_start),
            "fit_window": [fit_start, fit_stop],
            **ensemble_floor,
        },
        "individual_rate_result": {
            "mean_gamma": float(np.mean(individual_rates)),
            "paired_seed_bootstrap_95pct_interval": [
                float(individual_mean_ci[0]),
                float(individual_mean_ci[1]),
            ],
            "fitted_efoldings_range": [
                float(min(row["individual_fitted_efoldings"] for row in rows)),
                float(max(row["individual_fitted_efoldings"] for row in rows)),
            ],
            "fitted_efoldings_by_seed": {
                str(row["seed"]): row["individual_fitted_efoldings"] for row in rows
            },
        },
        "seeds": {str(row["seed"]): row for row in rows},
        "input_file_sha256": file_hashes,
    }
    args.output.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    csv_path = args.output.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
