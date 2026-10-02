#!/usr/bin/env python3
"""Reduce the frozen paired CIC/q050_x000 E2 validation without exclusions."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


ARMS = ("cic", "q050_x000")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def linear_fit(times: np.ndarray, values: np.ndarray, start: float, stop: float):
    mask = (
        (times >= start)
        & (times <= stop)
        & np.isfinite(values)
        & (values > 1.0e-300)
    )
    if np.count_nonzero(mask) < 5:
        raise RuntimeError("fewer than five finite positive fit samples")
    x = times[mask]
    y = np.log(values[mask])
    slope, intercept = np.polyfit(x, y, 1)
    fitted = slope * x + intercept
    residual = float(np.sum((y - fitted) ** 2))
    total = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = 1.0 - residual / total if total > 0.0 else 0.0
    return float(slope), float(intercept), float(r_squared), mask


def estimate_floor(
    times: np.ndarray,
    amplitudes: np.ndarray,
    fit_start: float,
    duration: float,
) -> dict:
    """Frozen E2 quietest pre-fit rolling-median floor on mean magnitude."""
    ensemble = np.mean(amplitudes, axis=0)
    dt = float(np.median(np.diff(times)))
    width = max(5, int(round(duration / dt)))
    stop_limit = int(np.searchsorted(times, fit_start, side="left"))
    if stop_limit < width:
        raise RuntimeError("insufficient pre-fit samples for floor estimate")
    candidates = []
    for begin in range(stop_limit - width + 1):
        end = begin + width
        candidates.append((float(np.median(ensemble[begin:end])), begin, end))
    floor, begin, end = min(candidates, key=lambda item: item[0])
    fit_index = int(np.searchsorted(times, fit_start, side="left"))
    start_amplitude = float(ensemble[fit_index])
    return {
        "noise_floor": floor,
        "floor_t_start": float(times[begin]),
        "floor_t_stop": float(times[end - 1]),
        "fit_start_amplitude": start_amplitude,
        "fit_start_snr": start_amplitude / max(floor, 1.0e-300),
        "definition": "quietest rolling median of mean seed magnitudes in a complete two-time-unit pre-fit window",
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty table {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_histories(root: Path, config: dict, config_hash: str) -> dict[str, dict]:
    loaded: dict[str, dict] = {}
    input_hashes: dict[str, str] = {}
    pairing: dict[int, dict[str, str]] = {int(seed): {} for seed in config["seeds"]}
    pairing_arrays: dict[int, dict[str, dict[str, str]]] = {
        int(seed): {} for seed in config["seeds"]
    }
    reuse_path = root / "cic" / "reuse_audit.json"
    reuse_audit = None
    if reuse_path.is_file():
        reuse_audit = json.loads(reuse_path.read_text(encoding="utf-8"))
        input_hashes[str(reuse_path.relative_to(root))] = sha256_file(reuse_path)
        if reuse_audit["config_sha256"] != config_hash:
            raise AssertionError("reused-CIC audit config hash mismatch")
        if reuse_audit["status"] != "PASS_PRODUCTION_RUNTIME_RECONSTRUCTION_AND_CIC_QUALITY":
            raise AssertionError("reused-CIC audit did not pass")
    reference_times = None
    for arm in ARMS:
        modes = []
        validity = []
        results = []
        for seed in config["seeds"]:
            directory = root / arm / f"seed{seed}"
            result_path = directory / "result.json"
            history_path = directory / "mode_history.npz"
            manifest_path = directory / "campaign_manifest.json"
            required_paths = [result_path, history_path]
            legacy_reuse = arm == "cic" and reuse_audit is not None
            if legacy_reuse:
                required_paths.append(manifest_path)
            for path in required_paths:
                if not path.is_file():
                    raise FileNotFoundError(path)
                input_hashes[str(path.relative_to(root))] = sha256_file(path)
            result = json.loads(result_path.read_text(encoding="utf-8"))
            if legacy_reuse:
                audit_seed = reuse_audit["seeds"].get(str(seed))
                if audit_seed is None:
                    raise AssertionError(f"missing reused-CIC audit for seed {seed}")
                state_hash = audit_seed[
                    "production_runtime_reconstructed_initial_state_sha256"
                ]
                array_hashes = {
                    name: audit_seed[f"{name}_sha256"]
                    for name in ("x", "y", "vx", "vy", "w_full", "w_delta")
                }
                if int(result["seed"]) != int(seed):
                    raise AssertionError(f"legacy seed mismatch in {result_path}")
            else:
                if result["config_sha256"] != config_hash:
                    raise AssertionError(f"config hash mismatch in {result_path}")
                if result["arm"] != arm or int(result["seed"]) != int(seed):
                    raise AssertionError(f"arm/seed mismatch in {result_path}")
                state_hash = result["initial_state_sha256"]
                array_hashes = result["initial_array_sha256"]
                if not result.get(
                    "initial_state_matches_production_runtime_reconstruction", False
                ):
                    raise AssertionError(
                        f"production-runtime state match not certified in {result_path}"
                    )
            if int(result["ppc"]) != int(config["loading"]["ppc"]):
                raise AssertionError(f"PPC mismatch in {result_path}")
            if [int(result["kx_mode"]), int(result["ky_mode"])] != config["mode"]:
                raise AssertionError(f"mode mismatch in {result_path}")
            if not bool(result["target_mode_projection"]):
                raise AssertionError(f"target-mode projection disabled in {result_path}")
            pairing[int(seed)][arm] = state_hash
            pairing_arrays[int(seed)][arm] = array_hashes
            with np.load(history_path, allow_pickle=False) as archive:
                times = np.asarray(archive["times"], dtype=np.float64)
                mode = np.asarray(archive["mode"], dtype=np.complex128)
                current_validity = np.asarray(
                    archive["delta_f_ratio"], dtype=np.float64
                )
                archived_seed = int(np.asarray(archive["seed"]).flat[0])
            if archived_seed != int(seed):
                raise AssertionError(f"seed mismatch in {history_path}")
            if reference_times is None:
                reference_times = times
            else:
                np.testing.assert_array_equal(reference_times, times)
            modes.append(mode)
            validity.append(current_validity)
            results.append(result)
        loaded[arm] = {
            "modes": np.stack(modes),
            "amplitudes": np.abs(np.stack(modes)),
            "validity": np.stack(validity),
            "results": results,
        }
    for seed, hashes in pairing.items():
        if set(hashes) != set(ARMS):
            raise AssertionError(f"missing pairing hash for seed {seed}")
        if hashes["cic"] != hashes["q050_x000"]:
            raise AssertionError(f"initial particle state differs within seed {seed}")
        for name in ("x", "y", "vx", "vy", "w_full", "w_delta"):
            if pairing_arrays[seed]["cic"][name] != pairing_arrays[seed]["q050_x000"][name]:
                raise AssertionError(
                    f"initial {name} array differs within seed {seed}"
                )
    if reference_times is None:
        raise RuntimeError("no histories loaded")
    loaded["times"] = reference_times
    loaded["input_sha256"] = input_hashes
    loaded["pairing"] = pairing
    loaded["pairing_arrays"] = pairing_arrays
    loaded["cic_reuse_audit"] = reuse_audit
    return loaded


def bootstrap(
    loaded: dict[str, dict],
    times: np.ndarray,
    start: float,
    stop: float,
    draws: int,
    seed: int,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    nseed = loaded["cic"]["amplitudes"].shape[0]
    indices = rng.integers(0, nseed, size=(draws, nseed))
    individual = {
        arm: np.asarray(
            [linear_fit(times, row, start, stop)[0] for row in loaded[arm]["amplitudes"]],
            dtype=np.float64,
        )
        for arm in ARMS
    }
    individual_cic = np.mean(individual["cic"][indices], axis=1)
    individual_q050 = np.mean(individual["q050_x000"][indices], axis=1)
    ensemble_cic = np.empty(draws, dtype=np.float64)
    ensemble_q050 = np.empty(draws, dtype=np.float64)
    for draw, sample in enumerate(indices):
        ensemble_cic[draw] = linear_fit(
            times,
            np.mean(loaded["cic"]["amplitudes"][sample], axis=0),
            start,
            stop,
        )[0]
        ensemble_q050[draw] = linear_fit(
            times,
            np.mean(loaded["q050_x000"]["amplitudes"][sample], axis=0),
            start,
            stop,
        )[0]
    return {
        "indices": indices,
        "individual_cic_gamma": individual_cic,
        "individual_q050_gamma": individual_q050,
        "individual_ratio": individual_q050 / individual_cic,
        "ensemble_cic_gamma": ensemble_cic,
        "ensemble_q050_gamma": ensemble_q050,
        "ensemble_ratio": ensemble_q050 / ensemble_cic,
    }


def interval(values: np.ndarray) -> list[float]:
    lo, hi = np.percentile(np.asarray(values, dtype=np.float64), [2.5, 97.5])
    return [float(lo), float(hi)]


def make_figure(
    output: Path,
    times: np.ndarray,
    loaded: dict[str, dict],
    seed_rows: list[dict],
    start: float,
    stop: float,
) -> None:
    import matplotlib as mpl

    mpl.use("Agg", force=True)
    import matplotlib.pyplot as plt

    mpl.rcParams.update(
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
    colors = {"cic": "#14213D", "q050_x000": "#8B1E1E"}
    labels = {"cic": "CIC", "q050_x000": "selected q050_x000"}
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8))
    for arm in ARMS:
        ensemble = np.mean(loaded[arm]["amplitudes"], axis=0)
        axes[0].semilogy(times, ensemble, color=colors[arm], label=labels[arm])
    axes[0].axvspan(start, stop, color="0.7", alpha=0.18, linewidth=0)
    axes[0].set(xlabel=r"$t\omega_{pe}$", ylabel=r"mean $|\widehat E_{12,1}|$")
    axes[0].legend(frameon=False)

    by_arm = {arm: [row for row in seed_rows if row["arm"] == arm] for arm in ARMS}
    seeds = np.asarray([row["seed"] for row in by_arm["cic"]], dtype=int)
    cic = np.asarray([row["individual_gamma"] for row in by_arm["cic"]])
    q050 = np.asarray([row["individual_gamma"] for row in by_arm["q050_x000"]])
    axes[1].plot(cic, q050, "o", color="#2A7F62", markersize=4)
    limits = [min(cic.min(), q050.min()), max(cic.max(), q050.max())]
    axes[1].plot(limits, limits, "--", color="0.5", linewidth=1)
    for seed, x, y in zip(seeds, cic, q050):
        axes[1].annotate(str(seed), (x, y), xytext=(3, 2), textcoords="offset points", fontsize=6)
    axes[1].set(
        xlabel=r"CIC individual $\widehat\gamma_s$",
        ylabel=r"q050 individual $\widehat\gamma_s$",
    )
    fig.tight_layout()
    for suffix in ("pdf", "svg"):
        fig.savefig(output / f"fig_q050_exact_width_validation.{suffix}")
    fig.savefig(output / "fig_q050_exact_width_validation.png", dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--config-sha256", required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if sha256_file(args.config) != args.config_sha256:
        raise AssertionError("frozen config hash mismatch")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    args.output.mkdir(parents=True)

    loaded = load_histories(args.input, config, args.config_sha256)
    times = loaded["times"]
    start = float(config["analysis"]["fit_t_start"])
    stop = float(config["analysis"]["fit_t_stop"])
    seed_rows: list[dict] = []
    arm_summaries: dict[str, dict] = {}
    for arm in ARMS:
        individual = []
        individual_r2 = []
        for seed, amplitudes in zip(config["seeds"], loaded[arm]["amplitudes"]):
            rate, _, r2, _ = linear_fit(times, amplitudes, start, stop)
            individual.append(rate)
            individual_r2.append(r2)
            seed_rows.append(
                {
                    "seed": seed,
                    "arm": arm,
                    "individual_gamma": rate,
                    "individual_r_squared": r2,
                    "max_delta_f_ratio": float(
                        np.max(loaded[arm]["validity"][len(individual) - 1])
                    ),
                    "initial_state_sha256": loaded["pairing"][int(seed)][arm],
                }
            )
        ensemble = np.mean(loaded[arm]["amplitudes"], axis=0)
        ensemble_rate, _, ensemble_r2, _ = linear_fit(times, ensemble, start, stop)
        floor = estimate_floor(
            times,
            loaded[arm]["amplitudes"],
            start,
            float(config["analysis"]["floor_window_duration"]),
        )
        arm_summaries[arm] = {
            "individual_rate_mean": float(np.mean(individual)),
            "individual_rate_sd": float(np.std(individual, ddof=1)),
            "individual_r_squared_min": float(np.min(individual_r2)),
            "individual_r_squared_median": float(np.median(individual_r2)),
            "ensemble_amplitude_gamma": ensemble_rate,
            "ensemble_amplitude_r_squared": ensemble_r2,
            "fitted_efoldings": ensemble_rate * (stop - start),
            "max_delta_f_ratio": float(np.max(loaded[arm]["validity"])),
            **floor,
        }

    draws = bootstrap(
        loaded,
        times,
        start,
        stop,
        int(config["analysis"]["bootstrap_draws"]),
        int(config["analysis"]["bootstrap_seed"]),
    )
    for arm, key in (
        ("cic", "individual_cic_gamma"),
        ("q050_x000", "individual_q050_gamma"),
    ):
        arm_summaries[arm]["individual_rate_ci"] = interval(draws[key])
    for arm, key in (
        ("cic", "ensemble_cic_gamma"),
        ("q050_x000", "ensemble_q050_gamma"),
    ):
        arm_summaries[arm]["ensemble_amplitude_gamma_ci"] = interval(draws[key])

    primary_ratio = (
        arm_summaries["q050_x000"]["individual_rate_mean"]
        / arm_summaries["cic"]["individual_rate_mean"]
    )
    secondary_ratio = (
        arm_summaries["q050_x000"]["ensemble_amplitude_gamma"]
        / arm_summaries["cic"]["ensemble_amplitude_gamma"]
    )
    primary_ci = interval(draws["individual_ratio"])
    secondary_ci = interval(draws["ensemble_ratio"])
    rules = config["acceptance"]
    tolerance = rules["rate_preservation_interval"]
    quality = {
        arm: {
            "fit_start_snr": arm_summaries[arm]["fit_start_snr"]
            >= rules["fit_start_snr_min"],
            "ensemble_linearity": arm_summaries[arm]["ensemble_amplitude_r_squared"]
            >= rules["fit_r_squared_min"],
            "all_individual_fits_linear": arm_summaries[arm]["individual_r_squared_min"]
            >= rules["individual_fit_r_squared_min"],
            "fitted_duration": arm_summaries[arm]["fitted_efoldings"]
            >= rules["fitted_efoldings_min"],
            "delta_f_validity": arm_summaries[arm]["max_delta_f_ratio"]
            < rules["delta_f_ratio_max"],
            "positive_individual_mean_growth": arm_summaries[arm]["individual_rate_ci"][0]
            > 0.0,
            "positive_ensemble_growth": arm_summaries[arm]["ensemble_amplitude_gamma_ci"][0]
            > 0.0,
        }
        for arm in ARMS
    }
    all_quality = all(all(values.values()) for values in quality.values())
    primary_acceptance = (
        all_quality
        and primary_ci[0] >= tolerance[0]
        and primary_ci[1] <= tolerance[1]
    )
    secondary_acceptance = (
        all_quality
        and secondary_ci[0] >= tolerance[0]
        and secondary_ci[1] <= tolerance[1]
    )

    summary = {
        "campaign_id": config["campaign_id"],
        "config_sha256": args.config_sha256,
        "cic_source": (
            "archived job 5201231 with deterministic reconstruction audit"
            if loaded["cic_reuse_audit"] is not None
            else "new paired run"
        ),
        "cic_historical_saved_particle_array_hash_available": (
            None
            if loaded["cic_reuse_audit"] is None
            else loaded["cic_reuse_audit"][
                "historical_saved_particle_array_hash_available"
            ]
        ),
        "pairing_basis": (
            None
            if loaded["cic_reuse_audit"] is None
            else loaded["cic_reuse_audit"]["pairing_evidence"]
        ),
        "status": "COMPLETE" if all_quality else "UNRESOLVED_QUALITY_GATE",
        "fit_window": [start, stop],
        "n_paired_seeds": len(config["seeds"]),
        "arms": arm_summaries,
        "primary_individual_rate_mean_ratio": primary_ratio,
        "primary_paired_95pct_interval": primary_ci,
        "primary_preservation_established": primary_acceptance,
        "secondary_ensemble_amplitude_ratio": secondary_ratio,
        "secondary_paired_95pct_interval": secondary_ci,
        "secondary_preservation_established": secondary_acceptance,
        "quality_gates": quality,
        "model_operating_ratio": config["model_prediction"]["q050_rate_ratio"],
        "model_operating_gate": config["model_prediction"]["q050_rate_ratio"]
        >= rules["model_operating_ratio_min"],
        "interpretation": (
            "Primary and secondary estimands are reported separately. An interval "
            "crossing 0.90 is unresolved for preservation; it is not evidence that "
            "distortion exceeds ten percent."
        ),
        "input_sha256": loaded["input_sha256"],
    }
    write_csv(args.output / "individual_seed_rates.csv", seed_rows)
    np.savez_compressed(args.output / "paired_bootstrap_draws.npz", **draws)
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    make_figure(args.output, times, loaded, seed_rows, start, stop)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
