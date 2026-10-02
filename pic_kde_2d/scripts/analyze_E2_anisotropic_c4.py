#!/usr/bin/env python3
"""Reduce the paired E2 anisotropic C4 rate, noise, and transverse audits."""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import sys

import numpy as np

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pic_kde_2d.validation.landau_fit import fit_gamma_ensemble_fixed_window


def read_csv(path: pathlib.Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: pathlib.Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_rate_group(path: pathlib.Path, seeds: int) -> dict:
    amplitudes = []
    seed_ids = []
    times = None
    sigma_x_dx = None
    sigma_y_dy = None
    validity = []
    for seed in range(seeds):
        seed_path = path / f"seed{seed}"
        metadata = json.loads((seed_path / "result.json").read_text(encoding="utf-8"))
        with np.load(seed_path / "mode_history.npz") as data:
            this_times = np.asarray(data["times"], dtype=np.float64)
            amplitudes.append(np.abs(np.asarray(data["mode"], dtype=np.complex128)))
            seed_ids.append(int(np.asarray(data["seed"]).reshape(-1)[0]))
            validity.append(float(np.max(np.asarray(data["delta_f_ratio"]))))
        this_x = float(metadata.get("sigma_x_dx", metadata["sigma_dx"]))
        this_y = float(metadata.get("sigma_y_dy", metadata["sigma_dx"]))
        if sigma_x_dx is None:
            sigma_x_dx, sigma_y_dy = this_x, this_y
        elif this_x != sigma_x_dx or this_y != sigma_y_dy:
            raise RuntimeError(f"mixed directional widths in {path}")
        if times is None:
            times = this_times
        else:
            np.testing.assert_allclose(times, this_times, rtol=0.0, atol=0.0)
    if sorted(seed_ids) != list(range(seeds)):
        raise RuntimeError(f"{path}: expected paired seeds 0..{seeds - 1}")
    assert times is not None and sigma_x_dx is not None and sigma_y_dy is not None
    order = np.argsort(seed_ids)
    return {
        "sigma_x_dx": sigma_x_dx,
        "sigma_y_dy": sigma_y_dy,
        "times": times,
        "amplitudes": np.asarray(amplitudes, dtype=np.float64)[order],
        "max_delta_f_ratio": max(validity),
    }


def fit_rate(group: dict, indices: np.ndarray, t_start: float, t_stop: float) -> float:
    fit = fit_gamma_ensemble_fixed_window(
        [row for row in group["amplitudes"][indices]],
        group["times"],
        t_fit_start=t_start,
        t_fit_stop=t_stop,
        amp_floor=1.0e-30,
    )
    return float(fit["gamma"])


def rate_statistics(
    baseline: dict,
    groups: list[dict],
    t_start: float,
    t_stop: float,
    nboot: int,
) -> list[dict]:
    """Paired bootstrap rates against sigma=0 and isotropic selected width."""
    all_groups = [baseline, *groups]
    n_seeds = baseline["amplitudes"].shape[0]
    full = np.arange(n_seeds)
    central = [fit_rate(group, full, t_start, t_stop) for group in all_groups]
    isotropic_index = next(
        index
        for index, group in enumerate(groups)
        if np.isclose(group["sigma_x_dx"], group["sigma_y_dy"])
    )
    rng = np.random.default_rng(79231)
    rate_samples = [[] for _ in all_groups]
    ratio_zero_samples = [[] for _ in all_groups]
    ratio_iso_samples = [[] for _ in groups]
    for _ in range(nboot):
        indices = rng.integers(0, n_seeds, size=n_seeds)
        rates = [fit_rate(group, indices, t_start, t_stop) for group in all_groups]
        if not all(np.isfinite(rate) and rate > 0.0 for rate in rates):
            continue
        for index, rate in enumerate(rates):
            rate_samples[index].append(rate)
            ratio_zero_samples[index].append(rate / rates[0])
        isotropic_rate = rates[isotropic_index + 1]
        for index, rate in enumerate(rates[1:]):
            ratio_iso_samples[index].append(rate / isotropic_rate)

    output = []
    for index, group in enumerate(groups):
        rate_array = np.asarray(rate_samples[index + 1])
        zero_array = np.asarray(ratio_zero_samples[index + 1])
        iso_array = np.asarray(ratio_iso_samples[index])
        output.append(
            {
                "sigma_x_dx": group["sigma_x_dx"],
                "sigma_y_dy": group["sigma_y_dy"],
                "gamma": central[index + 1],
                "gamma_ci_lo": float(np.percentile(rate_array, 2.5)),
                "gamma_ci_hi": float(np.percentile(rate_array, 97.5)),
                "gamma_ratio_to_cic": central[index + 1] / central[0],
                "ratio_to_cic_ci_lo": float(np.percentile(zero_array, 2.5)),
                "ratio_to_cic_ci_hi": float(np.percentile(zero_array, 97.5)),
                "gamma_ratio_to_isotropic": (
                    central[index + 1] / central[isotropic_index + 1]
                ),
                "ratio_to_isotropic_ci_lo": float(np.percentile(iso_array, 2.5)),
                "ratio_to_isotropic_ci_hi": float(np.percentile(iso_array, 97.5)),
                "rate_gate_pass": bool(
                    np.percentile(zero_array, 2.5) >= 0.9
                    and np.percentile(zero_array, 97.5) <= 1.1
                ),
                "sigma_y_invariance_pass": bool(
                    np.percentile(iso_array, 2.5) >= 0.99
                    and np.percentile(iso_array, 97.5) <= 1.01
                ),
                "max_delta_f_ratio": group["max_delta_f_ratio"],
                "fit_t_start": t_start,
                "fit_t_stop": t_stop,
                "n_seeds": n_seeds,
            }
        )
    return output


def paired_ratio_ci(
    numerator: np.ndarray,
    denominator: np.ndarray,
    nboot: int,
    seed: int,
) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, numerator.size, size=(nboot, numerator.size))
    samples = np.mean(numerator[indices], axis=1) / np.mean(
        denominator[indices], axis=1
    )
    return (
        float(np.mean(numerator) / np.mean(denominator)),
        float(np.percentile(samples, 2.5)),
        float(np.percentile(samples, 97.5)),
    )


def load_noise_rows(root: pathlib.Path, seeds: int) -> list[dict]:
    rows = []
    for seed in range(seeds):
        path = root / f"seed{seed}" / "anisotropic_noise_seed.csv"
        selected = read_csv(path)
        if any(int(row["seed"]) != seed for row in selected):
            raise RuntimeError(f"seed mismatch in {path}")
        rows.extend(selected)
    return rows


def noise_statistics(rows: list[dict], seeds: int, nboot: int) -> list[dict]:
    keys = sorted(
        {
            (float(row["sigma_x_dx"]), float(row["sigma_y_dy"]))
            for row in rows
        }
    )
    values = {}
    for key in keys:
        selected = [
            row
            for row in rows
            if (float(row["sigma_x_dx"]), float(row["sigma_y_dy"])) == key
        ]
        selected.sort(key=lambda row: int(row["seed"]))
        if len(selected) != seeds:
            raise RuntimeError(f"noise key {key} has {len(selected)} seeds")
        values[key] = {
            name: np.asarray([float(row[name]) for row in selected])
            for name in (
                "mise_time_avg",
                "ky0_error_mse_time_avg",
                "ky_nonzero_error_mse_time_avg",
                "x_profile_mse_time_avg",
                "max_neutrality_defect",
                "max_relative_charge_error",
            )
        }
        if np.var(values[key]["mise_time_avg"], ddof=1) == 0.0:
            raise RuntimeError(f"zero across-seed MISE variance at {key}")

    cic = values[(0.0, 0.0)]
    isotropic_key = next(
        key
        for key in keys
        if key != (0.0, 0.0) and np.isclose(key[0], key[1])
    )
    isotropic = values[isotropic_key]
    output = []
    for index, key in enumerate(key for key in keys if key != (0.0, 0.0)):
        current = values[key]
        reduction = paired_ratio_ci(
            cic["mise_time_avg"], current["mise_time_avg"], nboot, 81000 + index
        )
        extra = paired_ratio_ci(
            isotropic["mise_time_avg"],
            current["mise_time_avg"],
            nboot,
            82000 + index,
        )
        transverse = paired_ratio_ci(
            isotropic["ky_nonzero_error_mse_time_avg"],
            current["ky_nonzero_error_mse_time_avg"],
            nboot,
            83000 + index,
        )
        output.append(
            {
                "sigma_x_dx": key[0],
                "sigma_y_dy": key[1],
                "mise_mean": float(np.mean(current["mise_time_avg"])),
                "noise_reduction_vs_cic": reduction[0],
                "noise_reduction_vs_cic_ci_lo": reduction[1],
                "noise_reduction_vs_cic_ci_hi": reduction[2],
                "extra_noise_reduction_vs_isotropic": extra[0],
                "extra_noise_reduction_vs_isotropic_ci_lo": extra[1],
                "extra_noise_reduction_vs_isotropic_ci_hi": extra[2],
                "transverse_error_reduction_vs_isotropic": transverse[0],
                "transverse_error_reduction_vs_isotropic_ci_lo": transverse[1],
                "transverse_error_reduction_vs_isotropic_ci_hi": transverse[2],
                "ky0_error_mse_mean": float(
                    np.mean(current["ky0_error_mse_time_avg"])
                ),
                "ky_nonzero_error_mse_mean": float(
                    np.mean(current["ky_nonzero_error_mse_time_avg"])
                ),
                "x_profile_mse_mean": float(
                    np.mean(current["x_profile_mse_time_avg"])
                ),
                "max_neutrality_defect": float(
                    np.max(current["max_neutrality_defect"])
                ),
                "max_relative_charge_error": float(
                    np.max(current["max_relative_charge_error"])
                ),
                "noise_gate_pass": bool(reduction[1] > 1.0),
                "n_seeds": seeds,
            }
        )
    return output


def linear_fit(times: np.ndarray, amplitudes: np.ndarray) -> tuple[float, float, float]:
    mask = np.isfinite(amplitudes) & (amplitudes > 0.0)
    if np.count_nonzero(mask) < 5:
        return float("nan"), float("nan"), float("nan")
    x = times[mask]
    y = np.log(amplitudes[mask])
    slope, intercept = np.polyfit(x, y, 1)
    fitted = slope * x + intercept
    residual = float(np.sum((y - fitted) ** 2))
    total = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - residual / total if total > 0.0 else 0.0
    return float(slope), float(r2), float(slope * (x[-1] - x[0]))


def transverse_statistics(
    root: pathlib.Path,
    seeds: int,
    eval_t_min: float,
    eval_t_max: float,
    protected_mode: int,
) -> list[dict]:
    seed_audits = []
    for seed in range(seeds):
        with np.load(root / f"seed{seed}" / "anisotropic_live_audit.npz") as data:
            seed_audits.append({name: np.asarray(data[name]) for name in data.files})
    widths = seed_audits[0]["sigma_y_dy"]
    times = seed_audits[0]["times"]
    for audit in seed_audits[1:]:
        np.testing.assert_allclose(widths, audit["sigma_y_dy"], rtol=0.0, atol=0.0)
        np.testing.assert_allclose(times, audit["times"], rtol=0.0, atol=0.0)

    spectra = np.stack([audit["field_amplitude"] for audit in seed_audits], axis=0)
    profiles = np.stack([audit["x_profiles"] for audit in seed_audits], axis=0)
    neutrality = np.stack([audit["neutrality_defect"] for audit in seed_audits])
    charge = np.stack([audit["relative_charge_error"] for audit in seed_audits])
    ky_fraction = np.stack([audit["ky_power_fraction"] for audit in seed_audits])
    isotropic_index = int(np.argmin(np.abs(widths - 0.18445488487247555)))
    fit_mask = (times >= eval_t_min) & (times <= eval_t_max)
    ensemble_spectra = np.mean(spectra, axis=0)
    ensemble_profiles = np.mean(profiles, axis=0)
    output = []

    for width_index, sigma_y in enumerate(widths):
        protected = ensemble_spectra[width_index, :, protected_mode, 0]
        protected_peak = float(np.max(protected[fit_mask]))
        ky_peak = float(np.max(ensemble_spectra[width_index, fit_mask, :, 1:]))
        emergent = []
        for mx in range(ensemble_spectra.shape[2]):
            for my in range(1, ensemble_spectra.shape[3]):
                amplitude = ensemble_spectra[width_index, fit_mask, mx, my]
                gamma, r2, efoldings = linear_fit(times[fit_mask], amplitude)
                peak_ratio = float(
                    np.max(amplitude) / max(protected_peak, np.finfo(float).tiny)
                )
                if (
                    gamma > 0.0
                    and r2 >= 0.98
                    and efoldings >= 0.5
                    and peak_ratio >= 0.1
                ):
                    emergent.append(
                        {
                            "mx": mx,
                            "my": my,
                            "gamma": gamma,
                            "r2": r2,
                            "efoldings": efoldings,
                            "peak_to_protected": peak_ratio,
                        }
                    )
        reference_profile = ensemble_profiles[isotropic_index, fit_mask]
        profile_delta = (
            ensemble_profiles[width_index, fit_mask] - reference_profile
        )
        full_profile_relative_l2 = float(
            np.linalg.norm(profile_delta)
            / max(np.linalg.norm(reference_profile), np.finfo(float).tiny)
        )
        profile_modes = np.fft.fft(
            ensemble_profiles[width_index, fit_mask], axis=-1
        )
        reference_modes = np.fft.fft(reference_profile, axis=-1)
        protected_profile = profile_modes[:, protected_mode]
        protected_reference = reference_modes[:, protected_mode]
        protected_profile_relative_l2 = float(
            np.linalg.norm(protected_profile - protected_reference)
            / max(np.linalg.norm(protected_reference), np.finfo(float).tiny)
        )
        output.append(
            {
                "sigma_y_dy": float(sigma_y),
                "mean_ky_power_fraction": float(
                    np.mean(ky_fraction[:, width_index, fit_mask])
                ),
                "max_ky_amplitude_to_protected": (
                    ky_peak / max(protected_peak, np.finfo(float).tiny)
                ),
                "full_x_profile_relative_l2_vs_isotropic": (
                    full_profile_relative_l2
                ),
                "protected_profile_mode_relative_l2_vs_isotropic": (
                    protected_profile_relative_l2
                ),
                "max_neutrality_defect_live": float(
                    np.max(neutrality[:, width_index])
                ),
                "max_relative_charge_error_live": float(
                    np.max(charge[:, width_index])
                ),
                "resolved_emergent_ky_modes": emergent,
                "n_resolved_emergent_ky_modes": len(emergent),
                "ky_emergence_gate_pass": len(emergent) == 0,
                "profile_gate_pass": protected_profile_relative_l2 <= 0.05,
                "neutrality_gate_pass": bool(
                    np.max(neutrality[:, width_index]) <= 1.0e-12
                ),
            }
        )
    return output


def plot_result(path: pathlib.Path, rows: list[dict]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    sigma_y = np.asarray([row["sigma_y_dy"] for row in rows])
    noise = np.asarray([row["noise_reduction_vs_cic"] for row in rows])
    noise_lo = np.asarray([row["noise_reduction_vs_cic_ci_lo"] for row in rows])
    noise_hi = np.asarray([row["noise_reduction_vs_cic_ci_hi"] for row in rows])
    fidelity = np.asarray([row["gamma_ratio_to_cic"] for row in rows])
    fid_lo = np.asarray([row["ratio_to_cic_ci_lo"] for row in rows])
    fid_hi = np.asarray([row["ratio_to_cic_ci_hi"] for row in rows])
    fig, left = plt.subplots(figsize=(8.2, 5.2))
    right = left.twinx()
    left.errorbar(
        sigma_y,
        noise,
        yerr=[noise - noise_lo, noise_hi - noise],
        marker="o",
        color="#12664f",
        capsize=3,
        label="density-noise reduction vs CIC",
    )
    right.errorbar(
        sigma_y,
        fidelity,
        yerr=[fidelity - fid_lo, fid_hi - fidelity],
        marker="s",
        color="#bc4b2c",
        capsize=3,
        label=r"$\gamma_{12}/\gamma_{12,\mathrm{CIC}}$",
    )
    right.axhspan(0.0, 0.9, color="#d1495b", alpha=0.12)
    right.axhline(0.9, color="#d1495b", linestyle="--", linewidth=1)
    left.set_xlabel(r"$\sigma_y/\Delta y$ at fixed $\sigma_x=0.18445\Delta x$")
    left.set_ylabel("noise reduction")
    right.set_ylabel("protected-mode fidelity")
    left.set_xscale("log")
    handles = left.get_legend_handles_labels()[0] + right.get_legend_handles_labels()[0]
    labels = left.get_legend_handles_labels()[1] + right.get_legend_handles_labels()[1]
    left.legend(handles, labels, loc="center left")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rate-root", type=pathlib.Path, required=True)
    parser.add_argument("--rate-baseline", type=pathlib.Path, required=True)
    parser.add_argument("--noise-root", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--sigma-y-dy", default="0.18445488487247555,1,2,4")
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--nboot", type=int, default=5000)
    parser.add_argument("--fit-t-start", type=float, default=24.32)
    parser.add_argument("--fit-t-stop", type=float, default=59.68)
    parser.add_argument("--eval-t-min", type=float, default=0.5)
    parser.add_argument("--eval-t-max", type=float, default=4.0)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    widths = [float(value) for value in args.sigma_y_dy.split(",")]

    baseline = load_rate_group(args.rate_baseline, args.seeds)
    rate_groups = [
        load_rate_group(args.rate_root / f"sy_{index}", args.seeds)
        for index, _ in enumerate(widths)
    ]
    rate_rows = rate_statistics(
        baseline,
        rate_groups,
        args.fit_t_start,
        args.fit_t_stop,
        args.nboot,
    )
    noise_rows = noise_statistics(
        load_noise_rows(args.noise_root, args.seeds),
        args.seeds,
        args.nboot,
    )
    transverse_rows = transverse_statistics(
        args.noise_root,
        args.seeds,
        args.eval_t_min,
        args.eval_t_max,
        protected_mode=12,
    )
    by_width_noise = {row["sigma_y_dy"]: row for row in noise_rows}
    by_width_transverse = {row["sigma_y_dy"]: row for row in transverse_rows}
    rows = []
    for rate in rate_rows:
        width = rate["sigma_y_dy"]
        row = {**rate, **by_width_noise[width], **by_width_transverse[width]}
        row["all_gates_pass"] = bool(
            row["rate_gate_pass"]
            and row["sigma_y_invariance_pass"]
            and row["noise_gate_pass"]
            and row["ky_emergence_gate_pass"]
            and row["profile_gate_pass"]
            and row["neutrality_gate_pass"]
        )
        rows.append(row)
    selected = [
        row for row in rows if row["all_gates_pass"]
    ]
    recommendation = max(selected, key=lambda row: row["sigma_y_dy"]) if selected else None
    write_csv(args.output / "anisotropic_c4_summary.csv", rows)
    result = {
        "benchmark": "E2 two-stream protected edge m=12",
        "sigma_x_dx": rows[0]["sigma_x_dx"],
        "sigma_y_sweep_dy": widths,
        "rate_reference": "paired projected linearized delta-f sigma=0",
        "noise_reference": (
            "paired PPC-1024 CIC density on the frozen matrix trajectory"
        ),
        "transverse_audit": (
            "nonprojected full-f live anisotropic deposit/gather at PPC 32"
        ),
        "gates": {
            "protected_rate": "paired 95% ratio interval within [0.9, 1.1]",
            "sigma_y_invariance": (
                "paired 95% ratio to isotropic selected width within [0.99, 1.01]"
            ),
            "noise": "paired 95% noise-reduction lower bound > 1",
            "ky_emergence": (
                "no ky!=0 mode with R2>=0.98, >=0.5 e-folding, and "
                "peak amplitude >=0.1 of protected m=12"
            ),
            "profile": (
                "paired ensemble protected m=12 component of the ky=0 "
                "profile has relative L2 change <= 0.05"
            ),
            "neutrality": "background-subtracted relative k=0 defect <= 1e-12",
        },
        "rows": rows,
        "recommended_sigma_y_dy": (
            None if recommendation is None else recommendation["sigma_y_dy"]
        ),
        "recommended_noise_reduction_vs_cic": (
            None
            if recommendation is None
            else recommendation["noise_reduction_vs_cic"]
        ),
        "status": "pass" if recommendation is not None else "failed",
    }
    (args.output / "anisotropic_c4_result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    plot_result(args.output / "anisotropic_c4_tradeoff.png", rows)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
