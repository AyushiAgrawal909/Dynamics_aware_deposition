#!/usr/bin/env python3
"""Step 2.2: zero-parameter warm two-stream surrogate validation.

This is the warm-beam replacement for ``validate_E2_cold_surrogate.py``.  It
uses the actual frozen E2 transfer chain:

    T_eff^2(k, sigma) = sinc(k dx / 2)^4 * exp(-(k sigma)^2)

and inserts that coupling into the symmetric warm two-stream dispersion:

    1 + T_eff^2 * omega_p^2 / (2 k^2 v_th^2)
        * sum_{b=+,-} [1 + zeta_b Z(zeta_b)] = 0

where zeta_+ = (omega - k v0) / (sqrt(2) k v_th) and
zeta_- = (omega + k v0) / (sqrt(2) k v_th).

The validation has no fitted parameters.  It compares the predicted growth-rate
ratio to the measured E2 gamma sweep on the pre-stabilization range and checks
whether the predicted near-zero-growth crossing lands near the measured
stabilization at sigma ~= 1 dx.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pic_kde_2d.benchmarks.theory import Z_function  # noqa: E402


def read_csv(path: pathlib.Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def explicit_sinc(x: float | np.ndarray) -> float | np.ndarray:
    x_arr = np.asarray(x, dtype=np.float64)
    out = np.ones_like(x_arr)
    mask = np.abs(x_arr) > 1.0e-14
    out[mask] = np.sin(x_arr[mask]) / x_arr[mask]
    if np.ndim(x) == 0:
        return float(out)
    return out


def plasma_Z(zeta: complex) -> complex:
    """Stable plasma dispersion function for the E2 warm-beam regime.

    The E2 beams have zeta magnitudes O(10).  The dependency-free fallback in
    ``benchmarks.theory`` uses a power series for erf and is not stable there,
    so use the standard large-|zeta| asymptotic expansion before falling back.
    """
    z = complex(zeta)
    if abs(z) > 8.0:
        inv = 1.0 / z
        inv2 = inv * inv
        # Z(z) ~ -1/z * [1 + 1/(2z^2) + 3/(4z^4)
        #                 + 15/(8z^6) + 105/(16z^8)]
        series = 1.0 + 0.5 * inv2 + 0.75 * inv2**2 + 1.875 * inv2**3 + 6.5625 * inv2**4
        return -inv * series
    return Z_function(z)


def warm_dielectric(
    omega: complex,
    k: float,
    v0: float,
    vth: float,
    coupling: float,
    omega_p: float = 1.0,
) -> complex:
    """Warm symmetric two-stream dielectric for the E2 transfer coupling."""
    denom = math.sqrt(2.0) * k * vth
    zeta_plus = (omega - k * v0) / denom
    zeta_minus = (omega + k * v0) / denom
    susceptibility = (1.0 + zeta_plus * plasma_Z(zeta_plus)) + (
        1.0 + zeta_minus * plasma_Z(zeta_minus)
    )
    return 1.0 + coupling * omega_p * omega_p / (2.0 * k * k * vth * vth) * susceptibility


def _real_dielectric_for_growth(gamma: float, k: float, v0: float, vth: float, coupling: float) -> float:
    """Real part of D(omega=i*gamma).  Symmetry makes the imaginary part cancel."""
    return float(np.real(warm_dielectric(1j * gamma, k, v0, vth, coupling)))


def warm_growth_rate(
    k: float,
    v0: float,
    vth: float,
    coupling: float,
    gamma_max: float = 2.0,
    n_scan: int = 500,
) -> float:
    """Find the positive purely growing warm two-stream root, or return 0."""
    # Avoid exactly gamma=0 because the Landau contour can be numerically sharp
    # there; a tiny positive imaginary part selects the unstable branch.
    grid = np.concatenate(([1.0e-9], np.geomspace(1.0e-7, gamma_max, n_scan)))
    values = np.array([_real_dielectric_for_growth(float(g), k, v0, vth, coupling) for g in grid])

    intervals: list[tuple[float, float]] = []
    for i in range(len(grid) - 1):
        f0 = values[i]
        f1 = values[i + 1]
        if not (np.isfinite(f0) and np.isfinite(f1)):
            continue
        if f0 == 0.0:
            intervals.append((float(grid[i]), float(grid[i])))
        elif f0 * f1 < 0.0:
            intervals.append((float(grid[i]), float(grid[i + 1])))

    if not intervals:
        return 0.0

    roots = []
    for lo, hi in intervals:
        if lo == hi:
            roots.append(lo)
            continue
        flo = _real_dielectric_for_growth(lo, k, v0, vth, coupling)
        fhi = _real_dielectric_for_growth(hi, k, v0, vth, coupling)
        for _ in range(90):
            mid = 0.5 * (lo + hi)
            fmid = _real_dielectric_for_growth(mid, k, v0, vth, coupling)
            if abs(fmid) < 1.0e-12 or (hi - lo) < 1.0e-10:
                lo = hi = mid
                break
            if flo * fmid <= 0.0:
                hi = mid
                fhi = fmid
            else:
                lo = mid
                flo = fmid
        _ = fhi
        roots.append(0.5 * (lo + hi))

    # The physical instability is the largest positive growth root.
    return float(max(roots))


def transfer_coupling(sigma_dx: np.ndarray, args: argparse.Namespace) -> dict:
    dx = args.Lx / args.Nx
    k = 2.0 * math.pi * args.kx_mode / args.Lx
    mode_fraction = args.kx_mode / args.Nx
    if mode_fraction >= 0.25:
        raise AssertionError(f"alias safety failed: kx_mode/Nx={mode_fraction:.3f} >= 0.25")
    x = k * dx / 2.0
    s_cic_single = explicit_sinc(x) ** 2
    s_cic_deposit_gather = s_cic_single ** 2
    sigma_phys = sigma_dx * dx
    W2 = np.exp(-(k * sigma_phys) ** 2)
    coupling = s_cic_deposit_gather * W2
    return {
        "dx": dx,
        "k": k,
        "kdx": k * dx,
        "mode_fraction": mode_fraction,
        "s_cic_single": s_cic_single,
        "s_cic_deposit_gather": s_cic_deposit_gather,
        "coupling": coupling,
    }


def compute_prediction(sigma_dx: np.ndarray, args: argparse.Namespace) -> dict:
    transfer = transfer_coupling(sigma_dx, args)
    gamma = np.array(
        [
            warm_growth_rate(
                transfer["k"],
                args.v0,
                args.vth,
                float(c),
                gamma_max=args.gamma_max,
                n_scan=args.n_scan,
            )
            for c in transfer["coupling"]
        ],
        dtype=np.float64,
    )
    sigma_min = float(np.min(sigma_dx))
    transfer_ref = transfer_coupling(np.asarray([sigma_min]), args)
    gamma_ref = warm_growth_rate(
        transfer["k"],
        args.v0,
        args.vth,
        float(transfer_ref["coupling"][0]),
        gamma_max=args.gamma_max,
        n_scan=args.n_scan,
    )
    ratio = gamma / gamma_ref if gamma_ref > 0.0 else np.full_like(gamma, np.nan)
    transfer.update({"gamma_pred": gamma, "ratio_pred": ratio, "gamma_ref_pred": gamma_ref})
    return transfer


def first_crossing(x: np.ndarray, y: np.ndarray, threshold: float) -> float | None:
    """Return first x where y <= threshold using linear interpolation."""
    for i in range(1, len(x)):
        if y[i] <= threshold:
            if y[i - 1] == y[i]:
                return float(x[i])
            t = (threshold - y[i - 1]) / (y[i] - y[i - 1])
            return float(x[i - 1] + t * (x[i] - x[i - 1]))
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gamma-csv", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--Nx", type=int, default=128)
    parser.add_argument("--Lx", type=float, default=64.0)
    parser.add_argument("--kx-mode", type=int, default=12)
    parser.add_argument("--v0", type=float, default=0.8)
    parser.add_argument("--vth", type=float, default=0.05)
    parser.add_argument("--omega-p", type=float, default=1.0)
    parser.add_argument("--prekill-max-sigma", type=float, default=0.8)
    parser.add_argument("--measured-stabilization-sigma", type=float, default=1.0)
    parser.add_argument("--stabilization-ratio", type=float, default=0.05)
    parser.add_argument("--rmse-threshold", type=float, default=0.15)
    parser.add_argument("--crossing-rel-threshold", type=float, default=0.20)
    parser.add_argument("--gamma-max", type=float, default=2.0)
    parser.add_argument("--n-scan", type=int, default=500)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    rows = read_csv(args.gamma_csv)
    sigma = np.asarray([float(row["sigma_dx"]) for row in rows], dtype=np.float64)
    ratio = np.asarray([float(row["gamma_ratio"]) for row in rows], dtype=np.float64)
    gamma0 = float(rows[0]["gamma_sigma"])
    ratio_lo = np.asarray([float(row["ci_lo"]) / gamma0 for row in rows], dtype=np.float64)
    ratio_hi = np.asarray([float(row["ci_hi"]) / gamma0 for row in rows], dtype=np.float64)

    pred = compute_prediction(sigma, args)
    ratio_pred = np.asarray(pred["ratio_pred"], dtype=np.float64)

    prekill = sigma <= args.prekill_max_sigma
    rmse = float(np.sqrt(np.mean((ratio[prekill] - ratio_pred[prekill]) ** 2)))

    sig_dense = np.geomspace(0.2, 3.0, 240)
    pred_dense = compute_prediction(sig_dense, args)
    sigma_cross = first_crossing(sig_dense, np.asarray(pred_dense["ratio_pred"]), args.stabilization_ratio)
    crossing_rel_error = (
        abs(sigma_cross - args.measured_stabilization_sigma) / args.measured_stabilization_sigma
        if sigma_cross is not None
        else math.inf
    )
    rmse_pass = bool(rmse < args.rmse_threshold)
    crossing_pass = bool(crossing_rel_error < args.crossing_rel_threshold)
    passed = bool(rmse_pass and crossing_pass)

    # A0-like near-marginality diagnostic, matching the cold script's convention.
    A0_cold_equivalent = (args.omega_p / (float(pred["k"]) * args.v0)) ** 2 * float(
        pred["s_cic_deposit_gather"]
    )

    summary = {
        "A0_cold_equivalent": float(A0_cold_equivalent),
        "S_CIC_single": float(pred["s_cic_single"]),
        "S_CIC_deposit_gather": float(pred["s_cic_deposit_gather"]),
        "alias_safe": bool(pred["mode_fraction"] < 0.25),
        "k": float(pred["k"]),
        "kdx": float(pred["kdx"]),
        "mode_fraction": float(pred["mode_fraction"]),
        "v0": float(args.v0),
        "vth": float(args.vth),
        "gamma_ref_pred": float(pred["gamma_ref_pred"]),
        "sigma_stabilize_pred_dx": sigma_cross,
        "sigma_stabilize_measured_dx": args.measured_stabilization_sigma,
        "stabilization_ratio_threshold": args.stabilization_ratio,
        "stabilization_rel_error": crossing_rel_error,
        "prekill_sigma_max": args.prekill_max_sigma,
        "prekill_rmse": rmse,
        "rmse_pass": rmse_pass,
        "crossing_pass": crossing_pass,
        "passed": passed,
    }
    with (args.output / "warm_surrogate_validation.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)

    out_rows = []
    for i, sig in enumerate(sigma):
        out_rows.append(
            {
                "sigma_dx": sig,
                "gamma_ratio_measured": ratio[i],
                "gamma_ratio_pred": ratio_pred[i],
                "coupling": float(pred["coupling"][i]),
                "prekill_used_for_rmse": bool(prekill[i]),
            }
        )
    with (args.output / "warm_surrogate_curve.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(out_rows[0].keys()))
        writer.writeheader()
        writer.writerows(out_rows)

    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    ax.axhspan(-3.2, 0.0, color="#d8d8d8", alpha=0.28, label="stabilized / damped fit")
    ax.errorbar(
        sigma,
        ratio,
        yerr=[ratio - ratio_lo, ratio_hi - ratio],
        fmt="o",
        color="#9f2f22",
        capsize=3,
        label="PIC measurement",
    )
    ax.plot(
        sig_dense,
        pred_dense["ratio_pred"],
        color="#174a7c",
        linewidth=2.0,
        label="zero-parameter warm surrogate",
    )
    if sigma_cross is not None:
        ax.axvline(
            sigma_cross,
            color="#174a7c",
            linestyle="--",
            linewidth=1.2,
            label=rf"predicted $\gamma/\gamma_0={args.stabilization_ratio:.2f}$ at $\sigma={sigma_cross:.2f}$",
        )
    ax.axvline(
        args.measured_stabilization_sigma,
        color="black",
        linestyle="-.",
        linewidth=1.2,
        label="measured kill ~1.0",
    )
    ax.axvline(args.prekill_max_sigma, color="#555555", linestyle=":", linewidth=1.1, label="RMSE cutoff")
    ax.set_xscale("log")
    ax.set_xlim(0.22, 12.5)
    ax.set_ylim(-3.1, 1.1)
    ax.set_xlabel(r"$\sigma/\Delta x$")
    ax.set_ylabel(r"$\gamma_\sigma/\gamma_0$")
    ax.set_title("Step 2.2: warm surrogate validation")
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(args.output / "warm_surrogate_validation.png", dpi=220)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
