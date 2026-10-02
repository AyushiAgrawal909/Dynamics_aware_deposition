#!/usr/bin/env python3
"""Compare finite-grid plasma heating for the post-C4 filter family.

The homogeneous Maxwellian benchmark is intentionally under-resolved:

    lambda_D / dx = 0.16,  omega_pe * dt = 0.5,  v_th * dt / dx = 0.08.

Temperature is the centered two-dimensional velocity variance.  All methods
use the same initial particles for a given seed and the same matched filter on
deposit and gather.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import pathlib
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


THIS_DIR = pathlib.Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pic_kde_2d.core.grid2d import Grid2D
from pic_kde_2d.core.particles2d import init_landau_2d
from pic_kde_2d.core.pic_loop import pic_step
from pic_kde_2d.methods.spectral_W import make_W_gaussian
from pic_kde_2d.methods.spectral_W_wrappers import make_spectral_W_methods
from pic_kde_2d.conventions import marker_grid_shape


METHODS = (
    {
        "id": "cic",
        "label": "CIC",
        "color": "#5B5B5B",
        "linestyle": "--",
    },
    {
        "id": "c4_gaussian",
        "label": r"C4 Gaussian ($\sigma=1\,\Delta x$)",
        "color": "#2A7F62",
        "linestyle": "-",
    },
    {
        "id": "binomial_n4",
        "label": "Binomial ($n=4$)",
        "color": "#D17A22",
        "linestyle": "-",
    },
    {
        "id": "compensated_p1_n4",
        "label": "Compensated ($p=1$, $n=4$)",
        "color": "#7A5195",
        "linestyle": "-",
    },
    {
        "id": "compensated_p2_n4",
        "label": "Compensated ($p=2$, $n=4$)",
        "color": "#247BA0",
        "linestyle": "-",
    },
    {
        "id": "compensated_p3_n4",
        "label": "Compensated ($p=3$, $n=4$)",
        "color": "#C23B53",
        "linestyle": "-",
    },
)


def make_binomial_W(grid: Grid2D, passes: int) -> np.ndarray:
    """Exact repeated (1, 2, 1) / 4 response in both directions."""
    response_x = np.cos(0.5 * grid.KX * grid.dx) ** (2 * passes)
    response_y = np.cos(0.5 * grid.KY * grid.dy) ** (2 * passes)
    return np.asarray(response_x * response_y, dtype=np.float64)


def make_compensated_binomial_W(
    grid: Grid2D,
    passes: int,
    compensation_order: int,
) -> np.ndarray:
    """Truncated inverse-binomial compensation used by the post-C4 study."""

    def response(k: np.ndarray, spacing: float) -> np.ndarray:
        s = np.sin(0.5 * k * spacing) ** 2
        base = (1.0 - s) ** passes
        compensation = np.ones_like(s)
        coefficient = 1.0
        for order in range(1, compensation_order + 1):
            coefficient *= (passes + order - 1) / order
            compensation += coefficient * s**order
        return base * compensation

    return np.asarray(
        response(grid.KX, grid.dx) * response(grid.KY, grid.dy),
        dtype=np.float64,
    )


def filter_for_method(grid: Grid2D, method_id: str) -> np.ndarray:
    if method_id == "cic":
        return np.ones((grid.Nx, grid.Ny // 2 + 1), dtype=np.float64)
    if method_id == "c4_gaussian":
        return make_W_gaussian(grid, grid.dx, grid.dy)
    if method_id == "binomial_n4":
        return make_binomial_W(grid, passes=4)
    if method_id.startswith("compensated_p"):
        order = int(method_id.split("_p", 1)[1].split("_", 1)[0])
        return make_compensated_binomial_W(
            grid,
            passes=4,
            compensation_order=order,
        )
    raise ValueError(f"unknown method: {method_id}")


def centered_temperature(particles) -> float:
    weights = np.abs(np.asarray(particles.w, dtype=np.float64))
    norm = float(np.sum(weights))
    mean_vx = float(np.sum(weights * particles.vx) / norm)
    mean_vy = float(np.sum(weights * particles.vy) / norm)
    variance_2d = np.sum(
        weights
        * ((particles.vx - mean_vx) ** 2 + (particles.vy - mean_vy) ** 2)
    ) / norm
    return 0.5 * float(variance_2d)


def run_one(payload: tuple) -> dict:
    (
        method_id,
        seed,
        nx,
        ny,
        ppc,
        v_thermal,
        dt,
        n_steps,
        diag_stride,
    ) = payload
    grid = Grid2D(Nx=nx, Ny=ny, Lx=float(nx), Ly=float(ny))
    nx_marker, ny_marker = marker_grid_shape(nx, ny, ppc)
    particles = init_landau_2d(
        Nx_marker=nx_marker,
        Ny_marker=ny_marker,
        Lx=grid.Lx,
        Ly=grid.Ly,
        v_thermal=v_thermal,
        alpha=0.0,
        kx_mode=0,
        rng=np.random.default_rng(seed),
    )
    W = filter_for_method(grid, method_id)
    deposit, gather = make_spectral_W_methods(W)

    times = [0.0]
    temperatures = [centered_temperature(particles)]
    for step in range(1, n_steps + 1):
        pic_step(particles, grid, dt, deposit, gather)
        if step % diag_stride == 0:
            times.append(step * dt)
            temperatures.append(centered_temperature(particles))

    temperatures_array = np.asarray(temperatures, dtype=np.float64)
    ratio = temperatures_array / temperatures_array[0]
    return {
        "method_id": method_id,
        "seed": seed,
        "times": np.asarray(times, dtype=np.float64),
        "temperature_ratio": ratio,
        "temperature_initial": float(temperatures_array[0]),
    }


def bootstrap_mean_interval(
    curves: np.ndarray,
    rng: np.random.Generator,
    n_boot: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = np.mean(curves, axis=0)
    draw = rng.integers(0, curves.shape[0], size=(n_boot, curves.shape[0]))
    boot_means = np.mean(curves[draw, :], axis=1)
    lo, hi = np.percentile(boot_means, [2.5, 97.5], axis=0)
    return mean, lo, hi


def write_raw_csv(path: pathlib.Path, results: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["method_id", "seed", "omega_pe_t", "temperature_ratio"])
        for result in sorted(results, key=lambda row: (row["method_id"], row["seed"])):
            for time, ratio in zip(result["times"], result["temperature_ratio"]):
                writer.writerow(
                    [result["method_id"], result["seed"], f"{time:.10g}", f"{ratio:.16g}"]
                )


def write_summary_csv(
    path: pathlib.Path,
    times: np.ndarray,
    summaries: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["method_id", "omega_pe_t", "temperature_ratio_mean", "ci95_lo", "ci95_hi"]
        )
        for method in METHODS:
            mean, lo, hi = summaries[method["id"]]
            for time, value, lower, upper in zip(times, mean, lo, hi):
                writer.writerow(
                    [
                        method["id"],
                        f"{time:.10g}",
                        f"{value:.16g}",
                        f"{lower:.16g}",
                        f"{upper:.16g}",
                    ]
                )


def style_axis(ax) -> None:
    ax.grid(True, which="major", color="#D8D8D8", linewidth=0.65, alpha=0.75)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", length=3.5, width=0.8)


def plot_figure(
    output_png: pathlib.Path,
    output_pdf: pathlib.Path,
    times: np.ndarray,
    summaries: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    ppc: int,
    dt: float,
    v_thermal: float,
) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.5,
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "legend.fontsize": 8.7,
            "lines.linewidth": 2.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, ax = plt.subplots(figsize=(8.2, 5.5), constrained_layout=True)
    ax.axhline(1.0, color="#222222", linewidth=0.8, linestyle=":", zorder=1)

    for method in METHODS:
        mean, lo, hi = summaries[method["id"]]
        ax.fill_between(
            times,
            lo,
            hi,
            color=method["color"],
            alpha=0.10,
            linewidth=0,
            zorder=2,
        )
        ax.plot(
            times,
            mean,
            color=method["color"],
            linestyle=method["linestyle"],
            label=method["label"],
            zorder=3,
        )

    ax.set(
        xlabel=r"Normalized time, $\omega_{pe}t$",
        ylabel=r"Normalized temperature, $T(t)/T(0)$",
        xlim=(float(times[0]), float(times[-1])),
        ylim=(0.96, None),
        title="Finite-grid plasma heating: post-C4 method comparison",
    )
    style_axis(ax)
    ax.legend(loc="upper left", frameon=False, ncol=2, columnspacing=1.2)
    ax.text(
        0.985,
        0.045,
        (
            rf"$\lambda_D/\Delta x={v_thermal:.2f}$, "
            rf"$\omega_{{pe}}\Delta t={dt:.1f}$, PPC $={ppc}$"
            "\nmatched deposit/filter/gather; shaded: paired-seed 95% CI"
        ),
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8.3,
        color="#333333",
    )

    # The full-scale view is driven by CIC; this inset resolves the filtered methods.
    inset = ax.inset_axes([0.43, 0.45, 0.54, 0.39])
    non_cic_hi = []
    for method in METHODS[1:]:
        mean, lo, hi = summaries[method["id"]]
        non_cic_hi.append(float(np.max(hi)))
        inset.fill_between(times, lo, hi, color=method["color"], alpha=0.08, linewidth=0)
        inset.plot(
            times,
            mean,
            color=method["color"],
            linestyle=method["linestyle"],
            linewidth=1.35,
        )
    inset.axhline(1.0, color="#222222", linewidth=0.65, linestyle=":")
    inset.set_xlim(float(times[0]), float(times[-1]))
    inset.set_ylim(0.995, max(1.01, 1.04 * max(non_cic_hi)))
    inset.set_title("Filtered methods", fontsize=8.8, pad=2)
    inset.tick_params(labelsize=7.2, direction="out", length=2.5)
    inset.grid(True, color="#E3E3E3", linewidth=0.45)

    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    fig.savefig(output_pdf, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=pathlib.Path, default=THIS_DIR / "heating_post_c4")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--nx", type=int, default=64)
    parser.add_argument("--ny", type=int, default=16)
    parser.add_argument("--ppc", type=int, default=32)
    parser.add_argument("--v-thermal", type=float, default=0.16)
    parser.add_argument("--dt", type=float, default=0.5)
    parser.add_argument("--n-steps", type=int, default=2000)
    parser.add_argument("--diag-stride", type=int, default=10)
    parser.add_argument("--n-boot", type=int, default=4000)
    args = parser.parse_args()

    payloads = [
        (
            method["id"],
            seed,
            args.nx,
            args.ny,
            args.ppc,
            args.v_thermal,
            args.dt,
            args.n_steps,
            args.diag_stride,
        )
        for method in METHODS
        for seed in range(args.seeds)
    ]
    results = []
    if args.workers == 1:
        for payload in payloads:
            results.append(run_one(payload))
    else:
        # Threads avoid platform semaphore restrictions and NumPy releases the
        # GIL for the dominant array/FFT operations in each independent run.
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            future_to_payload = {executor.submit(run_one, payload): payload for payload in payloads}
            for future in as_completed(future_to_payload):
                results.append(future.result())

    reference_times = results[0]["times"]
    for result in results:
        if not np.array_equal(result["times"], reference_times):
            raise AssertionError("all runs must share the same diagnostic times")
        if not np.isclose(result["temperature_ratio"][0], 1.0, atol=1.0e-14):
            raise AssertionError("temperature curves must be normalized at t=0")
        if np.any(~np.isfinite(result["temperature_ratio"])):
            raise AssertionError("temperature curves must remain finite")

    rng = np.random.default_rng(20260819)
    summaries = {}
    final_rows = []
    for method in METHODS:
        method_results = sorted(
            (row for row in results if row["method_id"] == method["id"]),
            key=lambda row: row["seed"],
        )
        curves = np.stack([row["temperature_ratio"] for row in method_results])
        summaries[method["id"]] = bootstrap_mean_interval(curves, rng, args.n_boot)
        mean, lo, hi = summaries[method["id"]]
        fit_mask = reference_times >= 0.2 * reference_times[-1]
        slope = float(np.polyfit(reference_times[fit_mask], mean[fit_mask], 1)[0])
        final_rows.append(
            {
                "method_id": method["id"],
                "label": method["label"],
                "final_temperature_ratio_mean": float(mean[-1]),
                "final_ci95_lo": float(lo[-1]),
                "final_ci95_hi": float(hi[-1]),
                "late_heating_slope_per_omega_pe_t": slope,
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_csv = args.output_dir / "plasma_heating_raw.csv"
    summary_csv = args.output_dir / "plasma_heating_summary.csv"
    final_json = args.output_dir / "plasma_heating_final.json"
    output_png = args.output_dir / "plasma_heating_post_c4.png"
    output_pdf = args.output_dir / "plasma_heating_post_c4.pdf"
    write_raw_csv(raw_csv, results)
    write_summary_csv(summary_csv, reference_times, summaries)
    final_json.write_text(
        json.dumps(
            {
                "benchmark": {
                    "lambda_D_over_dx": args.v_thermal,
                    "omega_pe_dt": args.dt,
                    "thermal_cell_crossing_per_step": args.v_thermal * args.dt,
                    "nx": args.nx,
                    "ny": args.ny,
                    "ppc": args.ppc,
                    "n_seeds": args.seeds,
                    "omega_pe_t_end": float(reference_times[-1]),
                },
                "methods": final_rows,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    plot_figure(
        output_png,
        output_pdf,
        reference_times,
        summaries,
        args.ppc,
        args.dt,
        args.v_thermal,
    )
    print(json.dumps({"png": str(output_png), "pdf": str(output_pdf), "methods": final_rows}, indent=2))


if __name__ == "__main__":
    main()
