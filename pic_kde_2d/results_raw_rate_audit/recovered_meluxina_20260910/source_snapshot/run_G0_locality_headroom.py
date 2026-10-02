#!/usr/bin/env python3
"""Evaluate whether localized bandwidth control has enough field-noise headroom.

This is Gate G0 for H2. It is a deterministic covariance propagation, not a
PIC run and not a fitted logarithmic approximation. The calculation uses the
actual periodic spectral field map

    rho --CIC--Gaussian(source width)--> rho_s --(-i k / |k|^2)--> E

and independent, equal-variance density noise at every source cell. For a
packet-weighted field diagnostic, the variance is the squared Frobenius norm
of this linear map. A source-dependent C0 bandwidth profile is handled one
source column at a time, so no stationarity assumption is made.

The protected packet is uniform in y. Its Gaussian amplitude envelope has
95.45 percent of its one-dimensional mass inside the protected core. Widths
are sigma_safe in the core and sigma_free outside, with a raised-cosine blend
in log-width space. The blend is C1 and therefore satisfies the production C0
log-Cholesky requirement.

G0 passes only when:

1. the packet is spectrally narrow enough for damping and group velocity to be
   meaningful;
2. the local profile preserves the packet field within the registered error;
3. an admissible sigma_free gives at least the registered field-noise gain.

Density-MISE gain is reported but cannot pass G0 by itself.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np


@dataclass(frozen=True)
class Geometry:
    name: str
    intended: bool
    carrier_mode_x: int
    Nx: int
    Ny: int
    Lx: float
    Ly: float

    @property
    def dx(self) -> float:
        return self.Lx / self.Nx

    @property
    def dy(self) -> float:
        return self.Ly / self.Ny


def periodic_distance(x: np.ndarray, center: float, length: float) -> np.ndarray:
    """Shortest unsigned distance from x to center on a periodic interval."""
    displacement = np.abs(np.asarray(x, dtype=np.float64) - center)
    return np.minimum(displacement, length - displacement)


def c0_log_width_profile(
    x: np.ndarray,
    length: float,
    protected_fraction: float,
    blend_fraction: float,
    sigma_safe: float,
    sigma_free: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return a C1 mask and a width profile blended in log-width space."""
    if not 0.0 < protected_fraction < 1.0:
        raise ValueError("protected_fraction must lie in (0, 1)")
    if not 0.0 < blend_fraction < 0.5 * (1.0 - protected_fraction):
        raise ValueError("blend_fraction must be positive and fit outside the core")
    if sigma_safe <= 0.0 or sigma_free < sigma_safe:
        raise ValueError("require sigma_free >= sigma_safe > 0")

    radius = periodic_distance(x, 0.5 * length, length)
    core_radius = 0.5 * protected_fraction * length
    blend_width = blend_fraction * length
    mask = np.ones_like(radius)
    outside = radius >= core_radius + blend_width
    transition = (radius > core_radius) & ~outside
    mask[outside] = 0.0
    phase = (radius[transition] - core_radius) / blend_width
    mask[transition] = 0.5 * (1.0 + np.cos(np.pi * phase))

    log_width = mask * math.log(sigma_safe) + (1.0 - mask) * math.log(sigma_free)
    return mask, np.exp(log_width)


def packet_geometry(
    geometry: Geometry,
    protected_fraction: float,
    k_lambda_D: float,
    lambda_D: float,
) -> Dict[str, np.ndarray]:
    """Construct the packet and return its spectral-resolution diagnostics."""
    x = np.arange(geometry.Nx, dtype=np.float64) * geometry.dx
    radius = periodic_distance(x, 0.5 * geometry.Lx, geometry.Lx)
    core_width = protected_fraction * geometry.Lx
    envelope_sigma = core_width / 4.0
    amplitude = np.exp(-0.5 * (radius / envelope_sigma) ** 2)
    power_weight = amplitude**2
    power_weight /= np.sum(power_weight)

    k_carrier = 2.0 * np.pi * geometry.carrier_mode_x / geometry.Lx
    if not np.isclose(k_carrier * lambda_D, k_lambda_D, rtol=0.0, atol=1.0e-12):
        raise ValueError(
            f"{geometry.name}: carrier does not satisfy k*lambda_D={k_lambda_D}"
        )
    wavelength = 2.0 * np.pi / k_carrier
    carrier_cycles = core_width / wavelength
    relative_bandwidth = 1.0 / (k_carrier * envelope_sigma)
    density = amplitude * np.cos(k_carrier * (x - 0.5 * geometry.Lx))
    density -= np.mean(density)
    return {
        "x": x,
        "amplitude": amplitude,
        "power_weight": power_weight,
        "density": density,
        "k_carrier": np.asarray(k_carrier),
        "envelope_sigma": np.asarray(envelope_sigma),
        "carrier_cycles": np.asarray(carrier_cycles),
        "relative_bandwidth": np.asarray(relative_bandwidth),
    }


class SpectralVarianceModel:
    """Exact periodic covariance propagation for source-dependent widths."""

    def __init__(self, geometry: Geometry):
        self.geometry = geometry
        self.kx = 2.0 * np.pi * np.fft.fftfreq(geometry.Nx, d=geometry.dx)[:, None]
        self.ky = 2.0 * np.pi * np.fft.fftfreq(geometry.Ny, d=geometry.dy)[None, :]
        self.k2 = self.kx**2 + self.ky**2
        self.nonzero = self.k2 > 0.0
        self.cic = (
            np.sinc(self.kx * geometry.dx / (2.0 * np.pi)) ** 2
            * np.sinc(self.ky * geometry.dy / (2.0 * np.pi)) ** 2
        )
        self._kernel_cache: Dict[float, Tuple[np.ndarray, float, float]] = {}

    def kernel_power(self, sigma_cells: float) -> Tuple[np.ndarray, float, float]:
        """Return field power by x displacement and total field/density power."""
        key = round(float(sigma_cells), 12)
        if key in self._kernel_cache:
            return self._kernel_cache[key]
        sigma_x = sigma_cells * self.geometry.dx
        sigma_y = sigma_cells * self.geometry.dy
        gaussian = np.exp(
            -0.5 * (self.kx**2 * sigma_x**2 + self.ky**2 * sigma_y**2)
        )
        shape = self.cic * gaussian
        ex_hat = np.zeros_like(shape, dtype=np.complex128)
        ey_hat = np.zeros_like(shape, dtype=np.complex128)
        kx_grid = np.broadcast_to(self.kx, shape.shape)
        ky_grid = np.broadcast_to(self.ky, shape.shape)
        ex_hat[self.nonzero] = (
            -1j * kx_grid[self.nonzero] * shape[self.nonzero] / self.k2[self.nonzero]
        )
        ey_hat[self.nonzero] = (
            -1j * ky_grid[self.nonzero] * shape[self.nonzero] / self.k2[self.nonzero]
        )
        ex = np.fft.ifft2(ex_hat)
        ey = np.fft.ifft2(ey_hat)
        field_node_power = np.abs(ex) ** 2 + np.abs(ey) ** 2
        field_power_by_dx = np.sum(field_node_power, axis=1)
        field_power = float(np.sum(field_node_power))
        density_power = float(np.sum(np.abs(np.fft.ifft2(shape)) ** 2))
        result = field_power_by_dx, field_power, density_power
        self._kernel_cache[key] = result
        return result

    def local_variances(
        self, sigma_by_source_x: np.ndarray, packet_power_weight: np.ndarray
    ) -> Tuple[float, float]:
        """Packet-field and domain-density variances for the local profile."""
        nx = self.geometry.Nx
        target_index = np.arange(nx)
        field_variance = 0.0
        density_variance = 0.0
        for source_x, sigma_cells in enumerate(sigma_by_source_x):
            field_by_dx, _, density_power = self.kernel_power(float(sigma_cells))
            displacement = (target_index - source_x) % nx
            field_variance += float(
                np.dot(packet_power_weight, field_by_dx[displacement])
            )
            density_variance += density_power / nx
        return field_variance, density_variance

    def ideal_outside_deletion_gain(
        self,
        sigma_safe: float,
        packet_power_weight: np.ndarray,
        core_source_mask: np.ndarray,
    ) -> float:
        """Upper-bound gain if all source noise outside the core were deleted."""
        nx = self.geometry.Nx
        field_by_dx, global_variance, _ = self.kernel_power(sigma_safe)
        target_index = np.arange(nx)
        retained = 0.0
        for source_x in np.flatnonzero(core_source_mask):
            displacement = (target_index - source_x) % nx
            retained += float(np.dot(packet_power_weight, field_by_dx[displacement]))
        return global_variance / retained

    def packet_field(self, density: np.ndarray, sigma_by_source_x: np.ndarray) -> np.ndarray:
        """Apply the nonstationary deposit and spectral solve to a ky=0 packet."""
        nx = self.geometry.Nx
        kx = self.kx[:, 0]
        cic_x = np.sinc(kx * self.geometry.dx / (2.0 * np.pi)) ** 2
        smoothed_density = np.zeros(nx, dtype=np.float64)
        cache: Dict[float, np.ndarray] = {}
        for source_x, sigma_cells in enumerate(sigma_by_source_x):
            key = round(float(sigma_cells), 12)
            if key not in cache:
                sigma_x = float(sigma_cells) * self.geometry.dx
                transfer = cic_x * np.exp(-0.5 * (kx * sigma_x) ** 2)
                cache[key] = np.fft.ifft(transfer).real
            smoothed_density += density[source_x] * np.roll(cache[key], source_x)
        rho_hat = np.fft.fft(smoothed_density)
        ex_hat = np.zeros(nx, dtype=np.complex128)
        nonzero = kx != 0.0
        ex_hat[nonzero] = -1j * rho_hat[nonzero] / kx[nonzero]
        return np.fft.ifft(ex_hat).real


def load_config(path: Path) -> dict:
    with path.open() as stream:
        config = json.load(stream)
    if config["field_gain_threshold"] <= 1.0:
        raise ValueError("field_gain_threshold must exceed one")
    return config


def build_geometry(entry: dict, packet_config: dict) -> Geometry:
    k = packet_config["k_lambda_D"] / packet_config["lambda_D"]
    carrier_mode = int(entry["carrier_mode_x"])
    Lx = 2.0 * np.pi * carrier_mode / k
    return Geometry(
        name=str(entry["name"]),
        intended=bool(entry["intended"]),
        carrier_mode_x=carrier_mode,
        Nx=int(entry["Nx"]),
        Ny=int(entry["Ny"]),
        Lx=Lx,
        Ly=float(entry["Ly"]),
    )


def analyze_geometry(config: dict, geometry: Geometry) -> Tuple[List[dict], dict]:
    packet_config = config["packet"]
    protected_fraction = float(packet_config["protected_core_fraction_x"])
    blend_fraction = float(packet_config["blend_fraction_x"])
    sigma_safe = float(config["sigma_safe_cells"])
    packet = packet_geometry(
        geometry,
        protected_fraction,
        float(packet_config["k_lambda_D"]),
        float(packet_config["lambda_D"]),
    )
    x = packet["x"]
    core_radius = 0.5 * protected_fraction * geometry.Lx
    core_source_mask = (
        periodic_distance(x, 0.5 * geometry.Lx, geometry.Lx) <= core_radius
    )
    model = SpectralVarianceModel(geometry)
    _, global_field_variance, global_density_variance = model.kernel_power(sigma_safe)
    safe_profile = np.full(geometry.Nx, sigma_safe)
    reference_field = model.packet_field(packet["density"], safe_profile)
    reference_norm = float(np.dot(reference_field, reference_field))
    if reference_norm <= 0.0:
        raise FloatingPointError("protected packet reference field vanished")

    rows: List[dict] = []
    for sigma_free in config["sigma_free_cells"]:
        mask, sigma_profile = c0_log_width_profile(
            x,
            geometry.Lx,
            protected_fraction,
            blend_fraction,
            sigma_safe,
            float(sigma_free),
        )
        local_field_variance, local_density_variance = model.local_variances(
            sigma_profile, packet["power_weight"]
        )
        local_packet_field = model.packet_field(packet["density"], sigma_profile)
        packet_projection = float(
            np.dot(local_packet_field, reference_field) / reference_norm
        )
        packet_relative_error = float(
            np.linalg.norm(local_packet_field - reference_field)
            / np.sqrt(reference_norm)
        )
        support_radius = float(config["kernel_support_sigmas"]) * float(sigma_free)
        admissible = support_radius <= 0.5 * geometry.Nx
        rows.append(
            {
                "geometry": geometry.name,
                "intended": geometry.intended,
                "Nx": geometry.Nx,
                "Ny": geometry.Ny,
                "Lx": geometry.Lx,
                "Ly": geometry.Ly,
                "dx": geometry.dx,
                "carrier_mode_x": geometry.carrier_mode_x,
                "carrier_cycles_in_core": float(packet["carrier_cycles"]),
                "relative_packet_bandwidth": float(packet["relative_bandwidth"]),
                "sigma_safe_cells": sigma_safe,
                "sigma_free_cells": float(sigma_free),
                "sigma_free_over_Lx": float(sigma_free) / geometry.Nx,
                "admissible_kernel_support": admissible,
                "packet_field_projection": packet_projection,
                "packet_field_relative_error": packet_relative_error,
                "field_variance_global": global_field_variance,
                "field_variance_local": local_field_variance,
                "field_noise_gain": global_field_variance / local_field_variance,
                "density_variance_global": global_density_variance,
                "density_variance_local": local_density_variance,
                "density_noise_gain": global_density_variance / local_density_variance,
                "safe_mask_fraction": float(np.mean(mask >= 1.0 - 1.0e-14)),
                "blend_mask_fraction": float(
                    np.mean((mask > 1.0e-14) & (mask < 1.0 - 1.0e-14))
                ),
            }
        )

    geometry_valid = (
        float(packet["carrier_cycles"])
        >= float(config["minimum_carrier_cycles_in_protected_core"])
        and float(packet["relative_bandwidth"])
        <= float(config["maximum_relative_packet_bandwidth"])
    )
    admissible_rows = [
        row
        for row in rows
        if row["admissible_kernel_support"]
        and row["packet_field_relative_error"]
        <= float(config["maximum_protected_field_error"])
    ]
    best = max(admissible_rows, key=lambda row: row["field_noise_gain"])
    ideal_gain = model.ideal_outside_deletion_gain(
        sigma_safe, packet["power_weight"], core_source_mask
    )
    gate_pass = (
        geometry.intended
        and geometry_valid
        and best["field_noise_gain"] >= float(config["field_gain_threshold"])
    )
    summary = {
        "geometry": geometry.name,
        "intended": geometry.intended,
        "geometry_valid": geometry_valid,
        "carrier_cycles_in_core": float(packet["carrier_cycles"]),
        "relative_packet_bandwidth": float(packet["relative_bandwidth"]),
        "best_admissible_sigma_free_cells": best["sigma_free_cells"],
        "best_admissible_field_noise_gain": best["field_noise_gain"],
        "density_noise_gain_at_best_field_point": best["density_noise_gain"],
        "packet_field_error_at_best_field_point": best["packet_field_relative_error"],
        "ideal_outside_deletion_field_gain_upper_bound": ideal_gain,
        "field_gain_threshold": float(config["field_gain_threshold"]),
        "gate_pass": gate_pass,
    }
    return rows, summary


def write_csv(path: Path, rows: Iterable[dict]) -> None:
    rows = list(rows)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_results(rows: List[dict], summaries: List[dict], output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), constrained_layout=True)
    for summary in summaries:
        selected = [row for row in rows if row["geometry"] == summary["geometry"]]
        sigma = np.asarray([row["sigma_free_cells"] for row in selected])
        field = np.asarray([row["field_noise_gain"] for row in selected])
        density = np.asarray([row["density_noise_gain"] for row in selected])
        admissible = np.asarray([row["admissible_kernel_support"] for row in selected])
        label = summary["geometry"]
        axes[0].plot(sigma, field, marker="o", label=label)
        axes[0].plot(sigma[~admissible], field[~admissible], marker="x", linestyle="none")
        axes[1].plot(sigma, density, marker="o", label=label)
    for ax in axes:
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(alpha=0.25)
        ax.set_xlabel(r"$\sigma_{\rm free}/\Delta x$")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].axhline(
        summaries[0]["field_gain_threshold"],
        color="darkred",
        linestyle="--",
        label="G0 threshold",
    )
    axes[0].set_ylabel("packet-field noise gain over global C4")
    axes[1].set_ylabel("domain density-noise gain over global C4")
    axes[0].legend(fontsize=7)
    axes[1].legend(fontsize=7)
    fig.suptitle("G0: density headroom does not imply field headroom")
    fig.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    root = Path(__file__).resolve().parents[1]
    parser.add_argument(
        "--config", type=Path, default=root / "configs" / "h2_g0_locality.json"
    )
    parser.add_argument(
        "--output", type=Path, default=root / "results" / "G0_locality_headroom"
    )
    parser.add_argument(
        "--sigma-safe",
        type=float,
        help="override the global/protected C4 width from the JSON contract",
    )
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.sigma_safe is not None:
        if not np.isfinite(args.sigma_safe) or args.sigma_safe <= 0.0:
            raise ValueError("--sigma-safe must be finite and positive")
        config["sigma_safe_cells"] = float(args.sigma_safe)
        config["sigma_free_cells"] = [
            value
            for value in config["sigma_free_cells"]
            if value >= args.sigma_safe
        ]
        if not config["sigma_free_cells"]:
            raise ValueError("sigma-free scan has no value above --sigma-safe")
    args.output.mkdir(parents=True, exist_ok=True)
    rows: List[dict] = []
    summaries: List[dict] = []
    for entry in config["geometries"]:
        geometry = build_geometry(entry, config["packet"])
        geometry_rows, summary = analyze_geometry(config, geometry)
        rows.extend(geometry_rows)
        summaries.append(summary)
        print(
            f"{geometry.name}: valid={summary['geometry_valid']}, "
            f"best field gain={summary['best_admissible_field_noise_gain']:.4f}x "
            f"at sigma_free={summary['best_admissible_sigma_free_cells']:.3g} dx, "
            f"ideal deletion upper bound={summary['ideal_outside_deletion_field_gain_upper_bound']:.4f}x"
        )

    intended = [summary for summary in summaries if summary["intended"]]
    if len(intended) != 1:
        raise ValueError("exactly one geometry must be marked intended")
    decision = (
        "PROCEED_TO_CONSERVATION_PRECHECK"
        if intended[0]["gate_pass"]
        else "STOP_H2_HEADROOM"
    )
    result = {
        "gate": config["gate_name"],
        "decision": decision,
        "preregistered_threshold": config["field_gain_threshold"],
        "sigma_safe_cells": config["sigma_safe_cells"],
        "intended_geometry": intended[0]["geometry"],
        "summaries": summaries,
        "method": {
            "noise_covariance": "independent equal-variance source-cell density noise",
            "deposit": "principal CIC transfer times source-dependent periodic Gaussian",
            "field_operator": "continuum spectral -i*k/|k|^2 with k=0 removed",
            "diagnostic": "Gaussian-packet-power-weighted Ex+Ey variance",
            "fit_parameters": 0,
        },
    }
    write_csv(args.output / "g0_geometry_scan.csv", rows)
    with (args.output / "g0_summary.json").open("w") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    if not args.no_plot:
        plot_results(rows, summaries, args.output / "g0_headroom")
    print(f"G0 HEADROOM DECISION: {decision}")


if __name__ == "__main__":
    main()
