#!/usr/bin/env python3
"""Run small, controlled conservation tests before any production campaign.

The operator test compares exact spectral and periodic centered-finite-
difference Poisson/gradient pairs while holding the density and matched filter
fixed.  The time-step test uses the recovered Python PIC implementation, the
same initial particles at every dt, and a short strong-Landau trajectory.  It
is a diagnostic convergence test, not a replacement for a manuscript
measurement.
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import sys

import numpy as np


HERE = pathlib.Path(__file__).resolve().parent
DOWNLOADS = HERE.parents[1]
CAMPAIGN_ROOT = DOWNLOADS / "pic_kde_2d_landau_cv_stage"
if not CAMPAIGN_ROOT.is_dir():
    raise FileNotFoundError(CAMPAIGN_ROOT)
sys.path.insert(0, str(CAMPAIGN_ROOT))

from pic_kde_2d.benchmarks.theory import langmuir_omega_r  # noqa: E402
from pic_kde_2d.conventions import landau_box_length, marker_grid_shape, thermal_velocity  # noqa: E402
from pic_kde_2d.core.grid2d import Grid2D  # noqa: E402
from pic_kde_2d.core.particles2d import Particles2D, init_landau_2d_eigenmode  # noqa: E402
from pic_kde_2d.core.pic_loop import run_simulation  # noqa: E402
from pic_kde_2d.methods.cic import deposit_cic_2d, gather_cic_2d  # noqa: E402
from pic_kde_2d.methods.spectral_W import make_W_gaussian  # noqa: E402
from pic_kde_2d.methods.spectral_W_wrappers import make_spectral_W_gaussian_methods  # noqa: E402


def write_csv(path: pathlib.Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def apply_filter(array: np.ndarray, transfer: np.ndarray) -> np.ndarray:
    return np.fft.irfft2(np.fft.rfft2(array) * transfer, s=array.shape).real


def solve_operator(rho: np.ndarray, grid: Grid2D, formulation: str) -> tuple[np.ndarray, np.ndarray]:
    rho_hat = np.fft.rfft2(rho)
    if formulation == "spectral":
        denominator = grid.KX**2 + grid.KY**2
        dx_symbol = grid.KX
        dy_symbol = grid.KY
    elif formulation == "centered_fd":
        denominator = (
            4.0 * np.sin(0.5 * grid.KX * grid.dx) ** 2 / grid.dx**2
            + 4.0 * np.sin(0.5 * grid.KY * grid.dy) ** 2 / grid.dy**2
        )
        dx_symbol = np.sin(grid.KX * grid.dx) / grid.dx
        dy_symbol = np.sin(grid.KY * grid.dy) / grid.dy
    else:
        raise ValueError(formulation)
    inverse = np.zeros_like(denominator)
    inverse[denominator > 0.0] = 1.0 / denominator[denominator > 0.0]
    phi_hat = rho_hat * inverse
    ex = np.fft.irfft2(-1j * dx_symbol * phi_hat, s=rho.shape).real
    ey = np.fft.irfft2(-1j * dy_symbol * phi_hat, s=rho.shape).real
    return ex, ey


def operator_tests() -> list[dict]:
    grid = Grid2D(Nx=64, Ny=16, Lx=2.0 * np.pi / 0.4, Ly=8.0)
    rng = np.random.default_rng(9102026)
    rho = rng.normal(size=(grid.Nx, grid.Ny))
    rho -= np.mean(rho)
    rows = []
    for sigma_cells in (0.0, 1.0, 3.0):
        transfer = make_W_gaussian(grid, sigma_cells * grid.dx, sigma_cells * grid.dy)
        rho_deposited = apply_filter(rho, transfer)
        for formulation in ("spectral", "centered_fd"):
            ex, ey = solve_operator(rho_deposited, grid, formulation)
            force_x = float(np.sum(rho_deposited * ex) * grid.dx * grid.dy)
            force_y = float(np.sum(rho_deposited * ey) * grid.dx * grid.dy)
            abs_force_x = float(np.sum(np.abs(rho_deposited * ex)) * grid.dx * grid.dy)
            abs_force_y = float(np.sum(np.abs(rho_deposited * ey)) * grid.dx * grid.dy)
            relative_norm = float(np.hypot(force_x, force_y) / max(np.hypot(abs_force_x, abs_force_y), 1.0e-300))
            if relative_norm > 2.0e-15:
                raise AssertionError(f"{formulation} matched operator lost skew symmetry: {relative_norm}")
            rows.append(
                {
                    "formulation": formulation,
                    "nx": grid.Nx,
                    "ny": grid.Ny,
                    "sigma_cells": sigma_cells,
                    "force_x": force_x,
                    "force_y": force_y,
                    "abs_force_x": abs_force_x,
                    "abs_force_y": abs_force_y,
                    "relative_force_norm": relative_norm,
                }
            )
    return rows


def particle_force_tests() -> list[dict]:
    """Measure normalized total force through the complete matched PIC force."""
    grid = Grid2D(Nx=128, Ny=16, Lx=64.0, Ly=16.0)
    rng = np.random.default_rng(17)
    n_particles = 4096
    particles = Particles2D(
        x=rng.uniform(0.0, grid.Lx, n_particles),
        y=rng.uniform(0.0, grid.Ly, n_particles),
        vx=np.zeros(n_particles),
        vy=np.zeros(n_particles),
        w=rng.uniform(-1.0, -0.5, n_particles),
        Lx=grid.Lx,
        Ly=grid.Ly,
    )
    rows = []
    for sigma_cells in (0.0, 0.5, 3.0):
        transfer = make_W_gaussian(grid, sigma_cells * grid.dx, sigma_cells * grid.dy)
        rho = deposit_cic_2d(
            particles.x,
            particles.y,
            particles.w,
            grid.Nx,
            grid.Ny,
            grid.dx,
            grid.dy,
        )
        rho = apply_filter(rho, transfer)
        ex, ey = grid.poisson_solve(rho)
        ex = apply_filter(ex, transfer)
        ey = apply_filter(ey, transfer)
        ex_p, ey_p = gather_cic_2d(
            ex,
            ey,
            particles.x,
            particles.y,
            grid.Nx,
            grid.Ny,
            grid.dx,
            grid.dy,
        )
        force_x = particles.w * ex_p
        force_y = particles.w * ey_p
        net = float(np.hypot(np.sum(force_x), np.sum(force_y)))
        scale = float(np.sum(np.hypot(force_x, force_y)))
        relative = net / scale
        if relative > 5.0e-16:
            raise AssertionError(f"matched particle-force test lost momentum at sigma={sigma_cells}: {relative}")
        rows.append(
            {
                "nx": grid.Nx,
                "ny": grid.Ny,
                "n_particles": n_particles,
                "seed": 17,
                "sigma_cells": sigma_cells,
                "total_force_x": float(np.sum(force_x)),
                "total_force_y": float(np.sum(force_y)),
                "total_force_norm": net,
                "sum_particle_force_norms": scale,
                "normalized_total_force": relative,
            }
        )
    return rows


def build_particles(grid: Grid2D, ppc: int, seed: int, alpha: float):
    nx_marker, ny_marker = marker_grid_shape(grid.Nx, grid.Ny, ppc)
    return init_landau_2d_eigenmode(
        Nx_marker=nx_marker,
        Ny_marker=ny_marker,
        Lx=grid.Lx,
        Ly=grid.Ly,
        v_thermal=thermal_velocity(),
        alpha=alpha,
        kx_mode=1,
        omega_r=langmuir_omega_r(0.4),
        rng=np.random.default_rng(seed),
    )


def time_step_tests(output: pathlib.Path) -> list[dict]:
    grid = Grid2D(Nx=64, Ny=16, Lx=landau_box_length(1, 0.4), Ly=8.0)
    ppc = 4
    seed = 24680
    alpha = 0.5
    t_end = 5.0
    rows = []
    for case, sigma_cells in (("cic", 0.0), ("c4", 1.0)):
        for dt in (0.2, 0.1, 0.05, 0.025, 0.0125):
            n_steps = int(round(t_end / dt))
            if not np.isclose(n_steps * dt, t_end):
                raise AssertionError("t_end is not an integer number of steps")
            particles = build_particles(grid, ppc, seed, alpha)
            deposit, gather = make_spectral_W_gaussian_methods(
                grid,
                sigma_cells * grid.dx,
                sigma_cells * grid.dy,
            )
            history = run_simulation(
                particles,
                grid,
                dt,
                n_steps,
                deposit,
                gather,
                diag_stride=1,
                tracked_modes=[(1, 0)],
            ).to_arrays()
            total = history["total_energy"]
            field = history["field_energy"]
            charge = history["total_charge"]
            delta = total - total[0]
            row = {
                "case": case,
                "sigma_cells": sigma_cells,
                "nx": grid.Nx,
                "ny": grid.Ny,
                "ppc": ppc,
                "n_particles": int(particles.n_particles),
                "seed": seed,
                "alpha": alpha,
                "dt": dt,
                "n_steps": n_steps,
                "t_end": t_end,
                "max_abs_charge_change_over_initial": float(np.max(np.abs(charge - charge[0])) / abs(charge[0])),
                "max_abs_dE_over_E0": float(np.max(np.abs(delta)) / abs(total[0])),
                "final_dE_over_E0": float(delta[-1] / abs(total[0])),
                "max_abs_dE_over_peak_field": float(np.max(np.abs(delta)) / np.max(field)),
            }
            rows.append(row)
            np.savez_compressed(output / f"short_landau_{case}_dt_{str(dt).replace('.', 'p')}.npz", **history)
    for case in ("cic", "c4"):
        selected = sorted((row for row in rows if row["case"] == case), key=lambda row: row["dt"])
        if any(row["max_abs_charge_change_over_initial"] != 0.0 for row in selected):
            raise AssertionError(f"charge changed in {case} time-step scan")
        for row in selected:
            row["successive_refinement_difference_final_dE_over_E0"] = ""
            row["successive_refinement_order"] = ""
        # The energy drift itself can approach a spatial/non-Hamiltonian floor.
        # Convergence of the final energy is therefore measured with successive
        # differences E_h(T)-E_{h/2}(T), without assuming an exact energy value.
        for index in range(1, len(selected)):
            coarse = selected[index]
            finer = selected[index - 1]
            difference = abs(coarse["final_dE_over_E0"] - finer["final_dE_over_E0"])
            coarse["successive_refinement_difference_final_dE_over_E0"] = difference
            if index >= 2:
                half_difference = selected[index - 1]["successive_refinement_difference_final_dE_over_E0"]
                coarse["successive_refinement_order"] = float(np.log2(difference / half_difference))
    return rows


def make_plot(operator_rows: list[dict], particle_rows: list[dict], time_rows: list[dict], output: pathlib.Path) -> None:
    import matplotlib.pyplot as plt

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
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.45), constrained_layout=True)
    colors = {"spectral": "#172a46", "centered_fd": "#9b2226"}
    for formulation in ("spectral", "centered_fd"):
        selected = [row for row in operator_rows if row["formulation"] == formulation]
        axes[0].plot(
            [row["sigma_cells"] for row in selected],
            [row["relative_force_norm"] for row in selected],
            "o-",
            color=colors[formulation],
            label=formulation.replace("_", " "),
        )
    axes[0].plot(
        [row["sigma_cells"] for row in particle_rows],
        [row["normalized_total_force"] for row in particle_rows],
        "s--",
        color="#2a9d8f",
        label="complete particle force",
    )
    axes[0].set_yscale("log")
    axes[0].set_xlabel(r"Gaussian width $\sigma/\Delta x$")
    axes[0].set_ylabel(r"$\|\sum\rho\mathbf{E}\|/\sum|\rho\mathbf{E}|$")
    axes[0].set_title("matched exact operators")
    axes[0].legend(frameon=False, fontsize=7)
    axes[0].grid(alpha=0.25)
    for case, color in (("cic", "#172a46"), ("c4", "#9b2226")):
        selected = sorted((row for row in time_rows if row["case"] == case), key=lambda row: row["dt"])
        differences = [row for row in selected if row["successive_refinement_difference_final_dE_over_E0"] != ""]
        finite_orders = [float(row["successive_refinement_order"]) for row in selected if row["successive_refinement_order"] != ""]
        axes[1].loglog(
            [row["dt"] for row in selected],
            [row["max_abs_dE_over_E0"] for row in selected],
            "o:",
            color=color,
            alpha=0.55,
            label=f"{case.upper()} max drift",
        )
        axes[1].loglog(
            [row["dt"] for row in differences],
            [row["successive_refinement_difference_final_dE_over_E0"] for row in differences],
            "s-",
            color=color,
            label=f"{case.upper()} successive diff. (median $p={np.median(finite_orders):.2f}$)",
        )
    reference_x = np.asarray([0.025, 0.2])
    axes[1].loglog(reference_x, 0.03 * reference_x**2, "--", color="0.5", lw=0.8, label=r"$O(\Delta t^2)$")
    axes[1].set_xlabel(r"time step $\Delta t\,\omega_{pe}$")
    axes[1].set_ylabel("normalized energy error")
    axes[1].set_title("controlled time-step scan")
    axes[1].set_xticks([0.0125, 0.025, 0.05, 0.1, 0.2])
    axes[1].set_xticklabels(["0.0125", "0.025", "0.05", "0.1", "0.2"], rotation=25)
    axes[1].minorticks_off()
    axes[1].legend(frameon=False, fontsize=7)
    axes[1].grid(alpha=0.25)
    fig.savefig(output / "controlled_conservation_checks.pdf", bbox_inches="tight")
    fig.savefig(output / "controlled_conservation_checks.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=pathlib.Path, default=HERE / "results" / "conservation_audit" / "small_tests")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    operator_rows = operator_tests()
    particle_rows = particle_force_tests()
    time_rows = time_step_tests(args.output)
    write_csv(args.output / "matched_operator_force.csv", operator_rows)
    write_csv(args.output / "matched_particle_force.csv", particle_rows)
    write_csv(args.output / "time_step_energy.csv", time_rows)
    make_plot(operator_rows, particle_rows, time_rows, args.output)
    result = {
        "scope": "controlled diagnostic only; not a manuscript benchmark measurement",
        "operator_rows": operator_rows,
        "particle_force_rows": particle_rows,
        "time_step_rows": time_rows,
    }
    (args.output / "small_conservation_tests.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
