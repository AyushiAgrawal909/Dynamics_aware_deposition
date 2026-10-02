#!/usr/bin/env python3
"""Freeze the E2 (12,1) beam-drift scan before reading simulation results.

The design uses the audited warm, oblique grid/time-discrete dielectric.  It
tracks the purely growing branch, locates its numerical marginal point, and
chooses directional Gaussian widths from the numerical roots.  This script is
diagnostic only: it does not launch or read any scan simulation.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np


HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[1]
CAMPAIGN_ROOT = WORKSPACE / "pic_kde_2d_landau_cv_stage"
for candidate in (HERE, CAMPAIGN_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from recompute_dispersion import (  # noqa: E402
    E2_M12N1,
    continuous_residual,
    gaussian_force_transfer,
    midpoint_residual,
    solve_level,
)
from pic_kde_2d.conventions import marker_grid_shape  # noqa: E402
from pic_kde_2d.core.grid2d import Grid2D  # noqa: E402
from pic_kde_2d.core.particles2d import init_two_stream_2d  # noqa: E402
from pic_kde_2d.methods.cic import deposit_cic_2d  # noqa: E402
from pic_kde_2d.methods.delta_f import (  # noqa: E402
    deposit_delta_f_2d,
    init_delta_f_from_particles,
)


DRIFTS = (0.60, 0.70, 0.76, 0.80, 0.818)
SEEDS = tuple(range(200, 210))
FIXED_SIGMA_Y_DY = 0.2427346
SAFETY_RATE_RATIO = 0.95
ADMISSIBLE_RATE_RATIO = 0.90
FIT_START = 24.32
TARGET_CONSTRAINED_EFOLDINGS = 5.0
TAIL_AFTER_FIT = 2.0
QMAX = 8
LMAX = 1024
BOOTSTRAP_SAMPLES = 5000
BOOTSTRAP_SEED = 120912
REMOTE_GIT_REVISION = "ed6dd96d16d0b70a53bf9486fd78a270ab3486d6"

SOURCE_PATHS = (
    CAMPAIGN_ROOT / "scripts/run_linearized_delta_f_mode.py",
    CAMPAIGN_ROOT / "pic_kde_2d/core/particles2d.py",
    CAMPAIGN_ROOT / "pic_kde_2d/core/pic_loop_delta_f.py",
    CAMPAIGN_ROOT / "pic_kde_2d/methods/delta_f.py",
    CAMPAIGN_ROOT / "pic_kde_2d/methods/spectral_W.py",
    CAMPAIGN_ROOT / "pic_kde_2d/core/grid2d.py",
    HERE / "recompute_dispersion.py",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(value.dtype.str.encode("ascii"))
    digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
    digest.update(value.tobytes())
    return digest.hexdigest()


def bisect_zero(function, lower: float, upper: float, iterations: int = 52) -> float:
    low_value = float(function(lower))
    high_value = float(function(upper))
    if not np.isfinite(low_value) or not np.isfinite(high_value) or low_value * high_value > 0.0:
        raise RuntimeError(
            f"invalid root bracket [{lower}, {upper}]: {low_value}, {high_value}"
        )
    for _ in range(iterations):
        middle = 0.5 * (lower + upper)
        middle_value = float(function(middle))
        if low_value * middle_value <= 0.0:
            upper, high_value = middle, middle_value
        else:
            lower, low_value = middle, middle_value
    return 0.5 * (lower + upper)


def residual_real(
    gamma: float,
    case,
    *,
    level: str,
    qmax: int,
    lmax: int,
    force_transfer: float = 1.0,
) -> float:
    if level == "principal_cic":
        value = continuous_residual(
            complex(gamma, 0.0),
            case,
            qmax=0,
            filter_force_transfer=force_transfer,
        )
    elif level == "spatial_discrete":
        value = continuous_residual(
            complex(gamma, 0.0),
            case,
            qmax=qmax,
            filter_force_transfer=force_transfer,
        )
    elif level == "grid_time_discrete":
        value = midpoint_residual(
            complex(gamma, 0.0),
            case,
            qmax=qmax,
            lmax=lmax,
            filter_force_transfer=force_transfer,
        )
    else:
        raise ValueError(level)
    if abs(value.imag) > 1.0e-10:
        raise AssertionError(f"growing-branch residual is not real: {value}")
    return float(value.real)


def growing_root(
    case,
    *,
    level: str,
    qmax: int,
    lmax: int,
    force_transfer: float = 1.0,
) -> tuple[float, float, str]:
    function = lambda gamma: residual_real(  # noqa: E731
        gamma,
        case,
        level=level,
        qmax=qmax,
        lmax=lmax,
        force_transfer=force_transfer,
    )
    at_zero = function(0.0)
    if at_zero >= 0.0:
        return 0.0, at_zero, "no positive purely-growing root"
    upper = 0.05
    while function(upper) < 0.0:
        upper *= 2.0
        if upper > 1.0:
            raise RuntimeError("could not bracket growing root")
    root = bisect_zero(function, 0.0, upper)
    residual = function(root)
    if abs(residual) > 2.0e-11:
        raise AssertionError(f"root residual {residual} exceeds tolerance")
    return root, residual, "unstable purely-growing branch"


def marginal_drift(*, qmax: int, lmax: int) -> float:
    def function(v_beam: float) -> float:
        case = replace(E2_M12N1, vbeam=v_beam)
        return residual_real(
            0.0,
            case,
            level="grid_time_discrete",
            qmax=qmax,
            lmax=lmax,
        )

    return bisect_zero(function, 0.80, 0.83)


def width_for_rate_ratio(
    case,
    baseline_gamma: float,
    ratio: float,
    *,
    qmax: int,
    lmax: int,
) -> tuple[float, float, float]:
    target_gamma = ratio * baseline_gamma

    def function(sigma_y_dy: float) -> float:
        transfer = gaussian_force_transfer(case, 0.0, sigma_y_dy)
        return residual_real(
            target_gamma,
            case,
            level="grid_time_discrete",
            qmax=qmax,
            lmax=lmax,
            force_transfer=transfer,
        )

    upper = 0.1
    while function(upper) < 0.0:
        upper *= 2.0
        if upper > 8.0:
            raise RuntimeError("could not bracket directional Gaussian width")
    sigma = bisect_zero(function, 0.0, upper)
    transfer = gaussian_force_transfer(case, 0.0, sigma)
    residual = function(sigma)
    if abs(residual) > 2.0e-11:
        raise AssertionError(f"width root residual {residual} exceeds tolerance")
    return sigma, transfer, residual


def cadence_ceil(value: float, cadence: float) -> float:
    return float(math.ceil((value - 1.0e-13) / cadence) * cadence)


def loading_invariance_check() -> dict:
    """Verify the actual PPC-512 initializer changes only signed beam offsets."""
    case = E2_M12N1
    nx_marker, ny_marker = marker_grid_shape(case.nx, case.ny, 512)
    endpoints = (DRIFTS[0], DRIFTS[-1])
    bases = []
    delta_particles = []
    for drift in endpoints:
        base = init_two_stream_2d(
            Nx_marker=nx_marker,
            Ny_marker=ny_marker,
            Lx=case.lx,
            Ly=case.ly,
            v_beam=drift,
            v_thermal=case.vthermal,
            alpha=0.0,
            kx_mode=case.mx,
            ky_mode=case.my,
            rng=np.random.default_rng(SEEDS[0]),
        )
        delta = init_delta_f_from_particles(
            base,
            f0_v_thermal=case.vthermal,
            f0_density=1.0,
            alpha=1.0e-8,
            kx_phys=case.kx,
            ky_phys=case.ky,
            background_kind="two_stream",
            beam_velocity=drift,
        )
        bases.append(base)
        delta_particles.append(delta)

    lower, upper = bases
    delta_lower, delta_upper = delta_particles
    drift_difference = endpoints[1] - endpoints[0]
    beam_sign = np.rint((upper.vx - lower.vx) / drift_difference)
    if not np.all(np.isin(beam_sign, (-1.0, 1.0))):
        raise AssertionError("beam assignment changed with drift")
    thermal_lower = lower.vx - beam_sign * endpoints[0]
    thermal_upper = upper.vx - beam_sign * endpoints[1]
    grid = Grid2D(Nx=case.nx, Ny=case.ny, Lx=case.lx, Ly=case.ly)
    density_lower = deposit_cic_2d(
        lower.x,
        lower.y,
        lower.w,
        grid.Nx,
        grid.Ny,
        grid.dx,
        grid.dy,
    )
    density_upper = deposit_cic_2d(
        upper.x,
        upper.y,
        upper.w,
        grid.Nx,
        grid.Ny,
        grid.dx,
        grid.dy,
    )
    perturbation_lower = deposit_delta_f_2d(delta_lower, grid)
    perturbation_upper = deposit_delta_f_2d(delta_upper, grid)

    exact_equalities = {
        "x": bool(np.array_equal(lower.x, upper.x)),
        "y": bool(np.array_equal(lower.y, upper.y)),
        "vy": bool(np.array_equal(lower.vy, upper.vy)),
        "charge": bool(np.array_equal(lower.w, upper.w)),
        "initial_delta_weight": bool(
            np.array_equal(delta_lower.w_delta, delta_upper.w_delta)
        ),
        "initial_full_density": bool(np.array_equal(density_lower, density_upper)),
        "initial_delta_density": bool(
            np.array_equal(perturbation_lower, perturbation_upper)
        ),
    }
    if not all(exact_equalities.values()):
        raise AssertionError(f"drift pairing invariance failed: {exact_equalities}")
    maximum_thermal_residual_difference = float(
        np.max(np.abs(thermal_lower - thermal_upper))
    )
    if maximum_thermal_residual_difference > 3.0e-16:
        raise AssertionError("thermal samples changed with drift")
    negative = int(np.count_nonzero(beam_sign == -1.0))
    positive = int(np.count_nonzero(beam_sign == 1.0))
    if negative != positive or negative + positive != nx_marker * ny_marker:
        raise AssertionError("beam populations are not exactly symmetric")
    return {
        "seed": SEEDS[0],
        "ppc": 512,
        "marker_grid": [nx_marker, ny_marker],
        "particles": nx_marker * ny_marker,
        "drift_endpoints_checked": list(endpoints),
        "exact_equalities": exact_equalities,
        "negative_beam_particles": negative,
        "positive_beam_particles": positive,
        "maximum_thermal_residual_difference": maximum_thermal_residual_difference,
        "initial_full_density_sha256": array_sha256(density_lower),
        "initial_delta_density_sha256": array_sha256(perturbation_lower),
        "scope": (
            "initial density and perturbation are invariant; later snapshots are "
            "not claimed invariant"
        ),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    args.output.mkdir(parents=True)

    source_hashes = {
        str(path.relative_to(WORKSPACE)): sha256(path) for path in SOURCE_PATHS
    }
    remote_expected = {
        "pic_kde_2d_landau_cv_stage/scripts/run_linearized_delta_f_mode.py":
            "c0e849a96c4822d07263cb6de137296766f1ef531d182a10c9a3d39236143599",
        "pic_kde_2d_landau_cv_stage/pic_kde_2d/core/particles2d.py":
            "7bc3e88101d9071048c85eff204694c71e96f2a55f32b68082ff055f171bdb4e",
        "pic_kde_2d_landau_cv_stage/pic_kde_2d/core/pic_loop_delta_f.py":
            "cc2231527ba0e725d4929bd0b12c3ffb04b52cb6c69d6fc20c925e16d831c814",
        "pic_kde_2d_landau_cv_stage/pic_kde_2d/methods/delta_f.py":
            "e5c75624ae09cb94381fa3c519a82828528ca511ec3e73f9cf6dce6799f9e074",
        "pic_kde_2d_landau_cv_stage/pic_kde_2d/methods/spectral_W.py":
            "fc9d4091d53873be2519bac4c1a6edd7bcbd047913de56ee4ec7777b03053ca2",
        "pic_kde_2d_landau_cv_stage/pic_kde_2d/core/grid2d.py":
            "f3b3c04fb7bbb06ee67cf299ca2e2b0c7d4ae58f18660d2e083fae1381826299",
    }
    for path, expected in remote_expected.items():
        if source_hashes[path] != expected:
            raise AssertionError(f"local/remote source hash mismatch for {path}")

    convergence_settings = ((4, 1024), (8, 512), (8, 1024), (10, 2048))
    convergence_rows: list[dict] = []
    convergence_by_setting: dict[tuple[int, int], dict[float, dict]] = {}
    marginal_by_setting = {}
    for qmax, lmax in convergence_settings:
        marginal = marginal_drift(qmax=qmax, lmax=lmax)
        marginal_by_setting[(qmax, lmax)] = marginal
        convergence_by_setting[(qmax, lmax)] = {}
        for drift in DRIFTS:
            case = replace(E2_M12N1, vbeam=drift)
            baseline, baseline_residual, _ = growing_root(
                case,
                level="grid_time_discrete",
                qmax=qmax,
                lmax=lmax,
            )
            sigma_10, _, width_10_residual = width_for_rate_ratio(
                case,
                baseline,
                ADMISSIBLE_RATE_RATIO,
                qmax=qmax,
                lmax=lmax,
            )
            sigma_5, _, width_5_residual = width_for_rate_ratio(
                case,
                baseline,
                SAFETY_RATE_RATIO,
                qmax=qmax,
                lmax=lmax,
            )
            row = {
                "v_beam": drift,
                "qmax": qmax,
                "lmax": lmax,
                "marginal_v_beam": marginal,
                "baseline_gamma": baseline,
                "sigma_y_10pct_boundary_dy": sigma_10,
                "sigma_y_safety_5pct_dy": sigma_5,
                "baseline_root_residual": baseline_residual,
                "boundary_root_residual": width_10_residual,
                "safety_root_residual": width_5_residual,
            }
            convergence_rows.append(row)
            convergence_by_setting[(qmax, lmax)][drift] = row

    reference = convergence_by_setting[(QMAX, LMAX)]
    verification = convergence_by_setting[(10, 2048)]
    max_gamma_difference = max(
        abs(reference[drift]["baseline_gamma"] - verification[drift]["baseline_gamma"])
        for drift in DRIFTS
    )
    max_boundary_difference = max(
        abs(
            reference[drift]["sigma_y_10pct_boundary_dy"]
            - verification[drift]["sigma_y_10pct_boundary_dy"]
        )
        for drift in DRIFTS
    )
    max_safety_difference = max(
        abs(
            reference[drift]["sigma_y_safety_5pct_dy"]
            - verification[drift]["sigma_y_safety_5pct_dy"]
        )
        for drift in DRIFTS
    )
    marginal_difference = abs(
        marginal_by_setting[(QMAX, LMAX)] - marginal_by_setting[(10, 2048)]
    )
    if max(max_gamma_difference, max_boundary_difference, max_safety_difference) >= 2.0e-6:
        raise AssertionError("grid/time root or bandwidth truncation is not converged")
    if marginal_difference >= 2.0e-6:
        raise AssertionError("marginal drift is not converged")

    fixed_one_pass = math.exp(
        -0.5 * (E2_M12N1.ky * E2_M12N1.dy * FIXED_SIGMA_Y_DY) ** 2
    )
    fixed_force = gaussian_force_transfer(E2_M12N1, 0.0, FIXED_SIGMA_Y_DY)
    cadence = 0.02 * 2
    scan_rows: list[dict] = []
    for drift in DRIFTS:
        case = replace(E2_M12N1, vbeam=drift)
        continuum = float(solve_level(case, "continuum").real)
        principal, _, _ = growing_root(
            case, level="principal_cic", qmax=0, lmax=0
        )
        spatial, _, _ = growing_root(
            case, level="spatial_discrete", qmax=QMAX, lmax=0
        )
        baseline, baseline_residual, branch_status = growing_root(
            case,
            level="grid_time_discrete",
            qmax=QMAX,
            lmax=LMAX,
        )
        fixed_gamma, fixed_residual, fixed_status = growing_root(
            case,
            level="grid_time_discrete",
            qmax=QMAX,
            lmax=LMAX,
            force_transfer=fixed_force,
        )
        sigma_10, force_10, residual_10 = width_for_rate_ratio(
            case,
            baseline,
            ADMISSIBLE_RATE_RATIO,
            qmax=QMAX,
            lmax=LMAX,
        )
        sigma_5, force_5, residual_5 = width_for_rate_ratio(
            case,
            baseline,
            SAFETY_RATE_RATIO,
            qmax=QMAX,
            lmax=LMAX,
        )
        constrained_gamma, constrained_residual, constrained_status = growing_root(
            case,
            level="grid_time_discrete",
            qmax=QMAX,
            lmax=LMAX,
            force_transfer=force_5,
        )
        if abs(constrained_gamma / baseline - SAFETY_RATE_RATIO) > 2.0e-10:
            raise AssertionError("constrained-width rate target was not reproduced")
        fit_stop = cadence_ceil(
            FIT_START + TARGET_CONSTRAINED_EFOLDINGS / constrained_gamma,
            cadence,
        )
        t_end = fit_stop + TAIL_AFTER_FIT
        scan_rows.append(
            {
                "v_beam": drift,
                "v_beam_plus": drift,
                "v_beam_minus": -drift,
                "continuum_gamma": continuum,
                "principal_cic_gamma": principal,
                "spatial_discrete_gamma": spatial,
                "grid_time_discrete_cic_gamma": baseline,
                "baseline_branch_status": branch_status,
                "baseline_root_residual": baseline_residual,
                "fixed_sigma_x_dx": 0.0,
                "fixed_sigma_y_dy": FIXED_SIGMA_Y_DY,
                "fixed_filter_W_one_application": fixed_one_pass,
                "fixed_particle_force_W2": fixed_force,
                "fixed_predicted_gamma": fixed_gamma,
                "fixed_predicted_rate_ratio": fixed_gamma / baseline,
                "fixed_branch_status": fixed_status,
                "fixed_root_residual_or_epsilon_at_zero": fixed_residual,
                "sigma_y_10pct_boundary_dy": sigma_10,
                "boundary_particle_force_W2": force_10,
                "boundary_root_residual": residual_10,
                "constrained_sigma_x_dx": 0.0,
                "constrained_sigma_y_dy": sigma_5,
                "constrained_particle_force_W2": force_5,
                "constrained_predicted_gamma": constrained_gamma,
                "constrained_predicted_rate_ratio": constrained_gamma / baseline,
                "constrained_branch_status": constrained_status,
                "constrained_root_residual": constrained_residual,
                "fit_t_start": FIT_START,
                "fit_t_stop": fit_stop,
                "t_end": t_end,
                "n_steps": int(round(t_end / case.dt)),
                "predicted_cic_fit_efoldings": baseline * (fit_stop - FIT_START),
                "predicted_fixed_fit_efoldings": fixed_gamma * (fit_stop - FIT_START),
                "predicted_constrained_fit_efoldings": constrained_gamma
                * (fit_stop - FIT_START),
            }
        )

    loading = loading_invariance_check()
    source_identity = {
        "remote_git_revision_observed_before_freeze": REMOTE_GIT_REVISION,
        "local_source_sha256": source_hashes,
        "remote_source_sha256_expected_and_matched": remote_expected,
        "baseline_source_modified_for_campaign": False,
    }
    specification = {
        "campaign_id": "E2_beam_drift_scan_v1",
        "status": "FROZEN_BEFORE_SIMULATION_RESULTS",
        "scope": (
            "mode-projected linearized delta-f test of the linear E2 mechanism; "
            "not unrestricted nonlinear full-f fidelity"
        ),
        "selection_scope": (
            "initial density-based bandwidth objective is identical across drift "
            "values; no equality is claimed for later evolving snapshots"
        ),
        "mode": [12, 1],
        "grid": {"Nx": 128, "Ny": 16, "Lx": 64.0, "Ly": 16.0},
        "thermal_velocity": {"x": 0.05, "y": 0.05},
        "symmetric_beam_drifts": list(DRIFTS),
        "ppc": 512,
        "marker_grid": [2048, 512],
        "particles_per_history": 1048576,
        "alpha": 1.0e-8,
        "dt": 0.02,
        "diagnostic_stride": 2,
        "diagnostic_cadence": cadence,
        "target_mode_projection": True,
        "deposition": "perturbation-weight CIC, then one Gaussian W",
        "poisson_solver": "periodic spectral Poisson solve on the projected conjugate mode pair",
        "gather": "one Gaussian W on nodal E, then CIC interpolation",
        "filter_direction": "y only; sigma_x/dx=0",
        "arms": ["cic", "fixed_gaussian", "constrained_gaussian"],
        "fixed_gaussian_sigma_y_dy": FIXED_SIGMA_Y_DY,
        "fixed_gaussian_W_one_application": fixed_one_pass,
        "fixed_gaussian_particle_force_W2": fixed_force,
        "admissible_boundary": "grid/time-discrete filtered gamma / CIC gamma = 0.90",
        "safety_rule": (
            "constrained width solves grid/time-discrete filtered gamma / CIC gamma "
            "= 0.95, leaving five percentage points before the 0.90 boundary"
        ),
        "model": {
            "name": "audited warm oblique grid/time-discrete dielectric",
            "spatial_alias_qmax": QMAX,
            "temporal_alias_lmax": LMAX,
            "root_residual_tolerance": 2.0e-11,
            "cold_longitudinal_threshold_used": False,
            "marginal_v_beam": marginal_by_setting[(QMAX, LMAX)],
        },
        "seeds": list(SEEDS),
        "seed_count": len(SEEDS),
        "seed_count_rationale": (
            "existing ten-seed E2 directional Gaussian ratio has bootstrap SD "
            "0.00398 and 95% interval width 0.01535; ten paired fresh seeds are "
            "fixed because that precision resolves a 0.95 target from the 0.90 gate"
        ),
        "development_seed_pooling": False,
        "fit": {
            "observable": "slope of ln(mean_seed(abs(Ex_hat(12,1))))",
            "t_start": FIT_START,
            "t_stop_by_drift": {
                f"{row['v_beam']:.3f}": row["fit_t_stop"] for row in scan_rows
            },
            "window_selection": "fixed from numerical predictions before results",
            "target_constrained_model_efoldings": TARGET_CONSTRAINED_EFOLDINGS,
            "tail_after_fit": TAIL_AFTER_FIT,
            "no_seed_rejection": True,
        },
        "uncertainty": {
            "bootstrap_samples": BOOTSTRAP_SAMPLES,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "unit": "paired independent seed index; no temporal resampling",
        },
        "quality_rules": {
            "rate_preservation": "paired 95% filtered/CIC interval wholly inside [0.90,1.10]",
            "positive_growth": "central gamma > 0 and 95% gamma lower limit > 0",
            "fit_r_squared_min": 0.98,
            "fitted_efoldings_min": 2.0,
            "fit_start_snr_min": 5.0,
            "delta_f_ratio_max": 0.5,
            "unresolvable_policy": "report UNRESOLVED; do not add seeds or change windows",
            "model_accuracy_reporting": (
                "prediction-inside-measured-95%-interval and relative discrepancy "
                "are reported separately from rate preservation"
            ),
        },
        "duration_schedule": {
            f"{row['v_beam']:.3f}": {
                "fit_t_stop": row["fit_t_stop"],
                "t_end": row["t_end"],
                "n_steps": row["n_steps"],
            }
            for row in scan_rows
        },
        "budget": {
            "histories": len(DRIFTS) * 3 * len(SEEDS),
            "configurations_before_seed_replication": len(DRIFTS) * 3,
            "requested_nodes": 1,
            "requested_partition": "cpu",
            "requested_wall_hours_hard_max": 4.0,
            "concurrent_histories_max": 40,
            "batch_count": 4,
            "empirical_expected_cpu_node_hours": 2.83,
            "basis": (
                "job 5195640 used about 0.45 node-hour per ten-seed t=60 arm; "
                "four packed batches have maximum durations summing to 376.16"
            ),
        },
        "source_identity": source_identity,
        "scan_rows": scan_rows,
    }

    frozen_path = args.output / "frozen_campaign.json"
    frozen_path.write_text(
        json.dumps(specification, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_csv(args.output / "predicted_scan.csv", scan_rows)
    write_csv(args.output / "root_convergence.csv", convergence_rows)
    (args.output / "loading_invariance.json").write_text(
        json.dumps(loading, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    summary = {
        "status": "PASS",
        "simulation_results_read": False,
        "new_simulations_run": False,
        "frozen_campaign_sha256": sha256(frozen_path),
        "marginal_v_beam": marginal_by_setting[(QMAX, LMAX)],
        "selected_drifts": list(DRIFTS),
        "maximum_q8_l1024_vs_q10_l2048_gamma_difference": max_gamma_difference,
        "maximum_q8_l1024_vs_q10_l2048_boundary_difference": max_boundary_difference,
        "maximum_q8_l1024_vs_q10_l2048_safety_width_difference": max_safety_difference,
        "q8_l1024_vs_q10_l2048_marginal_difference": marginal_difference,
        "loading_invariance": loading,
    }
    (args.output / "preflight_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
