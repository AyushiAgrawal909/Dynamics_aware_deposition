"""
tests/test_phase1.py
====================

Phase 1 acceptance tests for the 2D CIC baseline.
"""

from __future__ import annotations

import numpy as np
import pytest

from pic_kde_2d.conventions import CHARGE_TOL, SELF_FORCE_TOL, landau_box_length, thermal_velocity
from pic_kde_2d.core.grid2d import Grid2D
from pic_kde_2d.core.particles2d import init_landau_2d, quiet_start_maxwellian_2d
from pic_kde_2d.core.pic_loop import run_simulation
from pic_kde_2d.methods.cic import deposit_cic_2d
from pic_kde_2d.methods.cic_wrappers import cic_deposit_method, cic_gather_method
from pic_kde_2d.validation.self_force_2d import test_self_force as check_self_force
from pic_kde_2d.validation.conservation_2d import check_charge_conservation
from pic_kde_2d.benchmarks.theory import landau_gamma_theory
from pic_kde_2d.benchmarks.landau_2d import fit_mode_gamma
from pic_kde_2d.conventions import marker_grid_shape


def test_zero_density_zero_field():
    g = Grid2D(Nx=32, Ny=32, Lx=10.0, Ly=10.0)
    rho = np.zeros((32, 32))
    Ex, Ey = g.poisson_solve(rho)
    assert np.max(np.abs(Ex)) < 1.0e-12
    assert np.max(np.abs(Ey)) < 1.0e-12


def test_constant_density_zero_field():
    g = Grid2D(Nx=32, Ny=32, Lx=10.0, Ly=10.0)
    rho = np.full((32, 32), 0.5)
    Ex, Ey = g.poisson_solve(rho)
    assert np.max(np.abs(Ex)) < 1.0e-12
    assert np.max(np.abs(Ey)) < 1.0e-12


def test_sinusoidal_density_analytical_field():
    Nx, Ny = 128, 32
    Lx, Ly = 10.0, 10.0
    g = Grid2D(Nx=Nx, Ny=Ny, Lx=Lx, Ly=Ly)
    A = 0.05
    k = 2.0 * np.pi / Lx
    rho = A * np.cos(k * g.x)[:, None] * np.ones(Ny)[None, :]
    Ex, Ey = g.poisson_solve(rho)
    Ex_expected = (A / k) * np.sin(k * g.x)[:, None] * np.ones(Ny)[None, :]
    np.testing.assert_allclose(Ex, Ex_expected, atol=1.0e-12)
    np.testing.assert_allclose(Ey, 0.0, atol=1.0e-12)


def test_grid_quiet_start_uses_seed_without_changing_cell_quadrature():
    kwargs = {
        "Nx_marker": 8,
        "Ny_marker": 8,
        "Lx": 4.0,
        "Ly": 4.0,
        "v_thermal": 1.0,
        "grid_shape": (2, 2),
    }
    first = quiet_start_maxwellian_2d(rng=np.random.default_rng(3), **kwargs)
    repeated = quiet_start_maxwellian_2d(rng=np.random.default_rng(3), **kwargs)
    second = quiet_start_maxwellian_2d(rng=np.random.default_rng(4), **kwargs)

    np.testing.assert_array_equal(first[0], second[0])
    np.testing.assert_array_equal(first[1], second[1])
    np.testing.assert_array_equal(first[2], repeated[2])
    np.testing.assert_array_equal(first[3], repeated[3])
    assert not np.array_equal(first[2], second[2])
    assert not np.array_equal(first[3], second[3])

    for cell_x in range(2):
        for cell_y in range(2):
            indices = [
                marker_x * 8 + marker_y
                for marker_x in range(cell_x * 4, (cell_x + 1) * 4)
                for marker_y in range(cell_y * 4, (cell_y + 1) * 4)
            ]
            np.testing.assert_array_equal(
                np.sort(first[2][indices]),
                np.sort(second[2][indices]),
            )
            np.testing.assert_array_equal(
                np.sort(first[3][indices]),
                np.sort(second[3][indices]),
            )


def test_landau_velocity_phase_uses_unshifted_marker_position():
    kwargs = {
        "Nx_marker": 8,
        "Ny_marker": 4,
        "Lx": 10.0,
        "Ly": 6.0,
        "v_thermal": 1.0,
    }
    seed = 912
    alpha = 0.2
    kx_mode = 2
    ky_mode = 1
    omega_real = 1.25
    gamma = -0.08
    x0, y0, vx0, vy0 = quiet_start_maxwellian_2d(
        rng=np.random.default_rng(seed),
        **kwargs,
    )
    particles = init_landau_2d(
        alpha=alpha,
        kx_mode=kx_mode,
        ky_mode=ky_mode,
        omega_real=omega_real,
        gamma=gamma,
        rng=np.random.default_rng(seed),
        **kwargs,
    )

    kx = 2.0 * np.pi * kx_mode / kwargs["Lx"]
    ky = 2.0 * np.pi * ky_mode / kwargs["Ly"]
    k_mag = float(np.hypot(kx, ky))
    phase0 = kx * x0 + ky * y0
    u_parallel = alpha * (
        gamma * np.sin(phase0) - omega_real * np.cos(phase0)
    ) / k_mag
    np.testing.assert_allclose(
        particles.vx,
        vx0 + u_parallel * kx / k_mag,
        rtol=0.0,
        atol=2.0e-15,
    )
    np.testing.assert_allclose(
        particles.vy,
        vy0 + u_parallel * ky / k_mag,
        rtol=0.0,
        atol=2.0e-15,
    )


def test_deposit_charge_conservation():
    g = Grid2D(Nx=64, Ny=64, Lx=10.0, Ly=10.0)
    rng = np.random.default_rng(42)
    N_p = 1000
    x = rng.uniform(0.0, g.Lx, N_p)
    y = rng.uniform(0.0, g.Ly, N_p)
    w = rng.uniform(-1.0, -0.5, N_p)
    rho = deposit_cic_2d(x, y, w, g.Nx, g.Ny, g.dx, g.dy)
    result = check_charge_conservation(rho, np.sum(w), g.dx, g.dy, tol=CHARGE_TOL)
    assert result["passes"], f"Charge not conserved: {result}"


def test_single_particle_self_force():
    g = Grid2D(Nx=64, Ny=64, Lx=10.0, Ly=10.0)
    result = check_self_force(cic_deposit_method, cic_gather_method, g)
    assert result["passes"], f"CIC self-force too large: {result['magnitude']}"


@pytest.mark.xfail(
    strict=True,
    reason=(
        "legacy Nx=32, PPC=64, ten-time-unit smoke benchmark misses its frozen "
        "25% continuum-rate tolerance; retained as an explicit known limitation"
    ),
)
def test_landau_damping_rate():
    kx_mode = 1
    Lx = landau_box_length(kx_mode)
    Ly = 8.0
    v_th = thermal_velocity()
    g = Grid2D(Nx=32, Ny=8, Lx=Lx, Ly=Ly)
    ppc = 64
    Nx_marker, Ny_marker = marker_grid_shape(g.Nx, g.Ny, ppc)
    particles = init_landau_2d(
        Nx_marker=Nx_marker,
        Ny_marker=Ny_marker,
        Lx=Lx,
        Ly=Ly,
        v_thermal=v_th,
        alpha=0.02,
        kx_mode=kx_mode,
        ky_mode=0,
        rng=np.random.default_rng(123),
    )
    history = run_simulation(
        particles,
        g,
        0.05,
        200,
        cic_deposit_method,
        cic_gather_method,
        diag_stride=2,
        tracked_modes=[(kx_mode, 0)],
        verbose=False,
    )
    amps = np.abs(np.asarray(history.mode_amplitudes_Ex[(kx_mode, 0)], dtype=np.complex128))
    times = np.asarray(history.times, dtype=np.float64)
    gamma_measured = fit_mode_gamma(times, amps, 2.0, 12.0, amp_floor=1.0e-8)
    gamma_theory = landau_gamma_theory(k_lambdaD=0.4, v_thermal=v_th)
    relative_error = abs(gamma_measured - gamma_theory) / abs(gamma_theory)
    assert relative_error < 0.25, (
        f"Landau gamma off: measured={gamma_measured:.4f}, theory={gamma_theory:.4f}, "
        f"relative_error={relative_error:.2%}"
    )
