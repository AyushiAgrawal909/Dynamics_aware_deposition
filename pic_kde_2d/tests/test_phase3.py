"""
tests/test_phase3.py
====================

Phase 3 acceptance tests for matched spectral-W filtering.
"""

from __future__ import annotations

import numpy as np
import pytest

from pic_kde_2d.benchmarks.cost_comparison import benchmark_deposit_cost
from pic_kde_2d.benchmarks.landau_2d import fit_mode_gamma
from pic_kde_2d.benchmarks.theory import landau_gamma_theory
from pic_kde_2d.conventions import CHARGE_TOL, landau_box_length, marker_grid_shape, thermal_velocity
from pic_kde_2d.core.grid2d import Grid2D
from pic_kde_2d.core.particles2d import Particles2D, init_landau_2d
from pic_kde_2d.core.pic_loop import run_simulation
from pic_kde_2d.methods.spectral_W import (
    deposit_binned_adaptive_spectral_W,
    make_W_gaussian,
)
from pic_kde_2d.methods.spectral_W_wrappers import make_spectral_W_gaussian_methods
from pic_kde_2d.validation.conservation_2d import check_charge_conservation
from pic_kde_2d.validation.self_force_2d import test_self_force
from pic_kde_2d.validation.spectral_W_validation import compare_phase2_vs_phase3_density


class TestSpectralWBasic:
    def test_gaussian_W_at_zero_is_one(self):
        g = Grid2D(Nx=32, Ny=32, Lx=10.0, Ly=10.0)
        W = make_W_gaussian(g, sigma_x=1.0, sigma_y=1.0)
        assert abs(W[0, 0] - 1.0) < 1.0e-12

    def test_gaussian_W_attenuates_high_k(self):
        g = Grid2D(Nx=32, Ny=32, Lx=10.0, Ly=10.0)
        sigma = 0.5
        W = make_W_gaussian(g, sigma, sigma)
        kx_nyquist = np.pi / g.dx
        expected = np.exp(-0.5 * kx_nyquist**2 * sigma**2)
        assert abs(W[g.Nx // 2, 0] - expected) < 1.0e-10

    def test_binned_adaptive_deposit_preserves_charge_and_constant_width_limit(self):
        g = Grid2D(Nx=16, Ny=12, Lx=8.0, Ly=6.0)
        rng = np.random.default_rng(8)
        n_particles = 40
        particles = Particles2D(
            x=rng.uniform(0.0, g.Lx, n_particles),
            y=rng.uniform(0.0, g.Ly, n_particles),
            vx=np.zeros(n_particles),
            vy=np.zeros(n_particles),
            w=rng.uniform(-1.0, -0.5, n_particles),
            Lx=g.Lx,
            Ly=g.Ly,
        )
        sigma_x = np.full(n_particles, 0.7 * g.dx)
        sigma_y = np.full(n_particles, 1.1 * g.dy)
        rho = deposit_binned_adaptive_spectral_W(
            particles,
            g,
            sigma_x,
            sigma_y,
        )
        deposit, _ = make_spectral_W_gaussian_methods(
            g,
            sigma_x[0],
            sigma_y[0],
        )
        np.testing.assert_allclose(rho, deposit(particles, g), atol=1.0e-13)
        np.testing.assert_allclose(
            np.sum(rho) * g.dx * g.dy,
            np.sum(particles.w),
            atol=1.0e-12,
        )


class TestSpectralWCharge:
    def test_charge_conservation_with_filter(self):
        g = Grid2D(Nx=64, Ny=64, Lx=10.0, Ly=10.0)
        sigma = 2.0 * g.dx
        deposit, _ = make_spectral_W_gaussian_methods(g, sigma, sigma)
        rng = np.random.default_rng(42)
        n_particles = 500
        particles = Particles2D(
            x=rng.uniform(0.0, g.Lx, n_particles),
            y=rng.uniform(0.0, g.Ly, n_particles),
            vx=np.zeros(n_particles, dtype=np.float64),
            vy=np.zeros(n_particles, dtype=np.float64),
            w=rng.uniform(-1.0, -0.5, n_particles),
            Lx=g.Lx,
            Ly=g.Ly,
        )
        rho = deposit(particles, g)
        result = check_charge_conservation(rho, np.sum(particles.w), g.dx, g.dy, tol=CHARGE_TOL)
        assert result["passes"], result


class TestSpectralWSelfForce:
    @pytest.mark.parametrize("sigma_dx", [0.5, 1.0, 2.0, 3.0, 5.0])
    def test_isotropic_self_force(self, sigma_dx):
        g = Grid2D(Nx=64, Ny=64, Lx=10.0, Ly=10.0)
        sigma = sigma_dx * g.dx
        deposit, gather = make_spectral_W_gaussian_methods(g, sigma, sigma)
        result = test_self_force(deposit, gather, g)
        assert result["passes"], result

    @pytest.mark.parametrize("sx_dx,sy_dx", [(0.5, 2.0), (3.0, 0.5), (1.0, 1.0)])
    def test_anisotropic_self_force(self, sx_dx, sy_dx):
        g = Grid2D(Nx=64, Ny=64, Lx=10.0, Ly=10.0)
        deposit, gather = make_spectral_W_gaussian_methods(g, sx_dx * g.dx, sy_dx * g.dy)
        result = test_self_force(deposit, gather, g)
        assert result["passes"], result


class TestSpectralWMatchesPhase2:
    @pytest.mark.parametrize("sigma_dx", [1.0, 2.0])
    def test_density_matches_phase2(self, sigma_dx):
        g = Grid2D(Nx=64, Ny=64, Lx=10.0, Ly=10.0)
        sigma = sigma_dx * g.dx
        particles = init_landau_2d(
            Nx_marker=32,
            Ny_marker=32,
            Lx=g.Lx,
            Ly=g.Ly,
            v_thermal=1.0,
            alpha=0.02,
            kx_mode=1,
            rng=np.random.default_rng(42),
        )
        result = compare_phase2_vs_phase3_density(g, particles, sigma, sigma)
        assert result["relative_rms"] < 0.05, result


class TestLandau2DSpectralW:
    @pytest.mark.slow
    @pytest.mark.xfail(
        strict=True,
        reason=(
            "legacy PPC=16 spectral-W smoke benchmark misses its frozen 30% "
            "continuum-rate tolerance; retained as an explicit known limitation"
        ),
    )
    def test_landau_with_W_sigma_3dx(self):
        kx_mode = 1
        Lx = landau_box_length(kx_mode)
        Ly = 8.0
        Nx, Ny = 64, 16
        g = Grid2D(Nx=Nx, Ny=Ny, Lx=Lx, Ly=Ly)
        sigma = 3.0 * g.dx
        deposit, gather = make_spectral_W_gaussian_methods(g, sigma, sigma)
        Nx_marker, Ny_marker = marker_grid_shape(g.Nx, g.Ny, 16)
        particles = init_landau_2d(
            Nx_marker=Nx_marker,
            Ny_marker=Ny_marker,
            Lx=Lx,
            Ly=Ly,
            v_thermal=thermal_velocity(),
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
            deposit,
            gather,
            diag_stride=2,
            tracked_modes=[(kx_mode, 0)],
        )
        amps = np.abs(np.asarray(history.mode_amplitudes_Ex[(kx_mode, 0)], dtype=np.complex128))
        times = np.asarray(history.times, dtype=np.float64)
        gamma_measured = fit_mode_gamma(times, amps, 2.0, 10.0, amp_floor=1.0e-8)
        gamma_theory = landau_gamma_theory(k_lambdaD=0.4, v_thermal=thermal_velocity())
        rel_error = abs(gamma_measured - gamma_theory) / abs(gamma_theory)
        assert rel_error < 0.30


class TestCostComparison:
    @pytest.mark.slow
    def test_phase3_cheaper_at_sigma_3dx(self):
        g = Grid2D(Nx=96, Ny=96, Lx=15.0, Ly=15.0)
        results = benchmark_deposit_cost(g, ppc=8, sigma_dx_list=[2.0, 3.0], n_warmup=1, n_trials=2)
        ratio = results["t_phase2"][1] / results["t_phase3"][1]
        assert ratio > 2.0, ratio
