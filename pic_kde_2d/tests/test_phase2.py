"""
tests/test_phase2.py
====================

Phase 2 acceptance tests for anisotropic matched Gaussian deposit and gather.
"""

from __future__ import annotations

import numpy as np
import pytest

from pic_kde_2d.conventions import CHARGE_TOL, landau_box_length, marker_grid_shape, thermal_velocity
from pic_kde_2d.core.grid2d import Grid2D
from pic_kde_2d.core.particles2d import init_landau_2d
from pic_kde_2d.core.pic_loop import run_simulation
from pic_kde_2d.methods.gaussian import deposit_gaussian_2d_aniso
from pic_kde_2d.methods.gaussian_wrappers import make_gaussian_methods
from pic_kde_2d.methods.sigma_selection import select_sigma_anisotropic_2d, sigma_max_matched
from pic_kde_2d.validation.conservation_2d import check_charge_conservation
from pic_kde_2d.validation.self_force_2d import test_self_force
from pic_kde_2d.validation.transfer_function_test import compare_mode_transfer_to_analytical, measure_transfer_function_deposit
from pic_kde_2d.benchmarks.landau_2d import fit_mode_gamma
from pic_kde_2d.benchmarks.theory import landau_gamma_theory


class TestGaussianChargeConservation:
    @pytest.mark.parametrize("sigma_dx", [0.5, 1.0, 2.0, 3.0])
    def test_deposit_conserves_charge(self, sigma_dx):
        g = Grid2D(Nx=64, Ny=64, Lx=10.0, Ly=10.0)
        rng = np.random.default_rng(42)
        N_p = 1000
        x = rng.uniform(0.0, g.Lx, N_p)
        y = rng.uniform(0.0, g.Ly, N_p)
        w = rng.uniform(-1.0, -0.5, N_p)
        sigma = sigma_dx * g.dx
        rho = deposit_gaussian_2d_aniso(x, y, w, sigma, sigma, g.Nx, g.Ny, g.dx, g.dy)
        result = check_charge_conservation(rho, np.sum(w), g.dx, g.dy, tol=CHARGE_TOL)
        assert result["passes"], f"Charge not conserved at sigma={sigma_dx}*dx: {result}"


class TestGaussianSelfForce:
    @pytest.mark.parametrize("sigma_dx", [0.5, 1.0, 1.5, 2.0, 3.0])
    def test_isotropic_self_force(self, sigma_dx):
        g = Grid2D(Nx=64, Ny=64, Lx=10.0, Ly=10.0)
        sigma = sigma_dx * g.dx
        deposit, gather = make_gaussian_methods(sigma, sigma)
        result = test_self_force(deposit, gather, g)
        assert result["passes"], f"Self-force too large at sigma={sigma_dx}*dx: |E|={result['magnitude']:.2e}"

    @pytest.mark.parametrize("sx,sy", [(0.5, 1.0), (1.0, 2.0), (3.0, 1.0)])
    def test_anisotropic_self_force(self, sx, sy):
        g = Grid2D(Nx=64, Ny=64, Lx=10.0, Ly=10.0)
        sigma_x = sx * g.dx
        sigma_y = sy * g.dy
        deposit, gather = make_gaussian_methods(sigma_x, sigma_y)
        result = test_self_force(deposit, gather, g)
        assert result["passes"], f"Anisotropic self-force too large at sx={sx}, sy={sy}: {result['magnitude']:.2e}"


class TestSigmaSelection:
    def test_spectral_preservation_rule(self):
        sigma_cells = sigma_max_matched(N=128, m_max=4, A_min=0.85)
        k_phys = 2.0 * np.pi * 4.0 / 128.0
        T = np.exp(-0.5 * k_phys**2 * sigma_cells**2)
        assert abs(T**2 - 0.85) < 1.0e-10


class TestTransferFunctionEmpirical:
    def test_uniform_deposit_low_noise(self):
        g = Grid2D(Nx=32, Ny=32, Lx=10.0, Ly=10.0)
        sigma = 1.5 * g.dx
        deposit, _ = make_gaussian_methods(sigma, sigma)
        result = measure_transfer_function_deposit(deposit, g, Nx_marker=64, Ny_marker=64)
        assert result["rms_nonzero_modes"] < 1.0e-3

    def test_selected_modes_match_analytic(self):
        g = Grid2D(Nx=32, Ny=32, Lx=10.0, Ly=10.0)
        sigma = 1.0 * g.dx
        deposit, _ = make_gaussian_methods(sigma, sigma)
        rows = compare_mode_transfer_to_analytical(
            deposit,
            g,
            sigma_x=sigma,
            sigma_y=sigma,
            modes=[(1, 0), (2, 0), (0, 1), (1, 1)],
        )
        for row in rows:
            assert row["relative_error"] < 0.05, row


class TestLandau2DGaussian:
    @pytest.mark.slow
    def test_landau_with_optimal_sigma(self):
        kx_mode = 1
        Lx = landau_box_length(kx_mode)
        Ly = 8.0
        v_th = thermal_velocity()
        Nx, Ny = 32, 8
        g = Grid2D(Nx=Nx, Ny=Ny, Lx=Lx, Ly=Ly)
        sigma_x_cells, _ = select_sigma_anisotropic_2d(Nx=Nx, Ny=Ny, mx_max=1, my_max=1, A_min=0.85)
        sigma_x = sigma_x_cells * g.dx
        sigma_y = 3.0 * g.dy
        deposit, gather = make_gaussian_methods(sigma_x, sigma_y)

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
            deposit,
            gather,
            diag_stride=2,
            tracked_modes=[(kx_mode, 0)],
        )
        amps = np.abs(np.asarray(history.mode_amplitudes_Ex[(kx_mode, 0)], dtype=np.complex128))
        times = np.asarray(history.times, dtype=np.float64)
        gamma_measured = fit_mode_gamma(times, amps, 2.0, 10.0, amp_floor=1.0e-8)
        gamma_theory = landau_gamma_theory(k_lambdaD=0.4, v_thermal=v_th)
        rel_error = abs(gamma_measured - gamma_theory) / abs(gamma_theory)
        assert rel_error < 0.25, (
            f"Landau gamma off with Gaussian sigma={sigma_x:.3f}: "
            f"measured={gamma_measured:.4f} vs theory={gamma_theory:.4f}"
        )
