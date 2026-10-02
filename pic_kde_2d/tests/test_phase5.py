"""
tests/test_phase5.py
====================

Phase 5 structural acceptance tests for delta-f control variates.
"""

from __future__ import annotations

import numpy as np

from pic_kde_2d.benchmarks.theory import landau_gamma_theory, langmuir_omega_r
from pic_kde_2d.conventions import Q_OVER_M, landau_box_length, marker_grid_shape, thermal_velocity
from pic_kde_2d.core.grid2d import Grid2D
from pic_kde_2d.core.particles2d import init_landau_2d
from pic_kde_2d.core.pic_loop_delta_f import (
    pic_step_delta_f,
    pic_step_linearized_delta_f,
    run_simulation_delta_f,
)
from pic_kde_2d.methods.delta_f import (
    activate_full_f_fallback,
    background_velocity_score,
    check_delta_f_validity,
    delta_weight_rhs,
    init_delta_f_from_particles,
    make_delta_f_methods,
)
from pic_kde_2d.methods.spectral_W import make_W_gaussian


def make_uniform_delta_f(alpha: float = 0.01):
    Lx = landau_box_length(1)
    grid = Grid2D(Nx=32, Ny=8, Lx=Lx, Ly=8.0)
    nx_marker, ny_marker = marker_grid_shape(grid.Nx, grid.Ny, 16)
    particles = init_landau_2d(
        Nx_marker=nx_marker,
        Ny_marker=ny_marker,
        Lx=grid.Lx,
        Ly=grid.Ly,
        v_thermal=thermal_velocity(),
        alpha=0.0,
        kx_mode=1,
        velocity_perturbation=False,
        rng=np.random.default_rng(42),
    )
    kx = 2.0 * np.pi / grid.Lx
    delta = init_delta_f_from_particles(
        particles,
        f0_v_thermal=thermal_velocity(),
        f0_density=1.0,
        alpha=alpha,
        kx_phys=kx,
    )
    return delta, grid


class TestDeltaFInitialization:
    def test_zero_alpha_has_zero_delta_weight(self):
        particles, _ = make_uniform_delta_f(alpha=0.0)
        assert np.all(particles.w_delta == 0.0)

    def test_cosine_delta_charge_is_neutral(self):
        particles, _ = make_uniform_delta_f(alpha=0.01)
        assert abs(np.sum(particles.w_delta)) < 1.0e-12

    def test_eigenmode_component_remains_charge_neutral(self):
        particles, grid = make_uniform_delta_f(alpha=0.0)
        base = init_landau_2d(
            Nx_marker=particles.x.size // 8,
            Ny_marker=8,
            Lx=grid.Lx,
            Ly=grid.Ly,
            v_thermal=thermal_velocity(),
            alpha=0.0,
            kx_mode=1,
            velocity_perturbation=False,
            rng=np.random.default_rng(12),
        )
        kx = 2.0 * np.pi / grid.Lx
        delta = init_delta_f_from_particles(
            base,
            f0_v_thermal=thermal_velocity(),
            f0_density=1.0,
            alpha=0.001,
            kx_phys=kx,
        )
        assert abs(np.sum(delta.w_delta)) < 1.0e-12


class TestDeltaFEvolution:
    def test_step_preserves_delta_charge_neutrality(self):
        particles, grid = make_uniform_delta_f(alpha=0.01)
        deposit, gather = make_delta_f_methods()
        pic_step_delta_f(particles, grid, 0.05, deposit, gather)
        assert abs(np.sum(particles.w_delta)) < 1.0e-12

    def test_validity_flags_large_weights(self):
        particles, _ = make_uniform_delta_f(alpha=0.001)
        assert check_delta_f_validity(particles)["valid"]
        particles.w_delta[0] = 100.0 * np.mean(np.abs(particles.w_full))
        assert not check_delta_f_validity(particles)["valid"]

    def test_validity_ratio_is_the_marker_maximum_not_an_rms_norm(self):
        particles, _ = make_uniform_delta_f(alpha=0.0)
        particles.w_full.fill(-2.0)
        particles.w_delta.fill(0.0)
        particles.w_delta[0] = 1.0

        result = check_delta_f_validity(particles, threshold=0.5)

        assert result["ratio"] == 0.5
        assert result["valid"] is False
        rms_ratio = np.sqrt(np.mean((particles.w_delta / particles.w_full) ** 2))
        assert rms_ratio < result["ratio"]

    def test_validity_ratio_is_invariant_to_equal_charge_weight_scaling(self):
        particles, _ = make_uniform_delta_f(alpha=0.01)
        before = check_delta_f_validity(particles)["ratio"]
        particles.w_full *= 7.0
        particles.w_delta *= 7.0
        after = check_delta_f_validity(particles)["ratio"]
        np.testing.assert_allclose(after, before, rtol=0.0, atol=2.0e-16)

    def test_full_f_fallback_switches_deposit_weights(self):
        particles, grid = make_uniform_delta_f(alpha=0.01)
        deposit, _ = make_delta_f_methods()
        rho_delta = deposit(particles, grid)
        activate_full_f_fallback(particles)
        rho_full = deposit(particles, grid)
        assert particles.use_full_f
        assert np.max(np.abs(rho_full - rho_delta)) > 0.0

    def test_linearized_step_free_streams_without_accelerating_markers(self):
        particles, grid = make_uniform_delta_f(alpha=0.01)
        deposit, gather = make_delta_f_methods()
        x_initial = particles.x.copy()
        y_initial = particles.y.copy()
        vx_initial = particles.vx.copy()
        vy_initial = particles.vy.copy()
        dt = 0.05

        pic_step_linearized_delta_f(particles, grid, dt, deposit, gather)

        np.testing.assert_array_equal(particles.vx, vx_initial)
        np.testing.assert_array_equal(particles.vy, vy_initial)
        np.testing.assert_allclose(
            particles.x,
            (x_initial + dt * vx_initial) % particles.Lx,
            atol=1.0e-14,
        )
        np.testing.assert_allclose(
            particles.y,
            (y_initial + dt * vy_initial) % particles.Ly,
            atol=1.0e-14,
        )

    def test_maxwellian_weight_rhs_matches_closed_form(self):
        particles, _ = make_uniform_delta_f(alpha=0.01)
        Ex = np.linspace(-0.2, 0.3, particles.n_particles)
        Ey = np.linspace(0.1, -0.4, particles.n_particles)
        rhs = delta_weight_rhs(particles, Ex, Ey, Q_OVER_M)
        expected = (
            Q_OVER_M
            * (Ex * particles.vx + Ey * particles.vy)
            / particles.f0_v_thermal**2
            * particles.w_full
        )
        np.testing.assert_allclose(rhs, expected, atol=1.0e-15)

    def test_two_stream_score_is_stable_and_matches_mixture_derivative(self):
        particles, _ = make_uniform_delta_f(alpha=0.0)
        particles.background_kind = "two_stream"
        particles.beam_velocity = 0.8
        particles.f0_v_thermal = 0.2
        particles.vx = np.asarray([-100.0, -0.6, 0.1, 0.8, 100.0])
        particles.vy = np.zeros_like(particles.vx)
        particles.w_full = np.full_like(particles.vx, -1.0)
        particles.w_delta = np.zeros_like(particles.vx)

        score_x, score_y = background_velocity_score(particles)
        assert np.all(np.isfinite(score_x))
        assert np.all(score_y == 0.0)

        v = particles.vx[2]
        beam = particles.beam_velocity
        vth = particles.f0_v_thermal
        step = 1.0e-6

        def log_mixture(value):
            plus = -0.5 * ((value - beam) / vth) ** 2
            minus = -0.5 * ((value + beam) / vth) ** 2
            return np.logaddexp(plus, minus)

        numerical = (log_mixture(v + step) - log_mixture(v - step)) / (2.0 * step)
        assert abs(score_x[2] - numerical) < 1.0e-7


class TestDeltaFLandau:
    @staticmethod
    def run_eigenmode(alpha: float):
        Lx = landau_box_length(1)
        grid = Grid2D(Nx=64, Ny=16, Lx=Lx, Ly=8.0)
        nx_marker, ny_marker = marker_grid_shape(grid.Nx, grid.Ny, 32)
        base = init_landau_2d(
            Nx_marker=nx_marker,
            Ny_marker=ny_marker,
            Lx=grid.Lx,
            Ly=grid.Ly,
            v_thermal=thermal_velocity(),
            alpha=0.0,
            kx_mode=1,
            velocity_perturbation=False,
            rng=np.random.default_rng(7),
        )
        delta = init_delta_f_from_particles(
            base,
            f0_v_thermal=thermal_velocity(),
            f0_density=1.0,
            alpha=alpha,
            kx_phys=2.0 * np.pi / Lx,
            omega_real=langmuir_omega_r(0.4),
            gamma=landau_gamma_theory(0.4),
        )
        W = make_W_gaussian(grid, 1.5 * grid.dx, 1.5 * grid.dx)
        deposit, gather = make_delta_f_methods(W=W)
        return run_simulation_delta_f(
            delta,
            grid,
            dt=0.05,
            n_steps=400,
            deposit_fn=deposit,
            gather_fn=gather,
            diag_stride=4,
            tracked_modes=[(1, 0)],
        )

    def test_landau_gamma_small_perturbation(self):
        history = self.run_eigenmode(0.001)
        times = np.asarray(history.times)
        amps = np.abs(np.asarray(history.mode_amplitudes_Ex[(1, 0)]))
        mask = (times >= 5.0) & (times <= 20.0) & (amps > 1.0e-12)
        gamma = float(np.polyfit(times[mask], np.log(amps[mask]), 1)[0])
        theory = landau_gamma_theory(0.4)
        assert abs(gamma - theory) / abs(theory) < 0.20
