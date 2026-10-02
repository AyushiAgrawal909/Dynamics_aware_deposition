import numpy as np

from pic_kde_2d.benchmarks.heterogeneous import (
    PeriodicFloorVonMisesProfile,
    PeriodicVonMisesProfile,
    importance_sampling_noise_profile,
    init_importance_sampled_landau_2d,
    init_profiled_plasma_2d,
    ion_background_on_grid,
)
from pic_kde_2d.core.grid2d import Grid2D
from pic_kde_2d.core.immobile_ions import (
    ImmobileIonBackground,
    neutrality_diagnostics,
    with_immobile_ions,
)
from pic_kde_2d.methods.cic import deposit_cic_2d
from pic_kde_2d.methods.spectral_W_wrappers import (
    make_binned_adaptive_spectral_W_methods,
    make_spectral_W_gaussian_methods,
)


def test_von_mises_profile_is_normalized_and_has_requested_dynamic_range():
    profile = PeriodicVonMisesProfile(Lx=10.0, kappa=np.log(100.0) / 2.0)
    x = np.linspace(0.0, 10.0, 100001)
    assert abs(np.trapezoid(profile.pdf(x), x) - 1.0) < 1.0e-9
    assert abs(profile.dynamic_range - 100.0) < 1.0e-12


def test_importance_weights_reconstruct_total_charge():
    particles, _ = init_importance_sampled_landau_2d(
        Nx_marker=256,
        Ny_marker=8,
        Lx=10.0,
        Ly=4.0,
        v_thermal=1.0,
        alpha=0.01,
        kx_mode=1,
        marker_kappa=np.log(100.0) / 2.0,
        rng=np.random.default_rng(3),
    )
    assert abs(np.sum(particles.w) + 40.0) / 40.0 < 3.0e-4


def test_randomized_stratified_positions_are_reproducible_and_seed_dependent():
    kwargs = {
        "Nx_marker": 128,
        "Ny_marker": 4,
        "Lx": 10.0,
        "Ly": 4.0,
        "v_thermal": 1.0,
        "alpha": 0.01,
        "kx_mode": 1,
        "marker_kappa": np.log(100.0) / 2.0,
        "randomized_positions": True,
    }
    first, _ = init_importance_sampled_landau_2d(
        rng=np.random.default_rng(3),
        **kwargs,
    )
    repeated, _ = init_importance_sampled_landau_2d(
        rng=np.random.default_rng(3),
        **kwargs,
    )
    second, _ = init_importance_sampled_landau_2d(
        rng=np.random.default_rng(4),
        **kwargs,
    )
    np.testing.assert_array_equal(first.x, repeated.x)
    assert not np.array_equal(first.x, second.x)
    assert abs(np.sum(first.w) + 40.0) / 40.0 < 3.0e-4


def test_noise_profile_distinguishes_weight_squared_from_aggregate_variance():
    profile = PeriodicVonMisesProfile(Lx=10.0, kappa=np.log(100.0) / 2.0)
    values = importance_sampling_noise_profile(
        np.array([0.0, 5.0]), profile, alpha=0.0, kx_mode=1
    )
    squared_ratio = values["squared_weight_amplification"][1]
    aggregate_ratio = values["aggregate_variance_amplification"][1]
    np.testing.assert_allclose(squared_ratio, 10000.0, rtol=1.0e-12)
    np.testing.assert_allclose(aggregate_ratio, 100.0, rtol=1.0e-12)


def test_floor_von_mises_profile_has_mean_one_and_requested_ratio():
    profile = PeriodicFloorVonMisesProfile(
        Lx=16.0,
        peak_to_floor=100.0,
        kappa=4.0,
    )
    x = np.linspace(0.0, profile.Lx, 200001)
    density = profile.density(x)
    assert abs(np.trapezoid(density, x) / profile.Lx - 1.0) < 1.0e-9
    np.testing.assert_allclose(
        np.max(density) / np.min(density),
        100.0,
        rtol=1.0e-10,
    )


def test_profiled_plasma_and_explicit_ions_are_charge_neutral():
    grid = Grid2D(Nx=32, Ny=8, Lx=16.0, Ly=4.0)
    profile = PeriodicFloorVonMisesProfile(grid.Lx, 10.0)
    particles = init_profiled_plasma_2d(
        128,
        16,
        grid.Lx,
        grid.Ly,
        1.0,
        profile,
        np.random.default_rng(3),
        randomized_positions=False,
    )
    ion = ImmobileIonBackground(
        ion_background_on_grid(profile, grid.x, grid.Ny),
        grid,
    )

    def electron_deposit(particles_here, grid_here):
        return deposit_cic_2d(
            particles_here.x,
            particles_here.y,
            particles_here.w,
            grid_here.Nx,
            grid_here.Ny,
            grid_here.dx,
            grid_here.dy,
        )

    total = with_immobile_ions(electron_deposit, ion)(particles, grid)
    diagnostics = neutrality_diagnostics(total, grid)
    assert abs(particles.total_charge() + ion.total_charge) < 1.0e-12
    assert diagnostics["relative_integrated_charge"] < 1.0e-13


def test_binned_uniform_width_matches_global_matched_chain():
    grid = Grid2D(Nx=16, Ny=8, Lx=8.0, Ly=4.0)
    profile = PeriodicFloorVonMisesProfile(grid.Lx, 10.0)
    particles = init_profiled_plasma_2d(
        32,
        16,
        grid.Lx,
        grid.Ly,
        1.0,
        profile,
        np.random.default_rng(5),
    )
    sigma_x = np.full(particles.n_particles, 0.7 * grid.dx)
    sigma_y = np.full(particles.n_particles, 0.7 * grid.dy)
    global_deposit, global_gather = make_spectral_W_gaussian_methods(
        grid,
        sigma_x[0],
        sigma_y[0],
    )
    local_deposit, local_gather = make_binned_adaptive_spectral_W_methods(
        grid,
        sigma_x,
        sigma_y,
        particles.w,
        n_bins=8,
    )
    np.testing.assert_allclose(
        local_deposit(particles, grid),
        global_deposit(particles, grid),
        rtol=0.0,
        atol=1.0e-14,
    )
    rng = np.random.default_rng(7)
    Ex = rng.normal(size=(grid.Nx, grid.Ny))
    Ey = rng.normal(size=(grid.Nx, grid.Ny))
    local_fields = local_gather(Ex, Ey, particles, grid)
    global_fields = global_gather(Ex, Ey, particles, grid)
    np.testing.assert_allclose(local_fields[0], global_fields[0], atol=1.0e-14)
    np.testing.assert_allclose(local_fields[1], global_fields[1], atol=1.0e-14)
