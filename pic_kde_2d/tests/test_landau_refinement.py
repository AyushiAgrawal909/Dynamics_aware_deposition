"""Focused checks for the Landau refinement physics and censoring path."""

import numpy as np

from pic_kde_2d.benchmarks.theory import solve_landau_dispersion
from pic_kde_2d.validation.landau_censored_fit import fit_censored_mode


def test_landau_dispersion_anchor():
    omega = solve_landau_dispersion(0.4)
    np.testing.assert_allclose(omega.real, 1.28505697, atol=2.0e-7)
    np.testing.assert_allclose(omega.imag, -0.06612796, atol=2.0e-7)


def test_censored_fit_excludes_floor_and_recurrence():
    times = np.linspace(0.0, 90.0, 901)
    gamma = -0.07
    omega = 1.28
    clean = np.exp((gamma - 1j * omega) * times)
    histories = np.stack([clean, clean * np.exp(1j * 1.0e-3), clean * np.exp(-1j * 1.0e-3)])
    histories[:, times >= 35.0] = 2.0e-2
    histories[:, times >= 75.0] += 5.0e-3 * np.exp(1j * omega * times[times >= 75.0])

    result = fit_censored_mode(
        histories,
        times,
        fit_t_start=5.0,
        fit_t_stop_max=60.0,
        noise_floor_start=70.0,
        noise_floor_stop=90.0,
        noise_factor=3.0,
        persistence_points=5,
        min_r_squared=0.95,
    )

    assert result["floor_censored"]
    assert result["fit_t_stop"] < 35.0
    assert result["quality_pass"]
    np.testing.assert_allclose(result["gamma"], gamma, atol=2.0e-3)
