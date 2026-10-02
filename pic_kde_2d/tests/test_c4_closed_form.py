import math

import pytest

from pic_kde_2d.benchmarks.theory import (
    matched_cic_force_coupling,
    rate_tolerance_crossing,
    solve_landau_dispersion_coupling,
    warm_two_stream_growth,
)


def test_matched_cic_transfer_anchor():
    assert matched_cic_force_coupling(8, 128, 0.0) == pytest.approx(
        0.9745931220856418, rel=1.0e-14
    )


def test_landau_closed_form_crossing():
    def rate(sigma):
        coupling = matched_cic_force_coupling(1, 64, sigma)
        return solve_landau_dispersion_coupling(0.4, coupling).imag

    crossing = rate_tolerance_crossing(
        rate, tolerance=0.10, sigma_upper=4.0, increasing_magnitude=True
    )
    assert crossing == pytest.approx(2.45344, abs=2.0e-4)


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        (8, 1.06229),
        (12, 0.19872),
    ],
)
def test_warm_two_stream_closed_form_crossing(mode, expected):
    k = 2.0 * math.pi * mode / 64.0

    def rate(sigma):
        coupling = matched_cic_force_coupling(mode, 128, sigma)
        return warm_two_stream_growth(k, 0.8, 0.05, coupling)

    crossing = rate_tolerance_crossing(rate, tolerance=0.10, sigma_upper=2.0)
    assert crossing == pytest.approx(expected, abs=2.0e-4)


def test_warm_two_stream_zero_coupling_has_zero_growth():
    k = 2.0 * math.pi * 8.0 / 64.0
    assert warm_two_stream_growth(k, 0.8, 0.05, 0.0) == 0.0


def test_m8_bias_calibration_predicts_tighter_m12_prior():
    def rate(mode, sigma):
        k = 2.0 * math.pi * mode / 64.0
        coupling = matched_cic_force_coupling(mode, 128, sigma)
        return warm_two_stream_growth(k, 0.8, 0.05, coupling)

    m8_ratio_at_measured_crossing = rate(8, 0.99) / rate(8, 0.0)
    sigma_divisor = math.sqrt(0.10 / (1.0 - m8_ratio_at_measured_crossing))
    assert sigma_divisor == pytest.approx(1.0773, abs=1.0e-4)
    assert 0.19872036040909125 / sigma_divisor == pytest.approx(0.1845, abs=1.0e-4)
