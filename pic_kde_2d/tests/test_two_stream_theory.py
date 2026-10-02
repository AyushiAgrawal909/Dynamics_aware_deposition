import numpy as np
import pytest

from pic_kde_2d.benchmarks.theory import two_stream_growth


def test_cold_two_stream_exact_maximum():
    v_beam = 0.8
    k_max = np.sqrt(3.0 / 8.0) / v_beam
    expected = 1.0 / (2.0 * np.sqrt(2.0))
    assert two_stream_growth(k_max, v_beam) == pytest.approx(expected, abs=1.0e-15)


def test_cold_two_stream_e2_modes():
    v_beam = 0.8
    Lx = 64.0
    k_m8 = 2.0 * np.pi * 8 / Lx
    k_m13 = 2.0 * np.pi * 13 / Lx
    assert two_stream_growth(k_m8, v_beam) == pytest.approx(
        0.353281859016369,
        abs=1.0e-15,
    )
    assert two_stream_growth(k_m13, v_beam) == 0.0


def test_cold_two_stream_rejects_invalid_beam_speed():
    with pytest.raises(ValueError):
        two_stream_growth(k=1.0, v_beam=0.0)
