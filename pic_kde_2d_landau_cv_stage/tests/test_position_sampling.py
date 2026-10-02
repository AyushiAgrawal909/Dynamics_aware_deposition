import numpy as np

from pic_kde_2d.core.particles2d import quiet_start_maxwellian_2d


def _positions(seed, jitter):
    return quiet_start_maxwellian_2d(
        8,
        4,
        2.0,
        1.0,
        1.0,
        np.random.default_rng(seed),
        position_jitter=jitter,
    )[:2]


def test_regular_quiet_start_positions_are_seed_independent():
    x0, y0 = _positions(0, False)
    x1, y1 = _positions(1, False)
    np.testing.assert_array_equal(x0, x1)
    np.testing.assert_array_equal(y0, y1)


def test_stratified_positions_are_reproducible_and_seed_dependent():
    x0, y0 = _positions(0, True)
    x0_repeat, y0_repeat = _positions(0, True)
    x1, y1 = _positions(1, True)
    np.testing.assert_array_equal(x0, x0_repeat)
    np.testing.assert_array_equal(y0, y0_repeat)
    assert not np.array_equal(x0, x1)
    assert not np.array_equal(y0, y1)

    marker_dx = 2.0 / 8
    marker_dy = 1.0 / 4
    base_x, base_y = np.meshgrid(np.arange(8), np.arange(4), indexing="ij")
    assert np.all(x0 >= base_x.ravel() * marker_dx)
    assert np.all(x0 < (base_x.ravel() + 1) * marker_dx)
    assert np.all(y0 >= base_y.ravel() * marker_dy)
    assert np.all(y0 < (base_y.ravel() + 1) * marker_dy)
