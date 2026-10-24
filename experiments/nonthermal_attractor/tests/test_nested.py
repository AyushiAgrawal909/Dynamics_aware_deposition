import numpy as np
import pytest

from attractor_analysis import nested as Nd


def test_prefixes_are_stratified_in_x_and_quantile():
    N = 1 << 14
    u, q = Nd.halton_2d(N, seed=5)
    rng = np.random.default_rng(0)
    for l in range(5):
        m = N >> l
        # x: every power-of-two prefix sits on a shifted lattice -> KS <= 1/m
        assert Nd.ks_uniform(u[:m]) <= 1.0 / m + 1e-12
        # velocity quantile: low discrepancy, far below iid sampling (~1/sqrt(m))
        ks_q = Nd.ks_uniform(q[:m])
        ks_iid = Nd.ks_uniform(rng.random(m))
        assert ks_q < 10 * np.log(m) / m
        assert ks_q < 0.2 * ks_iid or ks_q < 5.0 / m


def test_nested_subsets_preserve_totals_and_moments():
    vb, dvb = 1.0, 0.2
    N = 1 << 14
    x, v, w = Nd.quiet_load_1d1v(N, 50.0, Nd.waterbag_beams_inv_cdf(vb, dvb), seed=3)
    target_m2 = vb**2 + dvb**2 / 12
    for l, sl in enumerate(Nd.nested_load_indices(N, 5)):
        xs, vs, ws = Nd.subset(x, v, w, l)
        assert np.isclose(ws.sum(), w.sum())
        assert abs(np.average(vs, weights=ws)) < 5e-3
        assert np.average(vs**2, weights=ws) == pytest.approx(target_m2, rel=5e-3)
        assert np.all(np.abs(np.abs(vs) - vb) <= dvb / 2 + 1e-12)
        # nested: the level-l subset is a prefix of the level-0 load
        np.testing.assert_array_equal(xs, x[: N >> l])


def test_naive_every_other_subset_would_fail():
    # documents why the Halton prefix construction is needed
    N = 1 << 12
    j = np.arange(N)
    q = Nd.radical_inverse(j, 2)
    sub = q[:: 1 << 3]
    assert sub.max() < 1.0 / (1 << 3) + 1e-12  # collapsed into one corner


def test_gaussian_mixture_second_moment():
    vb, w_ = 1.0, 0.2 / np.sqrt(12)
    x, v, w = Nd.quiet_load_1d1v(1 << 15, 10.0, Nd.gaussian_beams_inv_cdf(vb, w_), seed=1)
    assert np.mean(v**2) == pytest.approx(vb**2 + w_**2, rel=2e-3)
