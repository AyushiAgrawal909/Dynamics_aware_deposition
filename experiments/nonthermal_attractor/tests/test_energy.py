import numpy as np
import pytest

from attractor_analysis import energy as E


def test_moments_match_direct_sums_and_overflow_not_clipped():
    rng = np.random.default_rng(0)
    v = rng.normal(size=200_000) * 1.3
    w = rng.uniform(0.5, 1.5, size=v.size)
    eps = E.particle_energy(v, mass=2.0)
    edges = E.common_edges(0.1, 10.0, n_log=20)  # deliberately low ceiling
    h = E.energy_histogram(eps, w, edges)
    mom = E.histogram_moments(h)
    assert mom["number"] == pytest.approx(w.sum(), rel=1e-12)
    assert mom["energy_exact"] == pytest.approx((w * eps).sum(), rel=1e-12)
    assert h.overflow_weight == pytest.approx(w[eps >= 10.0].sum(), rel=1e-12)
    assert h.overflow_weight > 0


def test_finite_bin_moment_error_is_second_order():
    T = 1.0
    errs = []
    for nb in (40, 80, 160):
        edges = np.linspace(0.0, 30.0, nb + 1)
        number, energy = E.maxwellian_1v_bin_content(edges, T)
        h = E.EnergyHistogram(edges, number / np.diff(edges), number, energy, number * 1e9, 0.0, 0.0)
        m = E.histogram_moments(h)
        errs.append(abs(m["finite_bin_energy_error"]))
    # the eps^-1/2 singularity in the first bin makes the leading error
    # O(d eps^{3/2}); away from it the midpoint rule is O(d eps^2)
    r1, r2 = errs[0] / errs[1], errs[1] / errs[2]
    assert r1 > 2.5 and r2 > 2.5


def test_maxwellian_shape_recovered():
    rng = np.random.default_rng(1)
    T, n = 0.7, 400_000
    v = rng.normal(size=n) * np.sqrt(T)
    eps = E.particle_energy(v)
    edges = E.common_edges(0.02, 15.0, n_log=30, n_lin=4)
    h = E.energy_histogram(eps, np.full(n, 1.0 / n), edges)
    expect, _ = E.maxwellian_1v_bin_content(edges, T, 1.0)
    sig = np.sqrt(expect / n)
    m = expect * n > 50
    z = (h.weight[m] - expect[m]) / sig[m]
    assert np.mean(z**2) < 2.0
    # including the integrable eps^-1/2 singular lowest bin
    assert abs(z[0]) < 4.0


def test_log_density_is_a_different_quantity():
    edges = E.common_edges(0.1, 10.0, n_log=10)
    h = E.energy_histogram(np.array([0.5, 1.0, 2.0]), np.ones(3), edges)
    assert not np.allclose(h.dN_dlneps(), h.N)


def test_tail_threshold_must_be_a_frozen_edge():
    edges = E.common_edges(0.1, 10.0, n_log=10)
    h = E.energy_histogram(np.array([0.5, 1.0, 2.0]), np.ones(3), edges)
    with pytest.raises(ValueError):
        E.tail_moments(h, 1.2345)
    tm = E.tail_moments(h, float(edges[-3]))
    assert tm["Q_N"] >= 0


def test_quantile_feature_and_support():
    rng = np.random.default_rng(2)
    n = 100_000
    eps = rng.exponential(size=n)
    edges = E.common_edges(0.05, 50.0, n_log=60)
    h = E.energy_histogram(eps, np.full(n, 1.0 / n), edges)
    eq = E.quantile_feature(h, 0.01)
    assert eq == pytest.approx(np.log(100.0), rel=0.03)
    assert np.isnan(E.quantile_feature(h, 1e-5))  # 1 particle above: unsupported


def test_core_tail_excess_detects_powerlaw_not_maxwellian():
    rng = np.random.default_rng(3)
    n = 400_000
    core = E.particle_energy(rng.normal(size=n))
    edges = E.common_edges(0.02, 1e4, n_log=60)
    h = E.energy_histogram(core, np.full(n, 1.0 / n), edges)
    assert E.core_tail_excess(h)["excess"] < 3
    tail = 2.0 / rng.uniform(size=n // 10)  # eps^-2 above 2
    h2 = E.energy_histogram(np.concatenate([core, tail]), np.full(n + n // 10, 1.0), edges)
    assert E.core_tail_excess(h2)["excess"] > 100
