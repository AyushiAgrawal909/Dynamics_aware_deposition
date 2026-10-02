import numpy as np
import pytest

from attractor_analysis import forecast as Fc
from attractor_analysis import scaling as Sc


def _snapshots(F, alpha, beta, times, noise=0.02, seed=0):
    eps = np.geomspace(1.0, 2000.0, 50)
    rng = np.random.default_rng(seed)
    N = np.array([t**alpha * F(t**beta * eps) for t in times])
    N *= np.exp(noise * rng.standard_normal(N.shape))
    sig = np.full(N.shape, noise)
    sup = N > 1e-12 * N.max()
    return Sc.SnapshotSet(np.asarray(times, float), eps, N, sig, sup, np.ones(eps.size, bool))


def test_curved_moving_cutoff_identified():
    F = lambda x: x**-2.0 * np.exp(-x / 50.0 - 2.0 / x)
    ss = _snapshots(F, -0.25, -0.5, [1.0, 1.5, 2.0, 2.5, 3.0])
    fit = Sc.fit_scaling(ss, n_grid_ab=21)
    assert fit["status"] == Sc.IDENTIFIED
    assert fit["alpha"] == pytest.approx(-0.25, abs=0.05)
    assert fit["beta"] == pytest.approx(-0.5, abs=0.05)


def test_stationary_powerlaw_is_degenerate():
    F = lambda x: x**-2.0
    ss = _snapshots(F, 0.0, 0.0, [1.0, 1.5, 2.0, 2.5, 3.0], seed=1)
    fit = Sc.fit_scaling(ss, n_grid_ab=21)
    assert fit["status"] == Sc.DEGENERATE
    comb = fit["identified_combination"]
    assert abs(comb["value"]) < 0.05


def test_forecast_masks_unsupported_and_is_exact_for_selfsimilar():
    F = lambda x: x**-2.0 * np.exp(-x / 50.0 - 2.0 / x)
    eps = np.geomspace(1.0, 2000.0, 50)
    N1 = F(eps)
    sup = np.ones(eps.size, bool)
    pred = Fc.scaling_forecast(eps, N1, sup, t_ref=1.0, t=4.0, alpha=-0.25, beta=-0.5, t0=0.0)
    truth = 4.0**-0.25 * F(4.0**-0.5 * eps)
    m = np.isfinite(pred)
    assert np.allclose(pred[m], truth[m], rtol=2e-2)
    assert (~m).any()  # energies needing template beyond eps[-1]... or below eps[0] are masked


def test_oracle_excluded_from_ranking():
    errs = {"scaling": {"rms_ln": 0.2, "coverage": 1.0}, "ORACLE_maxwellian": {"rms_ln": 0.01, "coverage": 1.0},
            "stationary": {"rms_ln": 0.3, "coverage": 1.0}}
    assert Fc.rank_forecasts(errs) == ["scaling", "stationary"]


def test_sector_scaling_powers():
    F = lambda x: x**-2.0 * np.exp(-x / 50.0 - 2.0 / x)
    edges = np.geomspace(0.5, 1e5, 400)
    mid = np.sqrt(edges[1:] * edges[:-1])
    times = np.array([1.0, 2.0, 3.0, 4.0])
    N_t = [t**-0.25 * F(t**-0.5 * mid) for t in times]
    r = Sc.sector_scaling_check(times, edges, N_t, 5.0, 100.0, -0.25, -0.5, 0.0)
    assert r["meas_number_power"] == pytest.approx(r["pred_number_power"], abs=0.02)
    assert r["meas_energy_power"] == pytest.approx(r["pred_energy_power"], abs=0.02)
