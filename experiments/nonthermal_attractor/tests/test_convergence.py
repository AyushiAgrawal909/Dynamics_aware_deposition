import numpy as np

from attractor_analysis import convergence as C


def _traj(levels, n_seeds, times, bias_fn, noise=0.05, seed=0):
    rng = np.random.default_rng(seed)
    base = 10.0 + 5.0 * np.sqrt(times)  # a physically moving feature
    out = {}
    # nested noise: the level-l estimate shares the finer level's noise
    eps0 = rng.standard_normal((n_seeds, len(times)))
    for l in range(levels):
        extra = rng.standard_normal((n_seeds, len(times)))
        out[l] = base + bias_fn(l) + noise * (2 ** (l / 2)) * (eps0 + 0.5 * extra)
    return out


def test_converging_feature_passes():
    t = np.linspace(1, 6, 11)
    F = _traj(5, 5, t, lambda l: 0.01 * 2**l)
    g = C.convergence_gate(t, F, (1, 6), tau=0.1)
    assert g["verdict"] == C.PASS


def test_regulator_offset_fails():
    t = np.linspace(1, 6, 11)
    # stationary feature whose value tracks ln N: offset per level never shrinks
    rng = np.random.default_rng(1)
    F = {l: 30.0 - 3.5 * l + 0.05 * rng.standard_normal((5, len(t))) for l in range(5)}
    g = C.convergence_gate(t, F, (1, 6), tau=0.1)
    assert g["verdict"] == C.FAIL


def test_too_few_levels_inconclusive():
    t = np.linspace(1, 6, 11)
    F = _traj(2, 5, t, lambda l: 0.0)
    assert C.convergence_gate(t, F, (1, 6))["verdict"] == C.INCONCLUSIVE


def test_onset_alignment_and_lnN_preference():
    t = np.linspace(0, 8, 81)
    ppc = {l: 8192 / 2**l for l in range(5)}
    rng = np.random.default_rng(2)
    F, ons = {}, {}
    for l in range(5):
        t0 = 1.0 + 0.3 * np.log(ppc[l] / 512.0) + 0.01 * rng.standard_normal(5)
        ons[l] = t0
        F[l] = np.array([np.where(t > a, 10 * np.sqrt(np.clip(t - a, 0, None)), 0.0) for a in t0])
        F[l] += 0.01 * 2 ** (l / 2) * rng.standard_normal(F[l].shape)
    g = C.convergence_gate(t, F, (2.5, 8.0), tau=0.1, onset_by_level=ons, regulator_by_level=ppc)
    assert g["verdict"] == C.PASS_ALIGNED
    assert g["onset_scaling"]["preferred"] == "lnN"


def test_onset_time_interpolates():
    t = np.array([0.0, 1.0, 2.0])
    assert C.onset_time(t, np.array([0.0, 1.0, 2.0]), 1.5) == 1.5
    assert np.isnan(C.onset_time(t, np.zeros(3), 1.0))
