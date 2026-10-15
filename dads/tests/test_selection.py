"""Root finding, continuation, objectives and the selection rule."""

import math

import numpy as np
import pytest
from scipy.integrate import quad

from dads import (Binomial, Gaussian, Grid, Kernel, ModeDispersion, ModelObjective, NoFilter,
                  Plasma, ProtectedMode, Selector, SnapshotObjective, Species, deposit)
from dads import roots as rt
from dads.config import ConfigError, build_selector
from dads.selection import unstable_modes


@pytest.fixture(scope="module")
def paper():
    grid = Grid([128, 16], [64.0, 16.0], shape="cic")
    plasma = Plasma([Species(-1, 1, 0.5, [0.8, 0], 0.05, "b+"),
                     Species(-1, 1, 0.5, [-0.8, 0], 0.05, "b-")]).resolve(2)
    return grid, plasma


@pytest.fixture(scope="module")
def paper_selection(paper):
    grid, plasma = paper
    obj = ModelObjective(grid, plasma, ppc=32, signal=[dict(mode=[12, 0], amplitude=1e-3)])
    return Selector(grid, plasma, [ProtectedMode((12, 1))], obj, tolerance=0.05, n_alias=8,
                    gaussian=dict(grid=31)).run()


# ------------------------------------------------------------------- roots
def test_marginal_coupling_matches_penrose(paper):
    """Principal-harmonic a_c from continuation equals the Penrose threshold."""
    grid, plasma = paper
    d = ModeDispersion(grid, plasma, (12, 1), n_alias=0)
    w0 = rt.select_branch(rt.find_roots(d))
    br = rt.continue_branch(d, w0, 1.0, 0.5)
    assert br.status == "marginal"
    k = d.k
    kn = np.linalg.norm(k)
    ub, vt = 0.8 * k[0] / kn, 0.05
    F = lambda u: 0.5 * (np.exp(-0.5 * ((u - ub) / vt) ** 2) + np.exp(-0.5 * ((u + ub) / vt) ** 2)) \
        / (math.sqrt(2 * math.pi) * vt)
    P = 2 * quad(lambda u: (F(u) - F(0.0)) / u**2, 1e-12, ub + 12 * vt, points=[ub], limit=400)[0]
    a_c = kn**2 / (P * float(grid.shape_transform(k)) ** 2)
    assert abs(br.a_marginal - a_c) < 2e-7


def test_square_root_law_near_marginality(paper):
    grid, plasma = paper
    d = ModeDispersion(grid, plasma, (12, 1), n_alias=0)
    w0 = rt.select_branch(rt.find_roots(d))
    br = rt.continue_branch(d, w0, 1.0, 0.5)
    ac = br.a_marginal
    g1 = rt.solve_at(d, br, ac + 1e-4).imag
    g2 = rt.solve_at(d, br, ac + 4e-4).imag
    assert abs(g2 / g1 - 2.0) < 0.01          # gamma ~ sqrt(a - a_c)


def test_most_unstable_verified_by_argument_principle(paper):
    grid, plasma = paper
    d = ModeDispersion(grid, plasma, (12, 1), n_alias=4)
    roots = rt.find_roots(d)
    assert rt.count_unstable(d, 1.0, im_min=1e-3) == sum(r.imag > 1e-3 for r in roots) == 1
    assert rt.verify_most_unstable(d, 1.0, roots[0].imag) is True


def test_landau_ratios_and_two_sided_tolerance():
    grid = Grid([64, 16], [2 * math.pi / 0.4, 8.0])
    plasma = Plasma([Species(-1, 1, 1, 0.0, 1.0)]).resolve(2)
    obj = ModelObjective(grid, plasma, ppc=1024, signal=[dict(mode=[1, 0], amplitude=0.01)])
    sel = Selector(grid, plasma, [ProtectedMode((1, 0), branch="least_damped", sign=1)], obj,
                   tolerance=0.05, n_alias=0, gaussian=dict(enabled=False))
    sel.analyse_modes()
    ma = sel.analyses[grid.order][0]
    assert ma.omega0.real > 0 and ma.gamma0 < 0
    st, w, R = ma.at(0.9904080138578515)
    assert abs(R - 1.016348411037295) < 1e-8
    a_lo = ma.lower_threshold(0.05)
    assert abs(ma.at(a_lo)[2] - 1.05) < 1e-8      # damping grows: upper side of the band
    cand = sel.evaluate(Kernel(grid.order, Gaussian((7.0, 0.0)), 2))
    assert cand.R[0] > 1.05 and not cand.admissible


# --------------------------------------------------------------- selection
def test_paper_selection_reproduced(paper_selection):
    res = paper_selection
    bins = [c for c in res.candidates if c.family == "binomial"]
    assert sum(c.status[0] == "growing" for c in bins) == 17
    assert sum(c.status[0] == "marginal" for c in bins) == 13
    assert sum(c.admissible for c in bins) == 11
    f = res.best.kernel.filter
    assert f.family == "binomial" and f.p == 4 and f.n == (8,)
    raw8 = next(c for c in bins if c.kernel.filter.p == 0 and c.kernel.filter.n == (8,))
    assert abs(raw8.a[0] - 0.13143) < 1e-5 and raw8.status[0] == "marginal"
    assert not res.unconstrained.admissible


def test_constrained_gaussian_on_boundary(paper_selection):
    g = paper_selection.best_by_family["gaussian"]
    assert g.admissible
    assert g.kernel.filter.h[0] < 1e-6                     # h_x = 0 as in the paper
    assert abs(g.R[0] - 0.95) < 1e-6                       # noise-only objective: boundary
    assert abs(g.a[0] - paper_selection.modes[0].lower_threshold(0.05)) < 1e-9


def test_density_only_coupling_relaxes_threshold(paper):
    grid, plasma = paper
    obj = ModelObjective(grid, plasma, ppc=32)
    kw = dict(tolerance=0.05, n_alias=0, gaussian=dict(enabled=False), binomial=dict(enabled=False))
    s2 = Selector(grid, plasma, [ProtectedMode((12, 1))], obj, coupling_power=2, **kw)
    s1 = Selector(grid, plasma, [ProtectedMode((12, 1))], obj, coupling_power=1, **kw)
    f = Gaussian((0.0, 0.2))
    c2 = (s2.analyse_modes(), s2.evaluate(Kernel(grid.order, f, 2)))[1]
    c1 = (s1.analyse_modes(), s1.evaluate(Kernel(grid.order, f, 1)))[1]
    assert abs(c2.a[0] - c1.a[0] ** 2) < 1e-14
    assert c1.R[0] > c2.R[0]


def test_alternative_shape_candidates(paper):
    grid, plasma = paper
    obj = ModelObjective(grid, plasma, ppc=32)
    res = Selector(grid, plasma, [ProtectedMode((12, 1))], obj, tolerance=0.05, n_alias=4,
                   shapes=["tsc"], gaussian=dict(grid=11), binomial=dict(p=[3, 4], n=[4, 8])).run()
    tsc0 = next(c for c in res.candidates if c.kernel.order == 2 and c.family == "none")
    # TSC weakens the protected coupling relative to CIC (S_TSC < S_CIC)
    assert tsc0.R[0] < 1.0


def test_auto_protect_finds_longitudinal_mode(paper):
    grid, plasma = paper
    found = unstable_modes(grid, plasma, mode_limit=[16, 2])
    assert found[0][0][1] == 0          # the fastest mode is longitudinal (m_y = 0)
    assert any(tuple(m) == (12, 1) for _, _, members in found for m in members)


# --------------------------------------------------------------- objectives
def test_snapshot_objective_equals_direct_mse():
    grid = Grid([16, 8], [4.0, 2.0], fft="complex")
    rng = np.random.default_rng(7)
    rho = rng.normal(size=(3, 4, 16, 8))
    ref = rho + rng.normal(scale=0.3, size=rho.shape)
    obj = SnapshotObjective(grid, rho, ref, select=[0, 1, 2], evaluate=[0])
    f = Binomial(1, 2)
    direct = np.mean((f.apply(rho, grid) - ref) ** 2)
    assert abs(obj.mse(Kernel(1, f), "density", "select") - direct) < 1e-12 * direct

    K = np.stack(grid.fft_wavenumbers(), axis=-1)
    k2 = np.sum(K * K, axis=-1)
    k2[0, 0] = 1.0

    def field(r):
        rh = np.fft.fftn(r, axes=(-2, -1))
        rh[..., 0, 0] = 0.0
        phi = rh / k2
        return [np.real(np.fft.ifftn(-1j * K[..., i] * phi, axes=(-2, -1))) for i in range(2)]

    Ef, Er = field(f.apply(rho, grid)), field(ref)
    direct_E = np.mean(sum((a - b) ** 2 for a, b in zip(Ef, Er)))
    assert abs(obj.mse(Kernel(1, f), "field", "select") - direct_E) < 1e-10 * direct_E

    # real-to-complex layout, as in numpy.fft.rfftn/irfftn solvers
    grid_r = Grid([16, 8], [4.0, 2.0], fft="rfft")
    obj_r = SnapshotObjective(grid_r, rho, ref, select=[0, 1, 2], evaluate=[0])
    kx = 2 * np.pi * np.fft.fftfreq(16, 0.25)
    ky = 2 * np.pi * np.fft.rfftfreq(8, 0.25)
    KX, KY = np.meshgrid(kx, ky, indexing="ij")
    kk = KX**2 + KY**2
    kk[0, 0] = 1.0

    def field_r(r):
        rh = np.fft.rfftn(r, axes=(-2, -1))
        rh[..., 0, 0] = 0.0
        phi = rh / kk
        return [np.fft.irfftn(-1j * Kc * phi, s=(16, 8), axes=(-2, -1)) for Kc in (KX, KY)]

    Ef, Er = field_r(f.apply(rho, grid_r)), field_r(ref)
    direct_r = np.mean(sum((a - b) ** 2 for a, b in zip(Ef, Er)))
    assert abs(obj_r.mse(Kernel(1, f), "field", "select") - direct_r) < 1e-10 * direct_r


def test_model_noise_matches_monte_carlo():
    grid = Grid([32, 16], [8.0, 4.0])
    plasma = Plasma([Species(-1, 1, 1.0, 0.0, 1.0)]).resolve(2)
    ppc = 10
    obj = ModelObjective(grid, plasma, ppc=ppc)
    rng = np.random.default_rng(11)
    N = ppc * grid.n_cells
    q = -1.0 * np.prod(grid.L) / N                 # rho = -1 on average
    filters = [NoFilter(), Binomial(1, 2), Gaussian((0.8, 0.5))]
    err = np.zeros(len(filters))
    R = 150
    for _ in range(R):
        x = rng.uniform(size=(N, 2)) * np.array(grid.L)
        rho = deposit(x, q, grid)
        for i, f in enumerate(filters):
            err[i] += np.mean((f.apply(rho, grid) + 1.0) ** 2)
    err /= R
    for i, f in enumerate(filters):
        model = obj.mse(Kernel(1, f))
        assert abs(err[i] / model - 1) < 0.03


# ------------------------------------------------------------------- config
def test_config_errors_are_helpful():
    cfg = dict(grid=dict(N=[16], L=[1.0]), species=[dict(q=-1, m=1, n=1, vth=1)],
               protected_modes=[dict(mode=[1])], tolerence=0.05)
    with pytest.raises(ConfigError, match="did you mean 'tolerance'"):
        build_selector(cfg)


def test_model_signal_bias_matches_monte_carlo():
    """Signal + noise: the expected error of a smoothing filter (1D, cosine signal)."""
    grid = Grid([64], [64.0])
    plasma = Plasma([Species(-1, 1, 1.0, 0.0, 1.0)]).resolve(1)
    ppc, A, m = 20, 0.05, 3
    obj = ModelObjective(grid, plasma, ppc=ppc, signal=[dict(mode=[m], amplitude=A)])
    rng = np.random.default_rng(5)
    N = ppc * grid.n_cells
    k = 2 * np.pi * m / grid.L[0]
    nodes = np.arange(64) * grid.dx[0]
    target = -(1 + A * np.cos(k * nodes) * float(grid.shape_transform(np.array([k]))))
    filters = [NoFilter(), Gaussian((6.0,))]
    err = np.zeros(2)
    R = 400
    for _ in range(R):
        # rejection sampling of n(x) = 1 + A cos(kx)
        x = rng.uniform(0, 64.0, size=3 * N)
        x = x[rng.uniform(size=x.size) < (1 + A * np.cos(k * x)) / (1 + A)][:N]
        rho = deposit(x, -64.0 / N, grid)
        for i, f in enumerate(filters):
            err[i] += np.mean((f.apply(rho, grid) - target) ** 2)
    err /= R
    for i, f in enumerate(filters):
        assert abs(err[i] / obj.mse(Kernel(1, f)) - 1) < 0.05


# ------------------------------------------------- robustness (review cases)
def test_argument_principle_with_cold_species():
    g = Grid([4096], [2 * math.pi / 0.5])
    p = Plasma([Species(-1, 1, 0.5, 1.0, 0.0), Species(-1, 1, 0.5, -1.0, 0.0)]).resolve(1)
    assert rt.count_unstable(ModeDispersion(g, p, (1,), n_alias=0)) == 1


@pytest.mark.parametrize("vth_beam", [0.0, 1e-3])
def test_auto_protect_finds_cold_beam_instability(vth_beam):
    g = Grid([256], [2 * math.pi / 0.25])
    p = Plasma([Species(-1, 1, 0.99, 0.0, 0.1), Species(-1, 1, 0.01, 4.0, vth_beam)]).resolve(1)
    found = unstable_modes(g, p, mode_limit=[3])
    assert found and found[0][0] == (1,) and abs(found[0][1] - 0.1353) < 1e-3


def test_aliasing_instability_is_counted():
    g = Grid([32], [32.0])
    p = Plasma([Species(-1, 1, 1.0, 3.0, 0.0)]).resolve(1)
    d = ModeDispersion(g, p, (3,), n_alias=3)
    w = rt.newton(d, complex(-17.08, 0.03))
    assert w.imag > 0.03
    assert rt.count_unstable(d) >= 1
    assert rt.verify_most_unstable(d, 1.0, 0.5 * w.imag) is False


def test_frequency_tolerance_is_unit_invariant():
    """A purely growing mode in units where omega ~ 3e8: no spurious frequency constraint."""
    s = 3e8
    g = Grid([64], [64.0 / s * 1.0])
    p = Plasma([Species(-1, 1, 0.5 * s**2, 1.0, 0.05), Species(-1, 1, 0.5 * s**2, -1.0, 0.05)]).resolve(1)
    obj = ModelObjective(g, p, ppc=32)
    out = []
    for ft in (None, 0.05):
        sel = Selector(g, p, [ProtectedMode((5,))], obj, tolerance=0.05, n_alias=2,
                       frequency_tolerance=ft, gaussian=dict(enabled=False),
                       binomial=dict(enabled=False))
        sel.analyse_modes()
        out.append(sel.analyses[g.order][0].lower_threshold(0.05))
    assert abs(out[0] - out[1]) < 1e-9


def test_nyquist_protected_mode_rejected():
    g = Grid([16], [16.0])
    p = Plasma([Species(-1, 1, 1.0, 0.0, 1.0)]).resolve(1)
    with pytest.raises(ValueError, match="Nyquist"):
        ModeDispersion(g, p, (8,))


def test_single_realisation_is_flagged(paper):
    grid, plasma = paper
    rng = np.random.default_rng(0)
    rho = rng.normal(size=(1, 2, 128, 16))
    obj = SnapshotObjective(grid, rho, rho + 0.1 * rng.normal(size=rho.shape))
    assert obj.held_out is False
    res = Selector(grid, plasma, [ProtectedMode((12, 1))], obj, n_alias=0,
                   gaussian=dict(grid=7), binomial=dict(p=[4], n=[8])).run()
    assert any("not independent" in w for w in res.warnings)
