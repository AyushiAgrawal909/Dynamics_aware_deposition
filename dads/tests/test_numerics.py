"""Unit tests: Z function, mesh transforms, filters, deposition, dispersion."""

import math
from itertools import product

import numpy as np
import pytest
from scipy.integrate import quad

from dads import Binomial, Gaussian, Grid, ModeDispersion, Plasma, Species, deposit
from dads.filters import binomial_axis_transfer, binomial_stencil
from dads.zfunc import Z, maxwell_response, maxwell_response_derivative


# ------------------------------------------------------------------ Z function
def test_z_at_origin_and_symmetry():
    assert abs(Z(0.0) - 1j * math.sqrt(math.pi)) < 1e-14
    z = 0.7 + 0.3j
    # Z(-z*) = -Z(z)* (reality of the Maxwellian)
    assert abs(Z(-np.conj(z)) + np.conj(Z(z))) < 1e-14


def test_response_derivative_matches_finite_difference():
    for z in (0.3 + 0.2j, 2.0 - 1.0j, 11.0 + 0.4j, 29.0 - 0.8j, 45.0 + 1.0j):
        h = 1e-6 * max(1.0, abs(z))
        fd = (maxwell_response(z + h) - maxwell_response(z - h)) / (2 * h)
        assert abs(fd - maxwell_response_derivative(z)) < 1e-6 * max(1.0, abs(fd)) + 1e-12


def test_response_continuous_across_asymptotic_switch():
    for phase in (0.05, -0.05, 0.4, -0.4):
        z0 = 30.0 * np.exp(1j * phase)
        za, zb = z0 * (1 - 1e-9), z0 * (1 + 1e-9)
        a, b = maxwell_response(za), maxwell_response(zb)
        slope = maxwell_response_derivative(z0)
        assert abs(b - a - slope * (zb - za)) < 1e-10 * abs(a)


def test_response_matches_principal_value_integral():
    # 1 + zeta Z(zeta) = (1/sqrt(pi)) int t exp(-t^2) / (t - zeta) dt for Im zeta > 0
    zeta = 0.8 + 0.5j
    re = quad(lambda t: (t * np.exp(-t * t) / (t - zeta)).real, -12, 12, limit=200)[0]
    im = quad(lambda t: (t * np.exp(-t * t) / (t - zeta)).imag, -12, 12, limit=200)[0]
    assert abs((re + 1j * im) / math.sqrt(math.pi) - maxwell_response(zeta)) < 1e-9


# ------------------------------------------------------------------- the mesh
@pytest.mark.parametrize("shape", ["ngp", "cic", "tsc", "pqs"])
def test_alias_power_closed_form(shape):
    g = Grid([16], [16.0], shape=shape)
    k = np.linspace(0.05, math.pi, 7)[:, None]
    p = np.arange(-4000, 4001)
    brute = [np.sum(g.shape_transform((kk + 2 * math.pi * p)[:, None]) ** 2) for kk in k[:, 0]]
    tol = 2e-4 if shape == "ngp" else 1e-12   # NGP converges only like 1/p
    assert np.allclose(g.alias_power(k), brute, atol=tol)


def test_operator_symbols_low_k_limit():
    g = Grid([64, 32], [10.0, 5.0], poisson="fd2", gradient="centered")
    k = np.array([1e-3, 2e-3])
    assert abs(g.laplacian_symbol(k) / np.sum(k * k) - 1) < 1e-6
    assert np.allclose(g.gradient_symbol(k), k, rtol=1e-6)


def test_aliases_principal_first_and_geometry():
    g = Grid([32, 8], [8.0, 4.0])
    k = g.k_of_mode((3, 1))
    q, geom = g.aliases(k, 3)
    assert np.allclose(q[0], k)
    assert abs(geom[0] - g.shape_transform(k) ** 2) < 1e-14   # spectral: q.k/k^2 = 1


def test_mode_validation():
    g = Grid([16], [1.0])
    with pytest.raises(ValueError):
        g.k_of_mode((9,))
    with pytest.raises(ValueError):
        g.k_of_mode((0,))


# -------------------------------------------------------------------- filters
@pytest.mark.parametrize("p,n", [(0, 1), (0, 8), (1, 2), (2, 4), (3, 4), (4, 8)])
def test_binomial_stencil_reproduces_transfer(p, n):
    taps = binomial_stencil(n, p)
    assert abs(taps.sum() - 1.0) < 1e-12
    assert np.allclose(taps, taps[::-1])
    theta = np.linspace(0, math.pi, 50)
    off = np.arange(-(taps.size // 2), taps.size // 2 + 1)
    W_stencil = np.real(np.exp(1j * np.outer(theta, off)) @ taps)
    W = binomial_axis_transfer(np.sin(theta / 2) ** 2, n, p)
    assert np.allclose(W_stencil, W, atol=1e-12)


@pytest.mark.parametrize("p", [0, 1, 2, 3, 4])
def test_compensation_order(p):
    n = 4
    s = np.array([1e-3, 2e-3])
    err = 1.0 - binomial_axis_transfer(s, n, p)
    # 1 - W = O(s^(p+1)): doubling s multiplies the error by 2^(p+1)
    assert abs(err[1] / err[0] - 2 ** (p + 1)) < 0.02 * 2 ** (p + 1)


def test_filters_bounded_and_apply_preserves_mean():
    g = Grid([32, 16], [8.0, 4.0])
    rng = np.random.default_rng(1)
    rho = rng.normal(size=g.N) + 3.0
    for f in (Gaussian((0.7, 1.3)), Binomial(2, 4), Binomial(0, (2, 0))):
        W = f.transfer_grid(g)
        assert W.max() <= 1 + 1e-12 and W.min() >= -1e-12
        out = f.apply(rho, g)
        assert abs(out.mean() - rho.mean()) < 1e-12


def test_gaussian_half_power():
    g = Grid([64], [64.0])
    f = Gaussian((1.0,))
    frac = f.half_power_wavenumber(g, 0)
    assert abs(frac * math.pi - math.sqrt(2 * math.log(2))) < 1e-9


# ------------------------------------------------------------------ deposition
def _bspline(t, order):
    t = np.abs(t)
    if order == 0:
        return np.where(t < 0.5, 1.0, np.where(t == 0.5, 0.5, 0.0))
    if order == 1:
        return np.maximum(1.0 - t, 0.0)
    if order == 2:
        return np.where(t < 0.5, 0.75 - t * t, np.where(t < 1.5, 0.5 * (1.5 - t) ** 2, 0.0))
    return np.where(t < 1, 2 / 3 - t * t + 0.5 * t**3, np.where(t < 2, (2 - t) ** 3 / 6, 0.0))


@pytest.mark.parametrize("shape,order", [("ngp", 0), ("cic", 1), ("tsc", 2), ("pqs", 3)])
def test_deposit_conserves_charge_and_matches_bspline(shape, order):
    g = Grid([16, 8], [4.0, 2.0], shape=shape)
    rng = np.random.default_rng(3)
    x = rng.uniform(0, 1, size=(5000, 2)) * np.array(g.L)
    q = rng.uniform(0.5, 1.5, size=5000)
    rho = deposit(x, q, g)
    assert abs(rho.sum() * np.prod(g.dx) - q.sum()) < 1e-9
    x0 = np.array([1.37, 0.81])
    one = deposit(x0[None, :], 1.0, g) * np.prod(g.dx)
    nodes = [np.arange(n) * d for n, d in zip(g.N, g.dx)]
    expect = np.ones(g.N)
    for ax in range(2):
        t = (nodes[ax] - x0[ax]) / g.dx[ax]
        t = (t + g.N[ax] / 2) % g.N[ax] - g.N[ax] / 2          # periodic distance
        w = _bspline(t, order)
        expect = expect * (w[:, None] if ax == 0 else w[None, :])
    assert np.allclose(one, expect, atol=1e-12)


# ------------------------------------------------------------------ dispersion
def _two_stream(grid):
    return Plasma([Species(-1, 1, 0.5, [0.8, 0], 0.05), Species(-1, 1, 0.5, [-0.8, 0], 0.05)]).resolve(2)


def test_analytic_derivative():
    g = Grid([128, 16], [64.0, 16.0])
    d = ModeDispersion(g, _two_stream(g), (12, 1), n_alias=3)
    for w in (0.1j, 0.5 + 0.05j, 0.9 - 0.02j):
        h = 1e-6
        fd = (d.G(w + h) - d.G(w - h)) / (2 * h)
        assert abs(fd - d.dG(w)) < 1e-6 * abs(fd)


def test_energy_scheme_principal_equals_momentum():
    g1 = Grid([128, 16], [64.0, 16.0], scheme="momentum")
    g2 = Grid([128, 16], [64.0, 16.0], scheme="energy")
    pl = _two_stream(g1)
    d1 = ModeDispersion(g1, pl, (12, 1), n_alias=0)
    d2 = ModeDispersion(g2, pl, (12, 1), n_alias=0)
    assert abs(d1.G(0.1j) - d2.G(0.1j)) < 1e-12


def test_leapfrog_temporal_sum_against_brute_force():
    """Analytic tail of the temporal alias sum against a long explicit sum."""
    g = Grid([128, 16], [64.0, 16.0])
    pl = _two_stream(g)
    dt = 0.2
    d = ModeDispersion(g, pl, (12, 1), n_alias=1, time="leapfrog", dt=dt, n_time_alias=6)
    dc = ModeDispersion(g, pl, (12, 1), n_alias=1)
    w = 0.3 + 0.1j
    wg = 2 * math.pi / dt
    brute = sum(dc.G(w - l * wg) for l in range(-20000, 20001))
    assert abs(d.G(w) - brute) < 1e-7 * abs(brute)


def test_leapfrog_tends_to_continuous():
    g = Grid([128, 16], [64.0, 16.0])
    pl = _two_stream(g)
    w = 0.12j
    dc = ModeDispersion(g, pl, (12, 1), n_alias=2)
    errs = []
    for dt in (0.08, 0.04):
        d = ModeDispersion(g, pl, (12, 1), n_alias=2, time="leapfrog", dt=dt)
        errs.append(abs(d.G(w) - dc.G(w)))
    assert errs[1] < 0.3 * errs[0]          # O(dt^2)
