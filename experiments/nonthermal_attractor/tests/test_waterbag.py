import numpy as np

from attractor_analysis import waterbag as W


def _exp_f(nx=8, nv=400, vlim=10.0, ve=6.0):
    v = (np.arange(nv) + 0.5) * (2 * vlim / nv) - vlim
    f1 = np.where(np.abs(v) < ve, np.exp(-np.abs(v)), 0.0)
    return np.tile(f1, (nx, 1)), 1.0 / nx, 2 * vlim / nv


def test_closure_volume_and_mass():
    f, dx, dv = _exp_f()
    edges = np.exp(np.arange(np.log(1e-4), np.log(2.0), 0.25))
    wc = W.waterbag_content(f, dx, dv, edges)
    c = W.closure(wc, f)
    assert np.isclose(c["volume_hist"], c["volume_direct"])
    assert np.isclose(c["mass_hist"], c["mass_direct"])


def test_eta_minus1_lower_edge_on_grid_distribution():
    f, dx, dv = _exp_f(ve=6.0)
    edges = np.exp(np.arange(np.log(1e-5), np.log(2.0), 0.25))
    wc = W.waterbag_content(f, dx, dv, edges)
    det = W.eta_minus1_range(wc)
    assert det["found"]
    assert abs(np.log(det["eta_min"] / np.exp(-6.0))) < 0.25
    assert not det["resolution_limited"]


def test_particle_floor_flag():
    rng = np.random.default_rng(0)
    N = 20_000
    a = rng.exponential(size=N)
    a = a[a < 8.0]
    v = np.where(rng.random(a.size) < 0.5, -a, a)
    x = rng.random(a.size)
    w = np.full(a.size, 2.0 / a.size)
    fh = W.reconstruct_f_particles(x, v, w, 1.0, 10.0, 8, 200)
    dxb, dvb = 1 / 8, 0.1
    floor = W.particle_eta_floor(w[0], dxb, dvb)
    edges = np.exp(np.arange(np.log(1e-6), np.log(2.0), 0.25))
    wc = W.waterbag_content(fh, dxb, dvb, edges, floor)
    det = W.eta_minus1_range(wc)
    # truth exp(-8) = 3.4e-4 is far below the floor: the estimate must be flagged
    assert np.exp(-8.0) < floor
    assert det["resolution_limited"]


def test_coarse_grain_conserves_mass():
    f, dx, dv = _exp_f()
    g = W.coarse_grain_grid(f, 2, 4)
    assert np.isclose(g.sum() * dx * 2 * dv * 4, f.sum() * dx * dv)
