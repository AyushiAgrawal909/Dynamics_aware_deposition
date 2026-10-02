"""T2 backend checks at CI size (V1-V3 subsets, diagnostics contract)."""
import numpy as np
import pytest
import yaml
from pathlib import Path

from attractor_analysis import schema
from attractor_sim import theory, validation as V
from attractor_sim.vlasov1d1v import Vlasov1D1V, waterbag_beams_f

HERE = Path(__file__).resolve().parents[1]


def test_waterbag_dispersion_cold_limit_and_reference():
    cold = theory.fastest_mode(1.0, 1e-6)
    assert cold["gamma_max"] == pytest.approx(1 / (2 * np.sqrt(2)), rel=1e-6)
    assert cold["k_star"] == pytest.approx(np.sqrt(3 / 8), rel=1e-4)
    fm = theory.fastest_mode(1.0, 0.2)
    assert fm["gamma_max"] == pytest.approx(0.3522, abs=2e-4)
    assert fm["k_star"] == pytest.approx(0.614, abs=1e-3)


def test_landau_root():
    w = theory.landau_root(0.5)
    assert w.real == pytest.approx(1.4157, abs=1e-3)
    assert w.imag == pytest.approx(-0.1534, abs=1e-3)


def test_pic_momentum_roundoff_and_neutrality():
    r = V.v1_pic_conservation(nx=32, ppc=64, steps=100)
    assert abs(r["charge"]) < 1e-10
    assert r["momentum_drift_max"] < 1e-12


@pytest.mark.slow
def test_pic_and_vlasov_landau():
    assert V.v2_landau_pic()["gamma_measured"] == pytest.approx(-0.1534, rel=0.05)
    assert V.v2_landau_vlasov()["gamma_measured"] == pytest.approx(-0.1534, rel=0.05)


@pytest.mark.slow
def test_waterbag_growth_both_backends():
    for b in ("pic", "vlasov"):
        r = V.v3_waterbag_growth(b)
        assert r["rel_err"] < 0.05, r


def test_vlasov_mass_conservation_and_positivity():
    L, nx, nv = 20.0, 32, 128
    f = waterbag_beams_f(L, nx, 3.0, nv, 1.0, 0.2, modes=[1, 2], amplitudes=[0.05, 0.05], phases=[0.0, 1.0])
    sim = Vlasov1D1V(L, nx, 3.0, nv, f)
    m0 = sim.budgets()["number"]
    for _ in range(50):
        sim.step(0.1)
    b = sim.budgets()
    assert abs(b["number"] + b["outflow_mass"] - m0) < 1e-10 * m0
    # v-advection is positivity-preserving; the spectral x-shift is not
    # (documented): undershoots stay small and are recorded as f_min
    assert b["f_min"] >= -1e-3 * f.max()


def test_waterbag_cell_average_second_moment():
    nv, vmax = 400, 3.0
    f = waterbag_beams_f(10.0, 4, vmax, nv, 1.0, 0.2)
    dv = 2 * vmax / nv
    v = -vmax + (np.arange(nv) + 0.5) * dv
    n = f[0].sum() * dv
    m2 = (f[0] * v * v).sum() * dv / n
    assert n == pytest.approx(1.0, rel=1e-12)
    assert m2 == pytest.approx(1.0 + 0.2**2 / 12, rel=1e-3)


def test_runner_writes_valid_bundles(tmp_path):
    from attractor_sim import run as R
    cfg = yaml.safe_load((HERE / "configs" / "t2_smoke.yaml").read_text())
    cfg["grid"]["nx"] = 128
    cfg["pic"]["ppc_levels"] = [64, 32]
    cfg["vlasov"] = {"v_max": 3.0, "dv_finest": 0.05, "levels": 1}
    cfg["reconstruction"] = {"nx_b": 16, "nv_b": 60, "v_lim": 3.0}
    cfg["diagnostics"]["every_steps"] = 10
    for backend, kw in (("pic1d1v", {}), ("vlasov1d1v", {}), ("pic1d1v", {"control": True})):
        p = R.run(cfg, backend, 1 if backend == "pic1d1v" else 0, 0, tmp_path, t_end=2.0, **kw)
        md, arr = schema.read_bundle(p)
        assert md["backend"] == backend
        assert np.allclose(arr["times"], np.arange(0, 2.0 + 1e-9, 0.5))
    md, arr = schema.read_bundle(tmp_path / "pic1d1v_L1_s0.npz")
    assert md["particles_per_cell"] == 32
    # moments of the histogram equal the direct budgets
    K_hist = arr["eps_bin_energy"].sum(axis=1) + arr["eps_overflow"][:, 1]
    assert np.allclose(K_hist, arr["kinetic_energy"], rtol=1e-12)
