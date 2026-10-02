import numpy as np
import pytest

from attractor_analysis import schema


def _minimal(nt=3, neps=4, neta=5):
    arrays = {
        "times": np.arange(nt, dtype=float),
        "number": np.ones(nt), "total_charge": -np.ones(nt), "momentum": np.zeros(nt),
        "kinetic_energy": np.ones(nt), "field_energy": np.zeros(nt), "total_energy": np.ones(nt),
        "eps_edges": np.linspace(0, 1, neps + 1), "N_eps": np.ones((nt, neps)),
        "eps_bin_weight": np.ones((nt, neps)), "eps_bin_energy": np.ones((nt, neps)),
        "eps_bin_neff": np.ones((nt, neps)), "eps_overflow": np.zeros((nt, 2)),
        "eta_edges": np.geomspace(1e-3, 1, neta + 1), "rho_eta": np.ones((nt, neta)),
        "rho_bin_volume": np.ones((nt, neta)), "rho_bin_mass": np.ones((nt, neta)),
        "rho_reconstruction": np.array([0.1, 0.1, 1e-3]), "k": np.array([0.0, 1.0]),
        "Ek2": np.zeros((nt, 2)),
    }
    md = {"schema_version": schema.SCHEMA_VERSION, "backend": "pic1d1v", "git_revision": "x", "config": {}, "normalization": "omega_pe=1",
          "seed": 0, "nested_level": 0, "dt": 0.1, "dx": 0.05, "staggering": "KDK synchronised",
          "wall_time_s": 1.0, "particles_per_cell": 64, "n_particles": 640, "macro_weight": 0.01}
    return md, arrays


def test_roundtrip(tmp_path):
    md, arrays = _minimal()
    path = schema.write_bundle(tmp_path / "run.npz", md, arrays)
    md2, arr2 = schema.read_bundle(path)
    assert md2["backend"] == "pic1d1v"
    assert md2["schema_version"] == schema.SCHEMA_VERSION
    np.testing.assert_array_equal(arr2["N_eps"], arrays["N_eps"])


@pytest.mark.parametrize("drop", ["N_eps", "rho_eta", "Ek2", "total_energy"])
def test_missing_array_rejected(drop):
    md, arrays = _minimal()
    del arrays[drop]
    with pytest.raises(schema.BundleError, match=drop):
        schema.validate(md, arrays)


def test_backend_specific_metadata_required():
    md, arrays = _minimal()
    del md["particles_per_cell"]
    with pytest.raises(schema.BundleError, match="particles_per_cell"):
        schema.validate(md, arrays)


def test_shape_mismatch_rejected():
    md, arrays = _minimal()
    arrays["N_eps"] = np.ones((3, 7))
    with pytest.raises(schema.BundleError, match="bins"):
        schema.validate(md, arrays)
