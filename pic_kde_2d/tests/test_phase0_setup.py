"""Phase 0 acceptance tests: verify project structure is correct."""

from __future__ import annotations

import pathlib


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_core_modules_importable():
    """All core modules must be importable, even if not implemented."""
    import pic_kde_2d.core.grid2d
    import pic_kde_2d.core.particles2d
    import pic_kde_2d.core.pic_loop


def test_methods_modules_importable():
    import pic_kde_2d.methods.cic
    import pic_kde_2d.methods.gaussian
    import pic_kde_2d.methods.spectral_W
    import pic_kde_2d.methods.adaptive_sigma
    import pic_kde_2d.methods.delta_f


def test_benchmarks_modules_importable():
    import pic_kde_2d.benchmarks.landau_2d
    import pic_kde_2d.benchmarks.two_stream_2d
    import pic_kde_2d.benchmarks.theory


def test_validation_modules_importable():
    import pic_kde_2d.validation.self_force
    import pic_kde_2d.validation.conservation
    import pic_kde_2d.validation.spectral_rule


def test_numpy_fft_conventions():
    """Verify the FFT convention is what we expect."""
    import numpy as np

    Nx, Ny = 16, 16
    a = np.random.rand(Nx, Ny)
    a_hat = np.fft.rfft2(a)
    assert a_hat.shape == (Nx, Ny // 2 + 1), f"Unexpected rfft2 shape: {a_hat.shape}"

    a_recovered = np.fft.irfft2(a_hat, s=(Nx, Ny))
    np.testing.assert_allclose(a, a_recovered, atol=1e-12)


def test_units_documented_in_readme():
    """README must mention key conventions."""
    content = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    assert "lambda_D = 1" in content or "Debye lengths" in content
    assert "omega_p" in content
    assert "q_e = -1" in content or "electron" in content.lower()
