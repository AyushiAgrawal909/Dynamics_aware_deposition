import importlib.util
from pathlib import Path
import sys

import numpy as np


SCRIPT = (
    Path(__file__).resolve().parents[1] / "scripts" / "run_G0_locality_headroom.py"
)
SPEC = importlib.util.spec_from_file_location("g0_locality", SCRIPT)
G0 = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = G0
SPEC.loader.exec_module(G0)


def test_c0_log_profile_is_safe_in_core_free_outside_and_continuous():
    x = np.linspace(0.0, 64.0, 4096, endpoint=False)
    mask, sigma = G0.c0_log_width_profile(
        x, 64.0, protected_fraction=0.1, blend_fraction=0.03,
        sigma_safe=1.0, sigma_free=8.0,
    )
    np.testing.assert_allclose(np.min(sigma), 1.0, rtol=0.0, atol=1.0e-14)
    np.testing.assert_allclose(np.max(sigma), 8.0, rtol=0.0, atol=1.0e-14)
    assert np.all((mask >= 0.0) & (mask <= 1.0))
    x_fine = np.linspace(0.0, 64.0, 8192, endpoint=False)
    _, sigma_fine = G0.c0_log_width_profile(
        x_fine, 64.0, protected_fraction=0.1, blend_fraction=0.03,
        sigma_safe=1.0, sigma_free=8.0,
    )
    assert np.max(np.abs(np.diff(sigma_fine))) < 0.51 * np.max(
        np.abs(np.diff(sigma))
    )
    center = len(x) // 2
    assert sigma[center] == 1.0
    np.testing.assert_allclose(sigma[0], 8.0, rtol=0.0, atol=1.0e-14)


def test_uniform_profile_matches_stationary_field_and_density_variance():
    geometry = G0.Geometry(
        name="test", intended=True, carrier_mode_x=1,
        Nx=16, Ny=8, Lx=2.0 * np.pi / 0.4, Ly=8.0,
    )
    model = G0.SpectralVarianceModel(geometry)
    packet = G0.packet_geometry(geometry, 0.4, 0.4, 1.0)
    _, field_global, density_global = model.kernel_power(1.0)
    field_local, density_local = model.local_variances(
        np.ones(geometry.Nx), packet["power_weight"]
    )
    np.testing.assert_allclose(field_local, field_global, rtol=2.0e-14)
    np.testing.assert_allclose(density_local, density_global, rtol=2.0e-14)


def test_field_variance_matches_direct_random_propagation():
    geometry = G0.Geometry(
        name="test", intended=True, carrier_mode_x=1,
        Nx=8, Ny=4, Lx=2.0 * np.pi / 0.4, Ly=8.0,
    )
    model = G0.SpectralVarianceModel(geometry)
    _, predicted, _ = model.kernel_power(1.5)

    # The trace is exact; Monte Carlo only independently checks normalization.
    rng = np.random.default_rng(123)
    samples = []
    shape = model.cic * np.exp(
        -0.5
        * (
            model.kx**2 * (1.5 * geometry.dx) ** 2
            + model.ky**2 * (1.5 * geometry.dy) ** 2
        )
    )
    for _ in range(20000):
        rho = rng.normal(size=(geometry.Nx, geometry.Ny))
        rho_hat = np.fft.fft2(rho) * shape
        ex_hat = np.zeros_like(rho_hat, dtype=np.complex128)
        ey_hat = np.zeros_like(rho_hat, dtype=np.complex128)
        kx = np.broadcast_to(model.kx, rho_hat.shape)
        ky = np.broadcast_to(model.ky, rho_hat.shape)
        ex_hat[model.nonzero] = (
            -1j * kx[model.nonzero] * rho_hat[model.nonzero] / model.k2[model.nonzero]
        )
        ey_hat[model.nonzero] = (
            -1j * ky[model.nonzero] * rho_hat[model.nonzero] / model.k2[model.nonzero]
        )
        ex = np.fft.ifft2(ex_hat).real
        ey = np.fft.ifft2(ey_hat).real
        samples.append(np.mean(ex**2 + ey**2))
    np.testing.assert_allclose(np.mean(samples), predicted, rtol=0.025)


def test_proposed_packet_is_not_narrowband():
    geometry = G0.Geometry(
        name="proposed", intended=False, carrier_mode_x=4,
        Nx=256, Ny=32, Lx=2.0 * np.pi * 4 / 0.4, Ly=8.0,
    )
    packet = G0.packet_geometry(geometry, 0.1, 0.4, 1.0)
    assert packet["carrier_cycles"] == 0.4
    assert packet["relative_bandwidth"] > 1.0
