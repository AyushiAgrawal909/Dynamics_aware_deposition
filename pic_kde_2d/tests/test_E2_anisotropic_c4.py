"""Unit checks for the E2 anisotropic C4 validation harness."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import numpy as np

from pic_kde_2d.core.grid2d import Grid2D
from pic_kde_2d.benchmarks.theory import (
    warm_two_stream_growth,
    warm_two_stream_oblique_growth,
)
from pic_kde_2d.methods.spectral_W import apply_W_to_grid, make_W_gaussian


ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


anisotropic = load_script("run_E2_anisotropic_c4_seed")
linearized = load_script("run_linearized_delta_f_mode")


def test_ky_zero_transfer_is_sigma_y_independent():
    grid = Grid2D(Nx=128, Ny=16, Lx=64.0, Ly=16.0)
    values = []
    for sigma_y_dy in (0.18445488487247555, 1.0, 2.0, 4.0, 8.0):
        W = make_W_gaussian(
            grid,
            0.18445488487247555 * grid.dx,
            sigma_y_dy * grid.dy,
        )
        values.append(float(W[12, 0] ** 2))
    np.testing.assert_allclose(values, values[0], rtol=0.0, atol=0.0)


def test_y_averaged_density_is_sigma_y_independent():
    rng = np.random.default_rng(42)
    grid = Grid2D(Nx=128, Ny=16, Lx=64.0, Ly=16.0)
    rho = rng.normal(size=(grid.Nx, grid.Ny))
    profiles = []
    for sigma_y_dy in (0.18445488487247555, 1.0, 2.0, 4.0):
        W = make_W_gaussian(
            grid,
            0.18445488487247555 * grid.dx,
            sigma_y_dy * grid.dy,
        )
        profiles.append(np.mean(apply_W_to_grid(rho, W, grid), axis=1))
    for profile in profiles[1:]:
        np.testing.assert_allclose(profile, profiles[0], rtol=0.0, atol=2.0e-16)


def test_rfft_error_components_close_parseval():
    rng = np.random.default_rng(7)
    error = rng.normal(size=(32, 10))
    ky0, ky_nonzero = anisotropic.rfft_parseval_components(error)
    np.testing.assert_allclose(
        ky0 + ky_nonzero,
        np.mean(error**2),
        rtol=2.0e-15,
        atol=2.0e-15,
    )


def test_background_neutrality_is_roundoff():
    rng = np.random.default_rng(8)
    rho = -1.0 + rng.normal(scale=0.1, size=(128, 16))
    assert anisotropic.background_neutrality(rho) < 1.0e-14


def test_directional_widths_override_isotropic_cli():
    args = argparse.Namespace(sigma_dx=3.0, sigma_x_dx=0.2, sigma_y_dy=4.0)
    assert linearized.resolved_sigma_cells(args) == (0.2, 4.0)
    args = argparse.Namespace(sigma_dx=1.5, sigma_x_dx=None, sigma_y_dy=None)
    assert linearized.resolved_sigma_cells(args) == (1.5, 1.5)


def test_oblique_projection_keeps_only_target_conjugate_pair():
    grid = Grid2D(Nx=32, Ny=16, Lx=32.0, Ly=16.0)
    x, y = np.meshgrid(grid.x, grid.y, indexing="ij")
    target = np.cos(2.0 * np.pi * (3.0 * x / grid.Lx + 2.0 * y / grid.Ly))
    contaminant = 0.4 * np.cos(
        2.0 * np.pi * (5.0 * x / grid.Lx + y / grid.Ly)
    )
    projected = linearized.project_density_to_mode(
        target + contaminant, grid, 3, 2
    )
    np.testing.assert_allclose(projected, target, rtol=0.0, atol=3.0e-15)


def test_oblique_warm_solver_reduces_to_aligned_solver_at_ky_zero():
    kx = 2.0 * np.pi * 12.0 / 64.0
    aligned = warm_two_stream_growth(kx, 0.8, 0.05, coupling=0.94)
    oblique = warm_two_stream_oblique_growth(
        kx, 0.0, 0.8, 0.05, coupling=0.94
    )
    np.testing.assert_allclose(oblique, aligned, rtol=0.0, atol=0.0)
