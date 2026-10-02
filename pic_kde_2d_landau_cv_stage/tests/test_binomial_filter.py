"""Tests for the production binomial-filter baseline and pass budgets."""

from __future__ import annotations

import numpy as np

from pic_kde_2d.core.grid2d import Grid2D
from pic_kde_2d.methods.sigma_selection import (
    binomial_pass_budget_for_gaussian_ceiling,
    equivalent_gaussian_sigma_cells,
)
from pic_kde_2d.methods.spectral_W import (
    make_W_binomial,
    make_W_compensated_binomial,
    make_W_gaussian,
)


def test_binomial_spectral_response_is_exact() -> None:
    grid = Grid2D(Nx=64, Ny=16, Lx=16.0, Ly=8.0)
    passes = 12
    W = make_W_binomial(grid, passes, 0)
    expected = np.cos(np.pi / grid.Nx) ** (2 * passes)
    np.testing.assert_allclose(W[1, 0], expected, rtol=0.0, atol=2.0e-15)
    np.testing.assert_allclose(W[0, 1], 1.0, rtol=0.0, atol=2.0e-15)


def test_gaussian_equivalent_matches_resolved_landau_mode() -> None:
    grid = Grid2D(Nx=64, Ny=16, Lx=16.0, Ly=8.0)
    passes = 12
    sigma = equivalent_gaussian_sigma_cells(passes)
    W_binomial = make_W_binomial(grid, passes, 0)
    W_gaussian = make_W_gaussian(grid, sigma * grid.dx, 0.0)
    np.testing.assert_allclose(
        W_binomial[1, 0],
        W_gaussian[1, 0],
        rtol=3.0e-5,
    )


def test_compensated_binomial_response_is_exact() -> None:
    grid = Grid2D(Nx=64, Ny=16, Lx=16.0, Ly=8.0)
    passes = 12
    W = make_W_compensated_binomial(grid, passes, 0)
    phase = np.pi / grid.Nx
    expected = np.cos(phase) ** (2 * passes) * (
        1.0 + passes * np.sin(phase) ** 2
    )
    np.testing.assert_allclose(W[1, 0], expected, rtol=0.0, atol=2.0e-15)
    np.testing.assert_allclose(W[0, 1], 1.0, rtol=0.0, atol=2.0e-15)


def test_compensation_preserves_low_k_and_suppresses_nyquist() -> None:
    grid = Grid2D(Nx=64, Ny=16, Lx=16.0, Ly=8.0)
    passes = 8
    ordinary = make_W_binomial(grid, passes, 0)
    compensated = make_W_compensated_binomial(grid, passes, 0)
    assert 1.0 - compensated[1, 0] < 0.02 * (1.0 - ordinary[1, 0])
    np.testing.assert_allclose(compensated[grid.Nx // 2, 0], 0.0, atol=1.0e-25)


def test_higher_order_compensation_matches_e2_anchors() -> None:
    grid = Grid2D(Nx=128, Ny=16, Lx=64.0, Ly=16.0)
    second = make_W_compensated_binomial(
        grid,
        2,
        0,
        compensation_order=2,
    )
    third = make_W_compensated_binomial(
        grid,
        4,
        0,
        compensation_order=3,
    )
    np.testing.assert_allclose(second[12, 0] ** 2, 0.9955208571006037, atol=2e-15)
    np.testing.assert_allclose(third[12, 0] ** 2, 0.9971375751242197, atol=2e-15)


def test_compensation_order_must_be_positive_integer() -> None:
    grid = Grid2D(Nx=64, Ny=16, Lx=16.0, Ly=8.0)
    for invalid in (0, -1, 1.5, True):
        with np.testing.assert_raises(ValueError):
            make_W_compensated_binomial(
                grid,
                1,
                compensation_order=invalid,
            )


def test_c4_pass_budgets_for_final_benchmarks() -> None:
    assert binomial_pass_budget_for_gaussian_ceiling(64, 1, 2.453) == 12
    assert (
        binomial_pass_budget_for_gaussian_ceiling(
            128,
            12,
            0.1844549,
        )
        == 0
    )
