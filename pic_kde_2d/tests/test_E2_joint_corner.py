"""Unit checks for the E2 joint corner rate-noise campaign."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

from pic_kde_2d.core.grid2d import Grid2D


ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(
        name, ROOT / "scripts" / f"{name}.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


noise_runner = load_script("run_E2_joint_corner_noise_seed")
analyzer = load_script("analyze_E2_joint_corner_sweep")


def test_config_parser_requires_cic_first():
    configs = noise_runner.parse_configs(
        "base:0:0,y018:0:0.18,joint08:0.08:0.21110810"
    )
    assert configs[0] == ("base", 0.0, 0.0)
    assert configs[-1] == ("joint08", 0.08, 0.21110810)


def test_joint_points_share_the_frozen_rate_budget():
    ax = (2.0 * np.pi * 12.0 / 128.0) ** 2
    ay = (2.0 * np.pi / 16.0) ** 2
    q08 = ax * 0.08**2 + ay * 0.21110810**2
    q12 = ax * 0.12**2 + ay * 0.16299273**2
    np.testing.assert_allclose(q08, 0.009093395731, rtol=0.0, atol=5.0e-10)
    np.testing.assert_allclose(q12, 0.009093395731, rtol=0.0, atol=5.0e-10)


def test_white_noise_proxy_is_normalized_and_monotone():
    grid = Grid2D(Nx=128, Ny=16, Lx=64.0, Ly=16.0)
    baseline = analyzer.white_noise_proxy(grid, 0.0, 0.0)
    narrow = analyzer.white_noise_proxy(grid, 0.0, 0.12)
    wide = analyzer.white_noise_proxy(grid, 0.0, 0.30)
    np.testing.assert_allclose(baseline, 1.0, rtol=0.0, atol=0.0)
    assert 1.0 < narrow < wide
