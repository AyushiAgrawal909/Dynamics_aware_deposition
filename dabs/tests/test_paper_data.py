"""Reproduction of the paper's reference-assisted joint selection.

Uses the compact spectral statistics of the ten paired reconstruction seeds
(examples/two_stream_paper/paper_spectral_stats.npz); they were computed from
snapshots regenerated with make_snapshots.py, which reproduce every archived
per-seed density and field error of the 115 candidates to round-off.
"""

from pathlib import Path

import pytest

from dabs import Gaussian, Grid, Kernel, NoFilter
from dabs.config import load, run
from dabs.objectives import SnapshotObjective

HERE = Path(__file__).resolve().parents[1] / "examples" / "two_stream_paper"
STATS = HERE / "paper_spectral_stats.npz"
pytestmark = pytest.mark.skipif(not STATS.exists(), reason="paper statistics not available")


@pytest.fixture(scope="module")
def result():
    return run(load(HERE / "two_stream_snapshots.yaml"))


def test_selected_kernel_and_heldout_reductions(result):
    best = result.best
    assert best.kernel.filter.family == "binomial"
    assert (best.kernel.filter.p, best.kernel.filter.n) == (4, (8,))
    N_rho = result.best.evaluation["density"][0]
    N_E = result.best.evaluation["field"][0]
    assert abs((1 - 1 / N_rho) - 0.7644) < 5e-5        # paper: 76.44 %
    assert abs((1 - 1 / N_E) - 0.3760) < 5e-5          # paper: 37.60 %


def test_paper_gaussian_reduction_factors():
    grid = Grid([128, 16], [64.0, 16.0])
    obj = SnapshotObjective.from_statistics(grid, STATS, select=range(5), evaluate=range(5, 10))
    g = Kernel(1, Gaussian((0.0, 0.1716392815470568)))
    base = Kernel(1, NoFilter())
    assert abs(obj.reduction_interval(g, base, "density")[0] - 1.0772) < 5e-5
    assert abs(obj.reduction_interval(g, base, "field")[0] - 1.0363) < 5e-5


def test_unconstrained_choice_removes_growth(result):
    assert not result.unconstrained.admissible
    assert "marginal" in result.unconstrained.status
