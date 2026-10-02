"""Tests for Wu-Qin CV censoring and update semantics."""

import numpy as np

from pic_kde_2d.methods.wu_qin_bandwidth import (
    WuQinWidthUpdater,
    identify_cv_minimum,
    silverman_local_width_factors,
)


def test_interior_cv_minimum_is_identified():
    result = identify_cv_minimum([0.1, 0.2, 0.4], [3.0, 1.0, 2.0])
    assert result.identified
    assert result.status == "interior"
    assert result.bandwidth == 0.2


def test_decreasing_cv_curve_is_right_censored():
    result = identify_cv_minimum([0.1, 0.2, 0.4], [3.0, 2.0, 1.0])
    assert not result.identified
    assert result.status == "right_censored"
    assert result.bandwidth is None
    assert result.lower_bound == 0.4


def test_uniform_distribution_holds_previous_width():
    updater = WuQinWidthUpdater(current_width=0.3)
    event = updater.step(p_value=0.2)
    assert event["reason"] == "uniform_hold"
    assert not event["triggered_cv"]
    assert updater.current_width == 0.3


def test_pvalue_halving_accepts_finite_target_with_paper_relaxation():
    updater = WuQinWidthUpdater(current_width=0.2)
    selection = identify_cv_minimum([0.1, 0.4, 0.8], [2.0, 1.0, 1.5])
    event = updater.step(p_value=0.009, selection=selection)
    assert event["triggered_cv"]
    assert event["accepted_target"]
    assert event["reason"] == "finite_cv_target"
    assert abs(updater.current_width - (0.05 * 0.4 + 0.95 * 0.2)) < 1.0e-15


def test_right_censored_target_holds_without_dynamics_cap():
    updater = WuQinWidthUpdater(current_width=0.2)
    selection = identify_cv_minimum([0.1, 0.4, 0.8], [3.0, 2.0, 1.0])
    event = updater.step(p_value=0.009, selection=selection)
    assert event["triggered_cv"]
    assert not event["accepted_target"]
    assert event["reason"] == "right_censored_hold"
    assert updater.current_width == 0.2


def test_right_censored_target_uses_explicit_dynamics_cap():
    updater = WuQinWidthUpdater(current_width=0.2)
    selection = identify_cv_minimum([0.1, 0.4, 0.8], [3.0, 2.0, 1.0])
    event = updater.step(p_value=0.009, selection=selection, dynamics_cap=0.35)
    assert event["accepted_target"]
    assert event["reason"] == "right_censored_cv_capped_by_dynamics"
    assert abs(updater.current_width - (0.05 * 0.35 + 0.95 * 0.2)) < 1.0e-15


def test_silverman_local_factors_preserve_geometric_mean():
    factors = silverman_local_width_factors([0.25, 1.0, 4.0], sensitivity=0.5)
    assert abs(float(np.mean(np.log(factors)))) < 1.0e-15
    assert factors[0] > factors[1] > factors[2]
