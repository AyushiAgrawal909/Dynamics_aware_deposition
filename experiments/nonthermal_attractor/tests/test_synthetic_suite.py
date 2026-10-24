"""Reduced-size run of the synthetic suite (the full run is
scripts/run_t1_synthetic_suite.py)."""
import pytest

from attractor_analysis import suite


@pytest.mark.slow
@pytest.mark.parametrize("case,expected", [("S1", "NO_TAIL"), ("S3", "UNIDENTIFIABLE_DEGENERATE"),
                                           ("S4b", "NO_CLAIM")])
def test_fit_cases(case, expected):
    assert suite.evaluate_fit_case(case, 2048, 3)["verdict"] == expected


@pytest.mark.slow
def test_s2_identified_and_forecast_wins():
    assert suite.evaluate_fit_case("S2", 2048, 4)["verdict"] == "IDENTIFIED_AND_FORECAST_WINS"


@pytest.mark.slow
def test_s5_regulator_dependent():
    assert suite.evaluate_gate_case("S5", 2048, 4, 3)["verdict"] == "FAIL_REGULATOR_DEPENDENT"


@pytest.mark.slow
def test_s7():
    assert suite.evaluate_s7([2048, 512])["verdict"] == "ETA_MIN_RECOVERED_OR_FLAGGED"
