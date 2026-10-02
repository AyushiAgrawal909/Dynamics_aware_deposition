"""Tests for the preregistered selector comparison contract."""

import json
from pathlib import Path

import numpy as np

from paper1_selector_comparison.scripts.reduce_noise_shards import (
    find_seed_spread_failures,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "paper1_selector_comparison" / "config" / "comparison_matrix.json"


def load_config():
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def test_matrix_has_frozen_shape():
    config = load_config()
    assert len(config["selectors"]) == 7
    assert len(config["benchmarks"]) == 2


def test_c4_widths_are_preregistered():
    config = load_config()
    c4 = next(row for row in config["selectors"] if row["id"] == "c4")
    assert c4["sigma_dx"] == {
        "weak_landau": 1.0,
        "two_stream_e2": 0.18445488487247555,
    }


def test_protocol_ppcs_are_frozen():
    config = load_config()
    benchmarks = {row["id"]: row for row in config["benchmarks"]}
    assert benchmarks["weak_landau"]["rate_protocol"]["ppc"] == 1024
    assert benchmarks["weak_landau"]["noise_protocol"]["ppc"] == 1024
    assert benchmarks["two_stream_e2"]["rate_protocol"]["ppc"] == 512
    assert benchmarks["two_stream_e2"]["noise_protocol"]["ppc"] == 32


def test_delta_f_is_not_ranked_as_selector():
    config = load_config()
    delta_f = next(row for row in config["selectors"] if row["id"] == "delta_f")
    assert delta_f["eligible"] is False


def test_statistical_gate_is_not_central_only():
    gates = load_config()["gates"]
    assert gates["require_rate_ci_inside_tolerance"] is True
    assert gates["require_noise_ci_lower_above_minimum"] is True


def test_silverman_rule_is_frozen_before_runs():
    config = load_config()
    silverman = next(row for row in config["selectors"] if row["id"] == "silverman")
    assert silverman["sigma_dx"]["weak_landau"] == 1.8329728493314705
    assert silverman["sigma_dx"]["two_stream_e2"] == 5.819326058515848
    assert "N_particles^(-1/6)" in silverman["formula"]


def test_naive_and_wu_qin_cv_are_separate_rows():
    selectors = {row["id"]: row for row in load_config()["selectors"]}
    assert selectors["loo_cv"]["expected_censoring"] == "right"
    assert selectors["wu_qin_ad_cv"]["initial_sigma_dx"] == 0.0
    rule = selectors["wu_qin_ad_cv"]["selection"]
    assert "p <= 0.01" in rule
    assert "factor of two" in rule
    assert "0.05" in rule


def test_safe_window_closed_interpretation_is_preregistered():
    text = (ROOT / "paper1_selector_comparison" / "README.md").read_text(encoding="utf-8")
    normalized = " ".join(text.split())
    assert "SAFE_WINDOW_CLOSED" in normalized
    assert "no statistically demonstrated smoothing benefit" in normalized


def test_e2_cv_precision_statement_is_numerically_correct():
    text = (ROOT / "paper1_selector_comparison" / "README.md").read_text(encoding="utf-8")
    normalized = " ".join(text.split())
    assert "representable in binary64" in normalized
    assert "underflows binary32" in normalized
    assert "must not be described as binary64 underflow" in normalized


def test_noise_reducer_rejects_zero_seed_variance():
    failures = find_seed_spread_failures({0.0: np.ones(10)})
    assert failures
    assert failures[0]["variance"] == 0.0


def test_noise_reducer_rejects_suspiciously_small_seed_variance():
    values = 1.0 + np.linspace(-1.0e-13, 1.0e-13, 10)
    assert find_seed_spread_failures({0.0: values})


def test_noise_reducer_accepts_stochastic_seed_variance():
    values = np.asarray([0.97, 1.02, 0.99, 1.04, 0.96])
    assert find_seed_spread_failures({0.0: values}) == []


def test_landau_noise_uses_rate_aligned_global_quiet_start():
    runner = (ROOT / "scripts" / "run_landau_mise_gate.py").read_text(encoding="utf-8")
    batch = (
        ROOT
        / "paper1_selector_comparison"
        / "slurm"
        / "stage1_landau_noise_packed_gpu.sbatch"
    ).read_text(encoding="utf-8")
    assert 'default="global"' in runner
    assert "--quiet-start-layout global" in batch
