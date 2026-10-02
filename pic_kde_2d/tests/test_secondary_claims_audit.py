"""Regression checks for the Paper 1 secondary-claims audit."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
RESULT = (
    ROOT
    / "reproducibility"
    / "results"
    / "secondary_claims_audit"
    / "secondary_claims_audit.json"
)


@pytest.fixture(scope="module")
def audit() -> dict:
    return json.loads(RESULT.read_text(encoding="utf-8"))


def test_delta_f_guard_is_the_documented_marker_maximum(audit):
    delta = audit["delta_f"]
    assert delta["validity_metric"] == "max_p |w_delta,p| / mean_p |w_full,p|"
    assert delta["validity_threshold"] == 0.5
    assert "not an L2 error norm" in delta["validity_metric_scope"]
    rows = {row["alpha"]: row for row in delta["rows"]}
    assert rows[0.001]["maximum_ratio"]["value"] == pytest.approx(0.11380959404411854)
    assert rows[0.001]["maximum_ratio"]["ci_lo"] == pytest.approx(0.11035471539606227)
    assert rows[0.001]["maximum_ratio"]["ci_hi"] == pytest.approx(0.1196974799258608)
    assert rows[0.01]["median_first_crossing_time"] == pytest.approx(6.1)
    assert rows[0.1]["median_first_crossing_time"] == 0.0


def test_production_delta_f_is_projected_and_linearized(audit):
    delta = audit["delta_f"]
    assert "frozen/free-streaming" in delta["production_operator"]
    assert "target mode only" in delta["deposition_projection"]


def test_finite_amplitude_observable_resolves_rebound_not_trapping(audit):
    nonlinear = audit["finite_amplitude"]
    assert nonlinear["full_f_rebound_energy_t30_40_over_t12_18"]["ci_lo"] > 4.0
    assert nonlinear["linearized_delta_f_energy_t30_40_over_t12_18"]["ci_hi"] < 0.12
    assert nonlinear["late_energy_full_f_over_delta_f_t40_60"]["ci_lo"] > 20.0
    assert "trapping itself is not directly measured" in nonlinear["scope"]


def test_cost_claim_is_scoped_to_fixed_width_time_stepping(audit):
    cost = audit["cost"]
    assert cost["scope"]["filter_application_included"] is True
    assert cost["scope"]["bandwidth_selection_included"] is False
    assert cost["scope"]["filter_construction_included"] is False
    assert cost["c4_persistent_filter_bytes"] == 4608
    assert cost["c4_total_ratio_vs_cic"]["value"] == pytest.approx(1.0022634936894088)
    assert cost["c4_filter_ms_per_step"] == pytest.approx(1.3189918948337436)


def test_no_learned_controller_claim_survives_the_stop_rule(audit):
    controller = audit["adaptive_and_learned"]
    assert controller["decision"] == "STOP_LEARNING_ANALYTIC_H1_NEGATIVE"
    assert controller["learned_controller_was_trained"] is False
    assert controller["learned_model_artifact_count"] == 0
    assert controller["D100_density_mise_local_over_global"]["ci_lo"] > 1.0
    assert controller["D100_field_mse_local_over_global"]["ci_hi"] < 0.67
    assert controller["D100_protected_error_local_over_global"]["ci_lo"] > 1.0


def test_g0_numbers_are_predictions_for_an_unimplemented_profile(audit):
    g0 = audit["g0_locality"]
    assert g0["requested_free_width_cells"] == 384.0
    assert g0["implemented_width_cap_cells"] == 20.0
    assert g0["predicted_gain_at_cap"] == pytest.approx(1.0000329989595405)
    assert g0["status"] == "STOP_H2_VARIATIONAL_PROFILE_NOT_IMPLEMENTED"
    assert "not run in dynamics" in g0["scope"]
