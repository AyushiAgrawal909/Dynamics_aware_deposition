from __future__ import annotations

import csv
import json
from pathlib import Path
import unittest


REPRO = Path(__file__).resolve().parents[1]
RESULT = REPRO / "results/e2_rate_estimator_diagnostic_20260911_r3"
SUMMARY = RESULT / "diagnostic_summary.json"
FIGURE_SOURCE = REPRO / "analyze_E2_beam_drift_scan.py"
AMENDMENT = (
    REPRO
    / "results/e2_beam_drift_scan_design_20260911/extension_budget_amendment_20260911.json"
)


class E2RateEstimatorDiagnosticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
        with (RESULT / "diagnostic_rate_ratios.csv").open(
            encoding="utf-8"
        ) as handle:
            cls.ratios = list(csv.DictReader(handle))

    def test_registered_result_is_explicitly_preserved(self) -> None:
        self.assertEqual(
            self.summary["status"],
            "POST_HOC_DIAGNOSTIC_REGISTERED_RESULTS_UNCHANGED",
        )
        self.assertAlmostEqual(
            self.summary["vbeam_080_cic_diagnosis"]["registered_gamma"],
            0.14474589901399387,
            places=14,
        )

    def test_exponential_mixture_reproduces_registered_estimator(self) -> None:
        row = self.summary["vbeam_080_cic_diagnosis"]
        self.assertAlmostEqual(row["individual_gamma_mean"], 0.12445080926380203)
        self.assertLess(
            abs(row["registered_minus_fitted_exponential_mixture"]), 2.0e-5
        )
        self.assertAlmostEqual(
            row[
                "fraction_of_registered_absolute_theory_gap_associated_with_estimator"
            ],
            0.7439014024103462,
        )
        self.assertLess(row["effective_seed_count_end"], row["effective_seed_count_start"])

    def test_diagnostic_ratios_are_not_substituted(self) -> None:
        selected = {
            row["arm"]: row
            for row in self.ratios
            if row["campaign"] == "registered_drift_scan_5201231"
            and row["v_beam"] == "0.8"
            and row["window"] == "registered_frozen"
        }
        self.assertAlmostEqual(
            float(selected["fixed_gaussian"]["ratio_of_individual_rate_means"]),
            0.9089110358946052,
        )
        self.assertAlmostEqual(
            float(selected["constrained_gaussian"]["ratio_of_individual_rate_means"]),
            0.9544602585726027,
        )

    def test_historical_source_revision_remains_unknown(self) -> None:
        comparison = self.summary["vbeam_080_campaign_comparison"]
        self.assertIsNone(comparison["source_revision_historical"])
        self.assertAlmostEqual(
            comparison["endpoint_effect_current_same_seeds"],
            0.0015985562550745736,
        )
        self.assertAlmostEqual(
            comparison["remaining_current_minus_historical_at_shared_endpoint"],
            0.016346401890474094,
        )

    def test_figure_wording_and_extension_scope(self) -> None:
        source = FIGURE_SOURCE.read_text(encoding="utf-8")
        self.assertIn("(c) Predicted admissible bandwidth", source)
        self.assertIn("No growing model branch;\\nmeasurement unavailable.", source)
        self.assertNotIn('marker="v"', source)
        amendment = json.loads(AMENDMENT.read_text(encoding="utf-8"))
        self.assertEqual(
            amendment["status"], "FROZEN_EXTENSION_BUDGET_BEFORE_EXTENSION_RESULTS"
        )
        self.assertEqual(amendment["extension_scope"]["histories"], 30)
        self.assertFalse(amendment["scientific_configuration_changed"])
        self.assertFalse(amendment["acceptance_rules_changed"])
        self.assertFalse(amendment["extension_results_inspected_when_frozen"])


if __name__ == "__main__":
    unittest.main()
