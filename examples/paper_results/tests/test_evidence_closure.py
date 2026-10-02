import csv
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "results/evidence_closure_20260911"


class EvidenceClosureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.summary = json.loads((RESULT / "evidence_closure.json").read_text(encoding="utf-8"))

    def test_historical_per_cell_scope_is_narrowed(self):
        support = self.summary["reconstructed_per_cell_support"]
        self.assertEqual(support["distinct_longitudinal_nodes"], 32)
        self.assertEqual(support["nodes_at_or_above_resonance"], 0)
        self.assertFalse(support["resonance_covered"])
        with (RESULT / "historical_landau_loading.csv").open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        table4 = next(row for row in rows if row["job_or_campaign"] == "4867515")
        self.assertEqual(table4["same_as_rejected_confirmation"], "YES")
        self.assertIn("density squared-error power", table4["quantity"])

    def test_e2_c4_has_direct_signal_evidence(self):
        signal = self.summary["e2_c4_signal_resolvability"]
        self.assertEqual(signal["signal_resolvability"], "PASS")
        self.assertAlmostEqual(signal["floor"]["fit_start_snr"], 11.213402344678167)
        self.assertAlmostEqual(signal["candidate_fitted_efoldings"], 4.113115973142947)
        self.assertAlmostEqual(signal["shared_cic_baseline_fitted_efoldings"], 4.483681269108223)
        self.assertEqual(signal["paired_bootstrap_valid"], 5000)

    def test_prediction_calibration_is_separate(self):
        with (RESULT / "e2_compensated_prediction_accuracy.csv").open(
            encoding="utf-8", newline=""
        ) as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 4)
        self.assertTrue(all(row["prediction_inside_measured_95pct_interval"] == "False" for row in rows))
        by_label = {row["label"]: row for row in rows}
        self.assertEqual(by_label["p2n2"]["measured_rate_preservation_ci_pass"], "True")
        self.assertEqual(by_label["p3n4"]["measured_rate_preservation_ci_pass"], "True")
        self.assertAlmostEqual(float(by_label["p2n2"]["relative_discrepancy_percent"]), 0.9318056242667794)
        self.assertAlmostEqual(float(by_label["p3n4"]["relative_discrepancy_percent"]), 0.5451801505383158)


if __name__ == "__main__":
    unittest.main()
