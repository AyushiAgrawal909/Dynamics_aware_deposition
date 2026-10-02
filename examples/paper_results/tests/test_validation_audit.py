import csv
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "results" / "validation_audit_v9"


def rows():
    with (RESULT / "validation_status.csv").open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


class ValidationAuditTests(unittest.TestCase):
    def setUp(self):
        self.rows = rows()
        self.by_condition = {row["condition"]: row for row in self.rows}

    def test_no_weak_landau_condition_passes_complete_gate(self):
        weak = [row for row in self.rows if row["benchmark"] == "weak Landau"]
        self.assertTrue(weak)
        self.assertFalse(any(row["overall_status"] == "PASS" for row in weak))
        self.assertEqual(self.by_condition["weak_c4"]["rate_gate"], "PASS")
        self.assertEqual(self.by_condition["weak_c4"]["duration_gate"], "FAIL")
        self.assertEqual(self.by_condition["weak_c4"]["overall_status"], "FAIL")

    def test_two_sided_rate_gate_is_applied(self):
        self.assertEqual(self.by_condition["weak_raw_n4"]["rate_gate"], "PASS")
        self.assertEqual(self.by_condition["weak_raw_n8"]["rate_gate"], "FAIL")
        self.assertGreater(float(self.by_condition["weak_raw_n8"]["rate_ratio_ci_hi"]), 1.10)
        self.assertEqual(self.by_condition["e2_gaussian_sigma01987"]["rate_gate"], "FAIL")
        self.assertLess(float(self.by_condition["e2_gaussian_sigma01987"]["rate_ratio_ci_lo"]), 0.90)

    def test_selected_loo_width_has_its_own_direct_rate(self):
        selected = self.by_condition["weak_loo_selected"]
        proxy = self.by_condition["weak_loo_proxy"]
        self.assertEqual(selected["setting"], "selected sigma=7.244293147028428")
        self.assertAlmostEqual(float(selected["rate_ratio"]), 1.822221359125454)
        self.assertEqual(selected["rate_gate"], "FAIL")
        self.assertEqual(selected["duration_gate"], "FAIL")
        self.assertEqual(selected["signal_resolvability"], "FAIL")
        self.assertEqual(selected["raw_traceability"], "PASS")
        self.assertEqual(proxy["setting"], "sigma=6.61")
        self.assertEqual(proxy["rate_gate"], "FAIL")

    def test_fresh_v2_confirmation_keeps_baseline_duration_separate(self):
        cic = self.by_condition["weak_confirm_v2_cic_absolute"]
        c4 = self.by_condition["weak_confirm_v2_c4"]
        lscv = self.by_condition["weak_confirm_v2_lscv"]
        self.assertEqual(cic["rate_gate"], "PASS")
        self.assertEqual(cic["duration_gate"], "FAIL")
        self.assertEqual(cic["signal_resolvability"], "PASS")
        self.assertAlmostEqual(float(cic["model_efoldings"]), 1.7213861112191615)
        self.assertEqual(c4["rate_gate"], "PASS")
        self.assertEqual(c4["duration_gate"], "PASS")
        self.assertEqual(c4["baseline_duration_gate"], "FAIL")
        self.assertEqual(c4["overall_status"], "FAIL")
        self.assertAlmostEqual(float(c4["rate_ratio"]), 0.9894153689137298)
        self.assertAlmostEqual(float(c4["rate_ratio_ci_lo"]), 0.9678835537488053)
        self.assertAlmostEqual(float(c4["rate_ratio_ci_hi"]), 1.0138274034763808)
        self.assertEqual(lscv["rate_gate"], "FAIL")
        self.assertEqual(lscv["duration_gate"], "PASS")

    def test_recovered_corner_and_longitudinal_scope_are_separate(self):
        corner = self.by_condition["e2_c4_corner"]
        self.assertEqual(corner["measured_mode"], "(12,1)")
        self.assertEqual(corner["raw_traceability"], "PASS")
        self.assertEqual(corner["overall_status"], "PASS")
        self.assertEqual(corner["baseline_duration_gate"], "PASS")
        self.assertAlmostEqual(float(corner["fit_start_snr"]), 11.213402344678167)
        for name in ("e2_comp_p2_n2", "e2_comp_p3_n4"):
            row = self.by_condition[name]
            self.assertEqual(row["measured_mode"], "(12,0)")
            self.assertEqual(row["overall_status"], "PASS")
        for name in ("e2_comp_p2_n2_corner", "e2_comp_p3_n4_corner"):
            row = self.by_condition[name]
            self.assertEqual(row["measured_mode"], "(12,1)")
            self.assertEqual(row["raw_traceability"], "PASS")
            self.assertEqual(row["rate_gate"], "PASS")
            self.assertEqual(row["baseline_duration_gate"], "PASS")
            self.assertEqual(row["overall_status"], "PASS")
        for name in ("e2_comp_p1_n1_corner", "e2_comp_p2_n4_corner"):
            self.assertEqual(self.by_condition[name]["rate_gate"], "FAIL")

    def test_compensated_prediction_accuracy_is_not_rate_preservation(self):
        for name in (
            "e2_comp_p1_n1_corner",
            "e2_comp_p2_n2_corner",
            "e2_comp_p2_n4_corner",
            "e2_comp_p3_n4_corner",
        ):
            row = self.by_condition[name]
            prediction = float(row["prediction_only_ratio"])
            self.assertFalse(
                float(row["rate_ratio_ci_lo"]) <= prediction <= float(row["rate_ratio_ci_hi"])
            )
            self.assertIn("outside its 95% interval", row["note"])
        self.assertEqual(self.by_condition["e2_comp_p2_n2_corner"]["rate_gate"], "PASS")
        self.assertEqual(self.by_condition["e2_comp_p3_n4_corner"]["rate_gate"], "PASS")

    def test_filtered_e2_uses_its_own_duration(self):
        raw = self.by_condition["e2_raw_n1"]
        self.assertAlmostEqual(float(raw["model_efoldings"]), 0.08721732740841422)
        self.assertLess(float(raw["model_efoldings"]), 2.0)
        p3 = self.by_condition["e2_comp_p3_n4"]
        self.assertAlmostEqual(float(p3["model_efoldings"]), 5.3128125590385755, places=12)

    def test_newly_recovered_delta_f_histories_close_traceability_gaps(self):
        gaussian = self.by_condition["e2_gaussian_sigma01845"]
        reference = self.by_condition["df_e2_m8_reference"]
        self.assertEqual(gaussian["raw_traceability"], "PASS")
        self.assertEqual(gaussian["overall_status"], "PASS")
        self.assertEqual(reference["raw_traceability"], "PASS")
        self.assertEqual(reference["overall_status"], "PASS")
        self.assertIn("recovered_delta_f_m12_sigma_4872745_20260911", gaussian["evidence"])
        self.assertIn("recovered_delta_f_reference_4872599_20260911", reference["evidence"])

    def test_new_mode_passes_and_silverman_is_not_a_measurement(self):
        new_mode = self.by_condition["e2_c4_new_mode_m10n1"]
        self.assertEqual(new_mode["measured_mode"], "(10,1)")
        self.assertEqual(new_mode["overall_status"], "PASS")
        self.assertAlmostEqual(float(new_mode["rate_ratio"]), 0.9891181308879333, places=13)
        silverman = self.by_condition["e2_silverman_erased"]
        self.assertEqual(silverman["measured_mode"], "not measured")
        self.assertEqual(silverman["model_admissibility"], "FAIL")
        self.assertEqual(silverman["rate_gate"], "NOT_MEASURED")
        self.assertEqual(silverman["overall_status"], "FAIL")

    def test_summary_matches_table(self):
        summary = json.loads((RESULT / "validation_summary.json").read_text())
        counts = {status: 0 for status in ("PASS", "FAIL", "UNRESOLVED")}
        for row in self.rows:
            counts[row["overall_status"]] += 1
        self.assertEqual(summary["counts"], counts)
        self.assertEqual(counts, {"PASS": 8, "FAIL": 31, "UNRESOLVED": 1})
        self.assertEqual(summary["weak_landau_overall_passes"], [])
        self.assertEqual(
            summary["e2_binding_corner_overall_passes"],
            ["e2_c4_corner", "e2_comp_p2_n2_corner", "e2_comp_p3_n4_corner"],
        )

    def test_direct_result_anchors_are_preserved(self):
        selected = self.by_condition["weak_loo_selected"]
        confirm = self.by_condition["weak_confirm_v2_c4"]
        self.assertAlmostEqual(float(selected["rate_ratio"]), 1.822221359125454)
        self.assertAlmostEqual(float(selected["rate_ratio_ci_lo"]), 1.4893354835524286)
        self.assertAlmostEqual(float(selected["rate_ratio_ci_hi"]), 2.213983246660343)
        self.assertAlmostEqual(float(confirm["rate_ratio"]), 0.9894153689137298)
        self.assertAlmostEqual(float(confirm["rate_ratio_ci_lo"]), 0.9678835537488053)
        self.assertAlmostEqual(float(confirm["rate_ratio_ci_hi"]), 1.0138274034763808)
        self.assertTrue(selected["evidence"])
        self.assertTrue(confirm["evidence"])


if __name__ == "__main__":
    unittest.main()
