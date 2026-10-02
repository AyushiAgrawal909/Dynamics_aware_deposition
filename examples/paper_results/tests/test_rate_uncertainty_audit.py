import csv
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "results" / "rate_uncertainty_audit_v7"


def rows(path: Path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


class RateUncertaintyAuditTests(unittest.TestCase):
    def test_all_e2_histories_are_longitudinal_not_corner(self):
        inventory = rows(RESULT / "e2_measured_mode_inventory.csv")
        self.assertEqual(len(inventory), 140)
        self.assertEqual({(r["kx_mode"], r["ky_mode"]) for r in inventory}, {("12", "0")})
        self.assertEqual({r["ppc"] for r in inventory}, {"512"})

    def test_reproduction_mismatch_is_gate_label_only(self):
        comparison = rows(RESULT / "reproduction_comparison.csv")
        self.assertEqual(len(comparison), 14)
        failures = [r for r in comparison if r["status"] == "FAIL"]
        self.assertEqual([r["campaign"] for r in failures], ["strong_alpha0p5_fit5"])
        detail = json.loads((RESULT / "strong_alpha0p5_fit5_comparison.json").read_text())
        self.assertLessEqual(detail["max_abs_numeric_difference"], 4e-15)
        self.assertTrue(all("gate_status" in example for example in detail["examples"]))

    def test_strong_window_sensitivity_and_duration(self):
        expected = {
            "strong_alpha0p5_fit3": (1.0101957559748984, 1.2419626837237332),
            "strong_alpha0p5_fit5": (1.0311577271061798, 0.6557207075423289),
        }
        for campaign, (ratio, efoldings) in expected.items():
            table = rows(RESULT / campaign / "strong_landau_fidelity.csv")
            selected = next(r for r in table if r["sigma_dx"] == "1.10785")
            self.assertAlmostEqual(float(selected["gamma_ratio_to_cic"]), ratio, places=13)
            self.assertAlmostEqual(float(selected["model_efoldings"]), efoldings, places=13)
            self.assertLess(float(selected["model_efoldings"]), 2.0)

    def test_alpha025_actual_cv_width_is_present_and_unresolved(self):
        table = rows(RESULT / "strong_alpha0p25_cv_exact" / "strong_landau_fidelity.csv")
        selected = next(r for r in table if r["sigma_dx"] == "2.8208226")
        self.assertAlmostEqual(float(selected["gamma_ratio_to_cic"]), 1.0479130760490991, places=13)
        self.assertEqual(selected["gate_status"], "UNRESOLVED")
        self.assertEqual(selected["bootstrap_quality_fraction"], "0.0")

    def test_low_ppc_bootstrap_is_not_resolved(self):
        diagnostics = rows(RESULT / "bootstrap_diagnostics_v2.csv")
        low_ppc = [r for r in diagnostics if r["setting"].startswith("ppc32:") and int(r["n_boot"])]
        self.assertEqual(len(low_ppc), 8)
        self.assertLessEqual(max(float(r["valid_fraction"]) for r in low_ppc), 0.416)
        self.assertLessEqual(max(float(r["quality_fraction"]) for r in low_ppc), 0.028)


if __name__ == "__main__":
    unittest.main()
