from __future__ import annotations

import json
from pathlib import Path
import unittest


REPRO = Path(__file__).resolve().parents[1]
RESULT = (
    REPRO
    / "results/selector_independent_audit/e2_new_mode_m10n1/E2_new_protected_mode.json"
)


class E2NewProtectedModeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.result = json.loads(RESULT.read_text(encoding="utf-8"))
        cls.rows = {row["label"]: row for row in cls.result["rows"]}

    def test_new_mode_and_seeds_are_independent(self) -> None:
        self.assertEqual(self.result["measured_mode"], [10, 1])
        self.assertEqual(
            self.result["seed_partition"]["new_seed_ids"], list(range(100, 110))
        )
        self.assertIn("disjoint", self.result["seed_partition"]["relationship_to_prior_campaigns"])

    def test_candidate_passes_fixed_gates(self) -> None:
        candidate = self.rows["c4"]
        self.assertEqual(self.result["status"], "PASS")
        for gate in ("rate_gate", "duration_gate", "signal_gate"):
            self.assertEqual(candidate[gate], "PASS")
        self.assertGreaterEqual(candidate["ratio_ci_lo"], 0.90)
        self.assertLessEqual(candidate["ratio_ci_hi"], 1.10)
        self.assertGreaterEqual(candidate["model_efoldings"], 2.0)
        self.assertGreaterEqual(candidate["fit_r_squared"], 0.98)
        self.assertGreaterEqual(candidate["fit_start_snr"], 5.0)

    def test_measured_ratio_anchor(self) -> None:
        candidate = self.rows["c4"]
        self.assertAlmostEqual(candidate["gamma_ratio_to_cic"], 0.9891181308879333, places=13)
        self.assertAlmostEqual(candidate["ratio_ci_lo"], 0.9889281872553365, places=13)
        self.assertAlmostEqual(candidate["ratio_ci_hi"], 0.9892525754644281, places=13)

    def test_zero_fit_parameter_prediction_agrees(self) -> None:
        prediction = self.result["grid_time_discrete_prediction"]
        self.assertEqual(prediction["fitted_parameters"], 0)
        self.assertAlmostEqual(prediction["candidate_ratio"], 0.9892473915549623, places=13)
        self.assertLess(
            abs(prediction["candidate_ratio"] - self.rows["c4"]["gamma_ratio_to_cic"]),
            2.0e-4,
        )


if __name__ == "__main__":
    unittest.main()
