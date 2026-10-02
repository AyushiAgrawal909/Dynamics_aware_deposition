"""Regression tests for the Paper 1 density/noise metric audit."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest


REPRO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPRO))

import audit_noise_metrics as audit  # noqa: E402


class NoiseMetricAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.result = audit.main()

    def test_archived_reducers_reproduce(self) -> None:
        self.assertEqual(self.result["status"], "PASS")
        self.assertEqual(
            self.result["archive_reproduction"][
                "all_archived_central_values_and_bootstrap_intervals"
            ],
            "matched",
        )

    def test_landau_c4_protocol_split(self) -> None:
        rows = self.result["weak_landau_sigma1_comparison"]
        self.assertAlmostEqual(rows[0]["noise_power_reduction"], 2.559393052975873)
        self.assertAlmostEqual(rows[1]["noise_power_reduction"], 5.643223264073615)
        self.assertIn("per-cell", rows[0]["quiet_start_layout"])
        self.assertIn("globally", rows[1]["quiet_start_layout"])

    def test_binomial_reducer_is_not_the_5p64_value(self) -> None:
        row = self.result["weak_landau_sigma1_comparison"][2]
        self.assertAlmostEqual(row["noise_power_reduction"], 5.8715086641784495)
        self.assertIn("n=2", row["filter"])

    def test_charge_and_parseval(self) -> None:
        checks = self.result["normalization_and_parseval_checks"]
        self.assertLess(
            max(row["absolute_charge_defect"] for row in checks["charge_rows"]),
            5.0e-15,
        )
        self.assertLess(checks["parseval"]["absolute_parseval_difference"], 3.0e-15)
        self.assertGreater(checks["parseval"]["k0_power_in_control"], 0.0)
        self.assertGreater(
            checks["parseval"]["protected_mode_pair_power_in_control"], 0.0
        )

    def test_exact_self_term_matches_explicit_leave_one_out(self) -> None:
        rows = self.result["loo_self_contribution_check"][
            "controlled_explicit_comparison"
        ]
        self.assertEqual(len(rows), 3)
        self.assertLess(max(row["absolute_difference"] for row in rows), 1.0e-15)
        self.assertEqual(
            self.result["loo_self_contribution_check"]["archived_self_term_rows"],
            540,
        )

    def test_archived_metric_definitions_are_explicit(self) -> None:
        definitions = self.result["metric_definition"]
        self.assertIn("error power", definitions["noise_reduction"])
        self.assertIn("not an amplitude ratio", definitions["noise_reduction"])
        self.assertIn("all resolved modes", definitions["fourier_scope"])
        self.assertIn("protected mode is included", definitions["fourier_scope"])
        self.assertIn("does not gather to particles", definitions["particle_force_warning"])


if __name__ == "__main__":
    unittest.main()
