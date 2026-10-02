"""Regression tests for the deterministic filter-transfer audit."""

from __future__ import annotations

import pathlib
import sys
import unittest

import numpy as np


AUDIT_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AUDIT_ROOT))

import measure_filter_transfers as audit  # noqa: E402


class FilterTransferAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.result = audit.run(subcells=8)

    def test_all_stages_match_independent_formula(self) -> None:
        for row in self.result["rows"]:
            self.assertAlmostEqual(row["W_implemented"], row["W_analytic"], places=12)
            self.assertAlmostEqual(
                row["density_ratio_measured"],
                row["density_ratio_predicted"],
                places=12,
            )
            self.assertAlmostEqual(
                row["grid_field_ratio_measured"],
                row["grid_field_ratio_predicted"],
                places=12,
            )
            self.assertAlmostEqual(
                row["particle_force_ratio_measured"],
                row["particle_force_ratio_predicted"],
                places=12,
            )

    def test_e2_p3_n4_anchor(self) -> None:
        anchor = self.result["e2_p3_n4"]
        self.assertAlmostEqual(
            anchor["measured_force_transfer"],
            0.9971375751242197,
            places=12,
        )
        self.assertLess(abs(anchor["relative_difference"]), 0.003)

    def test_both_axes_are_filtered_for_oblique_mode(self) -> None:
        row = next(
            row
            for row in self.result["rows"]
            if row["mode_case"] == "E2_oblique_m12_n1"
            and row["filter"] == "raw_binomial_n4"
        )
        x_only = (1.0 - np.sin(np.pi * 12 / 128) ** 2) ** 4
        self.assertLess(row["W_implemented"], x_only)
        self.assertEqual(row["passes_x"], 4)
        self.assertEqual(row["passes_y"], 4)


if __name__ == "__main__":
    unittest.main()
