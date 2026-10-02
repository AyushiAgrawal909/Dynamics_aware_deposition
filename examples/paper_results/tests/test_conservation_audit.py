#!/usr/bin/env python3
"""Regression tests for the Paper 1 conservation audit."""

from __future__ import annotations

import pathlib
import sys
import tempfile
import unittest

import numpy as np


HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import analyze_conservation as audit  # noqa: E402
import run_conservation_small_tests as small  # noqa: E402


class ConservationAuditTests(unittest.TestCase):
    def test_archived_energy_matrix_and_seed_level_bootstrap(self) -> None:
        per_seed, aggregate, _, metadata = audit.analyze_energy(nboot=400)
        self.assertEqual(len(per_seed), 90)
        self.assertEqual(len(aggregate), 9)
        self.assertTrue(all(row["max_abs_charge_change_over_initial_particle_charge"] == 0.0 for row in per_seed))
        by_case = {(row["benchmark"], row["case"]): row for row in aggregate}
        self.assertFalse(by_case[("landau_weak", "c4")]["slope_ci_excludes_zero"])
        self.assertTrue(by_case[("e2", "cic")]["slope_ci_excludes_zero"])
        self.assertIn("temporal samples are not resampled", metadata["uncertainty_definition"])

    def test_grad_h_components_are_not_substituted_for_norms(self) -> None:
        _, _, metadata = audit.analyze_momentum()
        ratios = metadata["grad_h_ratios"]
        self.assertAlmostEqual(ratios["grad_h_x_drift_on_over_off"], 0.5717812769266952, places=12)
        self.assertAlmostEqual(ratios["grad_h_norm_drift_on_over_off"], 1.0000136794945573, places=12)
        self.assertAlmostEqual(ratios["discontinuous_over_smooth_off_norm_drift"], 16.028417695339225, places=11)
        self.assertAlmostEqual(ratios["discontinuous_over_smooth_off_x_drift"], 1817.3207637235944, places=8)

    def test_hypre_force_floor_is_not_tolerance_controlled(self) -> None:
        rows, summary = audit.analyze_tolerance_scan()
        self.assertEqual(len(rows), 4)
        self.assertLess(summary["manual_residual_range"][0], 1.0e-11)
        self.assertGreater(summary["manual_residual_range"][1], 9.0e-6)
        self.assertLess(summary["force_x_fractional_span"], 1.0e-5)

    def test_matched_exact_operators_are_skew_to_roundoff(self) -> None:
        rows = small.operator_tests()
        self.assertEqual(len(rows), 6)
        self.assertLess(max(row["relative_force_norm"] for row in rows), 4.0e-17)

    def test_complete_matched_particle_force_is_at_roundoff(self) -> None:
        rows = small.particle_force_tests()
        self.assertEqual([row["sigma_cells"] for row in rows], [0.0, 0.5, 3.0])
        self.assertLess(max(row["normalized_total_force"] for row in rows), 5.0e-16)

    def test_short_pic_scan_has_second_order_successive_differences(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            rows = small.time_step_tests(pathlib.Path(directory))
        orders = [
            float(row["successive_refinement_order"])
            for row in rows
            if row["successive_refinement_order"] != ""
        ]
        self.assertEqual(len(orders), 6)
        self.assertTrue(np.all(np.asarray(orders) > 1.9))
        self.assertTrue(np.all(np.asarray(orders) < 2.4))


if __name__ == "__main__":
    unittest.main()
