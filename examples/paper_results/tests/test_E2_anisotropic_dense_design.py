from __future__ import annotations

import math
import json
from pathlib import Path
import sys
import unittest


REPRO = Path(__file__).resolve().parents[1]
if str(REPRO) not in sys.path:
    sys.path.insert(0, str(REPRO))

from e2_anisotropic_design import Q_BUDGET_GRID_TIME, make_design


RESULT = REPRO / "results/selector_independent_audit/e2_anisotropic_dense/anisotropic_dense_audit.json"


class E2AnisotropicDenseDesignTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rows = make_design()

    def test_design_size_and_unique_labels(self) -> None:
        self.assertEqual(len(self.rows), 85)
        self.assertEqual(len({row["label"] for row in self.rows}), 85)

    def test_every_point_respects_budget(self) -> None:
        self.assertLessEqual(max(row["q_total"] for row in self.rows), Q_BUDGET_GRID_TIME)
        self.assertTrue(all(row["q_total"] >= 0.0 for row in self.rows))

    def test_widths_reconstruct_q(self) -> None:
        coefficient_x = (2.0 * math.pi * 12.0 / 128.0) ** 2
        coefficient_y = (2.0 * math.pi / 16.0) ** 2
        for row in self.rows:
            reconstructed = (
                coefficient_x * row["sigma_x_dx"] ** 2
                + coefficient_y * row["sigma_y_dy"] ** 2
            )
            self.assertAlmostEqual(reconstructed, row["q_total"], places=15)

    def test_boundary_includes_both_axis_endpoints(self) -> None:
        boundary = [row for row in self.rows if row["q_fraction"] == 1.0]
        self.assertEqual(len(boundary), 21)
        self.assertEqual(boundary[0]["sigma_x_dx"], 0.0)
        self.assertEqual(boundary[-1]["sigma_y_dy"], 0.0)

    def test_heldout_result_selects_transverse_endpoint(self) -> None:
        result = json.loads(RESULT.read_text(encoding="utf-8"))
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["fixed_split"], {
            "train": [100, 101, 102, 103, 104],
            "test": [105, 106, 107, 108, 109],
        })
        for row in result["summary"]:
            self.assertEqual(row["train_selected_label"], "q100_x000")
            self.assertEqual(row["test_best_label"], "q100_x000")
            self.assertEqual(row["bootstrap_fraction_advantage_positive"], 1.0)
            self.assertEqual(row["train_selected_sigma_x_dx"], 0.0)
            self.assertAlmostEqual(
                row["train_selected_sigma_y_dy"], 0.24273459979982182, places=15
            )

    def test_heldout_result_does_not_claim_global_optimality(self) -> None:
        result = json.loads(RESULT.read_text(encoding="utf-8"))
        statement = result["optimality_statement"]
        self.assertIn("only discrete-design optimality", statement)
        self.assertIn("not continuous", statement)


if __name__ == "__main__":
    unittest.main()
