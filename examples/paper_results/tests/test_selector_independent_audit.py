from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest

import numpy as np


REPRO = Path(__file__).resolve().parents[1]
if str(REPRO) not in sys.path:
    sys.path.insert(0, str(REPRO))

import validate_selectors_independent as audit


RESULT = REPRO / "results/selector_independent_audit/selector_independent_audit.json"


class SelectorIndependentAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.result = json.loads(RESULT.read_text(encoding="utf-8"))
        cls.rows = {row["method"]: row for row in cls.result["rows"]}

    def test_fixed_split_is_disjoint(self) -> None:
        train = set(self.result["fixed_split"]["train"])
        test = set(self.result["fixed_split"]["test"])
        self.assertFalse(train & test)
        self.assertEqual(train | test, set(range(10)))

    def test_common_candidate_budget_and_selected_widths(self) -> None:
        self.assertEqual(self.result["training_budget_each_data_selector"]["candidate_widths"], 22)
        self.assertEqual(self.rows["MISE held-out"]["sigma_x_dx"], 7.0)
        self.assertEqual(self.rows["LOO-CV held-out"]["sigma_x_dx"], 12.0)

    def test_heldout_mise_matches_hindsight_on_fixed_split(self) -> None:
        mise = self.rows["MISE held-out"]
        hindsight = self.rows["test-set hindsight optimum"]
        self.assertEqual(mise["sigma_x_dx"], hindsight["sigma_x_dx"])
        self.assertAlmostEqual(mise["test_error_regret_vs_hindsight"], 0.0, places=15)
        self.assertAlmostEqual(mise["test_noise_power_reduction"], 4.991993816760017, places=12)

    def test_silverman_per_direction_is_not_the_run_filter(self) -> None:
        isotropic = self.rows["Silverman isotropized x-width"]
        directional = self.rows["Silverman per direction"]
        self.assertNotEqual(isotropic["sigma_y_dy"], directional["sigma_y_dy"])
        self.assertAlmostEqual(directional["sigma_y_dy"], audit.SILVERMAN_Y_DY, places=15)
        self.assertGreater(isotropic["test_noise_power_reduction"], directional["test_noise_power_reduction"])

    def test_all_balanced_split_counts(self) -> None:
        for selector in ("MISE", "LOO-CV"):
            self.assertEqual(self.result["split_sensitivity"][selector]["n_balanced_splits"], 252)

    def test_transfer_is_normalized_and_anisotropic(self) -> None:
        transfer = audit.gaussian_transfer(64, 16, 2.0 * np.pi / (0.4 * 64), 0.5, 1.0, 0.5)
        self.assertEqual(float(transfer[0, 0]), 1.0)
        self.assertNotEqual(float(transfer[1, 0]), float(transfer[0, 1]))


if __name__ == "__main__":
    unittest.main()
