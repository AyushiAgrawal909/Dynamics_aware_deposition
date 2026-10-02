from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import unittest

import numpy as np


REPRO = Path(__file__).resolve().parents[1]
if str(REPRO) not in sys.path:
    sys.path.insert(0, str(REPRO))

from analyze_E2_beam_drift_scan import bootstrap_drift, fit_quality


DESIGN = REPRO / "results/e2_beam_drift_scan_design_20260911"
TRANSFER = REPRO / "results/e2_beam_drift_scan_transfer_20260911"
CONFIG = DESIGN / "frozen_campaign.json"
CPU_LAUNCHER = REPRO / "meluxina/submit_E2_beam_drift_scan_cpu.sbatch"
TIMEOUT_RECORD = DESIGN / "campaign_timeout_5201231.json"
TIMEOUT_ANALYSIS = REPRO / "results/e2_beam_drift_scan_analysis_20260911_timeout_r6/E2_beam_drift_scan.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class E2BeamDriftScanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = json.loads(CONFIG.read_text(encoding="utf-8"))
        cls.preflight = json.loads(
            (DESIGN / "preflight_summary.json").read_text(encoding="utf-8")
        )
        cls.transfer = json.loads(
            (TRANSFER / "transfer_summary.json").read_text(encoding="utf-8")
        )
        cls.timeout = json.loads(TIMEOUT_RECORD.read_text(encoding="utf-8"))
        cls.timeout_analysis = json.loads(TIMEOUT_ANALYSIS.read_text(encoding="utf-8"))

    def test_design_was_frozen_before_results(self) -> None:
        self.assertEqual(self.config["status"], "FROZEN_BEFORE_SIMULATION_RESULTS")
        self.assertFalse(self.preflight["simulation_results_read"])
        self.assertFalse(self.preflight["new_simulations_run"])
        self.assertEqual(
            sha256(CONFIG),
            "e664460d392a216b953bc87d3859c3bce9e1cce2f1652ab9417c1a58b3baf446",
        )

    def test_branch_and_drift_anchors(self) -> None:
        self.assertEqual(
            self.config["symmetric_beam_drifts"],
            [0.60, 0.70, 0.76, 0.80, 0.818],
        )
        self.assertAlmostEqual(
            self.config["model"]["marginal_v_beam"],
            0.8191540596575886,
            places=13,
        )
        self.assertLess(
            self.preflight["maximum_q8_l1024_vs_q10_l2048_gamma_difference"],
            1.0e-7,
        )
        self.assertFalse(self.config["model"]["cold_longitudinal_threshold_used"])

    def test_fixed_and_constrained_controls(self) -> None:
        rows = self.config["scan_rows"]
        self.assertEqual({row["fixed_sigma_y_dy"] for row in rows}, {0.2427346})
        self.assertEqual(
            {row["fixed_particle_force_W2"] for row in rows},
            {0.9909549371154771},
        )
        for row in rows:
            self.assertAlmostEqual(
                row["constrained_predicted_rate_ratio"], 0.95, places=11
            )
            self.assertLess(
                row["constrained_sigma_y_dy"],
                row["sigma_y_10pct_boundary_dy"],
            )
        self.assertEqual(rows[-1]["fixed_predicted_rate_ratio"], 0.0)

    def test_actual_initializer_and_transfer_are_invariant(self) -> None:
        loading = self.preflight["loading_invariance"]
        self.assertTrue(all(loading["exact_equalities"].values()))
        self.assertEqual(loading["negative_beam_particles"], 524288)
        self.assertEqual(loading["positive_beam_particles"], 524288)
        self.assertLess(loading["maximum_thermal_residual_difference"], 3.0e-16)
        self.assertEqual(self.transfer["status"], "PASS")
        self.assertLess(self.transfer["maximum_absolute_error_W2"], 1.0e-14)

    def test_sample_windows_and_budget_are_fixed(self) -> None:
        self.assertEqual(self.config["seeds"], list(range(200, 210)))
        self.assertEqual(self.config["budget"]["histories"], 150)
        self.assertEqual(
            self.config["budget"]["configurations_before_seed_replication"], 15
        )
        self.assertEqual(self.config["budget"]["requested_wall_hours_hard_max"], 4.0)
        for row in self.config["scan_rows"]:
            self.assertGreaterEqual(row["predicted_constrained_fit_efoldings"], 5.0)
            self.assertAlmostEqual(row["t_end"] - row["fit_t_stop"], 2.0, places=12)

    def test_launcher_does_not_precreate_runner_output(self) -> None:
        launcher = CPU_LAUNCHER.read_text(encoding="utf-8")
        self.assertNotIn('mkdir -p "${out}/numba"', launcher)
        self.assertNotIn('>"${out}/run.log"', launcher)
        self.assertIn('cache=${ROOT}/numba_cache/${work_key}', launcher)
        self.assertIn('log=${ROOT}/runner_logs/${work_key}.log', launcher)
        self.assertIn('--output "${out}"', launcher)

    def test_timeout_is_not_silently_extended(self) -> None:
        self.assertEqual(self.timeout["job_state"], "TIMEOUT")
        self.assertEqual(self.timeout["complete_histories"], 120)
        self.assertEqual(self.timeout["missing_histories"], 30)
        self.assertEqual(self.timeout["missing_v_beam"], [0.818])
        self.assertFalse(self.timeout["continuation_submitted"])
        self.assertGreaterEqual(
            self.timeout["wall_hours_consumed"],
            self.config["budget"]["requested_wall_hours_hard_max"],
        )

    def test_timeout_censored_analysis_preserves_complete_gates(self) -> None:
        completeness = self.timeout_analysis["measurement_completeness"]
        self.assertEqual(completeness["complete_histories"], 120)
        self.assertEqual(completeness["unresolved_drift_values"], [0.818])
        complete = [
            row
            for row in self.timeout_analysis["rows"]
            if row["measurement_status"] == "COMPLETE"
        ]
        missing = [
            row
            for row in self.timeout_analysis["rows"]
            if row["measurement_status"] != "COMPLETE"
        ]
        self.assertEqual(len(complete), 12)
        self.assertEqual(len(missing), 3)
        self.assertTrue(
            all(
                row["rate_preservation_gate"] == "PASS"
                for row in complete
                if row["arm"] != "cic"
            )
        )
        self.assertTrue(
            all(
                row["rate_preservation_gate"] == "UNRESOLVED_BUDGET_TIMEOUT"
                for row in missing
            )
        )
        cic_0800 = next(
            row
            for row in complete
            if row["arm"] == "cic" and row["v_beam"] == 0.8
        )
        self.assertAlmostEqual(cic_0800["gamma"], 0.14474589901399387, places=14)
        self.assertFalse(cic_0800["prediction_inside_measured_95pct_interval"])

    def test_paired_reducer_on_exact_exponentials(self) -> None:
        times = np.linspace(0.0, 4.0, 81)
        seed_scale = np.linspace(0.9, 1.1, 10)[:, None]
        groups = []
        for gamma in (0.2, 0.19, 0.18):
            groups.append(
                {
                    "times": times,
                    "amplitudes": seed_scale * np.exp(gamma * times)[None, :],
                }
            )
        rates, ratios = bootstrap_drift(groups, 1.0, 4.0, 100, 99)
        for values in rates + ratios:
            self.assertEqual(values.size, 100)
        self.assertTrue(np.allclose(rates[0], 0.2, atol=2.0e-15))
        self.assertTrue(np.allclose(ratios[1], 0.95, atol=2.0e-14))
        self.assertTrue(np.allclose(ratios[2], 0.90, atol=2.0e-14))
        self.assertAlmostEqual(
            fit_quality(groups[0]["amplitudes"], times, 1.0, 4.0),
            1.0,
            places=14,
        )


if __name__ == "__main__":
    unittest.main()
