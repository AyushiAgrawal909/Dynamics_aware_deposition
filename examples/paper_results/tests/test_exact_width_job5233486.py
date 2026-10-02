import csv
import json
from pathlib import Path
import unittest


REPRO = Path(__file__).resolve().parents[1]
PACKAGE = REPRO.parents[1]
CAMPAIGN = (
    PACKAGE
    / "campaigns"
    / "E2_q050_x000_exact_width_vb080_v1_job5233486"
)
RAW = CAMPAIGN / "raw_remote_output"


class ExactWidthJob5233486Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = json.loads((RAW / "frozen_config.json").read_text(encoding="utf-8"))
        cls.summary = json.loads(
            (RAW / "analysis/summary.json").read_text(encoding="utf-8")
        )
        cls.audit = json.loads(
            (RAW / "runs/cic/reuse_audit.json").read_text(encoding="utf-8")
        )

    def test_frozen_estimands_and_verdicts(self):
        summary = self.summary
        self.assertEqual(summary["campaign_id"], "E2_q050_x000_exact_width_vb080_v1")
        self.assertEqual(summary["fit_window"], [24.32, 69.16])
        self.assertEqual(summary["n_paired_seeds"], 10)
        self.assertAlmostEqual(summary["primary_individual_rate_mean_ratio"], 0.9556447297592622)
        self.assertEqual(
            summary["primary_paired_95pct_interval"],
            [0.9452733092216409, 0.9640059783937596],
        )
        self.assertTrue(summary["primary_preservation_established"])
        self.assertAlmostEqual(summary["secondary_ensemble_amplitude_ratio"], 0.9706344447819394)
        self.assertEqual(
            summary["secondary_paired_95pct_interval"],
            [0.9553563586841584, 0.9737072604223515],
        )
        self.assertTrue(summary["secondary_preservation_established"])
        self.assertAlmostEqual(summary["model_operating_ratio"], 0.9513008502587932)

    def test_all_histories_and_registered_quality_gates_are_retained(self):
        self.assertEqual(self.config["seeds"], list(range(200, 210)))
        self.assertTrue(self.config["acceptance"]["no_seed_rejection"])
        histories = sorted((RAW / "runs").glob("*/seed*/mode_history.npz"))
        self.assertEqual(len(histories), 20)
        self.assertEqual(self.summary["status"], "COMPLETE")
        self.assertTrue(
            all(
                all(gates.values())
                for gates in self.summary["quality_gates"].values()
            )
        )
        with (RAW / "analysis/individual_seed_rates.csv").open(
            newline="", encoding="utf-8"
        ) as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 20)
        self.assertEqual({int(row["seed"]) for row in rows}, set(range(200, 210)))

    def test_production_runtime_reconstruction_and_array_pairing(self):
        self.assertEqual(
            self.audit["status"],
            "PASS_PRODUCTION_RUNTIME_RECONSTRUCTION_AND_CIC_QUALITY",
        )
        self.assertFalse(self.audit["historical_saved_particle_array_hash_available"])
        for seed in range(200, 210):
            archived = self.audit["seeds"][str(seed)]
            result = json.loads(
                (
                    RAW
                    / f"runs/q050_x000/seed{seed}/result.json"
                ).read_text(encoding="utf-8")
            )
            self.assertTrue(
                result["initial_state_matches_production_runtime_reconstruction"]
            )
            self.assertEqual(
                result["initial_state_sha256"],
                archived["production_runtime_reconstructed_initial_state_sha256"],
            )
            for name in ("x", "y", "vx", "vy", "w_full", "w_delta"):
                self.assertEqual(
                    result["initial_array_sha256"][name],
                    archived[f"{name}_sha256"],
                )


if __name__ == "__main__":
    unittest.main()
