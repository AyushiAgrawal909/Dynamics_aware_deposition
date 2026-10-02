import hashlib
import io
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest
from contextlib import redirect_stdout

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parents[1]
CONFIG = ROOT / "weak_landau_confirmatory_v1.json"
DIAGNOSIS = ROOT / "results/weak_landau_confirmatory_diagnosis"
PREFLIGHT = ROOT / "results/weak_landau_confirmatory_preflight_v5/preflight.json"
FAILED_V1_DIAGNOSIS = (
    ROOT
    / "results/weak_landau_confirmatory_stage1_failure_diagnosis_20260911_v4"
    / "stage1_loading_failure_diagnosis.json"
)
GLOBAL_PILOT_PREFLIGHT = (
    ROOT
    / "results/weak_landau_global_loading_pilot_remote_preflight_20260911"
    / "preflight.json"
)
V2_CONFIG = ROOT / "weak_landau_confirmatory_v2_global_velocity.json"
V2_PREFLIGHT = (
    ROOT
    / "results/weak_landau_confirmatory_v2_preflight_local_20260911_exact"
    / "preflight.json"
)
V2_STAGE1 = (
    ROOT
    / "results/weak_landau_confirmatory_v2_stage1_windows_20260911"
    / "window_decisions.json"
)
V2_STAGE2 = (
    ROOT
    / "results/weak_landau_confirmatory_v2_stage2_analysis_20260911"
    / "confirmatory_result.json"
)
V2_COMMON_WINDOW = (
    ROOT
    / "results/weak_landau_confirmatory_v2_common_window_sensitivity_20260911"
    / "common_window_sensitivity.json"
)
V2_WINDOW_DIAGNOSTIC = (
    ROOT
    / "results/weak_landau_confirmatory_v2_window_diagnostic_20260911_r2"
)
RUNNER = WORKSPACE / "pic_kde_2d/scripts/run_weak_landau_confirmatory.py"
V2_RUNNER = WORKSPACE / "pic_kde_2d/scripts/run_weak_landau_confirmatory_v2.py"
sys.path.insert(0, str(ROOT))
import analyze_weak_landau_confirmatory as confirmatory  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


class WeakLandauConfirmatoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = json.loads(CONFIG.read_text(encoding="utf-8"))
        cls.diagnosis = json.loads(
            (DIAGNOSIS / "window_diagnosis.json").read_text(encoding="utf-8")
        )
        cls.preflight = json.loads(PREFLIGHT.read_text(encoding="utf-8"))

    def test_historical_end_reasons_and_efoldings(self):
        cic = self.diagnosis["arms"]["CIC"]["fit"]
        c4 = self.diagnosis["arms"]["C4"]["fit"]
        self.assertEqual(cic["ending_reason"], "noise_threshold")
        self.assertEqual(c4["ending_reason"], "noise_threshold")
        self.assertAlmostEqual(cic["censor_t"], 26.3)
        self.assertAlmostEqual(c4["censor_t"], 29.0)
        self.assertAlmostEqual(cic["fit_t_stop"], 23.8)
        self.assertAlmostEqual(c4["fit_t_stop"], 26.5)
        self.assertAlmostEqual(cic["model_efoldings"], 1.3110652354002463)
        self.assertAlmostEqual(c4["model_efoldings"], 1.4907932086192146)
        self.assertNotEqual(
            cic["model_efoldings"], cic["signal_to_threshold_capacity_from_fit_start"]
        )

    def test_complex_histories_are_preserved(self):
        path = DIAGNOSIS / "weak_landau_complex_histories_and_envelopes.npz"
        with np.load(path, allow_pickle=False) as archive:
            self.assertEqual(archive["times"].shape, (901,))
            self.assertEqual(archive["cic_complex_mode_histories"].shape, (10, 901))
            self.assertEqual(archive["c4_complex_mode_histories"].shape, (10, 901))
            np.testing.assert_array_equal(
                archive["cic_coherent_complex_mode"],
                np.mean(archive["cic_complex_mode_histories"], axis=0),
            )

    def test_reference_two_efolding_time_and_fit_cap(self):
        gamma = abs(self.config["numerical_reference"]["continuum_gamma_for_context_only"])
        deadline = self.config["fit_rule"]["fit_t_start"] + 2.0 / gamma
        self.assertAlmostEqual(deadline, 35.2443934395903)
        self.assertGreater(self.config["fit_rule"]["fit_t_stop_max"], deadline)
        old_maximum = gamma * (30.0 - 5.0)
        self.assertLess(old_maximum, 2.0)

    def test_fixed_sample_and_comparator_provenance(self):
        confirmation = self.config["confirmation"]
        self.assertEqual(confirmation["n_seeds"], 40)
        self.assertEqual(
            confirmation["seed_stop_inclusive"] - confirmation["seed_start"] + 1,
            40,
        )
        self.assertTrue(confirmation["fixed_sample"])
        self.assertTrue(confirmation["sequential_seed_addition_forbidden"])
        comparator = self.config["arms"][2]
        self.assertEqual(comparator["id"], "lscv_job5008392")
        self.assertEqual(comparator["sigma_cells"], 7.244293147028428)
        self.assertIn("global", comparator["selection_independence_note"])

    def test_preflight_transfer_and_current_hashes(self):
        self.assertEqual(self.preflight["status"], "PASS")
        self.assertFalse(self.preflight["production_submitted"])
        self.assertTrue(self.preflight["paired_clone_check"]["all_hashes_match"])
        self.assertEqual(
            self.preflight["source_sha256"][
                "paper1_source_package/reproducibility/weak_landau_confirmatory_v1.json"
            ],
            sha256(CONFIG),
        )
        self.assertEqual(
            self.preflight["source_sha256"]["pic_kde_2d/scripts/run_weak_landau_confirmatory.py"],
            sha256(RUNNER),
        )
        by_arm = {row["arm"]: row for row in self.preflight["transfers"]}
        expected_c4 = math.exp(-((2.0 * math.pi / 64.0) ** 2))
        self.assertAlmostEqual(by_arm["c4_sigma1"]["measured_W2_particle_force"], expected_c4)
        self.assertLess(max(row["maximum_absolute_error"] for row in by_arm.values()), 2.0e-12)

    def test_analysis_is_explicitly_two_stage(self):
        lock = self.config["analysis_lock"]
        self.assertIn("compute no rate ratios", lock["stage_1"])
        self.assertIn("Refuse comparative analysis", lock["stage_2"])
        self.assertEqual(
            self.config["fit_rule"]["operation_order"],
            "average complex coefficients, then magnitude, then centered one-period geometric envelope, then one log-linear slope",
        )

    def test_failed_v1_loading_misses_resonance_and_has_coherent_regrowth(self):
        diagnosis = json.loads(FAILED_V1_DIAGNOSIS.read_text(encoding="utf-8"))
        self.assertEqual(diagnosis["status"], "FAIL_LOADING_ADMISSIBILITY")
        self.assertFalse(diagnosis["rate_ratios_computed"])
        loading = diagnosis["loading_implementation"]
        self.assertEqual(loading["longitudinal_distinct_unperturbed_velocities_per_cell"], 32)
        self.assertLess(loading["longitudinal_velocity_max"], loading["resonant_phase_velocity"])
        self.assertGreater(
            diagnosis["ensemble_behavior"]["confirmatory_cic"]["envelope_at_45_over_minimum_5_45"],
            3.0,
        )
        self.assertLess(
            diagnosis["ensemble_behavior"]["development_cic"]["envelope_at_45_over_minimum_5_45"],
            1.01,
        )

    def test_global_velocity_pilot_preflight_restores_resonance_support(self):
        preflight = json.loads(GLOBAL_PILOT_PREFLIGHT.read_text(encoding="utf-8"))
        self.assertEqual(preflight["status"], "PASS")
        self.assertEqual(preflight["pilot_seeds"], [202609180, 202609181, 202609182])
        self.assertTrue(preflight["spatial_loading"]["exact_requested_ppc"])
        support = preflight["velocity_support"]
        self.assertTrue(support["resonance_in_support"])
        self.assertGreater(support["longitudinal_velocity_max"], support["resonant_phase_velocity"])
        self.assertEqual(support["nodes_at_or_above_positive_resonance"], 690)

    def test_v2_fixed_sample_is_fresh_and_uses_global_velocity_assignment(self):
        config = json.loads(V2_CONFIG.read_text(encoding="utf-8"))
        confirmation = config["confirmation"]
        seeds = set(
            range(confirmation["seed_start"], confirmation["seed_stop_inclusive"] + 1)
        )
        self.assertEqual(len(seeds), 40)
        self.assertEqual(seeds, set(range(202609300, 202609340)))
        development = set()
        for item in config["development_data"].values():
            development.update(item["seeds"])
        self.assertFalse(seeds.intersection(development))
        self.assertTrue(confirmation["fixed_sample"])
        self.assertTrue(confirmation["sequential_seed_addition_forbidden"])
        self.assertTrue(
            config["benchmark"]["velocity_assignment_scope"].startswith("global")
        )

    def test_v2_preflight_has_exact_support_counts_and_current_hashes(self):
        preflight = json.loads(V2_PREFLIGHT.read_text(encoding="utf-8"))
        self.assertEqual(preflight["status"], "PASS")
        self.assertFalse(preflight["production_submitted"])
        support = preflight["velocity_support_check"]
        reference_support = preflight["reference_velocity_support_check"]
        self.assertEqual(support["nodes_at_or_above_positive_resonance"], 690)
        self.assertEqual(
            reference_support["nodes_at_or_above_positive_resonance"], 2758
        )
        self.assertNotIn(
            "estimated_nodes_at_or_above_positive_resonance", support
        )
        self.assertTrue(support["resonance_in_support"])
        self.assertTrue(reference_support["resonance_in_support"])
        self.assertEqual(
            preflight["source_sha256"][
                "paper1_source_package/reproducibility/weak_landau_confirmatory_v2_global_velocity.json"
            ],
            sha256(V2_CONFIG),
        )
        self.assertEqual(
            preflight["source_sha256"][
                "pic_kde_2d/scripts/run_weak_landau_confirmatory_v2.py"
            ],
            sha256(V2_RUNNER),
        )
        by_arm = {row["arm"]: row for row in preflight["transfers"]}
        expected_c4 = math.exp(-((2.0 * math.pi / 64.0) ** 2))
        self.assertAlmostEqual(
            by_arm["c4_sigma1"]["measured_W2_particle_force"], expected_c4
        )
        self.assertLess(
            max(row["maximum_absolute_error"] for row in by_arm.values()), 2.0e-12
        )

    def test_v2_stage1_is_rate_ratio_free_and_fails_only_cic_duration(self):
        stage1 = json.loads(V2_STAGE1.read_text(encoding="utf-8"))
        self.assertEqual(stage1["stage"], 1)
        self.assertFalse(stage1["rate_ratios_computed"])
        self.assertTrue(stage1["all_pairing_hashes_verified"])
        self.assertEqual(stage1["n_seeds"], 40)
        decisions = stage1["decisions"]
        self.assertEqual(decisions["cic"]["ending_reason"], "noise_threshold")
        self.assertEqual(decisions["c4_sigma1"]["ending_reason"], "noise_threshold")
        self.assertEqual(decisions["lscv_job5008392"]["ending_reason"], "noise_threshold")
        self.assertAlmostEqual(decisions["cic"]["accepted_t_stop"], 30.1)
        self.assertAlmostEqual(decisions["c4_sigma1"]["accepted_t_stop"], 35.6)
        self.assertAlmostEqual(decisions["lscv_job5008392"]["accepted_t_stop"], 28.0)
        self.assertLess(decisions["cic"]["fitted_amplitude_efoldings"], 2.0)
        self.assertGreaterEqual(
            decisions["c4_sigma1"]["fitted_amplitude_efoldings"], 2.0
        )
        self.assertGreaterEqual(
            decisions["lscv_job5008392"]["fitted_amplitude_efoldings"], 2.0
        )
        self.assertFalse(decisions["cic"]["central_fit_quality_pass"])
        self.assertTrue(decisions["c4_sigma1"]["central_fit_quality_pass"])

    def test_v2_stage2_uses_locked_windows_and_retains_failed_promotion(self):
        result = json.loads(V2_STAGE2.read_text(encoding="utf-8"))
        self.assertEqual(result["stage"], 2)
        self.assertEqual(result["promotion_status"], "FAIL")
        self.assertFalse(result["development_seeds_pooled"])
        self.assertEqual(result["n_fresh_seeds"], 40)
        self.assertEqual(
            result["window_decisions_sha256"],
            "cb724768fad3950d5ca181882285284b7f253d75643cee0c3e446fd1958b0cbf",
        )
        checks = result["promotion_checks"]
        self.assertTrue(checks["c4_to_cic_paired_interval"])
        self.assertTrue(checks["cic_absolute_rate_reference_interval"])
        self.assertTrue(checks["c4_two_efoldings"])
        self.assertFalse(checks["cic_two_efoldings"])
        comparison = result["rates"]["bootstrap"]["comparisons"]["c4_sigma1"]
        self.assertAlmostEqual(comparison["central_ratio_to_cic"], 0.9894153689137298)
        self.assertAlmostEqual(comparison["ratio_ci_lo"], 0.9678835537488053)
        self.assertAlmostEqual(comparison["ratio_ci_hi"], 1.0138274034763808)
        noise = result["same_campaign_noise_metrics"]["arms"]["c4_sigma1"]
        self.assertAlmostEqual(
            noise["density_error_power_reduction_vs_cic"]["central"],
            3.2586464941677002,
        )
        self.assertAlmostEqual(
            noise["grid_field_error_power_reduction_vs_cic"]["central"],
            1.6127965624275677,
        )

    def test_v2_common_window_is_secondary_and_paired(self):
        result = json.loads(V2_COMMON_WINDOW.read_text(encoding="utf-8"))
        self.assertEqual(result["analysis_role"], "secondary common-window sensitivity analysis")
        self.assertTrue(result["primary_result_preserved"])
        self.assertEqual(result["common_window"], [5.0, 30.1])
        self.assertEqual(result["n_fresh_seeds"], 40)
        self.assertFalse(result["development_seeds_pooled"])
        self.assertAlmostEqual(result["central"]["cic"]["gamma"], -0.0685811199688909)
        self.assertAlmostEqual(result["central"]["c4_sigma1"]["gamma"], -0.06986857364359501)
        paired = result["paired_bootstrap"]
        self.assertEqual(paired["valid_draws"], 5000)
        self.assertAlmostEqual(paired["c4_to_cic_ratio"], 1.0187727128878636)
        self.assertAlmostEqual(paired["ratio_ci_lo"], 1.0153585536490772)
        self.assertAlmostEqual(paired["ratio_ci_hi"], 1.0219305945128299)
        self.assertLess(result["central"]["cic"]["efoldings"], 2.0)
        self.assertLess(result["central"]["c4_sigma1"]["efoldings"], 2.0)
        self.assertTrue(result["window_sensitivity"]["point_estimate_direction_changes_across_one"])

    def test_v2_complex_history_and_envelope_diagnostic_matches_stage1(self):
        report = json.loads(
            (V2_WINDOW_DIAGNOSTIC / "window_diagnostic.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertTrue(report["diagnostic_only"])
        self.assertFalse(report["rate_ratios_recomputed"])
        self.assertIn("average the complex Fourier coefficient", report["primary_reducer_operation"])
        self.assertEqual(report["arms"]["cic"]["ending_reason"], "noise_threshold")
        self.assertEqual(report["arms"]["c4_sigma1"]["ending_reason"], "noise_threshold")
        self.assertAlmostEqual(report["arms"]["cic"]["fitted_amplitude_efoldings"], 1.7213861112191615)
        self.assertAlmostEqual(report["arms"]["c4_sigma1"]["fitted_amplitude_efoldings"], 2.076369551904831)
        table = np.genfromtxt(
            V2_WINDOW_DIAGNOSTIC / "complex_histories_and_envelopes.csv",
            delimiter=",",
            names=True,
        )
        self.assertEqual(table.shape, (901,))
        self.assertIn("cic_real", table.dtype.names)
        self.assertIn("cic_imag", table.dtype.names)
        self.assertIn("c4_sigma1_envelope", table.dtype.names)
        self.assertEqual(int(np.sum(table["cic_accepted"])), 252)
        self.assertEqual(int(np.sum(table["c4_sigma1_accepted"])), 307)

    def test_two_stage_reducer_on_synthetic_fixed_sample(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            low_root = root / "low"
            reference_root = root / "reference"
            config = json.loads(json.dumps(self.config))
            config["fit_rule"]["bootstrap_draws"] = 20
            config_path = root / "frozen.json"
            config_path.write_text(json.dumps(config, sort_keys=True) + "\n", encoding="utf-8")
            config_hash = sha256(config_path)
            runtime_source_sha256 = {"synthetic/source.py": "0" * 64}
            preflight_sha256 = "1" * 64
            times = np.linspace(0.0, 90.0, 901)
            snapshot_times = np.linspace(5.0, 30.0, 51)
            rates = {"cic": -0.0661, "c4_sigma1": -0.0660, "lscv_job5008392": -0.0800}
            scales = {"cic": 1.0, "c4_sigma1": 0.6, "lscv_job5008392": 0.4}
            for seed in confirmatory.expected_seeds(config):
                particle_hash = hashlib.sha256(f"particles:{seed}".encode()).hexdigest()
                seed_root = low_root / f"seed{seed}"
                seed_root.mkdir(parents=True)
                pairing = {
                    "seed": seed,
                    "initial_particle_sha256": particle_hash,
                    "arm_initial_hashes": {arm: particle_hash for arm in rates},
                    "all_three_arm_hashes_match": True,
                }
                (seed_root / "pairing_manifest.json").write_text(
                    json.dumps(pairing) + "\n", encoding="utf-8"
                )
                for arm, gamma in rates.items():
                    arm_root = seed_root / arm
                    arm_root.mkdir()
                    mode = 0.01 * np.exp((gamma + 1.285j) * times)
                    shape = (len(snapshot_times), 2, 2)
                    rho = np.full(shape, scales[arm], dtype=np.float64)
                    Ex = np.full(shape, 0.5 * scales[arm], dtype=np.float64)
                    Ey = np.full(shape, 0.25 * scales[arm], dtype=np.float64)
                    np.savez_compressed(
                        arm_root / confirmatory.HISTORY_NAME,
                        times=times,
                        mode=mode,
                        field_energy=np.ones_like(times),
                        kinetic_energy=np.ones_like(times),
                        total_charge=np.ones_like(times),
                        snapshot_times=snapshot_times,
                        rho_snapshots=rho,
                        Ex_snapshots=Ex,
                        Ey_snapshots=Ey,
                    )
                    metadata = {
                        "config_sha256": config_hash,
                        "seed": seed,
                        "ppc": 1024,
                        "arm": {"id": arm},
                        "initial_particle_sha256": particle_hash,
                        "source_sha256": runtime_source_sha256,
                        "preflight_sha256": preflight_sha256,
                    }
                    (arm_root / "run_parameters.json").write_text(
                        json.dumps(metadata) + "\n", encoding="utf-8"
                    )

                reference_arm = reference_root / f"seed{seed}" / "cic_reference"
                reference_arm.mkdir(parents=True)
                zeros = np.zeros((len(snapshot_times), 2, 2), dtype=np.float64)
                np.savez_compressed(
                    reference_arm / confirmatory.HISTORY_NAME,
                    snapshot_times=snapshot_times,
                    rho_snapshots=zeros,
                    Ex_snapshots=zeros,
                    Ey_snapshots=zeros,
                )
                reference_metadata = {
                    "config_sha256": config_hash,
                    "seed": seed,
                    "ppc": 4096,
                    "arm": {"id": "cic_reference"},
                    "source_sha256": runtime_source_sha256,
                    "preflight_sha256": preflight_sha256,
                }
                (reference_arm / "run_parameters.json").write_text(
                    json.dumps(reference_metadata) + "\n", encoding="utf-8"
                )

            windows_root = root / "windows"
            with redirect_stdout(io.StringIO()):
                confirmatory.select_windows(
                    config, config_path, config_hash, low_root, windows_root
                )
            decisions_path = windows_root / "window_decisions.json"
            decisions = json.loads(decisions_path.read_text(encoding="utf-8"))
            self.assertFalse(decisions["rate_ratios_computed"])
            self.assertEqual(decisions["decisions"]["cic"]["ending_reason"], "analysis_cap")
            result_root = root / "result"
            with redirect_stdout(io.StringIO()):
                confirmatory.comparative_analysis(
                    config,
                    config_path,
                    config_hash,
                    low_root,
                    reference_root,
                    decisions_path,
                    result_root,
                )
            result = json.loads(
                (result_root / "confirmatory_result.json").read_text(encoding="utf-8")
            )
            self.assertEqual(result["promotion_status"], "PASS")
            self.assertFalse(result["development_seeds_pooled"])
            c4_ratio = result["rates"]["bootstrap"]["comparisons"]["c4_sigma1"]
            self.assertGreaterEqual(c4_ratio["ratio_ci_lo"], 0.90)
            self.assertLessEqual(c4_ratio["ratio_ci_hi"], 1.10)


if __name__ == "__main__":
    unittest.main()
