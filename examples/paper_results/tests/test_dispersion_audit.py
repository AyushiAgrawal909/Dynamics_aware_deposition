"""Regression tests for the recovered Paper 1 dispersion calculation."""

from __future__ import annotations

import json
import math
import pathlib
import sys
import unittest


HERE = pathlib.Path(__file__).resolve().parent
REPRO = HERE.parent
sys.path.insert(0, str(REPRO))

import recompute_dispersion as audit  # noqa: E402


class DispersionAuditTests(unittest.TestCase):
    def test_continuum_and_principal_cic_anchors(self) -> None:
        landau = audit.solve_level(audit.LANDAU, "continuum")
        e2_m8 = audit.solve_level(audit.E2_M8, "principal_cic")
        e2_m12 = audit.solve_level(audit.E2_M12, "principal_cic")
        self.assertAlmostEqual(landau.real, -0.0661279586906626, places=11)
        self.assertAlmostEqual(landau.imag, -1.28505696965378, places=11)
        self.assertAlmostEqual(e2_m8.real, 0.3469724018712, places=9)
        self.assertAlmostEqual(e2_m12.real, 0.1446708324317, places=9)

    def test_alias_and_time_corrections_are_converged_and_small(self) -> None:
        principal = audit.solve_level(audit.E2_M12, "principal_cic")
        spatial4 = audit.solve_level(audit.E2_M12, "spatial_discrete", qmax=4)
        spatial8 = audit.solve_level(audit.E2_M12, "spatial_discrete", qmax=8)
        discrete512 = audit.solve_level(
            audit.E2_M12,
            "grid_time_discrete",
            qmax=8,
            lmax=512,
        )
        discrete1024 = audit.solve_level(
            audit.E2_M12,
            "grid_time_discrete",
            qmax=8,
            lmax=1024,
        )
        self.assertLess(abs(spatial8.real - spatial4.real), 1.0e-9)
        self.assertLess(abs(discrete1024.real - discrete512.real), 2.0e-8)
        self.assertLess(abs(spatial8.real - principal.real), 1.0e-5)
        self.assertLess(abs(discrete1024.real - spatial8.real), 7.0e-5)
        self.assertAlmostEqual(discrete1024.real, 0.144599903831, places=10)

    def test_principal_bandwidth_ceilings_reproduce_archived_values(self) -> None:
        landau = audit.find_ceiling(
            audit.LANDAU,
            "principal_cic",
            "isotropic",
            qmax=4,
            lmax=128,
        )
        e2_m8 = audit.find_ceiling(
            audit.E2_M8,
            "principal_cic",
            "isotropic",
            qmax=4,
            lmax=128,
        )
        e2_m12 = audit.find_ceiling(
            audit.E2_M12,
            "principal_cic",
            "isotropic",
            qmax=4,
            lmax=128,
        )
        self.assertAlmostEqual(landau, 2.4534381426, places=8)
        self.assertAlmostEqual(e2_m8, 1.0622934858, places=8)
        self.assertAlmostEqual(e2_m12, 0.1987203603, places=8)

    def test_compensated_campaign_did_not_run_binding_corner(self) -> None:
        launch = (
            audit.CAMPAIGN_ROOT / "scripts/submit_higher_order_comp_rates_cpu.sbatch"
        ).read_text(encoding="utf-8")
        self.assertIn("--benchmark two_stream --mode 12", launch)
        self.assertNotIn("--ky-mode", launch)
        rows = audit.filter_rate_table(qmax=4, lmax=128)
        p3_corner = next(
            row
            for row in rows
            if row["compensation_order"] == 3
            and row["passes"] == 4
            and row["evaluated_mode"] == "(12,1)"
            and row["theory_level"] == "grid_time_discrete"
        )
        self.assertIsNone(p3_corner["measured_rate_ratio"])
        self.assertIn("prediction only", p3_corner["status"])
        self.assertTrue(math.isclose(p3_corner["predicted_rate_ratio"], 0.968109, abs_tol=2e-6))

    def test_saved_equilibrium_is_the_actual_isotropic_e2_loading(self) -> None:
        path = REPRO / "results/dispersion_audit/dispersion_audit.json"
        with path.open("r", encoding="utf-8") as handle:
            result = json.load(handle)
        moments = result["e2_equilibrium_seeds"]
        self.assertEqual(len(moments), 10)
        self.assertTrue(all(row["particles"] == 1_048_576 for row in moments))
        self.assertTrue(all(row["negative_vx_count"] == 524_288 for row in moments))
        self.assertTrue(all(row["positive_vx_count"] == 524_288 for row in moments))
        self.assertLess(max(abs(row["vy_std"] - 0.05) for row in moments), 1.0e-6)
        self.assertLess(
            max(abs(row["vx_std_about_conditional_beam"] - 0.05) for row in moments),
            1.0e-6,
        )


if __name__ == "__main__":
    unittest.main()
