#!/usr/bin/env python3
"""Run T2 validations V1-V4 for both backends and write reports/T2_VALIDATION.md.

    python3 experiments/nonthermal_attractor/scripts/t2_validate.py
"""
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

from attractor_sim import validation as V  # noqa: E402


def main():
    t0 = time.time()
    r = {}
    r["V1_pic"] = V.v1_pic_conservation()
    r["V2_pic"] = V.v2_landau_pic()
    r["V2_vlasov"] = V.v2_landau_vlasov()
    r["V3_pic"] = V.v3_waterbag_growth("pic")
    r["V3_vlasov"] = V.v3_waterbag_growth("vlasov")
    r["V4_control"] = [V.v4_stable_control_heating(nx, 256, L=16.0, dt=dt, steps=int(20 / dt))
                       for nx in (256, 512) for dt in (0.05, 0.025)]
    r["V4_nonlinear"] = V.v4_nonlinear_refinement()
    wall = time.time() - t0
    g = lambda d: (d["gamma_measured"], d["gamma_theory"], abs(d["gamma_measured"] / d["gamma_theory"] - 1))
    nl = r["V4_nonlinear"]
    lines = [
        "# T2 validations (V1-V4)",
        "",
        f"Command: `python3 experiments/nonthermal_attractor/scripts/t2_validate.py` ({wall:.0f} s, 4 threads).",
        "",
        "| Check | Backend | Result | Criterion | Pass |",
        "|---|---|---|---|---|",
        f"| V1 charge neutrality | PIC | total charge {r['V1_pic']['charge']:.1e} | round-off | {abs(r['V1_pic']['charge']) < 1e-10} |",
        f"| V1 momentum (400 steps) | PIC | max drift {r['V1_pic']['momentum_drift_max']:.1e} (scale {r['V1_pic']['momentum_scale']:.1f}) | round-off | {r['V1_pic']['momentum_drift_max'] < 1e-12} |",
        f"| V1 energy drift (stable Maxwellian, t=40) | PIC | {r['V1_pic']['energy_drift_rel']:.1e} of K | reported | - |",
        "| V2 Landau k lambda_D = 0.5 | PIC | gamma = %.4f vs %.4f (%.1f%%) | 5%% | %s |" % (*g(r['V2_pic'])[:2], 100 * g(r['V2_pic'])[2], g(r['V2_pic'])[2] < 0.05),
        "| V2 Landau k lambda_D = 0.5 | Vlasov | gamma = %.4f vs %.4f (%.1f%%); recurrence T_R = %.0f | 5%% | %s |" % (*g(r['V2_vlasov'])[:2], 100 * g(r['V2_vlasov'])[2], r['V2_vlasov']['recurrence_time'], g(r['V2_vlasov'])[2] < 0.05),
        "| V3 waterbag growth, k = k* | PIC | gamma = %.4f vs %.4f (%.1f%%) | 5%% | %s |" % (*g(r['V3_pic'])[:2], 100 * g(r['V3_pic'])[2], g(r['V3_pic'])[2] < 0.05),
        "| V3 waterbag growth, k = k* | Vlasov | gamma = %.4f vs %.4f (%.1f%%) | 5%% | %s |" % (*g(r['V3_vlasov'])[:2], 100 * g(r['V3_vlasov'])[2], g(r['V3_vlasov'])[2] < 0.05),
        "| V4 stable control, dK/K at t=20 | PIC | " + "; ".join(
            f"dx/lambda_Db={c['dx_over_lambda_Db']:.2f}: {c['dK_over_K']:.1e}" for c in r["V4_control"]) + " | << 0.1 f_tail | see note |",
        f"| V4 nonlinear (t=30-60) dx halving | PIC | field energy {100 * nl['fe_rel_change_dx']:.1f}%, dK {100 * nl['dK_rel_change_dx']:.1f}% | reported | - |",
        f"| V4 nonlinear (t=30-60) dt halving | PIC | field energy {100 * nl['fe_rel_change_dt']:.1f}%, dK {100 * nl['dK_rel_change_dt']:.1f}% | reported | - |",
        "",
        "V3 seeds a single mode at k = k* in a box of 4 wavelengths and fits from t = 3/gamma until the mode "
        "reaches 5% of its saturated amplitude; the noise-free Vlasov run starts from a 1e-7 seed so that the "
        "linear phase is long compared with the beat of the two neutral roots.",
        "",
        "V4 note: the short stable-control runs only show that finite-grid heating is not present at t = 20; "
        "the decisive measurement is the stable-control arm of the scan at production length, which enters "
        "the frozen quality gate. The nonlinear halving runs use the same seed spectrum on a 4-wavelength box "
        f"at dx = {nl['dx']:.4f} (dx/lambda_Db = {nl['dx_over_lambda_Db']:.2f}); single seed, so differences "
        "include realisation noise after saturation.",
        "",
        "```json",
        json.dumps(r, indent=1, default=float),
        "```",
    ]
    (HERE / "reports" / "T2_VALIDATION.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:20]))


if __name__ == "__main__":
    main()
