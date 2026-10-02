# T2 analysis SMOKE: pipeline check only, NOT a physics result

This smoke run used a 4-wavelength box, ran to t = 100, and had 4 levels (1024 down to 128 ppc), 3 seeds and 2 Vlasov levels (`configs/t2_smoke.yaml`). It shows that bundles, the stable control, the frozen gate, the quality gate and the decision file work end to end on real backend output.

**The verdicts below must not be interpreted.** The run is far too short for tail evolution, and the box is too small for hole mergers. Under the T2 spec, a run this short is INCONCLUSIVE whatever the gate prints.

Gate outcome: **REGULATOR_RESULT**

| Feature | Verdict | Reason |
|---|---|---|
| eps_q | FAIL_REGULATOR_DEPENDENT | paired differences do not shrink and exceed tau x motion (includes features that differ between levels but barely move in time) |
| eta_min | FAIL_REGULATOR_DEPENDENT | paired differences do not shrink and exceed tau x motion (includes features that differ between levels but barely move in time) |

Onset model preferred: INCONCLUSIVE (Delta AIC 1.2890785473710764)

Quality gate (|dE|/K + control dK/K < 0.1 f_tail) pass fraction per level: L0: 1.00, L1: 1.00, L2: 1.00, L3: 1.00
