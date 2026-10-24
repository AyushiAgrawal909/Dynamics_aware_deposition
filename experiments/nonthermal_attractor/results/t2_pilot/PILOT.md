# T2 pre-freeze pilot (one level, one seed: not a gate result)

One production-length PIC run at the coarsest level, used only to close the protocol's open items before the freeze. A single level carries no level-to-level difference, so the regulator outcome cannot be read from it.

- **Run:** `pic1d1v`, 128 ppc (level 7), seed 0, production config, t_end = 4800 (12.4 min on 4 cores).
- **Check:** `python3 scripts/t2_pilot_check.py configs/t2_waterbag_dv020.yaml BUNDLE` → `pilot_check.json`.
- The bundle (8.6 MB) is kept outside the repo, in the project files under `nonthermal_attractor/t2_pilot/`.
- The run used the earlier reconstruction box (v_lim = 4); the frozen config raises it to 8.

## What it settled

| Open item | Pilot value | Frozen choice |
|---|---|---|
| eps_q (q = 0.01) counts | ≥ 5883 effective particles above eps_q at every window snapshot (need 100) | q = 0.01 kept |
| eta_min vs floor | Floor-limited at every window snapshot: eta_min / eta_floor ≥ 1.77 (need 10) | eta_min enters the gate only at levels resolved in ≥ 95% of window snapshots. With fewer than 3 such levels it is INCONCLUSIVE (unresolved), and the decision rests on eps_q |
| Vlasov v_max = 4 | 2.9e-3 of the mass beyond \|v\| = 3.2; 3.8e-6 beyond 6.3; none beyond 7.1 (one-particle floor 1.9e-6) | v_max = 10 (0.8 v_max = 8). The Vlasov arm cost rises 2.5× |
| Onset definition (eps_tail = 1, q = 1e-3) | Crossed at t = 10.1. That is beam saturation, unresolved by the 10/ω_p snapshots | eps_tail = 7.92, q_onset = 3e-4: crossed at t = 294, ≥ 157 particles at 128 ppc |

## Fraction of electrons above ε (this run, 128 ppc)

| ε | t = 50 | 100 | 200 | 500 | 1000 | 2000 | 3200 | 4800 |
|---|---|---|---|---|---|---|---|---|
| 1.0 | 1.6e-1 | 1.6e-1 | 1.5e-1 | 1.3e-1 | 1.3e-1 | 1.3e-1 | 1.3e-1 | 1.3e-1 |
| 3.15 | 7.1e-3 | 1.0e-2 | 8.6e-3 | 1.9e-2 | 2.0e-2 | 2.0e-2 | 2.0e-2 | 1.9e-2 |
| 5.0 | 5.7e-4 | 1.8e-3 | 1.8e-3 | 2.4e-3 | 2.4e-3 | 2.5e-3 | 2.4e-3 | 2.4e-3 |
| 7.92 | 5.7e-5 | 1.1e-4 | 2.2e-4 | 3.5e-4 | 3.6e-4 | 3.9e-4 | 4.0e-4 | 3.8e-4 |
| 9.98 | 0 | 9.5e-6 | 7.4e-5 | 9.3e-5 | 1.3e-4 | 1.4e-4 | 1.5e-4 | 1.6e-4 |

**Observation (single coarse run, not a claim):** at 128 ppc the tail up to ε ≈ 8 is in place by t ≈ 500 and then barely moves. Only the far tail (ε ≳ 10) is still growing at t = 4800.

Ewart et al. follow their tail to 3200/ω_pe, with beam widths of 2–50% of v_b and 20,000 ppc. Whether the early saturation here is physical or a 128-ppc effect is exactly what the nested ladder tests. It also means a T3 collapse fit on this window may come back STATIONARY.

Energy drift |ΔE|/K stays below 1.3e-4 throughout. The tail energy fraction at the end of the window is 0.54.
