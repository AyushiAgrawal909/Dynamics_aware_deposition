# Codex task T2 of 3: 1D1V PIC and Vlasov reference, cost estimate, and regulator-independence gate

**Prerequisite:** T1 finished with status `ANALYSIS_VALIDATED`, or `ANALYSIS_PARTIAL` with no failure in S3, S5 or S7. Read `experiments/nonthermal_attractor/reports/T1_REPORT.md` and `PRIOR_ART.md` first. T1 sets the minimum number of seeds and nested levels, and the Ewart et al. run parameters.

**Question this task decides:** On one physical preparation, is there a moving feature of the tail (in ρ(η, t) or N(ε, t)) whose time evolution converges as the regulator is removed? The regulator is particle number in PIC and velocity-grid resolution in Vlasov. Do the two codes agree on it in that limit?

T2 is informative either way:

- **Converged moving feature:** the forecasting programme (T3) is worth running.
- **Regulator-dependent tail, cutoff or onset:** a publishable result about what PIC evidence for universal tails means.

No scaling exponents, forecasts or attraction claims are made in T2.

## 0. Ground rules

- Work in `experiments/nonthermal_attractor/` on a new branch. Add `pic1d1v/`, `vlasov1d1v/`, `configs/`, `scripts/`, `results/` (small artifacts only), and `slurm/`.
- **HEXAPIC is not used in T2.** For an effectively 1D question it adds a 2D HYPRE iterative solve, three velocity components, a nodal-potential gather without a momentum guarantee, and y-independence to police. It enters only in a later 2D phase, where dimensionality is the scientific question.
- All output goes through the T1 data contract and must pass the T1 validator.
- **Freeze `protocol/T2_PROTOCOL.yaml` before running anything beyond the smoke and validation tests.** It is filled from T1's template and committed in its own commit before the scan runs. It records:
  - the preparation;
  - bins, the tail threshold, and the feature definitions;
  - the gate tolerance τ and the analysis window;
  - the regulator-scaling model comparison;
  - the quality gates.

## 1. Backend A: 1D1V spectral PIC (`pic1d1v/`)

Collapse the `pic_kde_2d` chain to 1D. The reference implementation is in these files:

- `pic_kde_2d/pic_kde_2d/methods/cic.py`: CIC deposit, and matched CIC gather from nodal E.
- `pic_kde_2d/pic_kde_2d/core/grid2d.py`: FFT Poisson solve, with E = −ik φ_k spectrally at nodes.

The chain is: CIC deposit, FFT Poisson with k = 0 removed, an explicit uniform immobile neutralizing background, spectral E at nodes, matched CIC gather, and leapfrog. This combination conserves momentum to roundoff. Test it, and keep it that way. No smoothing: fixed CIC is the unsmoothed baseline. Use numba if it is already available in the environment; otherwise use vectorised numpy. Report the measured particle-steps per second.

Loading: use quiet-start waterbag beams with **nested subsampling** via `nested_load_indices` from T1 §4. The load at N/2^ℓ is a stratified subset of the load at N, with weight × 2^ℓ.

**Validation, all before the scan:**

- **V1.** Charge neutrality, momentum conservation to roundoff, and the energy-drift budget on a stable Maxwellian.
- **V2.** Linear Landau damping at kλ_D = 0.5: γ = −0.1533 ω_p.
- **V3.** Linear growth of the chosen preparation against the waterbag dispersion relation (§3), within 5% on the three fastest modes.
- **V4.** Δx-halving and dt-halving on a short nonlinear run. Show that finite-grid heating of the stable control is below the quality gate (§5).

## 2. Backend B: 1D1V semi-Lagrangian Vlasov reference (`vlasov1d1v/`)

- Use Strang-split semi-Lagrangian advection (Cheng–Knorr) with a positivity-preserving, conservative interpolation (PFC or equivalent) and a spectral Poisson solve with the same background and normalization as the PIC.
- Choose the velocity domain [−v_max, v_max] so that the tail fits. Monitor the mass and energy flux through ±v_max. The run is invalid if more than 10⁻⁶ of the mass passes 0.8 v_max in the finest PIC run of the same preparation.
- **Validation:**
  - V2 and V3 as for PIC;
  - linear-Landau recurrence time 2π/(kΔv) seen and avoided;
  - mass conservation to roundoff;
  - the drift of L¹, L² and the entropy reported (coarse-graining by the interpolation is expected; report it, never call it physics).
- The Vlasov regulator is filamentation dissipation at the grid scale Δv. Call this a reference calculation, not collisionless truth.

## 3. The single preparation

- **Preparation:** symmetric counter-propagating electron waterbag beams, centred at ±v_b with full width Δv_b = 0.20 v_b, f₀ = n₀/(2Δv_b) on each interval, and an immobile neutralizing background. Second moment: v_b² + Δv_b²/12.
- **Why this width:** it sits inside Ewart et al.'s Lynden-Bell-agreement range (up to about Δv_b/v_b ≈ 0.25) and stays affordable. The narrower 0.15 v_b case gets a cost estimate only.
- **Linear theory:** each waterbag of density n₀/2 contributes −(ω_p²/2)/[(ω − k u)² − k²Δv_b²/4], so

  ε(k, ω) = 1 − (ω_p²/2) Σ_± 1/[(ω ∓ k v_b)² − k²Δv_b²/4].

  Solve this quartic for γ(k), the fastest-growing k*, and the unstable band. Check the cold limit (Δv_b → 0): γ_max = ω_p/(2√2) at k* v_b/ω_p = √(3/8). Reference values from a direct root solve: at Δv_b = 0.20 v_b, γ_max ≈ 0.352 ω_p at k* v_b/ω_p ≈ 0.614 (λ* ≈ 10.2 v_b/ω_p); at Δv_b = 0.15 v_b, γ_max ≈ 0.353 ω_p. Reproduce these numbers and don't trust them blindly.
- **Box:** L contains at least 16 fastest-growing wavelengths, to allow hole mergers. Run one L-doubling check at the middle regulator level, and record the finite-size limitation. Sixteen wavelengths is about 164 v_b/ω_p, or about 3,300 cells at Δx = 0.05 v_b/ω_p.
- **Grid:** resolve the beam Debye length λ_Db = (Δv_b/√12)/ω_p. Start with Δx ≤ λ_Db, about 0.05 v_b/ω_p, and confirm with V4.
- **Time step:** ω_p dt ≤ 0.1. Confirm with dt-halving, and report the largest |v| dt/Δx reached by tail particles.
- **Seed:** use a deterministic spectrum on the unstable band with recorded amplitudes and phases, shared by every run and both backends. **The amplitude must exceed the measured shot-noise floor of the lowest-N PIC level by at least 10×** (measure that floor from an unseeded load at t = 0 and after one plasma period), while staying linear (δn/n ≤ 10⁻³). Otherwise the low-N runs are silently noise-seeded and the meaning of the scan changes across levels.
- **End time:** at least 1.5× the tail-formation time reported by Ewart et al. (from PRIOR_ART.md), rescaled to this beam width via 1/γ_max. State the number used.

## 4. Cost estimate, then stop or scan

Run a timing smoke test of both backends, about 10³ steps each. Then write `reports/COST_ESTIMATE.md` covering:

- particle-steps (PIC) and cell-steps (Vlasov) for the full matrix in §5;
- wall time from the measured throughput;
- the same for Δv_b = 0.15 v_b;
- the scaling with L.

As a sanity anchor, about 10⁴ cells × 10³ ppc × 10⁵ steps ≈ 10¹² particle-steps is hours on one CPU node, not minutes.

- **Local budget default:** 8 CPU-node-hours, or the user's stated budget if larger. If the matrix fits, run it.
- **If it doesn't fit:**
  - generate Meluxina SLURM scripts in `slurm/`, with environment-specific lines marked `# PLACEHOLDER`;
  - run only the levels that fit, coarsest first, with all seeds at those levels;
  - stop and report.

**Never launch an underpowered run and interpret it.** The cheapest low-N runs are exactly where noise breaks the Casimirs fastest, so a short low-N pilot can produce a confident-looking false positive tail.

## 5. The regulator scan (decisive)

| Arm | Regulator levels | Seeds |
|---|---|---|
| PIC, nested | ppc spanning ≥ 2 decades in factor-2 nested levels (e.g. 64 → 8192), as T1's minimum requires | ≥ T1 minimum (default 4) |
| Vlasov | Δv ∈ {Δv_b/16, /32, /64, /128} at fixed Δx, plus one Δx-halving check | 1 (deterministic) |
| Stable control | A single waterbag at rest with the same width, at every PIC level | 2 |
| *(Optional, if budget remains)* pair plasma | Electron–positron version with the same ω_p normalization stated, at 3 PIC levels | ≥ 2 |

The optional pair-plasma arm is the sharpest single test. Ewart et al. find the pair case frozen (no tail). If increasing the noise flips it into the tail class, the universality claim weakens considerably; if it can't, the claim is strengthened. Either outcome is reportable. It is a distinct physical model, not another preparation.

**Measure on every run:**

- ρ(η, t) (primary) with T1's bin-insensitivity check;
- N(ε, t);
- Q_N, Q_E and f_tail;
- the moving features from T1 §2–§3;
- the tail onset time t_on, by a frozen definition;
- the budgets.

**Quality gate (frozen):** at the end of the analysis window, the following must hold for the run to count:

|ΔE_total|/K_total + (stable-control fractional ΔK at the same level) < 0.1 × f_tail

The PIC run must also conserve momentum to roundoff, and the Vlasov run must stay within the velocity-boundary flux limit.

## 6. Analysis and decision

Apply T1's `convergence_gate` to each frozen feature, using the PIC nested levels plus the Vlasov Δv sequence as the independent reference.

Pre-register the interpretation of the onset scaling:

- **t_on ∝ ln N (PIC), and ∝ ln(1/Δv) (Vlasov):** consistent with Casimir breaking at the noise or grid scale (Ewart's τ_c ~ γ⁻¹ ln(λ_D/l_c)). The tail itself may still be physical.
- **t_on ∝ N:** a discreteness-driven mechanism (Yoon school). Then the PIC tail is not a Vlasov property, and Vlasov should show the corresponding absence or delay.
- **No tail below some N:** a minimum-ppc threshold, which is a result in its own right. Report it.

Write `results/T2_DECISION.json` and `reports/T2_REPORT.md`, with a separate status (`SUPPORTED` / `REJECTED` / `INCONCLUSIVE` / `NOT_TESTED`) for each of:

1. numerical validity (V1–V4, quality gate);
2. tail detection;
3. regulator convergence of each feature;
4. PIC–Vlasov agreement;
5. onset-scaling model;
6. pair-plasma class flip.

**Gate outcome**, one of:

- `GO_T3`: at least one moving feature passes the convergence gate and agrees with Vlasov.
- `REGULATOR_RESULT`: features fail as regulator-dependent. Write up the PIC-evidence result, and recommend not running T3.
- `INCONCLUSIVE`: state exactly which runs, at what cost, would resolve it.

Include in the report:

- plots of raw N(ε) and ρ(η) with count support per level;
- feature trajectories by level with paired differences;
- onset time vs N and vs Δv;
- PIC vs Vlasov overlays;
- the budgets and the quality gate;
- a manifest linking every curve to backend, config, seed and level;
- which commands actually ran, wall time, and output size.

Do not fit scaling exponents, forecast, or test attraction in this task.
