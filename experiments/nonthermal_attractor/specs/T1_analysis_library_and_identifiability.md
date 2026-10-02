# Codex task T1 of 3: analysis library and synthetic identifiability suite (no physics runs)

Programme: Do nonthermal tails from electrostatic beam relaxation collapse onto a self-similar (nonthermal-fixed-point-like) form, and if so, can early data forecast later tail evolution?

This task is the first of three gated tasks:

- **T1 (this task):** analysis library, data contract, and a synthetic suite showing what the analysis can and cannot identify. **No plasma simulations are run.**
- **T2:** a 1D1V spectral PIC and a semi-Lagrangian Vlasov reference, a cost estimate, and a regulator-independence scan on one preparation. T2 is the decisive gate.
- **T3:** scaling fits, forecasting, and attraction. T3 runs only if T2 finds a regulator-converged moving feature.

T1 is complete when every synthetic test below passes, or when a failure is documented and understood. Do not start T2 work in this task.

## 0. Ground rules

- Work in the repository `dynamics_aware_deposition` on a new branch. Put everything under `experiments/nonthermal_attractor/` with subfolders `analysis/`, `tests/`, `synthetic/`, `reports/`, `protocol/`.
- Reuse the conventions in `pic_kde_2d/pic_kde_2d/conventions.py` (normalized units, `Q_OVER_M = -1`). Use numpy/scipy only. Add no new heavy framework.
- Use synthetic data only to validate the analysis. Keep it physically and file-wise separate from anything that will later hold plasma results.
- This is a statement of what is unproved: no tail, scaling regime, universal exponent, or fixed point is assumed to exist.

## 1. Data contract (the format T2 must write and T1 must read)

Write `analysis/schema.py` and `protocol/DATA_CONTRACT.md`. One run produces one HDF5 or `.npz` bundle with:

- **Metadata:** backend (`pic1d1v` | `vlasov1d1v` | `hexapic`), git revision, full config, normalization (ω_pe of the total electron density = 1; velocities in units of the beam speed v_b, lengths in v_b/ω_pe; state the conversion to λ_D), seed, nested-subsample level (see §4), dt, Δx, Δv (Vlasov only), particles per cell (PIC only), wall time, output size.
- **Exact diagnostic times** and the staggering convention (for leapfrog, which quantity lives at half-steps and how it is synchronized for the diagnostic).
- **Scalar budgets per snapshot:** particle number, total charge, total momentum, kinetic energy, discrete field energy consistent with the implemented operator, total energy, and drift relative to t = 0.
- **N(ε, t) histograms** on frozen common bin edges, plus underflow/overflow counts and their energies, and per-bin weighted effective counts (Kish n_eff).
- **ρ(η, t) waterbag-content histograms** on frozen η bins (§3), with the phase-space binning (Δx_b, Δv_b) used to reconstruct f.
- **Field spectrum** |E_k|² for all k, plus a few selected mode amplitudes.
- **Occasional x–v phase-space snapshots** (decimated) for structure diagnostics.

Write a validator that rejects a bundle missing any required field. T2 output must pass this validator before any analysis runs on it.

## 2. Energy-distribution estimators

Use these definitions exactly:

- N(ε, t) = dN_phys/dε, with N_b = Σ_{p∈b} w_p / Δε_b. The physical mass goes into ε, and the macro weight is used only in the sum.
- dN/d ln ε = ε N(ε). Plotting code must label which quantity it shows and must never overlay the two.
- The active energy for the 1D1V problem is ε = m v_x²/2, taken in the initial center-of-momentum frame, which is the lab frame for symmetric beams.
- Log-tail analysis uses strictly positive energies. Moment checks include the low-energy bins.
- There is no clipping and no per-snapshot renormalization.

The tests below must pass:

- Histogram number and energy moments match direct particle sums. Report the finite-bin moment error, with the expected O(Δε²) scaling demonstrated.
- For a 1V Maxwellian, N(ε) ∝ ε^(−1/2) e^(−ε/T): the analytic shape is recovered within sampling error, including the integrable ε^(−1/2) singularity in the lowest bin.

Tail statistics, with the threshold frozen in the protocol and not chosen per run:

- Q_N(t) = ∫_tail N dε
- Q_E(t) = ∫_tail ε N dε
- f_tail = Q_E / K_total

Moving-feature estimators (at least two, all supported by counts):

- (a) The energy ε_q(t) at which the complementary cumulative tail number falls to a fixed fraction q, for q at several values where the bin has ≥ 100 effective counts.
- (b) The upper cutoff of a power-law segment, defined by a frozen curvature criterion.
- (c) For ρ(η), the lower edge η_min(t) of the η^(−1) asymptotic range (§3).

A single largest particle energy is never a feature.

Uncertainty: the independent seed is the resampling unit. Do not treat snapshots as independent, and do not use Poisson errors for quiet-start loads. Bootstrap over seeds and report the method used.

## 3. Waterbag content ρ(η) (primary 1D1V observable)

In the Lynden-Bell framework used by Ewart et al. (arXiv:2409.01742), N(ε) is determined by ρ(η) and the total energy. The missing piece they name is a theory of how ρ(η) evolves in time. So ρ(η, t) is the primary observable, and N(ε, t) is secondary.

- Implement the phase-space reconstruction f̂(x, v) on a uniform (Δx_b, Δv_b) grid from either particles (PIC) or the grid f (Vlasov, optionally coarse-grained to the same Δx_b, Δv_b).
- ρ(η) is the phase-space volume per unit η occupied by cells with f̂ ∈ [η, η + dη). Check that ∫ρ dη equals the reconstructed phase volume and that ∫η ρ dη equals N.
- Coarse-grained ρ is **not** a conserved Casimir distribution. Binning-induced change must be separable from dynamics. Implement a bin-insensitivity check: recompute at (Δx_b, Δv_b) × {½, 1, 2} and report which features are stable.
- Implement the detector for the η^(−1) asymptotic range and its lower edge η_min(t), using a frozen local-slope criterion with bootstrap CI.
- Implement a discrete-particle floor: with N_ppc particles per cell and reconstruction cell (Δx_b, Δv_b), the smallest resolvable η is about one particle per bin. Report η_floor alongside η_min. A feature that tracks η_floor rather than time is flagged `RESOLUTION_LIMITED`.

## 4. Nested-subsample convergence gate (the core of T2's decision)

This is the most important function in T1.

- Specify the loading so that the load at level ℓ (N/2^ℓ particles, weight × 2^ℓ) is a **subset** of the load at level ℓ−1. With quiet start, each subset must itself be stratified. Use bit-reversed or base-2 van der Corput ordering of the stratified marker list, so that taking every 2^ℓ-th marker keeps it quiet. Provide `nested_load_indices(N, levels)` and a unit test showing every level is stratified (low discrepancy) and that its moments match the target.
- Implement `convergence_gate(feature_by_level, seeds)` with these steps:
  - Use paired differences Δ_ℓ(t) = F_ℓ(t) − F_{ℓ−1}(t) on the same seed. The differences are correlated, which is the point of nesting.
  - Gate **PASS** requires all of the following: |Δ| decreases monotonically with ℓ → finer; the finest paired difference is below a frozen tolerance τ relative to the feature's own motion over the analysis window; and the extrapolated feature trajectory is consistent with the independent reference (the Vlasov result in T2) within CI.
  - Gate **FAIL_REGULATOR_DEPENDENT** applies when the paired differences do not shrink, or when the feature's motion scales with ln N or N rather than with t.
  - Gate **INCONCLUSIVE** applies when there are too few levels or seeds, or when the CIs overlap both outcomes.
- Implement the two regulator-scaling fits T2 will pre-register. A feature onset time t_on(N) is fit to (i) a + b ln N (cascade or Casimir-breaking picture, τ_c ~ γ⁻¹ ln(λ_D/l_c)) and (ii) a + b N (discreteness-driven picture). Use the Vlasov analogue in Δv: t_on vs ln(1/Δv) or 1/Δv. Report the model comparison (AIC or held-out residual), not a forced choice.

## 5. Scaling inference with identifiability detection

Hypothesis: N(ε, t) = A★ s^α F(s^β ε/ε★), with s = (t − t₀)/t★ > 0. Apply the same form to ρ(η, t) with its own exponents.

- Fit only snapshots in a training window defined by the protocol. Profile t₀ over a bounded range and report its uncertainty.
- Use a common scaled overlap region with a minimum-overlap penalty, a loss weighted by the bootstrap uncertainty, and normalization preserved (no free per-snapshot amplitude).
- Outputs: α–β covariance, the profile-loss landscape, boundary-hitting flags, and sensitivity to the window, bins, and threshold.
- **Identifiability detector:** if F is locally a pure power law x^(−p) across the overlap, only α − pβ is identified. Detect this from the loss landscape, where a valley with Hessian eigenvalue ratio above a frozen threshold aligns with the direction (p, 1). Return `UNIDENTIFIABLE_DEGENERATE` with the measured combination, never separate exponents.
- Sector budgets: for scaled bins moving as ε_{1,2} = ε★ x_{1,2} s^(−β), compare the sector number and energy with s^(α−β) and s^(α−2β), and include the boundary-flux term. Keep fixed-threshold moments separate. For p ≈ 2, track both cutoffs, because the sector energy depends on ln(ε_max/ε_min).

## 6. Forecast and baselines (implemented and tested here, used in T3)

- Frozen-parameter forecast: N_pred(ε, t) = r^α N(r^β ε, t_ref) with r = (t − t₀)/(t_ref − t₀). Mask energies that would need template support outside the training data.
- Baselines, each given the same training information:
  - (1) a stationary distribution or stationary power-law tail;
  - (2) a 1V Maxwellian matched only to training-window energy;
  - (3) a one-scale heating model;
  - (4) a quasilinear diffusion baseline, implemented as a stub interface only, with its assumptions documented.
- An oracle Maxwellian built from future energy is labelled `ORACLE` and excluded from ranking.

## 7. Synthetic identifiability suite (the deliverable that decides T1)

Generate particle samples (not smooth curves) at several N and seeds, so the noise model is realistic, and write them through the data contract. Each case has a pre-stated expected verdict:

| # | Synthetic case | Expected verdict |
|---|---|---|
| S1 | 1V Maxwellian, stationary | No tail; estimators unbiased |
| S2 | Self-similar with a moving cutoff, known (α, β), curved F | (α, β) recovered within CI; held-out forecast beats all baselines |
| S3 | Stationary ε^(−2) tail between fixed cutoffs | `UNIDENTIFIABLE_DEGENERATE`, reporting α − 2β ≈ 0 |
| S4 | Maxwellian heating with T(t) ∝ t^a (non-self-similar tail) | No false attractor; heating baseline wins |
| S5 | **Resolution-tracking cutoff:** cutoff ε_c ∝ ln N, fixed in time | Convergence gate returns `FAIL_REGULATOR_DEPENDENT`; scaling fit must not be reported as physical |
| S6 | **Mixed:** physical moving cutoff plus an N-dependent onset delay t₀(N) = a + b ln N | Gate identifies ln N onset scaling and still recovers (α, β) after t₀ profiling |
| S7 | ρ(η) with a downward-extending η^(−1) range plus the particle floor η_floor | η_min(t) recovered where above the floor; flagged `RESOLUTION_LIMITED` where it touches the floor |

S5 and S6 are new relative to the earlier spec. They test the failure mode the physics most likely produces: features that move with resolution instead of time.

Report a confusion table (expected vs returned verdict) for all seven cases across N ∈ {2⁹, 2¹¹, 2¹³} particles per cell equivalents and ≥ 5 seeds.

## 8. Bounded prior-art note

Write `reports/PRIOR_ART.md` with a hard budget (about 1 hour; don't let it replace implementation). Give exact arXiv versions and the sections read for:

- Ewart et al., arXiv:2409.01742, and its published (PNAS 2025) version: the waterbag content ρ(η), the electron-only versus pair outcome classes, and the run parameters (grid, ppc, beam widths, run length). T2 will need these.
- Banik, Bhattacharjee & Sengupta, arXiv:2408.07127, including Appendix C.2 (self-similar solutions in the driven quasilinear model).
- Yoon et al. on spontaneous scattering and kappa tails (the discreteness-driven school; κ = 9/4).
- Park, Moore & Baalrud (2026) on PIC numerical thermalization time vs macroparticle weight.
- Any work on self-similarity or nonthermal fixed points in Vlasov tail formation, phase-space cascades, and electron-hole coarsening.

State what is already claimed, and what would remain novel: regulator-converged moving features, predictive transient scaling, and attraction in freely relaxing dynamics. Mark comparisons incomplete where a source wasn't accessible.

## 9. Deliverables and completion

- `analysis/` library with docstrings; `tests/` (pytest) covering §2–§7.
- `protocol/DATA_CONTRACT.md` and `protocol/PROTOCOL_TEMPLATE.yaml`. The protocol template has the fields T2/T3 must freeze before seeing data: bins, thresholds, training and held-out windows, gate tolerance τ, feature definitions, model-comparison rule, and quality gates.
- `reports/T1_REPORT.md`:
  - the S1–S7 confusion table;
  - which features are identifiable at which N;
  - the minimum number of seeds and nested levels the gate needs to separate ln N from N onset scaling at the effect sizes in S6 (this sets T2's run count);
  - known limitations.
- `reports/PRIOR_ART.md`.

Status line for this task: `ANALYSIS_VALIDATED` / `ANALYSIS_PARTIAL` (list the failing cases) / `ANALYSIS_FAILED`.

Do not run any plasma simulation, do not touch HEXAPIC, and do not write T2 code beyond the data-contract writer stub.
