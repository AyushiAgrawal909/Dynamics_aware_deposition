# Codex task T3 of 3: self-similar scaling, forecasting, and attraction (only if T2 returned `GO_T3`)

**Prerequisite:** `experiments/nonthermal_attractor/results/T2_DECISION.json` has gate outcome `GO_T3`. **If it doesn't, stop and report.** Work only with the features T2 marked regulator-converged, at the regulator levels T2 showed to be converged, and with the Vlasov reference run alongside. Reuse T1's library and T2's backends unchanged; any change to them requires re-running T1's suite and T2's validations V1–V4.

**Question:** Do the converged features evolve self-similarly with **identifiable** exponents? Do frozen early-time fits forecast later evolution better than simple baselines? Do different preparations approach the same scaled form?

## 0. Freeze the protocol first

Write `protocol/T3_PROTOCOL.yaml` and commit it before fitting. It must fix:

- the training and held-out windows;
- the t₀ profiling range;
- bins and the tail threshold (inherited from T2);
- the features;
- the parameter bounds;
- the forecast-error metrics;
- the baseline set;
- the attraction normalizations allowed (based on invariants, fixed in advance);
- which preparation is the untouched test;
- the quality gates.

After this commit, no held-out data is inspected for tuning. If a prediction fails, report the failure; do not rescue it by retuning.

## 1. Scaling fit (training window only)

- Fit N(ε, t) = A★ s^α F(s^β ε/ε★) and the analogous form for ρ(η, t), using T1 §5.
- Report:
  - the α–β covariance and profile-loss contours;
  - the t₀ uncertainty;
  - boundary hits;
  - sensitivity to window, bins and threshold;
  - seed-bootstrap CIs.
- If the identifiability detector returns `UNIDENTIFIABLE_DEGENERATE`, report the identified combination α − pβ and stop the exponent claim there. A stationary ε^(−2) tail is the expected degenerate case and is predicted by two different theories (Ewart et al.; Banik et al.). Its exponent therefore can't discriminate between them.
- Sector budgets: use the moving-boundary relations with the flux term. For p ≈ 2, track both cutoffs.

## 2. Forecast without refitting

- Freeze α, β, t₀, the scales and the template. Then predict N_pred(ε, t) = r^α N(r^β ε, t_ref) over the held-out window, masking unsupported energies.
- Metrics:
  - held-out distribution error;
  - feature-trajectory error;
  - Q_N and Q_E error;
  - the forecast horizon at which the error exceeds a frozen threshold.
- Evaluate on individual seeds and on the ensemble.
- Baselines get the same training information:
  - stationary;
  - matched-energy 1V Maxwellian;
  - one-scale heating;
  - quasilinear diffusion, only if its assumptions are justified, otherwise `NOT_TESTED`.
- The oracle Maxwellian built from future energy is reported as `ORACLE` and excluded from ranking.

## 3. Attraction ("turbulent amnesia")

Different preparations carry different Vlasov Casimirs. Matching density, momentum and energy does **not** match level content. Attraction therefore means the turbulence erases the memory of the initial ρ(η). Frame and report it that way.

- **Preparations:** at least three, plus one untouched test. All must sit inside the strong-turbulence regime (Δv_b/v_b ≤ 0.25 for waterbags), so that a negative result is not a basin boundary mistaken for failure. Record ρ(η, 0) for each. Defaults:
  - P1: the T2 waterbag preparation (Δv_b = 0.20 v_b);
  - P2: waterbag with Δv_b = 0.15 v_b, run only if T2's cost estimate allows; otherwise 0.25 v_b;
  - P3: two-Gaussian beams at ±v_b with w = Δv_b/√12, which has the same second moment as P1 but different level content;
  - P4, held out: a level-content-preserving rearrangement of P1. Cut each waterbag into slabs and shift them in v so that the same η levels occupy the same volume, preserving ρ(η, 0) exactly while changing the arrangement and the energy as documented. This distinguishes "same Casimirs → same endpoint" from "same moments → same endpoint".
- Each preparation is run at T2's converged level, with ≥ the T1 minimum number of seeds and the Vlasov reference. Seeds estimate sampling uncertainty only; they are not preparations.
- **Test:** do the scaled functions F_i agree within CI using only the pre-registered normalizations? Never warp trajectories individually to force agreement.

## 4. Stability exponents

Mark these `NOT_TESTED` unless there is a controlled family of small perturbations around one preparation (at least 3 amplitudes). With few preparations, the relaxation exponents λ_j in

N_tilde(x, s) = F(x) + Σ_j c_j s^(−λ_j) φ_j(x)

are unidentifiable. If attempted:

- separate the time-shift, normalization and exponent-fit tangent directions from physical residuals;
- call any residual-decay fit a diagnostic, not a derived spectrum;
- do not assert a closed Markovian equation for N; list the unresolved terms instead.

## 5. Deliverables

- `reports/T3_REPORT.md` with separate statuses (`SUPPORTED` / `REJECTED` / `INCONCLUSIVE` / `NOT_TESTED`) for:
  - approximate self-similarity;
  - scaling identifiability;
  - forecasting vs baselines;
  - attraction;
  - Casimir-vs-moment memory (P4);
  - stability exponents;
  - physical mechanism.
- Plots:
  - collapse with count support;
  - α–β loss landscape;
  - feature motion;
  - held-out forecasts vs baselines;
  - per-preparation scaled functions;
  - budgets;
  - PIC vs Vlasov.
  Include a manifest for every curve.
- `results/T3_SUMMARY.json` with the fit, forecast and quality summaries.
- `reports/NEXT_STEPS.md` with the smallest decisive next experiment and its cost. The natural candidate is the 2D phase in HEXAPIC, with wide-enough transverse domains for hole breakup (the current Ny = 16 domains are too narrow) and N(ε) as the primary 2D observable, since ρ(η) in 2D2V needs prohibitive counts per bin.

Recommend expanding the campaign only if all of the following hold:

- a regulator-converged moving feature;
- identifiable exponents;
- forecasts that beat all non-oracle baselines on held-out data;
- attraction that holds on the untouched preparation.

Make no PRL or priority claims from a collapse plot.
