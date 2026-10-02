# Missing validation data retrieval - 2026-09-21

## Scope and outcome

This is an existing-data analysis only. No simulation was submitted, no saved
campaign output was overwritten, and no manuscript file was edited.

| Question | Result |
|---|---|
| Complete 85-candidate rule | **PARTIAL.** The 5%-margin rule selects `q050_x000`; its error improvement is independently measured, but no dynamics history exists at that exact width. |
| Three-arm Landau reanalysis | **COMPLETE from existing histories.** Forty paired complex histories per arm were recovered and reduced over the fixed common interval `[5,28]`. CIC and the one-cell Gaussian fail the two-e-folding duration gate; the wide comparator passes duration but fails rate preservation. |
| Finite-marker/loading retrieval | **COMPLETE from existing data.** Exact current-initializer predictions, rates and loading moments were joined at `v_b=0.80`. |
| Controlled loading/PPC convergence | **MISSING.** No existing rate campaign varies PPC while controlling beam-sign/thermal-velocity covariance. A smallest-useful study is specified but was not submitted. |

## 1. Complete 85-candidate bandwidth rule

The recovered sweep is `e2_anisotropic_dense_5197174`. It contains CIC plus
four protected-mode budget shells (`q/q_10% = 0.25, 0.50, 0.75, 1.00`) and 21
x/y allocations on each shell: 85 candidates total. The fixed selection seeds
are `100--104`; the independent evaluation seeds are `105--109`.

The rate prediction was recomputed with the audited grid/time-discrete
continuum-velocity response (`qmax=8`, `lmax=1024`). The model-feasibility
rules used here are:

- 5% operating rule: predicted `gamma/gamma_CIC >= 0.95`;
- 10% acceptance boundary: predicted `gamma/gamma_CIC >= 0.90`.

All 85 candidates satisfy the inclusive 10% model boundary. Forty-three
candidates (CIC, the 0.25 shell and the 0.50 shell) satisfy the 5% operating
rule. Minimising either selection-seed objective within those 43 candidates
chooses the same point:

| Objective | Selected point | `(sigma_x/dx, sigma_y/dy)` | Predicted rate ratio | Held-out CIC/error ratio, 95% paired interval | Actual MSE decrease |
|---|---:|---:|---:|---:|---:|
| Full-grid density MSE | `q050_x000` | `(0, 0.1716393)` | 0.9513009 | 1.079880 `[1.077915, 1.081663]` | 7.40% |
| Pre-return grid-field MSE | `q050_x000` | `(0, 0.1716393)` | 0.9513009 | 1.038413 `[1.036210, 1.040776]` | 3.70% |

The constrained nested bootstrap selects `q050_x000` in 100% of 5,000 draws
for both objectives. This demonstrates independent held-out error improvement.

It does **not** complete the dynamic part of the rule. No rate history was found
at `sigma_y/dy = 0.1716392815`. The existing qualified `R=0.95` drift-scan arm
uses `0.1738592384`, 1.293% wider. Its individual-rate ratio is 0.95446
`[0.94407, 0.96310]`, and its registered ensemble-amplitude ratio is 0.96986
`[0.95376, 0.97301]`; these are useful nearby evidence but were not substituted
for the exact selected candidate. The earlier `q100_x000` point at
`sigma_y/dy=0.2427346` is the 10%-boundary optimum and is explicitly excluded
by the 5% operating rule.

Therefore the complete-rule verdict is: **independent error PASS; exact-width
dynamic acceptance UNRESOLVED**.

## 2. Three-arm weak-Landau reanalysis

The authoritative inputs are 120 binary histories under
`weak_landau_confirmatory_v2_production_20260911/low`: 40 seeds (`202609300--339`)
for each of CIC, the cell-isotropic one-cell Gaussian and the fixed CV-derived
wide Gaussian. Pairing manifests verify byte-identical initial particle states
within each seed across the three arms.

Common configuration: PPC 1024; global inverse-normal quiet-start velocities;
regular `2048 x 512` marker lattice; `64 x 16` field grid; `k lambda_D=0.4`;
`alpha=0.01`; `dt=0.05`; histories through `t=90` at cadence 0.1. The primary
observable is `abs(mean_seed(E_hat))`, followed by the frozen 49-sample centered
geometric envelope. The late floor is the minimum rolling median over complete
five-time-unit windows in `[70,90]`.

Paired-seed bootstrap, 5,000 draws, fixed common window `[5,28]`:

| Arm | Coherent gamma, 95% interval | Ratio to CIC, 95% interval | E-foldings | Signal/fit excluding duration | Two-e-fold duration |
|---|---:|---:|---:|---:|---:|
| CIC | -0.068791 `[-0.071438,-0.066209]` | 1 | 1.582 | PASS | **FAIL** |
| Gaussian `sigma=1 dx` | -0.070039 `[-0.072749,-0.067384]` | 1.01814 `[1.01486,1.02121]` | 1.611 | PASS | **FAIL** |
| Fixed CV-derived `sigma=7.2443 dx` | -0.129881 `[-0.143950,-0.117233]` | 1.88805 `[1.66753,2.14374]` | 2.987 | PASS | PASS |

Thus the one-cell Gaussian is rate-preserving on this common window but CIC
and the Gaussian still fail the registered duration requirement. The wide arm
has a measurable interval, but its rate ratio is far outside the 10% tolerance;
passing measurement duration is not a dynamics-preservation pass.

### Damping versus phase dephasing

For each arm the coherent log slope was decomposed exactly as

`gamma_coherent = gamma_mean(|E_s|) + slope(log coherence)`,

where `coherence = abs(mean(E_s))/mean(abs(E_s))`; all three terms use the same
49-sample geometric smoothing and `[5,28]` window.

| Arm | Mean-seed-magnitude slope | Coherence log slope | Smoothed coherence at 5 -> 28 | Seed phase-slope SD |
|---|---:|---:|---:|---:|
| CIC | -0.065958 | -0.002833 | 0.9974 -> 0.9247 | 0.01565 |
| Gaussian `sigma=1 dx` | -0.066994 | -0.003045 | 0.9974 -> 0.9183 | 0.01514 |
| Fixed CV-derived wide | -0.088064 | -0.041816 | 0.9936 -> 0.3523 | 0.34343 |

The wide filter therefore changes both individual-amplitude decay and phase
coherence. Relative to CIC, the coherence term accounts for 63.8% of the
additional coherent log-slope, with paired 95% interval `[57.4%,68.8%]`. The
illustrative 73.2% attribution is outside that interval under this explicit
decomposition and is not supported. This is an ensemble log-slope accounting
identity, not a causal intervention; the saved data demonstrate strong
dephasing but do not prove its unique cause.

## 3. Finite-marker predictions and loading convergence

The exact current audited initializer can reconstruct the `v_b=0.80`, seeds
`200--209`, PPC-512 states. The joined table contains all three arms and every
seed's initialization-only finite-marker root, fitted individual rate and
loading moments.

| Arm | Mean finite-marker prediction | Mean fitted individual rate | Prediction/measurement correlation | RMSE |
|---|---:|---:|---:|---:|
| CIC | 0.124198 | 0.124451 | 0.999885 | 0.000488 |
| Fixed Gaussian boundary control | 0.112651 | 0.113115 | 0.999648 | 0.000891 |
| Interior constrained Gaussian | 0.118435 | 0.118783 | 0.999808 | 0.000656 |

For CIC the observed-rate correlation with realised beam-sign/thermal-vx
covariance is -0.94646. This supports finite-marker loading as the leading
explanation of the seed spread in these ten realisations. It does not by itself
establish a PPC convergence law or causality. The predictor is an
initialization-only principal represented-mode, diagonal `+k` approximation;
it omits spatial aliases and finite-sample `+/-k` mixing and is not the exact
response of the complete update.

Historical seeds `0--9` remain nominal reconstructions: their original marker
arrays and exact source revision were never saved, so their loading diagnostics
cannot be upgraded to exact provenance.

A read-only MeluXina inventory found dynamic E2 outputs only at PPC 512:
120 drift-scan histories, 30 extension histories, 50 compensated-corner
histories and 20 new-mode histories. The PPC-32/1024 anisotropic data are static
error/reference measurements, not a dynamics convergence study. The historical
PPC-2048 m=8 aggregate is a different mode/provenance and lacks the required
paired covariance control.

### Smallest useful missing study - not submitted

If this question must be closed later, the smallest useful design is CIC only
at the existing `(12,1)`, `v_b=0.80` configuration, PPC `{128,512,2048}`, ten
fixed paired seeds per PPC: 30 histories. Use nested regular positions and
antithetic per-cell beam/thermal quartets so beam balance and
`mean(sign*thermal_vx)` are zero to roundoff at every PPC. Save the exact arrays,
initialization-only roots and complex histories; retain the fixed
`[24.32,69.16]` measurement window and stop at `t=71.16`.

Using completed job 5195640 as the empirical basis, the estimate is 2.90 CPU
node-hours, with a reviewable hard cap of 3.65 node-hours. No part of this study
was launched.

## Artifacts

- `bandwidth_85_complete_rule.csv`: all candidates, objectives, predictions,
  feasibility flags and fixed seed split.
- `bandwidth_85_per_seed_objectives.csv`: all 850 seed-candidate objective rows.
- `bandwidth_complete_rule_selection.csv` and JSON: selected points, held-out
  intervals and exact-width dynamics status.
- `landau_individual_complex_histories.csv`: all 108,120 complex samples.
- `landau_histories_and_envelopes.csv`: ensemble histories, envelopes and
  coherence diagnostics.
- `landau_three_arm_common_window.csv`, `landau_phase_decomposition.csv` and
  `landau_paired_bootstrap_draws.npz`: central and resampled results.
- `e2_vb080_finite_marker_loading_joined.csv`: exact current per-seed prediction,
  measurement and loading data.
- `e2_existing_ppc_loading_inventory.csv` and
  `e2_smallest_useful_convergence_study.json`: evidence gap and bounded design.
- `analysis_manifest.json`: no-run/no-manuscript-edit record.

The complete reconstruction command and read-only remote inventory commands are
recorded in `COMMANDS_AND_PROVENANCE.md`.
