# Audit changelog

## 2026-09-12 research-completion closeout

- Recovered separately frozen MeluXina extension job `5202227`: exit `0:0`,
  elapsed `02:21:40`, all 30/30 requested histories present. Verified the
  original campaign hash `e664460d...af446`, amendment hash
  `fc9ae62f...15700`, seeds, arms, widths, time grids and source revision.
  The original job `5201231` remains preserved as a 120/150 four-hour timeout.
- Extended `analyze_E2_beam_drift_scan.py` to accept only a complete, strictly
  validated extension block. Generated the non-overwriting combined result
  `e2_beam_drift_scan_analysis_20260912_complete_extension_r9`. Signal-invalid
  `.818` rows remain in CSV/JSON as descriptive values but are omitted from
  measured curves. Panel (c) is labelled “Predicted admissible bandwidth”; the
  no-branch/unavailable statement is caption metadata rather than an in-scale
  marker. The final PDF was rendered and visually checked.
- Added `complete_E2_estimator_comparison.py` and
  `diagnose_E2_seed_rate_variation.py`. Their non-overwriting outputs cover the
  direct compensated corner, historical Table-5 corner, all completed drift
  points and the extension. Registered results remain primary; the individual-
  rate paired bootstrap, mixture/weighting, leave-one-out, window and finite-
  marker calculations are explicitly post-hoc sensitivities.
- Added `audit_exact_filter_and_spectral_operators.py` and
  `OPERATOR_AND_TRANSFER_AUDIT.md`. The exact binomial/Gaussian and spectral/
  finite-difference corrections affect equation wording and table metadata,
  not existing particle histories, pass ceilings or E2 roots.
- Added `measure_E2_interior_noise_replay.py`. At the exact `R=.95`, `.80`
  interior width it reproduces the compatible saved anchors to `5e-15` before
  reporting density and pre-gather field error-power reductions; the boundary
  result remains separate.
- Added the claim map, campaign manifest, novelty matrix, factual figure
  captions, coauthor response, manuscript-correction ledger, compact estimator
  figure and final research-completion report. `introduction.tex` was not
  modified.
- Added `rebuild_research_completion.py`. Its first end-to-end invocation
  completed every reducer but failed at the final manifest because a Python
  boolean was written as `false`; that partial output is retained. Corrected
  the literal to `False`; the `_r2` rerun passed. After adding caption metadata,
  the final-current-code `_r3` rerun also passed with 35 hashed products and no
  simulation submission. Canonical and rebuilt key CSVs are byte-identical.
- Final affected regression results: 84/84 `unittest` tests and 26/26
  secondary-claim `pytest` tests pass. PDF/PNG figure renderings were visually
  inspected; JSON manifests parse successfully.

## 2026-09-11 E2 estimator diagnosis and fixed-design extension

- Preserved the registered arithmetic-mean-amplitude rates and paired
  intervals. On the frozen `vbeam=0.80` CIC window, the registered rate is
  `0.1447459`; the ten individual-seed rates have mean `0.1244508`, SD
  `0.0233284`, median `0.1227335`, and range `[0.0938376,0.1638252]`.
- Reconstructed an arithmetic mixture from the fitted seed exponentials. Its
  rate `0.1447311` matches the registered rate within `1.48e-5`. The effective
  amplitude-contributor count falls from `7.405` to `3.280`, the largest share
  rises from `0.2213` to `0.4808`, and rate versus contribution-change
  correlation is `0.7225`. The estimator component is 74.39% of the registered
  absolute theory gap; the residual mean-individual difference is 5.61% under
  the manuscript convention.
- Matched the current endpoint to the historical corner window. This changes
  CIC by only `0.0015986`. The remaining shared-endpoint campaign difference is
  `0.0163464`; seeds differ (`200..209` versus `0..9`) and the historical
  source revision was not recorded, so those two effects cannot be isolated.
  Current transfer regression remains a roundoff-level pass.
- Renamed panel (c) to "Predicted admissible bandwidth" and removed the red
  triangle that could resemble a measured failed point. The factual caption
  states "No growing model branch; measurement unavailable." The figure data
  products in timeout `r8` are byte-identical to registered `r6`; only that
  rendering changed.
- Froze an explicit one-node, eight-hour amendment for exactly the 30 original
  `vbeam=0.818` histories, with the same arms, seeds, window and gates. Submitted
  it separately as job `5202227`; the four-hour job `5201231` remains recorded
  as incomplete. No extension result had been inspected at freeze/submission.
- Added five estimator/extension regressions; the full suite passes 84/84.
- Compiled the authoritative manuscript to the retained 37-page
  `Dynamics_constrained_Deposition_verified_20260911_r4.pdf`, SHA-256
  `be7ee1a4...5af9`; the structural audit passes and the revised figure plus
  diagnostic-text pages were rendered and visually checked.

## 2026-09-11 E2 launcher incident and replacement

- Diagnosed GPU job `5201218` as a pre-simulation launcher failure, not a
  completed campaign.  Slurm reports `FAILED`, exit `1:0`, elapsed 20 seconds;
  every one of the 40 attempted first-batch processes raised
  `FileExistsError` because the launcher created its runner-owned output path.
  No mode histories, result JSON files, or campaign manifests were written.
- Preserved the complete 43-file failed tree plus scheduler logs under
  `pic_kde_2d/results_raw_rate_audit/recovered_meluxina_20260911/`.  The remote
  and local tree aggregate SHA-256 is `ac857b19...0020`.
- Moved only launcher cache/log paths to sibling trees, retained the runner's
  non-overwrite guard, and added a regression for the interface.  The patched
  launcher SHA-256 is `440c435f...f15`; after timeout/result anchors were added,
  focused tests pass 9/9 and the full suite passes 79/79.
- Submitted replacement GPU job `5201231`.  It reached the frozen four-hour
  limit after 120/150 histories: every arm completed at drifts `.60`, `.70`,
  `.76`, and `.80`; the `.818` block was not written.  The 513-file remote and
  local archives match at SHA-256 `0d00fe2d...75114`.  No continuation was
  submitted because that would silently expand the preregistered budget.
- Extended the reducer with an explicitly evidenced timeout-censored mode;
  strict completeness remains the default and was observed to fail at the
  first missing file.  All eight completed filtered comparisons pass their
  rate-CI, signal and duration rules; `.818` is
  `UNRESOLVED_BUDGET_TIMEOUT`.  The fixed filter's measured ratio changes from
  `.99278` to `.94064` while its transfer remains constant, and the constrained
  ratios are `.95059`, `.95173`, `.95546`, and `.96986`.
- Added the visually checked three-panel figure and a scope-limited account to
  the authoritative manuscript.  The absolute CIC prediction error grows from
  0.91% to 18.85%, so model calibration is kept separate from rate
  preservation.  The frozen config SHA-256 `e664460d...af446` is unchanged.
- Compiled the authoritative source to the retained 37-page
  `Dynamics_constrained_Deposition_verified_20260911_r3.pdf`, SHA-256
  `fdb97e6c...9e73`.  The structural audit passes with 86 labels, 64 cited
  keys and nine mirrored figures; the new figure page was rendered at 180 dpi
  and visually checked.  The complete audit suite passes 79/79.

## 2026-09-11 E2 beam-drift preregistration

- Added and froze the warm-oblique numerical design for a symmetric E2
  `(12,1)` beam-drift scan before reading any scan simulation.  The five drifts
  are `0.60,0.70,0.76,0.80,0.818`; the grid/time marginal point is
  `0.8191540597` and is converged through `qmax=10,lmax=2048`.
- Verified with the actual PPC-512 initializer that paired positions, charges,
  thermal samples, beam membership and initial full/delta-f densities are
  invariant under the drift change.  Measured all frozen Gaussian transfers
  against the implementation to roundoff.
- Frozen three arms, ten fresh seeds `200..209`, 5000 paired resamples,
  per-drift duration, all fit/signal rules, and a 0.95 constrained-rate target
  that leaves five percentage points inside the 10% boundary.  The fixed
  filter is `(0,0.2427346)` with complete matched transfer
  `W^2=0.9909549371` at `(12,1)`.
- Added non-overwriting runner/reducer, regression tests, and the three-panel
  figure generator.  The suite passes 76/76 tests.
- CPU jobs `5201211/5201212` never started (`AssocGrpGRESMinutes`) and were
  cancelled.  A recorded scheduler-only amendment moved the unchanged design
  to GPU job `5201218`, preserving the four-hour hard cap.

## 2026-09-11 evidence closure

- Added a non-overwriting common-window reducer for the existing fresh 40-seed
  weak-Landau histories.  It verifies the raw and Stage-1 locks, uses the
  intersection `[5,30.1]`, and returns C4/CIC
  `1.018773 [1.015359,1.021931]` from 5000 paired resamples.  The registered
  arm-specific `0.989415 [0.967884,1.013827]` result remains primary; both
  common-window arms fail the two-e-fold duration gate.
- Added `audit_evidence_closure.py`.  It records that Table-4 noise job
  `4867515` used the same repeated per-cell 32x32 velocity tensor as the
  rejected confirmation and has zero longitudinal nodes at the resonant phase
  velocity.  The values are retained only as density-error measurements for
  that constructed ensemble.  No recovered Landau rate history is identified
  as per-cell; the oldest exact revisions remain unknown.
- Applied the existing pre-emergence-floor estimator to the recovered E2 C4
  corner histories.  Fit-start SNR is `11.213`, candidate duration is 4.113
  e-foldings and shared-window CIC duration is 4.484; the explicit signal and
  baseline-duration checks pass without a new simulation.
- Quantified compensated E2 prediction calibration separately: all four
  grid/time point predictions lie outside the measured 95% intervals, while
  p2/n2 and p3/n4 still pass the measured 10% preservation criterion.
- Regenerated the validation matrix without overwriting earlier versions as
  `validation_audit_v9` (8 PASS, 31 FAIL, 1 UNRESOLVED).  The expanded suite
  passes 70/70 tests.
- Compiled the authoritative source to the new 36-page
  `Dynamics_constrained_Deposition_verified_20260911_r2.pdf`, SHA-256
  `627c0919...8c50`.  The structural audit passes (85 labels, 64 citations,
  eight figures), and affected pages were rendered at 180 dpi and visually
  checked.  Previous outputs remain unchanged.

## 2026-09-09

- Added `REPRODUCIBILITY_MANIFEST.md` with recovered run settings and explicit
  revision status.
- Added `MISSING_INFORMATION.md`; unknown settings remain unknown.
- Added `build_hash_manifest.py`, which hashes inputs read-only and writes
  `SHA256SUMS.txt` beside this file.
- No simulation output was modified, renamed, or deleted.
- No production simulation was submitted.
- Installed `pypdf` and `pymupdf` only in
  `/private/tmp/paper1_audit_pdf_deps` to inspect the supplied PDF.  This did not
  alter the project environment.
- Added `NUMERICAL_FORMULATION.md`, documenting the recovered Python and C++
  particle--field chains, exact smoothing locations, gather semantics, HYPRE
  configuration, boundary treatment, and PETSc status.
- Ran the dependency-free existing-test baseline: 9/9 selected import, FFT,
  and momentum tests passed.  The active environment lacks `pytest`, so the
  pytest-dependent C4 test was explicitly left unexecuted.
- Added `measure_filter_transfers.py` and its `unittest` regression.  These are
  deterministic operator tests only; they do not advance a plasma simulation.
- Added `FILTER_TRANSFER_AUDIT.md` and machine-readable JSON/CSV.  All 24
  controlled mode/filter rows matched independent transfer predictions to
  floating-point precision; 3/3 regression tests passed.
- Corrected the manifest paths for the binomial campaigns: their preserved
  implementations are in `pic_kde_2d_landau_cv_stage/`, not the newer
  top-level Python copy.
- Added `recompute_dispersion.py`, five `unittest` regressions,
  `DISPERSION_AUDIT.md`, and machine-readable dispersion/convergence/ceiling/
  finite-marker tables.  These are read-only diagnostic calculations; no
  production simulation was submitted and no historical output was changed.
- Recovered the exact E2 transverse loading (`vthermal_y=.05`), `Ly=16`,
  `dy=1`, `dt=.02`, delta-f PPC 512 and `alpha=1e-8`.  Separated continuum,
  principal-CIC, spatial-alias and explicit-midpoint predictions.
- Found that the compensated-filter E2 launchers measured `(12,0)`, not the
  binding corner `(12,1)`.  Updated `paper1.tex` to distinguish measured
  longitudinal results from predicted corner values and to mark direct corner
  certification as outstanding.
- Corrected the E2 benchmark description and table in `paper1.tex`, removed
  the obsolete settings note, and labelled the existing bandwidth limits as
  principal-CIC values alongside the independently computed grid/time values.
- Copied the original energy job `5048437` and HEXAPIC momentum jobs `4855533`
  and `4858823` from MeluXina without modifying their contents.
- Added `analyze_conservation.py`, `run_conservation_small_tests.py`, six
  `unittest` regressions, `CONSERVATION_AUDIT.md`, machine-readable conservation
  tables and three audit figures.
- Submitted the diagnostic-only, four-point, one-step PFMG tolerance job
  `5195488`.  Its launcher is preserved as `run_hypre_tolerance_scan.sbatch`;
  raw output records source and executable hashes.  No production physics
  simulation was launched.
- Established that the 42.8% grad-h change and 1817-fold discontinuity result
  are x-component comparisons.  The corresponding two-dimensional norm ratios
  are `1.00001368` and `16.0284`.
- Established that the one-step variable-width force does not decrease as the
  achieved PFMG residual is tightened from `9.60e-6` to `9.89e-12`; revised
  the manuscript to avoid attributing that floor to solver tolerance.
- Recomputed energy and charge from all 90 raw Python histories.  Charge is
  bitwise constant; eight of nine seed-bootstrap energy-slope intervals exclude
  zero.  Replaced the manuscript's unsupported “no secular growth” and
  spectral/HYPRE aggregate claims with explicit metrics and scope.
- Added controlled matched-operator and complete particle-force checks.  Both
  spectral and periodic centered-finite-difference exact operators are at the
  skew-adjoint round-off floor; a short fixed-space scan shows second-order
  successive time-step differences while retaining a nonzero spatial/coupling
  energy floor.
- Copied 720 rate-history NPZ files and their metadata/archived reductions from
  MeluXina into the separate `pic_kde_2d/results_raw_rate_audit/` tree.  Existing
  local summary directories were not merged or overwritten.
- Added `rebuild_rate_analysis.py`, `summarize_rate_bootstraps.py`, five
  regressions, `RATE_UNCERTAINTY_AUDIT.md`, and the complete Task-6 rebuild under
  `results/rate_uncertainty_audit_v7/`.  No production simulation was launched.
- Regenerated 14 rate tables from raw histories.  Thirteen match archived CSVs;
  strong alpha 0.5 with fit start 5 reproduces all numbers but records the
  current `UNRESOLVED` label versus historical `FAIL`.
- Established that Landau fits use a coherent seed-ensemble trace and that E2
  uses a mean of seed amplitudes; bootstraps preserve paired seed indices and
  never resample time.  Corrected the manuscript's per-seed/fixed-window claim.
- Corrected the alpha-0.5 “2.2 e-foldings” error: 2.2 is the fit duration, while
  the fit has 0.656 model e-foldings.  Added the fit-start-3 result (1.242
  e-foldings), retained both conditional rate ratios, and marked both
  finite-amplitude cells duration-unresolved.
- Narrowed the domain-boundary claim: the selected width crosses the analytic
  ceiling between alpha 0.25 and 0.5, but the current data do not establish a
  measured unsafe-to-safe validation crossover.

## 2026-09-10

- The user identified
  `Dynamics_constrained_Deposition/introduction.tex` as the authoritative
  manuscript.  Ported the evidence-supported Task 1--7 changes previously
  made in `paper1_source_package/paper1.tex` into that file, adapting them to
  its existing layout.  The package copy is retained as a non-authoritative
  audit artifact rather than deleted or silently reverted.

- Treated the user's confirmation that all historical jobs were submitted as
  a requirement to search MeluXina before labelling local artifacts missing.
  Recovered, without changing cluster originals, the raw campaigns
  `landau_binomial_strong_5048414`, `E2_binomial_5048416`,
  `p1_landau_rates_4866573`, `e2_joint_corner_4913408`, and
  `landau_ppc_dynamics_5049265` into the separate
  `results_raw_rate_audit/recovered_meluxina_20260909/` tree.
- Extended `rebuild_rate_analysis.py` to rebuild the recovered campaigns and
  added strict per-campaign E2 mode assertions.  The 19-table rebuild is in
  `results/rate_uncertainty_audit_v8/`: 18 reproduce exactly within the stated
  numerical comparison tolerance; one differs only in a historical/current
  gate label.  Fixed `--list` so it no longer requires a dummy output path.
- Added `reassess_validation.py`, `VALIDATION_AUDIT.md`, six regression tests
  and `results/validation_audit_v3/`.  The gate is two-sided `[0.90,1.10]`,
  requires at least two fitted model e-foldings, and separates signal quality,
  protected-mode scope and raw traceability.  All 25 cumulative audit tests
  pass.
- Reclassified measured sub-two-e-fold cells as `FAIL (duration)`, not
  `UNRESOLVED`.  No weak-Landau condition passes the complete gate.  The
  recovered Gaussian E2 `(12,1)` C4 row passes; compensated p2/n2 and p3/n4
  pass only on their actually measured longitudinal `(12,0)` scope.
- Recovered the strong raw-binomial sweep.  At n=16, the fit-start-3 rate CI
  passes but spans 1.311 e-foldings; the fit-start-5 ratio is 1.17925 and fails
  both rate and duration.  Removed the manuscript claim that all tested strong
  pass counts pass.
- Established that the weak LOO-CV selected width is
  `7.244293147028428 dx`, while the reported ratio `1.690` was measured at
  sigma `6.61`.  Revised the manuscript to keep proxy and selected-width
  evidence separate.
- Added and smoke-tested `analyze_E2_compensated_corner.py`; on the recovered
  Gaussian corner it reproduces every central rate to better than `1e-12` and
  computes each candidate's own e-fold span and fit-start SNR.
- Submitted direct compensated E2 binding-corner array `5195640` and dependent
  reducer `5195641`, after matching local/remote runner and filter SHA-256
  hashes.  Outputs go to a new tier-2 validation directory.
- Completed `5195640/41` with exit code zero. Direct `(12,1)` ratios are p1/n1
  `0.83807 [0.81930,0.84994]` (FAIL), p2/n2
  `0.95608 [0.95120,0.95920]` (PASS), p2/n4
  `0.78885 [0.76359,0.80469]` (FAIL), and p3/n4
  `0.97342 [0.97047,0.97529]` (PASS); every arm passes duration and signal and
  all 5000 paired resamples are valid. Copied all 50 histories and reducer
  products locally; the local/tier2 aggregate manifest hash is
  `1376b97196782c33193fdb6c36f400579ffbab5bc4aa604bc98c57908f006b7f`.
- Submitted weak-Landau three-seed alpha-0.02 duration pilot `5195658` and
  dependent reducer `5195659`.  Promotion to ten seeds is conditional on an
  unrelaxed two-e-fold and linear-rate check.
- Submitted paired exact-LOO-width weak-Landau array `5195682` and dependent
  reducer `5195683`, after a MeluXina metadata search found no prior dynamics
  run at that width.  Historical outputs are not overwritten.
- Completed `5195682/83` with exit code zero.  The exact selected-width rate
  ratio is `1.82222 [1.48934,2.21398]`; the candidate spans `1.62678` fitted
  model e-foldings and is floor-censored with zero quality-passing bootstrap
  fraction.  It fails rate, duration and signal quality.  Copied all 20 raw
  histories and reducer products locally and matched every per-file SHA-256
  against the unchanged tier2 originals.
- Revised the abstract, protocol, tables, captions, limitations and conclusion
  to apply the two-sided gate, identify duration failures, remove the
  unsupported “only selector passing both benchmarks” claim, relabel
  “CI-safe” as “rate-CI-safe,” and stop assigning a proxy rate to the selected
  LOO-CV width.  Pending jobs are not cited as completed results.
- Jobs `5195658/59` completed.  The three-seed alpha-0.02 weak-Landau pilot
  yielded no finite baseline rate because both Gaussian arms were already
  below five times their measured late floor at the fit start.  The pilot was
  preserved locally and was not expanded to ten seeds, as required by its
  pre-registered promotion rule.
- Recovered the two Gaussian Landau-noise campaigns (`4867515`, `4868286`),
  Landau and E2 raw-binomial noise campaigns (`5048431`, `5048435`), and the
  two exact-self LOO-CV campaigns (`5008392`, `5008393`) from MeluXina into a
  separate preservation tree.  Sorted per-file SHA-256 comparisons against
  the remote originals were empty; no source result was overwritten.
- Added `audit_noise_metrics.py`, six regression tests,
  `NOISE_METRIC_AUDIT.md`, and machine-readable Task-8 outputs under
  `results/noise_metric_audit/`.  All 34 Gaussian, 44 filter-family and seven
  E2 directional central values and bootstrap intervals reproduce.
- Established that the reported density “noise reduction” is a ratio of
  ensemble-mean, time-averaged full-grid squared errors against a paired
  high-PPC CIC density.  It is a power ratio over all resolved modes, not an
  amplitude ratio or a protected-mode-excluded high-wavenumber integral.
- Verified electron species normalization, total-charge preservation and the
  current exact particle-phase LOO self correction.  Explicit deletion agrees
  with the quadratic-form correction to at most `8.33e-17` in three controlled
  filter cases.
- Resolved the weak-Landau `2.559` versus `5.643` discrepancy: the metric,
  reference, PPC, window and 51-snapshot cadence match, but the former job uses
  randomized per-cell velocity quadrature and the latter global shuffling.
  Also established that Fig. 7a imports `5.643` from the Gaussian reducer; the
  raw-binomial reducer's matched-low-k `n=2` value is `5.872`.
- Revised the authoritative manuscript to define the discrete metric and its
  in-sample status, disclose the mixed Table-4 protocols, distinguish density,
  pre-gather grid-field and complete particle-force quantities, and document
  the exact LOO self term.  Figure-axis source text now labels the plotted
  quantities as density-error power reductions.
- Recovered all 192 selector snapshot files from job `5049182`, inspected their
  NPZ/JSON schemas, and verified the remote/local aggregate SHA-256
  `3e1c254f...f0f2b`.  Added `validate_selectors_independent.py` with a fixed
  seed-0--4 training/seed-5--9 test split and equal 22-width budgets for MISE
  and LOO-CV.
- The held-out weak-Landau audit selects 7 cells for MISE and 12 for LOO-CV,
  giving density-error power reductions `4.99199 [4.95617,5.02573]` and
  `4.94856 [4.91313,4.98055]`.  All 252 balanced five/five splits retain those
  widths.  C4 at one cell gives `3.32021 [3.30651,3.33788]`; the two-cell
  analytic-grid candidate is not promoted because its historical rate upper CI
  is `1.10959`.
- Found that the weak-Landau Silverman campaign applies the longitudinal width
  isotropically in cell units.  The intended per-direction rule instead gives
  `(1.83297,0.458243)` cells and a held-out density reduction `3.29838`, versus
  `4.44132` for the historical scalar proxy.  The supplied E2 noise and rate
  launchers also disagree with each other and neither implements the configured
  `(5.81933,0.727416)` directional pair.  Revised the manuscript to separate
  model inadmissibility from the missing exact directional measurement.
- Added and smoke-tested the predeclared 85-point admissible E2 anisotropic
  design, then submitted jobs `5197174/84`.  Both density and pre-gather field
  selectors choose `(sigma_x,sigma_y)=(0,0.2427346)` on independent seeds; the
  held-out reductions are `1.15837` and `1.07509`.  The closest sampled-point
  advantages are only `0.2908%` and `0.2181%`, so the manuscript now claims
  discrete-design rather than global optimality.  The 43-file tree has matching
  remote/local aggregate SHA-256 `d43ffbe5...425b`.
- Submitted independent projected E2 mode `(10,1)` jobs `5197185/86` only after
  the implementation and transfer checks passed.  On new seeds `100..109`, the
  measured C4/CIC ratio is `0.989118 [0.988928,0.989253]`, compared with the
  zero-fit-parameter grid/time prediction `0.989247`; 6.976 e-foldings,
  `R^2=.999993` and SNR 232.8 pass every fixed gate.  Copied the unchanged
  63-file tree and matched aggregate SHA-256 `2f7d4267...f5c4`.
- Added `SELECTOR_INDEPENDENT_AUDIT.md` and selector/anisotropy/new-mode
  regressions.  The cumulative dependency-free audit suite now passes 49/49
  tests.  No historical output was overwritten.
- Recovered all 15 delta-f boundary histories from job `4873222`, all five
  full-f overlay histories from `4873285`, all 15 repeat/method records from
  cost job `5049224`, and the G0 headroom/precheck artifacts.  Preserved copies
  match MeluXina aggregate hashes `70abf806...36e`, `ee6d8cb3...d0ebe`,
  `467e3ee0...29b3`, `d17ee858...5fc2`, and `707bd94b...b12`.
- Added `pic_kde_2d/reproducibility/audit_secondary_claims.py`, corrected the
  delta-f boundary reducer to bootstrap independent seed means, added eight
  targeted regression tests, and recorded the full Task-10 audit in
  `pic_kde_2d/reproducibility/SECONDARY_CLAIMS_AUDIT.md`.
- Documented that the production delta-f reference free-streams frozen
  equilibrium markers, evolves perturbation weights by explicit midpoint,
  deposits only `w_delta`, and projects protected runs to one real Fourier
  conjugate pair.  The `0.5` marker-maximum rule is now described as a
  particle-count/tail-dependent heuristic rather than proof of validity.
- Added a paired finite-amplitude target-mode energy diagnostic.  Full-f energy
  rebounds by `4.3776 [4.2568,4.5129]`, whereas projected linearized delta-f
  falls to `0.09159 [0.07119,0.11337]`; the saved data do not contain particle
  phase space or harmonic spectra, so trapping and saturation are not claimed.
- Corrected the controller claim: at H1 contrast 100 the analytic local rule
  improves field MSE but worsens density MISE and protected-mode error.  Its
  preregistered failure stopped the ladder before any learned model was
  trained.  Relabeled the 384-cell G0 statement as an unimplemented analytic
  headroom calculation.
- Audited the cost campaign.  Live filtering is included (`1.318992 ms/step`),
  but selection, construction and initialization are not.  The authoritative
  manuscript now reports only fixed-width steady-state stepping overhead and
  explicitly withdraws any end-to-end selector-efficiency interpretation.
- Backed up the pre-Task-11 authoritative TeX/BibTeX and both figure trees in
  `pre_task11_20260910/` before regeneration; no historical result was changed.
- Regenerated all eight manuscript figures from archived arrays/reducers or
  explicitly labelled analytic transfers. The erasure generator inspected the
  JSON schema before locating per-seed NPZ arrays and passed all requested
  ratio, slope, floor and forbidden-label assertions. The compensator check
  gives `W^2=0.997137575` and attenuation `0.002862425`.
- Found and fixed a figure-provenance error during visual QA: `fig_pareto`
  previously labelled historical compensated `(12,0)` rates as binding-corner
  evidence. It now loads direct job `5195640`, includes the failing p2/n4 arm,
  and keeps separate filter-family noise coordinates explicit. `fig_transfer`
  now loads the held-out 85-point E2 audit.
- Updated the E2 matrix from stale four-point in-sample noise values to the
  held-out dense-design values, while distinguishing the exact selected width
  `0.2427346 dy` from the rounded direct dynamics width `0.2428 dy`.
- Corrected the SD-KDE venue/BibTeX record and narrowed the production-code
  filtering sentence to claims supported by versioned Smilei and WarpX
  documentation. The final structural audit has no missing citations or
  references.
- Added `audit_manuscript.py`, `CITATION_AUDIT.md`, `TABLE_PROVENANCE.md`,
  `REPRODUCIBILITY.md`, and `POINT_BY_POINT_RESPONSE.md`. The script checks
  figure hashes and 300-dpi metadata, numeric anchors, labels/citations,
  placeholders and upright tables.
- Re-ran the final test sets after manuscript/figure changes: the dependency-
  free core audit passes 49/49 and the Task-10 pytest selection passes 26/26.
- Compiled the authoritative source with `elsarticle` and Tectonic 0.17.0 to a
  34-page PDF. Converted the two overflowing wide tables to wrapped `tabularx`
  layouts, rendered every page with Poppler, visually inspected the complete
  document and selected pages at 180 dpi, and removed duplicated “Appendix
  Appendix B” references. Remaining overfull boxes are at most 3.41 pt and do
  not visibly spill content.
- Reconstructed the exact PPC-1024 weak-Landau CIC/C4 window decisions from
  the ten preserved complex histories. The runs extend to `t=90`, but the
  first persistent five-floor crossings at 26.3/29.0 plus the centered-
  envelope look-ahead guard leave accepted endpoints 23.8/26.5 and fitted
  e-foldings 1.311/1.491. The old `t=30` cap is independently incapable of
  spanning two reference e-foldings from `t=5`.
- Added a frozen, fixed-sample 40-seed confirmation specification with exact
  per-cell loading, three paired PPC-1024 arms, a PPC-4096 CIC reference,
  same-campaign density/grid-field error powers, a fully discrete absolute-
  rate gate, and a two-stage window-before-ratio lock. The exact comparator is
  the independently selected `7.244293147028428`-cell LOO-CV width from job
  `5008392`, not the 6.61 proxy.
- Added the confirmatory runner, reducer, four MeluXina launchers and seven
  tests. The miniature run and protected-mode transfer preflight pass; C4 has
  measured `W=0.9951924507`, `W^2=0.9904080139`, with maximum analytic/
  measured error `2.22e-16`. A separate MeluXina-native smoke/preflight passed
  and its exact record is required by every simulation task. Only then were
  the fixed low-PPC and PPC-4096 reference arrays submitted as `5199341/42`;
  the window and ratio stages remain deliberately unsubmitted.
- Reconciled the subsequent Stage-1 artifact without inspecting any rate
  ratio.  It verifies all 40 paired low-PPC histories and the frozen source,
  configuration and preflight hashes, but all three arms censor at `t=5.0`
  with no accepted fit.  The unexpectedly large late floors are now an open
  diagnostic issue; the zero-e-fold outcome is not promoted as physics.
- Preserved completed and running reference tasks in array `5199342`, canceled
  only its seven still-pending elements `33..39`, and resubmitted those exact
  missing seeds as packed GPU-partition job `5200213`.  Added the non-
  overwriting continuation launcher and recorded that the NumPy/Numba solver
  remains CPU-backed despite GPU-partition placement.
- GPU continuation `5200213` completed all seven missing PPC-4096 references;
  the final reference tree contains exactly 40 seeds and no partials.  Copied
  the complete 182-MiB low/reference tree locally and matched its aggregate
  manifest SHA-256 to MeluXina: `91a4ce38...f7d0`.
- Resolved the failed Stage-1 floor anomaly to a loading-admissibility error.
  The v1 per-cell tensor supplies only 32 distinct unperturbed longitudinal
  velocity nodes per cell, with maximum `2.15387`, below the benchmark phase
  velocity `3.21264`.  CIC/C4 are seed-coherent through the target window and
  regrow by factors `3.78/3.94` from their pre-45 minima, so the late estimator
  measured deterministic recurrence rather than random noise.  Stage 2 was
  not run.
- Added a strict JSON/CSV/PDF loading-failure diagnostic and two regressions.
  Its comparison against the preserved global-velocity ten-seed histories
  shows continued damping through the two-e-fold target.  The visually checked
  diagnostic figure is 7.0 by 2.8 inches and the focused suite passes 9/9.
- Added and preflighted a three-fresh-seed diagnostic that changes only the
  velocity assignment to the historical global inverse-CDF shuffle while
  preserving exact spatial PPC.  The corrected support has 1,048,576 distinct
  longitudinal nodes, maximum `4.90096`, and 690 nodes at or above resonance.
  Submitted the non-confirmatory two-arm pilot as GPU-partition job `5200354`;
  no replacement 40-seed campaign is authorized until this pilot is reduced.
- Job `5200354` completed exit zero and all three fresh pilot seeds were copied
  without overwriting prior outputs.  Under the frozen pilot-only `[5,20]`
  diagnostic, CIC and C4 give rates `-0.0700812` and `-0.0715235`, with
  `R^2=0.99887/0.99884` and `1.051/1.073` fitted e-foldings.  The pilot passes
  its loading-behaviour rule and computes no rate ratio.  Applying the future
  confirmatory floor/censor rule diagnostically leaves only `1.045/1.134`
  e-foldings, as expected for three seeds; this does not relax the two-e-fold
  gate or replace the planned fixed sample.
- Froze v2 before production with 40 new seeds `202609300..202609339`, global
  inverse-CDF longitudinal velocities, exact 1024-particle-per-cell spatial
  loading, the same three arms and PPC-4096 reference, and the original
  two-stage analysis lock.  The launch check now counts midpoint-quantile
  support exactly: 690 low-PPC and 2758 reference nodes lie at or above the
  positive resonance.  Local and MeluXina smoke/preflight checks pass, all
  three arm clones match, and measured single-mode transfers differ from the
  analytic values by at most `2.22e-16`; the focused suite passes 11/11.
- Submitted only the frozen v2 simulations as GPU-partition arrays `5200458`
  (40 PPC-1024 paired three-arm histories) and `5200457` (40 PPC-4096 CIC
  references).  Each four-element array packs ten seeds per task and is
  throttled to two tasks.  The numerical solver is CPU-backed; one GPU is
  requested solely for partition placement.  The window-only and comparative
  stages remain unsubmitted, and no v1 or pilot seed will be pooled.
- At the user's request, canceled only the never-started task 0 of each v2 GPU
  array and repartitioned those seeds without overlap.  Four low seeds
  (`300..303`) run on CPU array `5200486`, five references (`300..304`) on CPU
  array `5200487`, and GPU remainder jobs `5200485/88` cover low `304..309`
  and reference `305..309`; original GPU array tasks 1--3 retain seeds
  `310..339`.  Historical maxima from CPU jobs `5199341/42` bound the selected
  nine CPU runs at `25013 s = 6.948` node-hours, 187 seconds below the stated
  seven-hour remainder.  Slurm admitted six immediately and held the last
  three references under `AssocGrpGRESMinutes`; every low and reference seed
  remains covered exactly once.
- All 17 active v2 Slurm elements completed with exit zero; the CPU slice used
  `24825 s = 6.8958` node-hours, 375 seconds below the stated seven-hour
  remainder.  The remote tree has exactly 40 final low and 40 final reference
  seeds, no partials and no nonempty stderr.  Copied all 360 files locally;
  the sorted per-file aggregate SHA-256 matches MeluXina at
  `47019511...8a7a1`.
- Ran and froze the v2 window-only stage before inspecting any comparative
  ratio.  CIC ends at `t=30.1` because of its persistent five-floor crossing
  at `32.6`, giving `gamma=-0.0685811`, `R^2=.999674` and only `1.7214`
  fitted amplitude e-foldings.  C4 ends at `35.6` from its crossing at `38.1`
  and gives `gamma=-0.0678552`, `R^2=.998440`, `2.0764` e-foldings.  The exact
  LOO-CV comparator ends at `28.0`, giving `2.9873` e-foldings.  Thus the
  preregistered campaign promotion already fails the CIC two-e-fold condition;
  Stage 2 remains unrun and the Stage-1 file hash is `cb724768...0cbf`.
- After separately reporting Stage 1, ran the locked comparative reducer on
  the byte-preserved archive.  The C4/CIC rate ratio is `0.989415`
  `[0.967884,1.013827]` and the CIC absolute-rate ratio to the preregistered
  fully discrete reference is `1.033434 [0.989774,1.078710]`; both rate
  intervals pass.  The campaign nevertheless remains FAIL because CIC has
  only `1.721386` e-foldings, and the paired full-quality bootstrap fraction
  is zero.  Same-campaign C4 squared-error-power reductions are
  `3.25865 [3.25180,3.26500]` for density and
  `1.61280 [1.60345,1.62155]` for the pre-return-filter grid field.  The exact
  LOO-CV comparator gives ratio `1.89383 [1.67902,2.13875]`, density reduction
  `4.61153`, and grid-field reduction `0.63238` (worse than CIC).
- Materialized the requested CIC/C4 coherent complex histories, one-period
  geometric envelopes, frozen fit lines, floors and censor markers in
  `results/weak_landau_confirmatory_v2_window_diagnostic_20260911_r2`.  The
  postprocessor reproduces the frozen slopes and fitted e-foldings to its
  strict numerical assertions and recomputes no rate ratio.  The PDF was
  rendered and visually checked; its source CSV retains real and imaginary
  coefficients at all 901 diagnostic times.  The initial output directory,
  which contains only its completed CSV after the first plotting invocation
  ended during font-cache initialization, is retained rather than overwritten;
  the `_r2` directory is the complete artifact.
- Regenerated the interim 40-row validation audit (`6 PASS`, `31 FAIL`,
  `3 UNRESOLVED`) and passed all then-current dependency-free regressions.
  Compiled the revised
  authoritative manuscript to a visually checked 35-page PDF at
  `Dynamics_constrained_Deposition/output/Dynamics_constrained_Deposition_verified_20260911.pdf`;
  SHA-256 is `9eb969950930841b1f7d71508e62bfccdf2cfc0eea52a9c70cb9d14f9bd3243a`.
- A final read-only, job-ID-guided MeluXina search found the previously missing
  delta-f reference (`4872599`), isotropic m12 Gaussian (`4872745`), and full
  1--14 band (`4873221`) raw trees under `/mnt/tier2/users/u103576/logs`.
  Copied them without overwriting earlier archives.  Remote/local aggregates
  match at `534466ba...3453` and `cb3fa4ba...5c52`; the m12 native-file hash
  matches at `1e4a721f...9e35`, while its remote sigma-zero symlink was
  deliberately materialized into a self-contained local tree.  Current
  reducers reproduce the archived tables to at most `3.38e-14`.
- Regenerated the non-overwriting validation audit as v7.  Restored raw
  traceability changes the 40-row count to `8 PASS`, `31 FAIL`, `1 UNRESOLVED`:
  the longitudinal Gaussian 0.184455 and delta-f m8 reference rows now pass.
  Weak-Landau and E2 binding-corner conclusions are unchanged.  The cumulative
  dependency-free suite now passes 65/65 tests.
- On 2026-09-14, added `complete_scientific_evidence_20260914.py`, a read-only
  reducer over existing histories and source tables.  Its final non-overwriting
  output is `results/scientific_evidence_completion_20260914_r3/`: a merged E2
  estimand/predictor table, finite-marker residual table, weak-Landau PPC-4096
  fit, finite-reference decomposition, practical-benefit source table, and
  four vector figures with 300-dpi PNG companions.  Earlier `_r1` and `_r2`
  attempts are retained; `_r1` exposed a JSON nesting assumption and `_r2`
  was complete but had overlapping practical-figure labels.
- Added `audit_E2_finite_reference_covariance.py`.  It deterministically
  replays the already verified ten-seed PPC-32/PPC-1024 E2 short-window
  calculation, refuses overwrite, and asserts every archived scalar MSE to
  absolute tolerance `5e-15` before writing the missing variance/covariance
  decomposition.  This is an estimator reconstruction, not a new parameter
  campaign or production simulation.
- Added `DISCRETE_FINITE_MARKER_TECHNICAL_SUPPLEMENT.tex`, which derives the
  continuum-equilibrium, grid/time-discrete, and seed-specific finite-marker
  responses from the actual research source.  It labels the finite-marker
  residual as a principal-mode diagonal approximation, identifies all inputs,
  excludes fitted rates from prediction, maps equations to functions/hashes,
  and records unavailable historical revisions.  Tectonic compilation and
  Poppler inspection produced a seven-page PDF with embedded fonts in the
  `_r3` evidence directory.
- Added `EVIDENCE_RECONCILIATION_20260914.csv`,
  `HISTORICAL_LANDAU_DATASET_CLASSIFICATION_20260914.csv`,
  `PAPER1_SCIENTIFIC_EVIDENCE_COMPLETION_20260914.md`,
  `PROPOSED_SCIENTIFIC_REPLACEMENTS_20260914.tex`, and
  `REVIEW_RESPONSE_SCIENTIFIC_EVIDENCE_20260914.md`.  Updated the focused
  novelty ledger to distinguish Evstatiev et al. (2021) on noise/error and
  arbitrary-width charge-conserving shapes from the 2013 variational paper.
  No manuscript source was edited and no MeluXina job was submitted.
- Added `build_scientific_evidence_manifest_20260914.py`, a non-overwriting
  package hasher covering the new analyses, documentation, operative source
  files, selector provenance record, generated assets and unchanged manuscript
  hashes.  It records `simulation_submitted=false` and
  `manuscript_edited=false` in the final evidence directory.
  Its first invocation stopped before writing output because relative result
  directories were compared with the absolute workspace root.  Resolving each
  path before hashing corrected the defect; no partial manifest existed.
- The first successful scientific-evidence manifest is retained.  A final
  path-existence check then corrected one obsolete weak-Landau directory name,
  expanded one figure path, and classified the not-yet-written finite-marker
  supplement correctly as requiring implementation analysis at task start.
  `scientific_evidence_manifest_r2.json` is the authoritative post-correction
  50-file record; no numerical result changed.
- On 2026-09-14, added
  `analyze_bounded_floor_estimand_reference_20260914.py` for the requested
  bounded existing-data correction.  It directly compares the forty-seed
  PPC-1024 and PPC-4096 CIC histories, records coherent amplitude, geometric
  envelope, phase coherence and incoherent standard-error diagnostics, builds
  like-for-like E2 individual-rate/predictor tables, and bootstraps every term
  of the finite-reference decomposition.  It runs no PIC simulation and
  refuses to overwrite an existing output directory.  Early versioned
  directories `_r1`--`_r11` are retained: `_r1`--`_r2` exposed a numerical
  tolerance/cancellation issue in the Gram decomposition, `_r3`--`_r5`
  exposed the non-headless Matplotlib backend, and later directories record
  successive non-overwriting figure and interval-presentation corrections.
  The complete inspected output is
  `results/bounded_floor_estimand_reference_correction_20260914_r12/`.
- Corrected `DISCRETE_FINITE_MARKER_TECHNICAL_SUPPLEMENT.tex`: repaired delta
  indices, midpoint fractions, a missing backslash in `\qquad`, mathematical
  spacing and natural logarithms; added the exact finite-reference identity
  and its interpretation.  Tectonic produced a nine-page PDF.  All nine pages
  were rendered with Poppler and inspected; no malformed equation remains.
- Corrected proposed replacement I in
  `PROPOSED_SCIENTIFIC_REPLACEMENTS_20260914.tex`: when `W` is the one-way
  amplitude transfer, the matched particle-field amplitude contains `W^2`
  and its modal power contains `|W|^4`; the pre-return grid-field power is a
  separate `|W|^2` metric.
- Added `BOUNDED_FLOOR_ESTIMAND_REFERENCE_REPORT_20260914.md` and
  `PROPOSED_BOUNDED_WRITING_REPLACEMENTS_20260914.tex`.  These are a writing
  handoff, not manuscript edits.  The authoritative manuscript hashes remain
  `44a76c...26be6`, `59bc6e...cdb9`, and `4c4004...1827`; no job was submitted.
