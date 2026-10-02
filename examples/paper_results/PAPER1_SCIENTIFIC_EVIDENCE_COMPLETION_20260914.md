# Paper 1 scientific-evidence completion

Evidence date: 2026-09-14. This work starts from the verified 2026-09-12
repository state and uses the research implementation
`pic_kde_2d_landau_cv_stage`, not the supplied baseline HEXAPIC archive. It
does not edit either manuscript and does not submit a simulation.

## Outcome

The evidence supports a focused paper with three connected findings:

1. minimising a spatial density-error objective need not preserve a selected
   kinetic mode;
2. an implementation-audited constraint on the exact round-trip transfer can
   exclude destructive widths and, with a predeclared interior margin,
   preserve the tested projected linear response; and
3. the practical benefit at a fragile E2 operating point is real but modest in
   the compatible finite-reference grid metrics, whereas larger noise
   reductions are available only at settings or in campaigns with narrower
   validation scope.

This evidence does **not** establish unrestricted nonlinear full-`f` fidelity,
particle-force noise reduction at the E2 interior width, a universal optimum,
an exact full finite-marker eigensystem, superiority over the complete
Wu--Qin adaptive KDE method, or end-to-end speedup.

The following established outcomes are unchanged:

- the `v_b=0.818` extension completed, but its physical rates remain
  unresolved;
- the fresh weak-Landau confirmation fails because CIC has insufficient
  fitted duration;
- every originally registered rate, window and verdict remains in the record;
- exact source revisions that were never recorded remain unknown.

## 1. Reconciliation before new analysis

The machine-readable ledger is
`EVIDENCE_RECONCILIATION_20260914.csv`. A concise rendering is below. Here,
“complete” may mean that a requested experiment has a defensible failed or
unresolved disposition; it never means that a scientific gate was relaxed.

| Requested item | Classification at start | Final disposition | Exact evidence |
|---|---|---|---|
| Numerical formulation, coupling and transfer audit | Already complete | Complete; reused | `NUMERICAL_FORMULATION.md`, `FILTER_TRANSFER_AUDIT.md`, `OPERATOR_AND_TRANSFER_AUDIT.md` |
| Continuum and grid/time dispersion | Already complete | Complete; reused | `DISPERSION_AUDIT.md`, `results/dispersion_audit/` |
| Finite-marker mathematical documentation | Required existing-data/implementation analysis | Complete as supplement | `DISCRETE_FINITE_MARKER_TECHNICAL_SUPPLEMENT.tex` and compiled PDF in `results/scientific_evidence_completion_20260914_r3/` |
| E2 estimand/predictor comparison | Required existing-data analysis | Complete | `e2_estimand_comparison.csv`, `finite_marker_prediction_residuals.csv` |
| `v_b=.818` extension | Already complete computationally | Physical rate unresolved | `results/research_completion_rebuild_check_20260912_r3/extension_audit/` |
| Seed heterogeneity | Already complete | Complete; reused and formalised | `ESTIMATOR_AND_SEED_DIAGNOSTIC.md`, `results/e2_seed_rate_variation_20260912_with_extension/` |
| Exact `R=.95` interior benefit | Already complete | Complete in compatible grid metrics | `results/e2_interior_noise_replay_20260911/` |
| E2 finite-reference contribution | Required existing-data analysis | Complete deterministic replay | `results/e2_finite_reference_covariance_20260914/` |
| Weak-Landau 4096-PPC duration and finite reference | Required existing-data analysis | Complete; duration still fails | `weak_landau_ppc4096_reference_fit.json`, `weak_landau_finite_reference_decomposition.csv` |
| 7.2443-cell selector provenance | Complete but absent/ambiguous in paper | Complete | archived job-5008392 JSON plus `HISTORICAL_LANDAU_DATASET_CLASSIFICATION_20260914.csv` |
| New full-`f` E2 rate/force-noise campaign | Genuinely new measurement | Not required for the narrowed claim | It remains necessary only for a simultaneous end-to-end claim |
| Exact old revisions/full finite-marker eigensystem | Evidence unavailable | Unresolved | `MISSING_INFORMATION.md`; supplement Secs. 4.3 and 10 |
| Public DOI and CRediT roles | Author input required | Open coauthor action | `Paper1_Rewritten/AUTHOR_REVIEW.md` |

No earlier campaign, transfer test, conservation audit or regression suite was
repeated. The only replay performed here is a bounded deterministic
reconstruction of an existing ten-seed E2 error calculation whose saved
outputs retained scalar MSEs but not the arrays needed to decompose reference
variance and covariance.

## 2. Discrete and finite-marker response mathematics

The self-contained source and compiled artifact are:

- `DISCRETE_FINITE_MARKER_TECHNICAL_SUPPLEMENT.tex`;
- `results/scientific_evidence_completion_20260914_r3/DISCRETE_FINITE_MARKER_TECHNICAL_SUPPLEMENT.pdf`.

The supplement keeps three objects separate:

1. **continuum equilibrium:** a velocity-integral dielectric for the specified
   warm equal-beam equilibrium;
2. **grid/time numerical response:** the same equilibrium susceptibility with
   CIC, exact filter, spatial-alias and midpoint time-alias factors; and
3. **finite-marker predictor:** a seed-specific, principal-mode diagonal
   response constructed from the initialized marker velocities, beam signs,
   equilibrium source weights and explicit-midpoint phases.

For marker (j), the implemented diagonal construction uses

\[
\Omega_j=\mathbf k\!\cdot\!\mathbf v_j,\qquad
u_j=e^{-i\Omega_j\Delta t},\qquad
h_j=e^{-i\Omega_j\Delta t/2},
\]

\[
a_j=\widetilde W_{\mathbf k}^{,2}
\frac{iS_{\rm CIC}^2(\mathbf k)(\mathbf k\!\cdot\!\mathbf g_j)}
{|\mathbf k|^2N_p},\qquad c_N=\sum_jh_ja_j,
\]

and solves

\[
\mathcal R_N(\lambda)=1-\Delta t\sum_j
\frac{u_ja_j}{e^{\lambda\Delta t}-u_j}
-\frac{\Delta t^2}{2}c_N\sum_j
\frac{h_ja_j}{e^{\lambda\Delta t}-u_j}=0.
\]

This is the numerical construction actually used for the seed diagnostic. It
is **not** the exact eigenresponse of the complete finite-marker update. It
omits spatial aliases in the finite sum, finite-sample coupling between the
retained (+\mathbf k) target and its conjugate, and nonlinear/full-`f`
response. The target and conjugate are nevertheless both retained by the
mode-projected time integrator, so the omission is documented rather than
silently equated to the code.

All predictor inputs are identified as one of:

- frozen configuration: grid, domain, mode, `dt`, filter and PPC;
- initialization: positions, velocities, beam memberships, weights and RNG
  correlations; or
- numerical-analysis specification: alias/time truncation and root tolerance.

Measured histories and fitted rates enter only after root construction, for
comparison. A fitted-exponential reconstruction of an arithmetic ensemble is
explicitly labelled an estimator diagnostic rather than an independent
prediction.

The supplement maps every equation to the operative function and SHA-256.
The surrounding Git identifier `ed6dd96...` is not claimed as the revision of
untracked production Python files. Their file hashes and frozen run manifests
are the operative revision identifiers. HYPRE and PETSc do not enter this
spectral Python chain.

## 3. E2 estimands and matched comparisons

### 3.1 Four quantities answer four different questions

For a fixed fit window (I):

| Quantity | Definition | Scientific question |
|---|---|---|
| Continuum-equilibrium eigenvalue, `lambda_eq` | zero of the continuum dielectric | What does the infinite homogeneous equilibrium predict? |
| Realised finite-marker predictor, `lambda_N,s` | zero of the initialization-only residual above | What principal-mode response is predicted for this marker quadrature, within the stated approximations? |
| Individual fitted rate, `gamma_hat_s` | slope of `log(abs(E_hat_s))` | What rate is measured in one realised history on this window? |
| Arithmetic-mean-amplitude rate, `gamma_hat_amp` | slope of `log(mean_s(abs(E_hat_s)))` | How does the registered positive-amplitude ensemble mixture grow? |

The individual-rate mean is the slope of the geometric-mean amplitude when
all seeds use identical time points and weights. It is not called unbiased.
The registered estimator is also not called unbiased. For exponential seed
amplitudes, faster seeds receive increasing weight in the arithmetic mean;
this is a property of the estimand, not a bootstrap defect.

Paired intervals resample the ten seed identities 5,000 times using the same
indices for baseline and candidate. Time samples and amplitude-contributor
counts are not treated as independent observations.

### 3.2 Registered primary and individual-rate sensitivity

Full source rows, including historical Table 5 and compensated job 5195640,
are in `results/scientific_evidence_completion_20260914_r3/e2_estimand_comparison.csv`.

| Drift | Arm | Registered amplitude ratio [95% paired CI] | Ratio of individual-rate means [95% paired CI] | Initialization finite-marker ratio |
|---:|---|---|---|---:|
| .60 | fixed | .992782 [.992684,.992871] | .992752 [.992668,.992844] | .992750 |
| .60 | interior | .950593 [.949907,.951224] | .950382 [.949791,.951027] | .950353 |
| .70 | fixed | .985949 [.985373,.986363] | .985671 [.985253,.986111] | .985701 |
| .70 | interior | .951726 [.949695,.953173] | .950736 [.949271,.952279] | .950854 |
| .76 | fixed | .972694 [.969710,.974265] | .970566 [.968597,.972540] | .970483 |
| .76 | interior | .955457 [.950522,.958042] | .951925 [.948678,.955180] | .951785 |
| .80 | fixed | .940636 [.907972,.946915] | .908911 [.887432,.926656] | .907024 |
| .80 | interior | .969862 [.953758,.973010] | .954460 [.944068,.963103] | .953595 |

The `.80` fixed individual-rate interval crosses 0.90. Therefore preservation
is **not established by that sensitivity estimand**; this does not establish
distortion greater than 10%. The registered fixed-arm result remains primary
and passes its own interval. The `.80` interior interval lies wholly within
`[.90,1.10]` under both estimands.

Historical Table-5 and compensated results are retained, but the finite-marker
roots for them are explicitly “nominal current-initializer reconstructions”
because neither the original marker arrays nor exact source revision was
saved. For job 5195640, the individual-rate ratios are:

| Compensator | Registered ratio | Individual-rate ratio [95% paired CI] | Nominal finite-marker ratio |
|---|---:|---|---:|
| p1/n1 | .838068 | .812242 [.774277,.839887] | .800635 |
| p2/n2 | .956084 | .949961 [.940921,.956642] | .947785 |
| p2/n4 | .788855 | .752403 [.698837,.791276] | .733877 |
| p3/n4 | .973416 | .969766 [.964359,.973761] | .968511 |

`p2/n2` and `p3/n4` preserve the projected rate under both estimands. This is
separate from calibration: the grid/time continuum point predictions lie
outside the narrow registered intervals, with central discrepancies 0.932%
and 0.545%, respectively; `p1/n1` and `p2/n4` discrepancies are 4.428% and
6.486%.

### 3.3 Current finite-marker predictor residuals

At `.80`, mean observed-minus-predicted individual-rate residuals and seedwise
RMSEs are:

| Arm | Mean residual | RMSE | Prediction/observation status |
|---|---:|---:|---|
| CIC | `+2.528e-4` | `4.884e-4` | exact reconstruction of current audited initializer |
| fixed Gaussian | `+4.641e-4` | `8.913e-4` | exact current initialization, approximate response |
| interior Gaussian | `+3.487e-4` | `6.564e-4` | exact current initialization, approximate response |

Across the 120 signal-valid histories at drifts `.60,.70,.76,.80`, seedwise
prediction/observation correlations are 0.9982--0.99994. Residuals quantify
approximation/calibration error and are not folded into a pass gate. Their
source table is `finite_marker_prediction_residuals.csv`.

### 3.4 Why individual rates differ

Existing-data diagnostics close the alternatives without deleting seeds:

- initial target amplitudes are seed-invariant to round-off and phases differ
  by at most `1.8e-15` rad;
- fixed-window-grid shifts at `.80` are approximately `0.00115`, far smaller
  than the CIC seed SD `0.02333`;
- transient/window effects are secondary over the frozen accepted windows;
- realised beam-sign/thermal-velocity covariance correlates `-0.9465` with
  the `.80` CIC rate; and
- the initialization-only finite-marker construction predicts the seed
  ordering and scale (`r=.999885`, RMSE `4.88e-4` for `.80` CIC).

The fitted-exponential mixture reproduces the registered `.80` CIC slope to
`1.48e-5`, while the effective amplitude-contributor count falls from 7.41 to
3.28. This diagnoses the estimator weighting but does not change the number of
independent seed realizations, which remains ten.

### 3.5 Completed `v_b=.818` extension

MeluXina job 5202227 completed all 30 histories with exit `0:0` in 02:21:40.
The frozen campaign hash and budget-amendment hash match, no duplicate was
submitted, and the original 120/150 four-hour timeout remains preserved.

The saved reducer outputs are descriptive only: every arm has fit-start SNR
below 5, 3/10 seeds per arm later exceed the linear-`delta f` guard, and the
arithmetic mean becomes nearly single-seed dominated. The computation is
complete; the physical rate, rate preservation and stabilisation questions
remain unresolved. The nominal “no growing fixed-filter branch” prediction is
not a measured stabilisation point.

## 4. Practical E2 benefit and the finite reference

The predeclared `.80` interior setting is
`sigma_y=0.1738592384458977 Delta y`, corresponding to the tested `R=.95`
interior prescription. The compatible ten-seed full-`f` replay compares PPC 32
with same-seed but independently generated PPC 1024 over `t=[.5,4]`. It gives:

| Metric | CIC/interior error-power ratio [95% seed bootstrap] |
|---|---:|
| Density grid MSE | 1.081314 [1.079778,1.082929] |
| Pre-return, pre-gather grid-field MSE | 1.038554 [1.037245,1.040182] |

The model-boundary width `0.2427346 Delta y` is kept separate; its larger
reductions are not assigned to the interior setting.

The new deterministic covariance replay exactly reproduces the four saved
scalar MSEs to absolute tolerance `5e-15`. The identity

\[
E_s[(X_s-R_s)^2]=(\bar X-\bar R)^2+\operatorname{Var}_sX+
\operatorname{Var}_sR-2\operatorname{Cov}_s(X,R)
\]

shows that the PPC-1024 reference contributes 2.69% (CIC density), 2.72%
(CIC field), 2.91% (interior density) and 2.82% (interior field) of pair MSE.
The `-2 covariance` contribution is only -0.16% to -0.24%. The observed
few-percent improvements therefore are not dominated by the reference
variance, but the estimand remains a finite-reference grid error. It is not a
continuum error or particle-force measurement.

## 5. Landau comparator, loading, and references

### 5.1 What the 7.2443-cell value is

The exact source is recovered job 5008392:
`pic_kde_2d/results_raw_rate_audit/recovered_meluxina_20260909/noise_metric_campaigns/landau_lscv_static_5008392/`.

It used:

- only the initial spatial sample (`n_steps=0`, evaluation `[0,0]`);
- PPC 1024 and seeds 0--9;
- stratified-jitter particle positions;
- a 27-width search grid with an exact particle-phase-averaged CIC quadratic
  self-contribution;
- no external high-PPC or analytic density reference; it is a LOO-CV density
  criterion; and
- quadratic interpolation of ten interior minima, giving
  `7.2442931470 Delta x` with interval `[7.2441376283,7.2444180515]`.

It did not use evolving snapshots. The later loading correction replaced the
invalid per-cell velocity construction for dynamics; velocities do not enter
this `t=0` spatial density objective. However, the confirmation uses a
different spatial protocol (deterministic lattice plus perturbation), so the
loading correction does not make job 5008392 the optimum of the corrected
confirmation campaign.

Selection seeds 0--9 are independent of the primary v2 evaluation seeds
202609300--202609339. The defensible name is therefore **fixed CV-derived
comparator**. It is not “the CV optimum of the corrected campaign.”

### 5.2 Historical Landau datasets

`HISTORICAL_LANDAU_DATASET_CLASSIFICATION_20260914.csv` classifies every
retained group by role, loading evidence, provenance and measurement status.
The central distinctions are:

- the rejected v1 confirmation genuinely used only 32 distinct longitudinal
  velocity nodes per cell and is invalid for resonant dynamics;
- job 4867515 Table-4 density errors used that construction, but remain valid
  as density-estimator measurements for that constructed ensemble only;
- no recovered historical Landau **rate** dataset is identified as using the
  rejected per-cell construction; several archived rate campaigns explicitly
  record global loading, while the oldest exact source revisions remain
  unknown;
- density-only evidence never becomes qualified dynamical evidence; and
- current global-loading runs do not retroactively identify an unknown old
  initializer.

The rewritten paper's blanket phrase “historical density metrics from
repeated 32-node-per-cell loading” is too broad. Only the identified
per-cell/Table-4 datasets should carry that statement.

### 5.3 Existing PPC-4096 reference and new-run decision

All forty saved PPC-4096 CIC histories were fitted with the same frozen
49-sample envelope and floor rule. The result is:

- floor `4.06958e-4`;
- persistent crossing `t=25.4`;
- accepted window `[5,22.9]`;
- `gamma=-0.0670563`, `R^2=.999829`;
- **1.200 fitted amplitude e-foldings**.

Thus higher PPC did not recover the two-e-fold requirement and does not
justify a still larger Landau campaign by ideal `1/sqrt(N)` reasoning alone.
The primary PPC-1024 v2 result remains C4/CIC
`.989415 [.967884,1.013827]`, with C4 2.076 e-foldings but CIC only 1.721.
The weak-Landau complete validation still fails.

No new Landau validation run is justified for the narrowed paper. A future
campaign would first need a specific design change that extends a usable CIC
measurement interval—rather than only more seeds, more PPC, or a longer
noise-dominated tail—and would require a new frozen specification and explicit
authorization. Consequently no new configuration or cost cap is invented
here.

The weak-Landau pair-MSE reference contribution is metric dependent: 19.50%
for CIC density, 63.54% for C4 density and 89.93% for the 7.2443-cell density
comparator; corresponding pre-return grid-field fractions are 19.52%, 31.48%
and 12.34%. Especially for wide-filter density, the reduction must be called a
finite-reference comparison, not a continuum-error reduction.

## 6. Exact-transfer and operator corrections

The implementation is internally consistent:

- Gaussian, raw-binomial and compensated filters are applied once after CIC
  deposition and once at CIC gather;
- saved pre-gather fields carry one factor `W`; particle force carries `W^2`;
- all 24 controlled mode/stage tests agree with the analytic transfer to
  round-off; and
- production E2 roots already use exact filter transfers and the spectral
  Green function.

Two documentation corrections remain:

1. `sigma_eq=sqrt(n/2) Delta x` is only the leading low-wavenumber Gaussian
   match to `cos(k Delta x/2)^(2n)`. At E2 `m=12,n=4`, the proxy differs by
   0.512% one-way and 1.022% round trip; and
2. the Python spectral force response uses `k/|k|^2`. The expression
   `D_h/Lambda_h` belongs only to the separate finite-difference/HYPRE chain.

These corrections change equations, labels and interpretation—not the
existing histories, exact transfer tables, frozen widths, reductions or E2
roots. The HYPRE and PETSc statements remain separate: HYPRE/PFMG belongs to
the auxiliary C++ prototype, while PETSc is not implemented in the audited
source.

## 7. Figures and traceable source tables

The new evidence package is
`results/scientific_evidence_completion_20260914_r3/`:

- `fig_e2_estimands.pdf/.svg/.png`;
- `fig_e2_finite_marker_predictions.pdf/.svg/.png`;
- `fig_e2_smoothing_budget.pdf/.svg/.png`;
- `fig_practical_benefit_vs_rate.pdf/.svg/.png`;
- `e2_estimand_comparison.csv`;
- `finite_marker_prediction_residuals.csv`;
- `e2_smoothing_budget_source.csv`;
- `practical_benefit_comparison.csv`;
- `weak_landau_ppc4096_reference_fit.json`; and
- `weak_landau_finite_reference_decomposition.csv`.

The `.818` missing measurement is shown as unavailable, not as a measured
stabilised point. The bandwidth panel is titled “Predicted admissible
bandwidth.” The practical figure uses filled markers only for same-campaign
comparisons and open markers for compatible split campaigns.

## 8. Strongest supported findings

1. In the projected E2 drift scan, the precomputed `R=.95` interior settings
   pass the signal, duration and paired 10% rate gates at all four resolved
   drifts. At `.80`, the registered ratio is
   `.969862 [.953758,.973010]`; the individual-rate sensitivity is
   `.954460 [.944068,.963103]`.
2. The current initialization-only finite-marker predictor explains the
   individual-rate spread at the resolved E2 drifts, including `.80` CIC
   (`r=.999885`, RMSE `4.88e-4`). This supports a loading-sensitive numerical
   interpretation, not an exact full-system eigenvalue claim.
3. At the exact `.80` E2 interior width, compatible full-`f` data show modest
   finite-reference reductions: 1.0813x density and 1.0386x pre-return
   grid-field error power. The reference variance is only about 3% of pair MSE.
4. The direct compensated corner establishes projected-rate preservation for
   p2/n2 and p3/n4 under both estimands; p3/n4's registered ratio is
   `.973416 [.970473,.975293]`. This is a useful filter setting even though
   prediction calibration to the narrow measured interval is imperfect.
5. The fresh weak-Landau same-campaign result directly demonstrates the
   density/dynamics trade-off, but the complete validation fails the CIC
   duration gate. It is evidence, not a certification pass.

## 9. Remaining coauthor decisions and genuinely missing evidence

| Issue | Present status | Decision needed |
|---|---|---|
| Central claim scope | Evidence supports projected-linear protection plus separately measured grid benefit | Approve narrowed wording; otherwise commission a new full-`f` E2 rate/particle-force campaign |
| Exact old revisions | Irrecoverable from present artifacts | Keep explicitly unknown |
| `.818` physical rate | Histories complete; SNR/linearity fail | Keep unresolved; do not plot a measured rate |
| Weak-Landau complete pass | Fails CIC duration at PPC1024 and PPC4096 diagnostic | Do not claim complete certification; no repeat recommended from current evidence |
| Historical loading sentence | Current rewritten wording overgeneralises | Restrict per-cell warning to identified datasets |
| Public archive/DOI and CRediT roles | Not scientific-computation tasks | Authors must supply before submission |
| Complete finite-marker eigenproblem | Not implemented/saved | Keep predictor labelled approximate; derive/implement only if central validation is expanded |

No present evidence requires a broad simulation campaign. The sole precisely
justified missing experiment is conditional: an end-to-end E2 full-`f`
comparison with the exact interior filter, matched CIC, saved particle-force
diagnostics and a predeclared linear measurement rule is needed only if the
paper insists on claiming simultaneous full-`f` dynamic fidelity and
particle-force noise benefit.

## 10. Commands, verification and manuscript preservation

Commands executed for the new analyses were:

```text
python3 complete_scientific_evidence_20260914.py \
  --output results/scientific_evidence_completion_20260914_r3

python3 audit_E2_finite_reference_covariance.py \
  --saved-summary results/e2_interior_noise_replay_20260911/interior_noise_summary.csv \
  --output results/e2_finite_reference_covariance_20260914

tectonic -X compile DISCRETE_FINITE_MARKER_TECHNICAL_SUPPLEMENT.tex \
  --outdir results/scientific_evidence_completion_20260914_r3 \
  --keep-intermediates

python3 -m py_compile complete_scientific_evidence_20260914.py \
  audit_E2_finite_reference_covariance.py
```

The covariance replay refuses to overwrite outputs, reconstructs the exact
archived seed protocol, and asserts each saved scalar to absolute tolerance
`5e-15`. The supplement is seven pages, compiles without overflow, has embedded
fonts, and was rendered with Poppler for visual inspection. All four figure
PDFs were rendered and inspected; PDF/SVG sources are vector and PNGs are
300 dpi.

The final 50-file package hash record is
`results/scientific_evidence_completion_20260914_r3/scientific_evidence_manifest_r2.json`.
The first manifest is retained but superseded because a subsequent ledger
check corrected one evidence-directory name and expanded one figure path; no
scientific value changed.

The manuscripts were not edited. Their SHA-256 values at completion are:

```text
44a76c8175cfa48ece6f84bdc8e1bf22bd3795a92af822aa47af988317226be6  Dynamics_constrained_Deposition/introduction.tex
59bc6e55e96972f1328c23528f384cdd5e8c41c6582e2655260845001602cdb9  Paper1_Rewritten/main.tex
4c4004830bc182376022a31569706bd1c7e44eb0415cbada895745d1011d1827  paper1_source_package/paper1.tex
```

Proposed scientific replacements are staged separately in
`PROPOSED_SCIENTIFIC_REPLACEMENTS_20260914.tex`; none has been applied.
