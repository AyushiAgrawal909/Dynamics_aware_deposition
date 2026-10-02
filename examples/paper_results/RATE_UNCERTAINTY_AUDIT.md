# Task 6 — rate estimation and uncertainty audit

Audit date: 2026-09-09.

## Outcome

The recovered raw histories regenerate every available Landau and compensated
filter rate and 95% seed-bootstrap interval.  Thirteen of fourteen reductions
match their archived CSVs numerically; the largest absolute cross-environment
difference is `1.13e-11`.  The remaining comparison has identical numerical
values to `3.33e-15`, but the current strong-Landau analyzer changes the
historical gate label from `FAIL` to `UNRESOLVED` when only fit quality or
duration fails.  This is a real analysis-code revision and is retained as a
mismatch, not silently normalized.

The audit also establishes that the paper did **not** fit ten independent
per-seed rates.  Each bootstrap draw resamples seed indices, reconstructs a
seed-ensemble signal, and fits that signal.  Time samples are never resampled.

No production simulation was submitted for this task.  Historical output was
not edited, moved, renamed, or overwritten.  Failed/partial audit outputs
`rate_uncertainty_audit` through `rate_uncertainty_audit_v6` are retained.  The
completed immutable-by-convention reduction is
`results/rate_uncertainty_audit_v7/`.

## Evidence and commands

Raw histories were recovered to the separate tree
`pic_kde_2d/results_raw_rate_audit/`.  The inventory made by
`rebuild_rate_analysis.py` is
`results/rate_uncertainty_audit_v7/raw_history_inventory.csv`:

| campaign | rate histories | seeds | samples per trace |
|---|---:|---|---|
| Gaussian weak-Landau grid | 180 | 0–9 | 901 |
| Gaussian weak-Landau MISE follow-up | 20 | 0–9 | 901 |
| Gaussian weak-Landau CV follow-up | 20 | 0–9 | 901 |
| raw-binomial weak Landau | 80 | 0–9 | 901 |
| first-order compensated Landau + E2 | 120 | 0–9 | 901 / 1501 |
| higher-order compensated Landau + E2 | 160 | 0–9 | 901 / 1501 |
| strong Landau, alpha 0.5 | 60 | 0–9 | 451 |
| strong Landau, alpha 0.25 | 80 | 0–9 | 601 |

The exact commands, observables, and measured modes are recorded in
`results/rate_uncertainty_audit_v7/commands.csv`.  The top-level command was:

```sh
python3 paper1_source_package/reproducibility/rebuild_rate_analysis.py \
  --output paper1_source_package/reproducibility/results/rate_uncertainty_audit_v7
```

Omitted bootstrap acceptance counts were regenerated with:

```sh
MPLCONFIGDIR=/private/tmp/p1-rate-mpl \
python3 paper1_source_package/reproducibility/summarize_rate_bootstraps.py \
  --rebuild-root paper1_source_package/reproducibility/results/rate_uncertainty_audit_v7 \
  --output paper1_source_package/reproducibility/results/rate_uncertainty_audit_v7/bootstrap_diagnostics_v2.csv
```

The completed reduction used Python 3.14.6, NumPy 2.5.1 and Matplotlib 3.11.1;
SciPy is not installed and is not used by these fit paths.  Historical Python
campaigns used the MeluXina Python 3.11.10 module, but their exact NumPy and
Matplotlib versions remain unknown.  The comparison tolerance is `1e-10`
relative only to accommodate last-bit cross-version least-squares/root
differences; no physical rate, CI, fit-quality or duration threshold was
changed.

Recovered analyzer hashes:

| file | SHA-256 |
|---|---|
| `pic_kde_2d/scripts/run_landau_sigma_refinement.py` | `43a546ec0409bd2ce09d9faef6daf4f65ac9c692ec2ce0cb35ece25d9ad4f32f` |
| `pic_kde_2d_landau_cv_stage/scripts/analyze_landau_binomial_passes.py` | `2042989d371503189848e0af5719a2ced7ca62bd9e02bab0acb29537ceb0cfc4` |
| `pic_kde_2d_landau_cv_stage/scripts/analyze_strong_landau_fidelity.py` | `abab108c7e4115dae9b0e21acb4c89d7957d339059f35f961214ccbf2ca1dde1` |
| `pic_kde_2d_landau_cv_stage/scripts/analyze_E2_binomial_passes.py` | `384beee9ba219d9b36f3f5aee84a20399d976f4126b2fa1c27de146e5860b8e8` |
| `pic_kde_2d_landau_cv_stage/pic_kde_2d/validation/landau_censored_fit.py` | `103e7446b484743f8d0bcfc6aaaf1b3c98e2272e6bbd919f87be494106e90229` |
| `pic_kde_2d_landau_cv_stage/pic_kde_2d/validation/strong_landau_fit.py` | `7991c4c12641dfbf28cd3ee4d39ba6c8d92bc3ed9cfc8b49710bac2300b658c8` |

The preserved stage E2 analyzer is not self-contained: its two bare imports
are absent from `pic_kde_2d_landau_cv_stage/scripts`.  The rebuild therefore
used `analyze_E2_mode_growth.py` (`20776bcd...b0a`) and
`analyze_linearized_delta_f_m12_sigma.py` (`73e5c197...529`) from the supplied
top-level research tree.  This mixed-source dependency is evidence of
correspondence, not proof of the exact historical source revision.

## Numerical definitions recovered from code

### Observable and rate convention

- The tabulated `gamma` is the slope of the logarithm of **field-mode
  amplitude**, in units of omega-pe.  A fit of field-mode energy would return
  `2*gamma`; no table in this audit silently substitutes the energy rate.
- Weak and strong Landau use
  `abs(mean_seed(complex Ex_hat(1,0)))`.  A centered geometric moving average
  of `log(amplitude)` over one Langmuir period (4.88942, 49 samples in the weak
  campaign) treats the oscillatory beating.
- E2 compensated runs use `mean_seed(abs(Ex_hat(12,0)))`, not coherent complex
  averaging.  Every one of the 140 recovered E2 `result.json` files records
  `kx_mode=12`, `ky_mode=0`, PPC 512, alpha `1e-8`, and `dt=0.02`.

### Windows and censoring

- Weak Landau: candidate interval `[5,30]`.  The noise floor is the quietest
  five-time-unit rolling median in `[70,90]`.  The fit stops before the first
  five-point-persistent crossing below `5*floor`, with half an envelope width
  removed to avoid centered-window look-ahead.  Minimum quality is 30 points,
  duration 5, 0.5 model e-foldings, and R-squared 0.75.
- Weak paired bootstrap: the same ten seed indices are drawn for CIC and the
  filtered arm, but each resampled arm estimates its own floor and censoring
  time.  Therefore the weak comparison is paired in seeds but does **not** use
  one fixed common time window.
- Strong Landau: the common stop is the earlier detected bounce minus half an
  envelope period.  Both paired arms and every bootstrap draw use this fixed
  central-data window.  The fit start is an input and is materially important.
- E2: the CIC mean-amplitude trace selects one common `[24.32,59.68]` growth
  window; every compensated setting and bootstrap draw uses it.  The E2
  analyzer rejects only a non-positive baseline rate or non-finite rate.  It
  has no per-draw R-squared/duration rejection for the filtered arms.

### Uncertainty unit

The ten independently loaded seeds are the independent realizations.  A draw
samples ten seed indices with replacement and applies identical indices to
both arms.  No diagnostic time point is treated as an independent bootstrap
observation.  Reported percentile intervals are conditional on the recovered
window/censoring rule; they do not include fit-window selection uncertainty.

## Reproduction results

`results/rate_uncertainty_audit_v7/reproduction_comparison.csv` gives all 14
comparisons.  Numerical values and intervals reproduce for all campaigns.
Two old binomial CSV schemas lack newer provenance columns; this is recorded as
`PASS_WITH_SCHEMA_EXTENSION`.  The only non-numerical mismatch is:

| campaign | archived | rebuilt | numerical difference |
|---|---|---|---:|
| strong alpha 0.5, fit start 5 | `FAIL` | `UNRESOLVED` | <= `3.33e-15` |

The current analyzer implements the manuscript's distinction: a fit that
misses duration/quality is unresolved, even when its conditional rate CI is
inside tolerance.  The exact historical source revision that emitted `FAIL`
is not present.

### Bootstrap acceptance and rejected fits

The full table is `bootstrap_diagnostics_v2.csv`.

- All 2,000 draws are numerically valid for the MISE and CV Gaussian follow-ups,
  all weak binomial/compensated comparisons, both strong-Landau campaigns, and
  all compensated E2 comparisons.
- Weak Gaussian at PPC 1024 has >= 98.7% valid draws at every tested width.
  Sigma 12 has only 20.6% quality-passing draws and must not support a
  confidence-qualified rate claim, despite its finite central slope.
- Weak Gaussian at PPC 32 is not rate-resolved: away from the baseline only
  2.4%–41.6% of draws are numerically valid and at most 2.8% pass both fit
  quality checks.  The CIC central fit itself fails quality.  These cells are
  unresolved, not validation measurements.
- Strong alpha 0.5 with fit start 5 and strong alpha 0.25 with fit start 3 have
  100% numerically valid draws but 0% quality-passing draws under their stated
  duration/R-squared requirements.  Their narrow CIs are conditional rate
  intervals, not validation passes.
- The E2 reductions retain all 2,000 draws, but the analyzer does not evaluate
  per-draw fit quality for filtered arms.  The common CIC window has
  R-squared 0.99946 and 5.406 e-foldings; this does not by itself certify a
  filtered trace whose fitted rate is near zero or negative.

### Strong-Landau fit-window sensitivity

At the alpha 0.5 CV width, the same raw histories give:

| window | gamma CIC | gamma filtered | ratio [95% CI] | R-squared | model e-foldings | strict 2-e-fold verdict |
|---|---:|---:|---|---:|---:|---|
| `[3,7.2]` | -0.292721 | -0.295705 | 1.01020 [1.00986, 1.01052] | 0.9910 | 1.242 | unresolved |
| `[5,7.2]` | -0.289049 | -0.298055 | 1.03116 [1.03063, 1.03166] | 0.9555 | 0.656 | unresolved |

Both conditional CIs meet the 10% rate criterion, but the inferred damage
changes from 1.0% to 3.1%.  The previously stated “2.2 e-foldings” was a unit
error: 2.2 is the **time duration** of the second window, whose fitted decay is
only 0.656 e-foldings.

At alpha 0.25 and the actual CV width 2.8208226 cells, the exact rebuilt result
is ratio 1.047913 `[1.047233,1.048585]`, gamma CIC -0.229885, gamma filtered
-0.240899, window `[3,7]`, R-squared 0.99490, and 0.964 e-foldings.  It meets
the conditional rate gate but fails the duration gate.

### Compensated E2 rates

The rebuilt longitudinal results reproduce the paper's numerical rows:

- first order, one pass: 0.91161 `[0.89519,0.91875]`, rate FAIL;
- second order, two passes: 0.97287 `[0.96807,0.97502]`, longitudinal PASS;
- third order, four passes: 0.98275 `[0.97972,0.98411]`, longitudinal PASS.

They are measurements of `(12,0)`, not the binding `(12,1)` corner.  They may
support longitudinal statements only.  Transfer/dispersion values for the
corner remain predictions.

## Recovered after the initial audit and remaining limitations

1. The initially missing E2 raw-binomial, Gaussian corner, and direct
   compensated `(12,1)` histories were recovered in earlier MeluXina passes.
   A final job-ID search on 2026-09-11 also recovered the complete delta-f
   reference (`4872599`), isotropic Gaussian m12 (`4872745`), and 1--14 band
   (`4873221`) trees. Current matching reducers reproduce their archived CSVs
   to `3.38e-14`, `3.66e-15`, and zero maximum absolute numeric difference.
2. The preserved E2 stage analyzer is missing two helper source files and must
   be combined with the current top-level research tree.
3. Historical dependency versions and exact analyzer revisions remain unknown.
4. The E2 compensated analyzer lacks filtered-arm per-draw fit-quality checks.
5. Bootstrap intervals quantify independent-seed uncertainty conditional on a
   selected window.  They do not quantify window-selection systematic error.

## Manuscript changes justified now

`paper1.tex` was revised to:

- define field-amplitude versus field-energy rates explicitly;
- replace the inaccurate fixed-window/per-seed-fit description with the actual
  coherent Landau and amplitude-averaged E2 ensemble procedures;
- state paired seed resampling, no temporal bootstrap, and weak-arm independent
  censoring;
- report low-PPC and finite-amplitude bootstrap-quality failures;
- show both alpha 0.5 fit starts and correct 2.2 time units to 0.656 e-foldings;
- mark both finite-amplitude cells UNRESOLVED on the strict duration gate; and
- narrow the “domain boundary” claim: the selected width crosses the analytic
  ceiling between alpha 0.25 and 0.5, but the available runs do not establish a
  measured unsafe-to-safe validation crossover.

Task 7 must now apply the two-sided 10% rate, duration, model-admissibility and
resolvability tests separately to every claimed validation row.  No additional
simulation campaign is justified until those statuses are enumerated and the
missing direct E2 corner measurement is designed against the verified filter
transfer.
