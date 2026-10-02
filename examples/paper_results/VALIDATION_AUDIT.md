# Task 7 — validation-gate audit

Audit date: 2026-09-11.  This report separates model admissibility, the
two-sided rate criterion, fitted duration, signal quality, protected-mode
coverage and raw traceability.  A complete `PASS` requires every component to
pass.  A measured component outside its criterion is `FAIL`; `UNRESOLVED` is
reserved for a missing required measurement or source artifact.  Predictions
never satisfy a measurement gate.

## Acceptance rules applied without relaxation

- Rate: the paired-seed 95% interval of the field-amplitude rate ratio must lie
  wholly inside `[0.90, 1.10]`.  The upper limit matters for damping, and the
  lower limit matters for growth.
- Duration: the measured arm must span at least 2.0 fitted model e-foldings.
  A signal-to-floor capacity such as `log(A0/(5 floor))` is reported
  separately and is not substituted for fitted duration.
- Signal: the benchmark-specific fit-quality rule must pass.  E2 uses at least
  two candidate-rate e-foldings and fit-start SNR at least 5.  The historical
  Gaussian corner reducer did not store a floor, but the recovered raw
  histories now give SNR `11.213` under the existing pre-emergence-floor
  estimator, in addition to `R^2`, duration, delta-f validity and 5000/5000
  valid paired resamples.
- Baseline duration: retrospective `NOT_APPLICABLE` values mean only that the
  original candidate-only schema did not define an additional baseline gate;
  they are not automatic passes.  The shared-window E2 C4 baseline is now
  checked explicitly and passes with 4.484 e-foldings.  The v2 weak-Landau
  campaign preregistered both arm-specific durations and its CIC baseline
  fails with 1.721 e-foldings.
- Coverage: a longitudinal `(12,0)` E2 rate does not certify the binding
  protected-set corner `(12,1)`.
- Traceability: an archived summary is insufficient for a complete pass until
  its raw independent-seed histories can be regenerated.

The current frozen machine-readable matrix is
`results/validation_audit_v9/validation_status.csv`; earlier versions are
retained, and generation is refused if a target directory already exists.

## Evidence recovered from MeluXina

The user's statement that all earlier jobs were submitted was treated as a
search directive, not as permission to call absent local files nonexistent.
Read-only searches of both `/home/users/u103576` and
`/mnt/tier2/users/u103576` recovered:

- `landau_binomial_strong_5048414`: 2.9 MiB remotely, eight pass counts and ten
  seeds per count;
- `E2_binomial_5048416`: 4.5 MiB remotely, eight pass counts and ten seeds per
  count;
- `p1_landau_rates_4866573`: the weak-Landau Silverman, CIC and erasure-width
  histories;
- `e2_joint_corner_4913408`: seven `(12,1)` rate arms with ten histories each,
  plus ten noise seeds;
- `landau_ppc_dynamics_5049265`: 240 histories including the PPC-4096 CIC/C4
  arms.

Byte-for-byte copies are under
`pic_kde_2d/results_raw_rate_audit/recovered_meluxina_20260909/`; the cluster
originals were not renamed or modified.  Rebuilding nineteen reductions in
`results/rate_uncertainty_audit_v8/` reproduced eighteen archived tables
numerically.  The remaining table (`strong_alpha0p5_fit5`) reproduces every
number to `3.33e-15`; only the current semantic label `UNRESOLVED` differs from
the historical `FAIL`.

## Current validation result

After the direct corner, independent-mode and fresh weak-Landau confirmation
campaigns completed and the final delta-f histories were recovered, the strict
matrix contains 40 conditions: 8 pass, 31 fail and 1 remains unresolved.

| Scope | Complete result | Reason |
|---|---:|---|
| weak Landau, all recovered selectors and filters | no pass | every measured row fails duration or rate; the exact LOO-CV selected width now fails rate, duration and signal quality |
| weak Landau, fresh 40-seed C4 confirmation | fail (baseline duration) | C4 ratio 0.98942 `[0.96788,1.01383]` and 2.076 C4 e-foldings pass, but paired CIC has only 1.721 e-foldings |
| strong Landau, Gaussian CV widths | fail | rate intervals pass, but only 0.656–1.242 fitted e-foldings |
| strong Landau, raw binomial n=16, fit start 3 | fail | rate CI passes; duration is 1.311 e-foldings |
| strong Landau, raw binomial n=16, fit start 5 | fail | rate and duration both fail; ratio 1.17925 `[1.17618,1.18171]` |
| E2 Gaussian C4, binding `(12,1)` | **pass** | ratio 0.91735 `[0.90782,0.92316]`, 4.113 e-foldings, `R^2=0.9959`, recovered-history SNR 11.213; shared CIC baseline 4.484 e-foldings |
| E2 isotropic Gaussian 0.184455, longitudinal `(12,0)` | **pass at `(12,0)` only** | ratio 0.92727 `[0.91428,0.93337]`, 5.013 e-foldings, SNR 22.50; all 50 sweep histories recovered |
| delta-f E2 reference, mode `(8,0)` | **pass at `(8,0)` only** | absolute-rate ratio 1.01354 `[1.00487,1.02222]`, 8.158 e-foldings, `R^2=0.9971`; all 30 reference histories recovered |
| E2 Gaussian transverse width 0.30 | fail | ratio 0.87104 `[0.85596,0.88027]` |
| E2 raw binomial n=1, longitudinal `(12,0)` | fail | ratio 0.01613, SNR 0.379 and only 0.087 candidate e-foldings |
| E2 compensated p1/n1, longitudinal | fail | lower CI 0.8952 is below 0.90 |
| E2 compensated p2/n2, longitudinal | **pass at `(12,0)` only** | ratio 0.97287 `[0.96807,0.97502]`, 5.259 candidate e-foldings |
| E2 compensated p3/n4, longitudinal | **pass at `(12,0)` only** | ratio 0.98275 `[0.97972,0.98411]`, 5.313 candidate e-foldings |
| E2 compensated p2/n4, longitudinal | fail | ratio 0.87535 `[0.85145,0.88554]` |
| compensated p1/n1 at binding corner | fail | direct ratio 0.83807 `[0.81930,0.84994]`; duration and signal pass |
| compensated p2/n2 at binding corner | **pass** | direct ratio 0.95608 `[0.95120,0.95920]`, 4.287 e-foldings, SNR 13.89 |
| compensated p2/n4 at binding corner | fail | direct ratio 0.78885 `[0.76359,0.80469]`; duration and signal pass |
| compensated p3/n4 at binding corner | **pass** | direct ratio 0.97342 `[0.97047,0.97529]`, 4.364 e-foldings, SNR 15.75 |
| Gaussian C4 at independent mode `(10,1)` | **pass** | direct ratio 0.989118 `[0.988928,0.989253]`, 6.976 e-foldings, `R^2=0.999993`, SNR 232.8; new seeds `100..109` |
| configured directional E2 Silverman pair | fail (model) / rate unresolved | its analytic budget exceeds the limit, but the recovered scalar launchers do not implement the configured pair and no measured rate is substituted |

The eight complete passes are the Gaussian C4 corner, the independent Gaussian
`(10,1)` transfer test, the two historical longitudinal compensated-filter
conditions, the new p2/n2 and p3/n4 binding-corner conditions, the recovered
longitudinal Gaussian 0.184455 row, and the recovered delta-f mode-8 reference.
Three of these cover the E2 binding corner. No weak-Landau condition passes the
complete gate.

The fresh weak-Landau result is kept distinct from recovered development
campaigns. It uses seeds `202609300..202609339`, global inverse-normal
velocity assignment, exact PPC-1024 spatial loading in each low arm and a
same-seed independently generated PPC-4096 CIC error reference. Its rate-free
Stage-1 decision file was hashed as `cb724768...0cbf` before Stage 2. The CIC
absolute-rate ratio to the fully discrete reference is
`1.03343 [0.98977,1.07871]`; C4/CIC is
`0.98942 [0.96788,1.01383]`. Same-campaign C4 density and pre-return-filter
grid-field squared-error-power reductions are
`3.25865 [3.25180,3.26500]` and `1.61280 [1.60345,1.62155]`. These rate and
noise results pass their individual criteria, but cannot override the CIC
duration failure. The 120 single-run fits are diagnostics; the primary rate
averages complex coefficients coherently over all 40 seeds before taking the
envelope.

The registered arm-specific result above remains primary.  A secondary
common-window fit on the locked intersection `[5,30.1]` gives
`gamma_CIC=-0.0685811`, `gamma_C4=-0.0698686`, and the paired ratio
`1.018773 [1.015359,1.021931]` from 5000/5000 valid seed resamples.  This is a
2.94-percentage-point shift from the primary `0.989415` estimate and reverses
the small point shift about unity.  Both common-window arms remain below two
fitted e-foldings, so no gate or conclusion changes.

For compensated E2 at `(12,1)`, all four independently computed grid/time
point predictions lie below and outside the measured 95% intervals.  Relative
central discrepancies are 4.43% (p1/n1), 0.932% (p2/n2), 6.49% (p2/n4), and
0.545% (p3/n4).  The p2/n2 and p3/n4 measured intervals still pass the
separate 10% rate-preservation test; this is not evidence that the dispersion
prediction is calibrated to the narrower statistical intervals.

The archived E2 binomial tables repeated the CIC `fit_efoldings=5.406` value in
every filtered row.  The audit recomputes candidate duration as
`gamma_candidate * (59.68 - 24.32)`.  That correction changes the raw n=1 arm
to 0.087 e-foldings and the p3/n4 arm to 5.313; it does not relax a gate.

## Weak-Landau duration diagnosis

Reanalysis of recovered PPC-scaling histories confirms the failure:

| PPC | method | rate ratio `[95% CI]` | fitted e-foldings | verdict |
|---:|---|---|---:|---|
| 1024 | C4 sigma=1 | 0.99430 `[0.96292,1.05110]` | 1.491 | rate pass, duration fail |
| 4096 | C4 sigma=1 | 1.02078 `[0.99763,1.03065]` | 1.155 | rate pass, duration fail |

The apparent PPC-4096 improvement reported elsewhere as 1.809 “resolved
e-foldings” is `log(A0/(5 floor))`; the actual fitted product
`|gamma| * (22.1-5.0)` is 1.155.  Increasing PPC alone did not produce a usable
window in these quiet-start realizations.  Extending the same decayed,
floor-dominated signal cannot fix the gate.

## New, pre-registered MeluXina work

These jobs use current source files whose SHA-256 hashes were matched between
the local audited copy and `/home/users/u103576/hexapic-main/pic_kde_2d` before
submission.  Outputs are new directories below
`/mnt/tier2/users/u103576/paper1_validation`; no historical result is
overwritten.

1. Direct compensated E2 corner: array `5195640` and dependent reduction
   `5195641` completed with exit code zero. Paired seeds 0–9, PPC 512, alpha
   `1e-8`, `dt=0.02`, `t=60`, projected `(12,1)`, fixed window
   `[24.32,59.68]`; configurations CIC, p1/n1, p2/n2, p2/n4 and p3/n4. The
   reducer was first tested on the recovered Gaussian corner and reproduced
   all central rates to better than `1e-12`. Direct results are p1/n1
   `0.83807 [0.81930,0.84994]` FAIL, p2/n2
   `0.95608 [0.95120,0.95920]` PASS, p2/n4
   `0.78885 [0.76359,0.80469]` FAIL, and p3/n4
   `0.97342 [0.97047,0.97529]` PASS. Every arm passes the duration and signal
   tests and all 5000 paired resamples are valid. All 50 histories and reducer
   products were copied locally; the local and tier2 aggregate manifest hashes
   both equal `1376b97196782c33193fdb6c36f400579ffbab5bc4aa604bc98c57908f006b7f`.
2. Weak-duration signal pilot: array `5195658`, dependent reduction `5195659`,
   both completed successfully as jobs. Three paired seeds at PPC 1024 and
   alpha 0.02 were run for CIC/C4 and CIC/p1-n16. The late coherent floors were
   `0.00351` (CIC) and `0.00342` (C4), leaving both baselines below the
   five-floor threshold at the candidate fit start. The reducers therefore
   returned no finite rate, zero model e-foldings and zero candidate bootstrap
   quality. The pilot failed the signal precondition and was not expanded.
3. Exact LOO-CV width: array `5195682`, dependent reduction `5195683`, both
   completed with exit code zero.  Paired CIC and sigma
   `7.244293147028428` at PPC 1024, alpha 0.01, ten seeds and 2000 paired
   bootstraps give a rate ratio `1.82222 [1.48934,2.21398]`, `1.62678` fitted
   model e-foldings and `R^2=0.99501`.  Despite the high central-fit `R^2`, the
   fit is floor-censored and zero bootstrap replicates pass the complete
   quality test.  The row therefore fails rate, duration and signal quality.
   This closes the width/rate mismatch without assigning the 6.61 proxy to the
   selected width.  All 20 histories and reducer outputs were copied locally;
   per-file SHA-256 values match the tier2 originals.
4. Fresh global-loading confirmation: the fixed 40-seed PPC-1024 low arms and
   paired PPC-4096 CIC references were completed by jobs
   `5200457/58`, `5200485/86/87/88`; all 17 active Slurm elements exited zero.
   Stage 1 was frozen before any ratio was inspected.  CIC ended at `30.1`
   because of its persistent five-floor crossing at `32.6`, giving 1.721
   fitted e-foldings.  C4 ended at `35.6` because of its crossing at `38.1`,
   giving 2.076 e-foldings.  The locked Stage-2 ratio is
   `0.98942 [0.96788,1.01383]`, but the complete campaign fails the unchanged
   baseline-duration gate.  The data were not pooled with development seeds.
5. Historical delta-f recovery: a job-ID-guided search found complete raw
   per-seed trees for jobs `4872599`, `4872745`, and `4873221` below
   `/mnt/tier2/users/u103576/logs`.  The local preservation copy contains 30
   reference, 50 logical m12-sweep, and 140 band histories.  Current matching
   reducers reproduce the archived tables to at most `3.38e-14`; exact source
   revisions remain unknown.  This closes two traceability-only unresolved
   rows but changes no measured central value or confidence interval.

## Commands

Core local reductions and checks:

```text
python3 paper1_source_package/reproducibility/rebuild_rate_analysis.py \
  --output paper1_source_package/reproducibility/results/rate_uncertainty_audit_v8
python3 paper1_source_package/reproducibility/reassess_validation.py \
  --output paper1_source_package/reproducibility/results/validation_audit_v9
python3 -m unittest discover \
  -s paper1_source_package/reproducibility/tests -p 'test_*.py' -v
```

Cluster submission commands and the uploaded-file hashes are preserved in the
terminal audit trail and `AUDIT_CHANGELOG.md`.  The exact launchers and reducer
are in `reproducibility/meluxina/` and
`analyze_E2_compensated_corner.py`.

## Manuscript changes justified now

- Replaced the one-sided lower-bound wording by the required two-sided 10%
  interval.
- Changed sub-two-e-fold cells from “UNRESOLVED” to `FAIL (duration)` while
  preserving their separate conditional rate results.
- Removed the claim that one selector passes both benchmarks; the recovered
  evidence supports a complete E2 C4 pass but no complete weak-Landau pass.
- Relabelled “CI-safe” filter limits as “rate-CI-safe” and stated the weak
  duration failures in tables, captions, abstract and conclusion.
- Removed the assignment of the sigma-6.61 rate to the selected LOO-CV width
  7.2443 and replaced the previously unresolved row by its direct measured
  failure after jobs `5195682/83` completed.
- Replaced the strong-binomial “all tested pass” claim with the recovered
  window-sensitive n=16 result.
- Kept the historical longitudinal compensated E2 measurements separate from
  the new direct binding-corner measurements; verified-transfer dispersion
  predictions remain in a separate column and never determine a gate.
- Applied all evidence-supported manuscript corrections to the authoritative
  `Dynamics_constrained_Deposition/introduction.tex` identified by the user;
  the earlier package file is retained as a non-authoritative audit artifact.
- Added the fresh 40-seed global-loading confirmation without pooling the
  failed v1 or three-seed pilot. Reported the passing rate intervals, the C4
  duration pass, the CIC duration failure, and same-campaign density/field
  noise benefits separately; the complete weak-Landau verdict remains FAIL.

## Remaining unresolved issues

- The fresh 40-seed campaign establishes C4's conditional rate preservation and
  its own two-e-fold duration, but the paired CIC baseline supplies only 1.721
  fitted e-foldings.  A future campaign seeking a complete weak-Landau pass
  would need a predeclared design that lengthens the usable baseline window;
  adding seeds sequentially or extending only the decayed tail is not justified.
- Compensated-corner noise reductions currently come from the separately
  measured filter-family noise campaign. The new corner job measures rate only
  and must not be described as a new noise measurement.
- The sole remaining `UNRESOLVED` validation row is the Wu--Qin AD-gated case:
  it held `sigma=0` and reproduces CIC, so it does not exercise a smoothing
  method and cannot establish an adaptive-filter validation pass.
