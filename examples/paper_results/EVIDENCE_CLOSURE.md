# Common-window and evidence-closure audit

Audit date: 2026-09-11. No new simulations were run. Historical outputs and
the registered v2 Stage-1/Stage-2 products were not modified.

## Common-window sensitivity (secondary)

The registered arm-specific analysis remains primary:

| analysis | CIC window | C4 window | gamma CIC | gamma C4 | C4/CIC [paired 95% CI] | duration result |
|---|---|---|---:|---:|---:|---|
| registered primary | `[5,30.1]` | `[5,35.6]` | -0.0685811 | -0.0678552 | 0.989415 `[0.967884,1.013827]` | FAIL: CIC 1.721 e-foldings |
| common-window sensitivity | `[5,30.1]` | `[5,30.1]` | -0.0685811 | -0.0698686 | 1.018773 `[1.015359,1.021931]` | FAIL: CIC/C4 1.721/1.754 e-foldings |

The common-window reducer reads all 40 raw complex histories, preserves the
registered operation order, and uses 5000 paired seed-index resamples. All
draws are valid. Equalising the windows changes the point estimate by 0.02936
(2.94 percentage points) and reverses its direction about unity. Both
intervals remain inside `[0.90,1.10]`; neither supplies a complete duration
pass.

Machine evidence:
`results/weak_landau_confirmatory_v2_common_window_sensitivity_20260911/`.

## Historical loading

Job `4867515`, which supplied the Table-4 C4 and Silverman density-error
powers, used the same repeated per-cell `32x32` inverse-normal velocity tensor
as the rejected v1 confirmation. At PPC 1024 it has 32 distinct longitudinal
nodes, `|vx|max=2.1538747`, and zero nodes at or above the resonant phase
velocity `3.2126424`.

Those values remain valid measurements of density squared error for that
constructed ensemble. They are not evidence of resonant Landau dynamics and
must not be combined with global-loading rate histories as a same-campaign
validation. No recovered Landau rate history is identified as per-cell. The
oldest rate metadata predate `quiet_start_layout`, so their exact historical
source revisions remain unknown rather than inferred.

## E2 C4 signal and baseline duration

The historical joint reducer stored no explicit floor. Applying the existing
quietest two-time-unit rolling-median pre-emergence estimator to the ten
recovered C4 histories gives:

- floor `5.86025e-10` on `[8.16,10.12]`;
- ensemble-mean fit-start amplitude `6.57133e-9`;
- fit-start SNR `11.2134`, above the threshold 5;
- `R^2=0.995913`, 4.1131 candidate e-foldings, 5000/5000 valid paired
  resamples, and maximum delta-f ratio `0.0002667` in the fit window.

The CIC baseline shares `[24.32,59.68]` and spans 4.4837 e-foldings. Thus both
the explicit signal and the newer baseline-duration checks pass. In older
matrix rows, `NOT_APPLICABLE` meant only that the retrospective candidate-only
schema had no additional baseline gate; it did not mean an automatic baseline
pass. The weak-Landau v2 campaign instead preregistered separate checks because
the arm-specific censoring windows differ.

## Compensated E2 prediction calibration

| setting | grid/time prediction | measured ratio [95% CI] | relative central discrepancy | prediction in CI | measured 10% preservation |
|---|---:|---:|---:|---:|---:|
| p1/n1 | 0.800959 | 0.838068 `[0.819297,0.849935]` | 4.43% | no | fail |
| p2/n2 | 0.947175 | 0.956084 `[0.951195,0.959197]` | 0.932% | no | pass |
| p2/n4 | 0.737692 | 0.788855 `[0.763593,0.804685]` | 6.49% | no | fail |
| p3/n4 | 0.968109 | 0.973416 `[0.970473,0.975293]` | 0.545% | no | pass |

All four predictions are below and outside the measured 95% intervals. The
p2/n2 and p3/n4 settings remain useful under the separate measured 10% rate
preservation criterion; that finding is not a claim of interval-level model
calibration.

## Commands

```bash
python3 paper1_source_package/reproducibility/analyze_weak_landau_common_window_sensitivity.py \
  --input-low paper1_source_package/reproducibility/results/weak_landau_confirmatory_v2_production_20260911/low \
  --window-decisions paper1_source_package/reproducibility/results/weak_landau_confirmatory_v2_stage1_windows_20260911/window_decisions.json \
  --primary-result paper1_source_package/reproducibility/results/weak_landau_confirmatory_v2_stage2_analysis_20260911/confirmatory_result.json \
  --output /a/new/common-window/output

MPLCONFIGDIR=/tmp/paper1-mpl \
python3 paper1_source_package/reproducibility/audit_evidence_closure.py \
  --output /a/new/evidence-closure/output

python3 paper1_source_package/reproducibility/reassess_validation.py \
  --output /a/new/validation-output

python3 -m unittest discover \
  -s paper1_source_package/reproducibility/tests -p 'test_*.py' -v
```

The recorded regression result is 70/70 passing. The current validation matrix
is `results/validation_audit_v9`: 8 PASS, 31 FAIL and 1 UNRESOLVED, unchanged
in count from v7.

## Manuscript changes

The authoritative `Dynamics_constrained_Deposition/introduction.tex` now:

- labels the common-window result as secondary and the registered arm-specific
  result as primary;
- narrows the job-4867515 density-error interpretation;
- reports the recovered-history E2 C4 SNR and CIC baseline duration;
- explains retrospective `NOT_APPLICABLE` baseline cells; and
- separates compensated-filter prediction calibration from measured rate
  preservation.

The verified 36-page PDF is
`Dynamics_constrained_Deposition/output/Dynamics_constrained_Deposition_verified_20260911_r2.pdf`,
SHA-256 `627c09193ae6e1f4d807dcfe483ac5c68bcb13eb85a5be2a2baab4af0ae98c50`.

## Unresolved, without an automatic run request

- Exact source revisions for the oldest Python campaigns remain unknown.
- Other retrospective rows whose baseline-duration field is `NOT_APPLICABLE`
  establish only the candidate-arm condition stated in their source reducers;
  no baseline pass is inferred.
- The fresh weak-Landau confirmation still fails because its CIC baseline has
  only 1.721 fitted e-foldings. The sensitivity analysis does not repair it.
