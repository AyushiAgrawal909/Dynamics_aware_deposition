# E2 estimator and seed-heterogeneity diagnostic

Audit date: 2026-09-12. This is a post-hoc sensitivity analysis. All registered
rates, windows, bootstraps and validation decisions remain primary and
unchanged.

## Estimands

The registered estimator is the slope of
`ln(mean_seed(abs(E_hat_s(t))))`. It estimates the growth of an arithmetic
ensemble-mean amplitude. The individual-rate diagnostic fits
`slope(ln(abs(E_hat_s(t))))` for every seed and averages those slopes. With the
same time samples and regression weights, that latter mean equals the slope of
the log geometric-mean amplitude. Neither estimand is automatically the
continuum eigenvalue.

For amplitudes `A_s exp(gamma_s t)`, the arithmetic mean increasingly weights
the faster-growing seeds. The fitted-exponential-mixture reconstruction and
the complete time-dependent seed shares verify this mechanism rather than
assuming it.

## Coverage and uncertainty

`complete_E2_estimator_comparison.py` covers all five completed drift values
and three arms, all seven historical Table-5 corner labels, and all five
direct compensated-corner arms from job 5195640. It writes individual rates,
fit diagnostics, time-dependent shares, effective seed counts and
leave-one-seed-out influence.

The labelled sensitivity intervals use 5,000 percentile resamples with RNG
seed 20260911. The unit is the paired seed; one index matrix is reused across
the baseline and candidate arms. Time samples are never resampled as
independent observations. With ten seeds, the intervals are conditional on
the observed seed sample and do not cover systematic or unobserved-tail error.

## Main estimator result at `v_b=0.80`

| Arm | registered mean-amplitude rate | mean individual rate (SD) | mixture reconstruction | effective seeds start -> end |
|---|---:|---:|---:|---:|
| CIC | 0.1447458990 | 0.1244508093 (0.0233284274) | 0.1447311054 | 7.405 -> 3.280 |
| fixed Gaussian | 0.1361532043 | 0.1131147140 (0.0253546119) | 0.1361416484 | 7.054 -> 3.080 |
| constrained Gaussian | 0.1403834906 | 0.1187833516 (0.0242867027) | 0.1403691373 | 7.234 -> 3.179 |

Thus the exponential mixture reproduces each registered slope within
`1.5e-5`, while the CIC largest seed share rises from 0.221 to 0.481. The
registered ratios remain 0.940636 (fixed) and 0.969862 (constrained). The
ratio-of-individual-rate-means sensitivities are respectively 0.908911 with
95% interval `[0.887432,0.926656]` and 0.954460 with interval
`[0.944068,0.963103]`. The constrained conclusion is insensitive to estimator;
the fixed arm sits on different sides of the 0.90 boundary under the two
estimands and must be described as estimator-sensitive.

## Direct compensated corner

All arms use the preserved `[24.32,59.68]` window and pass the registered
signal/duration checks. The registered and individual-rate sensitivities are:

| Candidate | registered ratio | ratio of individual-rate means (95% paired interval) | exact-transfer prediction |
|---|---:|---:|---:|
| `p=1,n=1` | 0.838068 | 0.812242 `[0.774277,0.839887]` | 0.800959 |
| `p=2,n=2` | 0.956084 | 0.949961 `[0.940921,0.956642]` | 0.947175 |
| `p=2,n=4` | 0.788855 | 0.752403 `[0.698837,0.791276]` | 0.737692 |
| `p=3,n=4` | 0.973416 | 0.969766 `[0.964359,0.973761]` | 0.968109 |

The `p=2,n=2` and `p=3,n=4` rate-preservation findings survive the estimator
sensitivity. Prediction calibration remains separate: all four predictions
are outside the narrow registered measured intervals, with registered central
discrepancies of 4.428%, 0.932%, 6.486%, and 0.545%, respectively.

## Cause of individual-seed differences

The saved `t=0` target-mode amplitude is identical across seeds within every
arm and its phase agrees to at worst `1.8e-15` rad. It therefore cannot explain
within-arm spread. A fixed, predeclared window grid changes the `v_b=0.80` CIC
mean individual rate from 0.123548 to 0.124696, far below its 0.023328 seed SD.
The corresponding `v_b=0.818` range is 0.048100--0.048539 versus SD 0.043118.
Fit-window choice and transient curvature are secondary, not dominant.

For the current audited initializer, the seed-specific principal-mode
finite-marker response explains the spread:

| Drift/arm | observed mean (SD) | finite-marker mean (SD) | correlation | RMSE |
|---|---:|---:|---:|---:|
| 0.800 CIC | 0.124451 (0.023328) | 0.124198 (0.023588) | 0.999885 | 0.000488 |
| 0.818 CIC | 0.048539 (0.043118) | 0.048258 (0.044263) | 0.997924 | 0.002897 |
| 0.818 fixed | 0.028131 (0.043480) | 0.027130 (0.044219) | 0.994422 | 0.004560 |
| 0.818 constrained | 0.047933 (0.043128) | 0.047677 (0.044218) | 0.997863 | 0.002910 |

At `v_b=0.80`, observed CIC rate correlates `-0.9465` with the reconstructed
beam-sign/thermal-velocity covariance. The finite-marker calculation includes
the actual equal marker weights, signed beam populations, projection and
explicit midpoint update; it is not an unweighted histogram. Its documented
approximation omits spatial aliases and finite-sample `+/-k` mixing.

Historical marker arrays and their exact source revision were not saved.
Seeds 0--9 can only be reconstructed with the current audited initializer and
are therefore nominal, not exact, historical loading diagnoses.

## Completed extension at `v_b=0.818`

Job 5202227 completed 30/30 requested histories within the unchanged eight-hour
amendment (elapsed 02:21:40). The original job 5201231 remains a separately
recorded 120/150 four-hour timeout.

The frozen mean-amplitude outputs are CIC 0.114044, fixed 0.101783 and
constrained 0.113689. They are not accepted physical rate measurements: every
arm starts below SNR 5, three of ten seeds in each arm cross the linear
`delta-f<0.5` gate, and the final effective seed count is only 1.02--1.04. The
registered fixed/CIC ratio 0.892489 and constrained/CIC ratio 0.996893 are
descriptive only. The individual-rate ratios are 0.579550
`[0.164060,0.803543]` and 0.987506 `[0.971357,0.994675]`, also descriptive.

The nominal fixed-filter dispersion has no positive growing branch, but its
finite-marker roots reproduce the observed seedwise behaviour. An increasing
arithmetic mean is therefore not evidence of a resolved continuum instability
or of physical stabilisation.

## Run decision

No higher-PPC history campaign was launched. Existing data identify a specific
finite-marker cause of the seed spread. Higher PPC would answer a narrower
question—convergence of the `v_b=0.818` finite-marker branch toward the
continuum marginal prediction—but that result is not necessary for the
narrowed paper claim.

Machine-readable outputs are under
`results/e2_estimator_comparison_complete_20260912_with_extension/`,
`results/e2_seed_rate_variation_20260912_with_extension/`, and
`results/e2_vb0818_extension_audit_20260912/`. The supporting figure and its
source CSVs are in `results/e2_estimator_seed_diagnostic_20260912/`.
