# C4 closed-form dispersion validation

## Frozen transfer

For grid mode `m` on `N` cells, the matched CIC/filter/gather coupling is

```text
T_eff^2 = sinc(pi*m/N)^4 * exp[-(2*pi*m*sigma_dx/N)^2].
```

The same coupling is inserted into the kinetic Maxwellian dielectric for
Landau damping and the symmetric warm-beam dielectric for two-stream growth.
No transfer or rate parameter is fitted.

## Ten-percent rate ceilings

| benchmark | mode | analytic sigma_R/dx | comparison width/dx | predicted rate ratio at comparison width |
|---|---:|---:|---:|---:|
| Landau | 1 | 2.453438 | 1.00 | 1.016348 |
| warm two-stream | 8 | 1.062293 | 0.99 | 0.913842 |
| warm two-stream | 12 | 0.198720 | 0.22 | 0.875883 |

The Landau value is `2.453 dx`, not `2.6 dx`; the values at `2 dx` and `3 dx`
bracket the crossing. The analytic predictions at both widths lie inside the
existing paired measurement intervals, so the measured data do not reject the
kinetic dispersion shift.

The isolated `m=8` crossing closely reproduces the measured central threshold.
For the protected `m=12` edge, the fully consistent warm+CIC result places the
central 10% crossing at `0.199 dx`.

## One-point bias correction

At the measured isolated-`m=8` crossing, `sigma=0.99 dx`, the closed form gives
`gamma_sigma/gamma_0=0.913842`: 8.616% predicted damage where the measurement
defines 10%. In the local `damage proportional to sigma^2` regime, the
calibration is

```text
damage multiplier = 0.10 / 0.086158 = 1.1606
sigma divisor      = sqrt(1.1606)    = 1.0773
```

It maps the raw closed-form crossings as follows:

```text
m=8:  1.062293 / 1.0773 = 0.9860 dx  (measured: 0.99 dx)
m=12: 0.198720 / 1.0773 = 0.1845 dx  (prediction to test)
```

The `0.1845 dx` value was preregistered as the bias-corrected prior for the
linearized-delta-f comparison.

The primary statistical comparison uses all four nonzero sweep points to
compare the raw and bias-corrected rate-ratio curves with a paired bootstrap
RMSE difference. A two-width comparison is secondary. Before interpreting
either model, the measured ratios must decrease monotonically with bandwidth,
with every adjacent decrease supported by its paired interval. Failure of this
power pre-check makes the model result unresolved.

The sweep passed the power pre-check: every adjacent rate-ratio decrease has a
positive paired 95% lower bound. The raw model is decisively favored, with RMSE
`0.01493` versus `0.03071` and a paired corrected-minus-raw RMSE interval
`[0.01548, 0.01578]`. Thus the one-point `m=8` damage multiplier does not
transfer to `m=12`.

## Protected-band edge

The projected linearized-delta-f `m=1-14` sweep identifies clean exponential
growth for `m=2-12`. Mode `m=1` has unresolved positive growth and is not
classified as stable. Modes `m=13-14` have no resolved exponential growth:
their long-window log-amplitude fits have `R^2<0.01`, with final-to-initial
amplitude ratios `0.662` and `0.235`, respectively. The measured upper edge is
therefore `m=12`, which remains the binding protected mode. Its measured
unsmoothed rate is `0.15291 [0.14166, 0.16025]`, versus warm theory `0.14467`.

## Matrix implication

The production width is `0.1844549 dx`. On binding `m=12`, its measured rate
ratio is `0.92727 [0.91428, 0.93337]`, so the all-protected-mode rate gate
passes. The raw `0.1987204 dx` crossing is not confidence-supported because its
lower bound is `0.89984`. The paired full-f noise run at the exact production
width gives reduction `1.20050 [1.19768, 1.20338]`. Both E2 gates therefore
pass; the safe window is narrow but statistically nonzero.

## Mechanism

Smoothing reduces the effective plasma coupling and moves every mode toward
greater stability. Positive two-stream growth falls toward marginality; the
magnitude of negative Landau damping increases. These are opposite movements
of the signed rate but the same coupling mechanism. At large bandwidth, the
coherent mode is erased into the estimator floor and a fitted rate ceases to
represent the dispersion root.

## Reproduction

```bash
python3 scripts/validate_c4_closed_form.py
pytest -q tests/test_c4_closed_form.py
```

The production C4 E2 width is frozen at `0.18445488487247555 dx`.
