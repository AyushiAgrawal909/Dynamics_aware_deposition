# Corrected existing-evidence interpretation

## Three-arm Landau result on `[5,28]`

The existing numerical results are retained unchanged:

| Arm | Coherent rate | Ratio to CIC, paired 95% interval | Fitted e-foldings | Duration gate |
|---|---:|---:|---:|---|
| CIC | `-0.0687909` | `1` | `1.582` | fail |
| one-cell Gaussian | `-0.0700391` | `1.01814 [1.01486,1.02121]` | `1.611` | fail |
| fixed CV-derived wide filter | `-0.129881` | `1.88805 [1.66753,2.14374]` | `2.987` | pass |

For each arm the exact same-window identity is

```text
slope log |mean_s(E_s)|
  = slope log mean_s(|E_s|)
  + slope log( |mean_s(E_s)| / mean_s(|E_s|) ).
```

The wide-minus-CIC change in the coherence term divided by the wide-minus-CIC
change in the coherent slope is `0.6381`, with paired interval
`[0.5745,0.6878]`. This is descriptive accounting for the saved ensemble, not a
causal attribution. The earlier toy calculation is not compared with this
fraction because it uses a different denominator and answers a different
question.

The first term above is the log-slope of the arithmetic mean of positive seed
magnitudes. It is not the arithmetic mean of the individual-seed log-amplitude
slopes. The coherent slope is also distinct: it includes phase cancellation in
the complex seed mean. The machine-readable corrected wording is in
`evidence/reconciled/landau_three_arm_summary_corrected_interpretation.json`.

## Density confidence interval discrepancy

The held-out central density ratio for q050_x000 is identically
`1.079880132634031` in both reductions. The archived candidate CSV reports
`[1.0778063603594845,1.081663262724321]`; the later report stated
`[1.077914994575217,1.081663262724321]`.

The discrepancy is only the Monte Carlo bootstrap stream. The archived reducer
uses one RNG initialized at `271828` and consumes density and field draws for
all 85 candidates in order before reaching q050_x000. The later retrieval
restarted a fresh stream at the selected candidate. Replaying the archived
stream reproduces the archived interval to floating-point precision. The
archived interval is therefore authoritative for that fixed-candidate result.
The nested complete-rule interval, `[1.0779122787198236,1.081703711461456]`,
remains separately labelled because it also resamples selection.

## Loading convergence remains separate

The proposed PPC scan uses nested positions and antithetic per-cell
beam/thermal quartets. It would measure convergence of that modified loading
scheme. It would not directly explain the original global-shuffle scheme's
rate spread without a bridge comparison at fixed PPC that changes only the
loading construction. The observed covariance/rate correlation is descriptive
and is not a causal demonstration. No loading-convergence run is part of the
exact-width campaign.
